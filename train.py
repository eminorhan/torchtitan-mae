# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time
from datetime import timedelta

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.model import MaskedAutoencoder, model_configs
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import parallelize_mae, ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

from torchvision.utils import save_image


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))
            yield

    return context


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):

    # set up logger
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # set determinism, use seed == None to skip deterministic training
    utils.set_determinism(job_config.training.seed)

    # init distributed
    world_size = int(os.environ['WORLD_SIZE'])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)

    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    
    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    # build dataloader
    data_loader = build_data_loader(
        job_config.training.data_dir,
        job_config.training.subdir_name,
        job_config.training.batch_size,
        (job_config.model.img_size, job_config.model.img_size, job_config.model.img_size),
        job_config.training.resolution,
        job_config.training.num_workers,
        dp_degree,
        dp_rank,
    )

    # build model (using meta init)
    model_config = model_configs[job_config.model.size]
    model_config.img_size = job_config.model.img_size
    model_config.patch_size = job_config.model.patch_size
    model_config.mask_ratio = job_config.model.mask_ratio

    with torch.device("meta"):
        model = MaskedAutoencoder.from_model_args(model_config)

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    eff_seq_len = int((1 - model_config.mask_ratio) * (model_config.img_size // model_config.patch_size) ** 3)
    num_flop_per_token = utils.get_num_flop_per_token(utils.get_num_params(model), model_config, eff_seq_len)  # TODO: is this still ~correct for MAE architecture?

    # parallelization: apply PT-D TP, activation checkpointing, torch.compile, DP
    parallelize_mae(model, world_mesh, parallel_dims, job_config)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    model.to_empty(device=init_device)
    model.init_weights()
    model.train()

    model_parts = [model]

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(f"GPU memory usage for model: {gpu_mem_stats.max_reserved_gib:.2f}GiB ({gpu_mem_stats.max_reserved_pct:.2f}%)")
    logger.info(f"Total number of parameters: {utils.get_num_params(model)}")

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load()

    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be plotted. This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {"loss_metrics/global_avg_loss": train_state.global_avg_losses[idx], "loss_metrics/global_max_loss": train_state.global_max_losses[idx]}
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)
    train_context = get_train_context(parallel_dims.loss_parallel_enabled, job_config.experimental.enable_compiled_autograd)

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"effective sequence length {eff_seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler, maybe_enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            ntokens_since_last_log += int(batch.shape[0] * (model_config.img_size // model_config.patch_size) ** 3)
            data_loading_times.append(time.perf_counter() - data_load_start)

            batch = batch.cuda()
            optimizers.zero_grad()

            # ###### visualize (NOTE: this is for debug purposes, will be removed later)
            model.eval()
            with torch.no_grad():
                _, comparison = model(batch, visualize=True)

                if torch.distributed.get_rank() == 0:

                    comparison = comparison[0].permute(0, 2, 1, 3, 4)

                    a = comparison[0, ::(model_config.img_size // 8), :, :, :]
                    b = comparison[1, ::(model_config.img_size // 8), :, :, :]
                    c = comparison[2, ::(model_config.img_size // 8), :, :, :]

                    vis = torch.cat((a, b, c), 0)
                    vis = vis.expand(-1, 3, -1, -1)

                    save_image(vis, f'sample.jpg', nrow=8, padding=1, normalize=True, scale_each=True)
            model.train()
            # ###### end visualize
            
            # run forward / backward
            with train_context():
                loss = model(batch)
                loss.backward()
            
            # clip gradients
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(m.parameters(), job_config.training.max_norm, foreach=True)

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # log metrics
            if (train_state.step == 1 or train_state.step % job_config.metrics.log_freq == 0):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = utils.dist_mean(avg_loss, dp_mesh), utils.dist_max(max_loss, dp_mesh)
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (time_delta * parallel_dims.model_parallel_size)
                # model FLOPS utilization; for its definition and calculation, please refer to the PaLM paper: https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.red}lr: {optimizers.optimizers[0].param_groups[0]['lr']:.6f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}wps: {round(wps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(train_state.step, force=(train_state.step == job_config.training.steps))

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(timeout=timedelta(seconds=job_config.comm.train_timeout_seconds), world_mesh=world_mesh)

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()