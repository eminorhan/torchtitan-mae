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
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import parallelize_dino, ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchvision.utils import save_image

from dinov3.eval.segmentation.models import build_segmentation_decoder

# visualization related imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import imageio
import io


def print_parameter_status(model):
    """
    Iterates over all named parameters of a PyTorch model and prints their
    name and whether they require a gradient (i.e., are being trained).
    """
    logger.info("Parameter Training Status:")
    logger.info("-" * 30)
    for name, param in model.named_parameters():
        status = "TRAINING" if param.requires_grad else "FROZEN"
        logger.info(f"{name:<50} | Requires Grad: {param.requires_grad} ({status})")
    logger.info("-" * 30)
    logger.info("\n")


def visualize_slices_3d(
        inputs: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        step: int,
        sample_idx: int = 0,
        overlay_alpha: float = 0.3,
        fps: int = 10
    ):
    """
    Visualizes slices from a 3D volume as a side-by-side GIF animation of predictions and ground truth.

    Args:
        inputs (torch.Tensor): The input volume (B, C, D, H, W). Assumes first channel is image data.
        preds (torch.Tensor): The model output logits (B, num_classes, D, H, W).
        targets (torch.Tensor): The ground truth labels (B, 1, D, H, W).
        num_classes (int): The total number of segmentation classes.
        step (int): Step number, used for saving the output file.
        sample_idx (int): The index of the sample in the batch to visualize.
        overlay_alpha (float): Transparency of the mask overlay (0.1 is faint, 0.5 is half-opaque).
        fps (int): Frames per second for the output GIF.
    """
    # Convert logits and select sample 
    pred_masks = torch.argmax(preds, dim=1)  # Shape: (B, D, H, W)

    input_sample = inputs[sample_idx]
    pred_mask_sample = pred_masks[sample_idx]
    target_mask_sample = targets[sample_idx]

    # Move tensors to CPU and numpy
    input_image = input_sample[0].cpu().numpy()
    pred_mask = pred_mask_sample.cpu().numpy()
    # Squeeze the channel dimension (1) from the target mask
    target_mask = target_mask_sample.squeeze(0).cpu().numpy() # Shape (D, H, W)

    # Create a consistent colormap
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0])  # Set background class (index 0) to transparent
    custom_cmap = ListedColormap(new_colors)
    bounds = np.arange(-0.5, num_classes, 1)
    norm = BoundaryNorm(bounds, custom_cmap.N)

    # Generate frames for the GIF
    frames = []
    depth = input_image.shape[0]

    for slice_idx in range(0, depth, 4):  # plot every k-th slice
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # fig.suptitle(f"Slice {slice_idx + 1} / {depth}", fontsize=14)

        # Left Plot: Predictions
        ax = axes[0]
        ax.imshow(input_image[slice_idx], cmap='gray')
        ax.imshow(pred_mask[slice_idx], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title("Prediction")
        ax.axis('off')

        # Right Plot: Ground truth
        ax = axes[1]
        ax.imshow(input_image[slice_idx], cmap='gray')
        ax.imshow(target_mask[slice_idx], cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title("Ground truth")
        ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        # Convert the matplotlib plot to an image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        
        # Close the figure to free up memory
        plt.close(fig)

    # Save the frames as a GIF animation
    output_filename = f"step_{step}_3d_animation.gif"
    imageio.mimsave(output_filename, frames, fps=fps)
    print(f"Saved animation to {output_filename}")


def visualize_slices_2d(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    step: int,
    overlay_alpha: float = 0.3
    ):
    """
    Visualizes all 2D images in a batch with their predicted and ground truth masks.

    Args:
        inputs (torch.Tensor): The input image tensor (B, C, H, W). Assumes the first channel is the image data.
        preds (torch.Tensor): The model output logits (B, num_classes, H, W).
        targets (torch.Tensor): The ground truth labels (B, 1, H, W).
        num_classes (int): The total number of segmentation classes.
    """
    # Get batch size
    batch_size = inputs.shape[0]

    # Convert prediction logits to discrete class labels
    pred_masks = torch.argmax(preds, dim=1)  # Shape: (B, H, W)

    # Create a consistent colormap for all classes
    colors = plt.cm.get_cmap('gist_ncar', num_classes)
    new_colors = colors(np.linspace(0, 1, num_classes))
    new_colors[0, :] = np.array([0, 0, 0, 0])
    custom_cmap = ListedColormap(new_colors)
    bounds = np.arange(-0.5, num_classes, 1)
    norm = BoundaryNorm(bounds, custom_cmap.N)

    # Set up the plot for the entire batch. `squeeze=False` ensures axes is always 2D.
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8.5), squeeze=False)

    # Loop through each sample in the batch
    for i in range(batch_size):
        # Prepare data for the i-th sample
        input_image = inputs[i, 0].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()
        target_mask = targets[i].cpu().numpy() # Extract from channel dim

        # Top Row: Prediction
        ax = axes[0, i]
        ax.imshow(input_image, cmap='gray')
        ax.imshow(pred_mask, cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title(f"Prediction (Sample {i})")
        ax.axis('off')

        # Bottom Row: Ground truth
        ax = axes[1, i]
        ax.imshow(input_image, cmap='gray')
        ax.imshow(target_mask, cmap=custom_cmap, norm=norm, alpha=overlay_alpha)
        ax.set_title(f"Ground truth (Sample {i})")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"step_{step}_2d.jpeg", bbox_inches='tight')
    plt.close(fig)


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

    assert len(job_config.model.crop_size) in (2, 3), f"model.crop_size must have 2 or 3 elements, but got {len(job_config.model.crop_size)}"

    # build dataloader
    data_loader = build_data_loader(
        job_config.training.batch_size,
        job_config.data.dataset_folder,
        tuple(job_config.model.crop_size),
        dp_rank,
        job_config.data.base_seed,
        job_config.data.augment
    )

    # build model skeleton (TODO: maybe try 'meta' init here). NOTE: we load the pretrained weights during ckpt.load() below
    backbone = torch.hub.load(job_config.model.dinov3_repo_folder, job_config.model.backbone, source="local", use_fa3=job_config.model.use_fa3, pretrained=False)
    model = build_segmentation_decoder(backbone, decoder_type=job_config.model.head, num_classes=job_config.model.num_classes)

    if torch.distributed.get_rank() == 0:
        logger.info(f"Model: {model}")  # check if the parameters are being trained or frozen

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # parallelization: apply PT-D TP, activation checkpointing, torch.compile, DP
    parallelize_dino(model, world_mesh, parallel_dims, job_config)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    model.to(device=init_device)
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
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # cross-entropy loss for 2D or 3D data
    if len(job_config.model.crop_size) == 2:
        def loss_fn(preds, labels):
            # resample predictions if necessary
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = torch.nn.functional.interpolate(input=preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            return torch.nn.functional.cross_entropy(preds, labels)
    else:
        def loss_fn(preds, labels):
            # resample predictions if necessary
            if preds.shape[-3:] != labels.shape[-3:]:
                preds = torch.nn.functional.interpolate(input=preds, size=labels.shape[-3:], mode="trilinear", align_corners=False)
                # preds = preds.clamp(min=-100., max=100.)
            return torch.nn.functional.cross_entropy(preds, labels)

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )

    if torch.distributed.get_rank() == 0:
        print_parameter_status(model)  # check if the parameters are being trained or frozen

    with maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler, maybe_enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            inputs, targets = next(data_iterator)
            data_loading_times.append(time.perf_counter() - data_load_start)

            inputs = inputs.cuda()
            targets = targets.cuda()

            optimizers.zero_grad()
            
            # run forward / backward
            with train_context():
                preds = model(inputs)
                loss = loss_fn(preds, targets)
                # need to free before bwd to avoid peaking memory
                del preds
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

            # # ###### visualize (NOTE: this is for debug purposes, will be removed later)
            # if train_state.step % job_config.metrics.log_freq == 0:
            #     model.eval()
            #     total_val_loss = 0
            #     num_val_samples = 0
            #     with torch.no_grad():
            #         # TODO: a bit hacky, ideally we should have separate train/val loaders
            #         for val_inputs, val_targets in data_loader.dataset.validation_iterator(): 
            #             val_inputs = val_inputs.unsqueeze(0).cuda()
            #             val_targets = val_targets.unsqueeze(0).cuda()
            #             val_preds = model(val_inputs) 
            #             val_loss = loss_fn(val_preds, val_targets)
            #             total_val_loss += val_loss.item()
            #             num_val_samples += 1
        
            #         avg_val_loss = total_val_loss / num_val_samples
            #         avg_val_loss = utils.dist_mean(avg_val_loss, dp_mesh)  # reduce val loss across ranks

            #         # visualize some examples
            #         if torch.distributed.get_rank() == 0:
            #             logger.info(f"--- Validation at step {train_state.step}: Average validation loss = {avg_val_loss} ---")
                        
            #             if len(job_config.model.crop_size) == 2:
            #                 # Use the detached tensor
            #                 val_preds = torch.nn.functional.interpolate(
            #                     input=val_preds, 
            #                     size=val_targets.shape[-2:], 
            #                     mode="bilinear", 
            #                     align_corners=False
            #                 )
            #                 visualize_slices_2d(
            #                     val_inputs,
            #                     val_preds,
            #                     val_targets,
            #                     job_config.model.num_classes,
            #                     train_state.step
            #                 )
            #             else:
            #                 # Use the detached tensor
            #                 val_preds = torch.nn.functional.interpolate(
            #                     input=val_preds, 
            #                     size=val_targets.shape[-3:], 
            #                     mode="trilinear", 
            #                     align_corners=False
            #                 )
            #                 visualize_slices_3d(
            #                     val_inputs,
            #                     val_preds,
            #                     val_targets,
            #                     job_config.model.num_classes,
            #                     train_state.step,
            #                 )
                            
            #             # Delete preds_vis var to free memory immediately
            #             del val_preds

            #     model.train()
            # # ###### end visualize

            # log train metrics
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

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
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
                )

                losses_since_last_log.clear()
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