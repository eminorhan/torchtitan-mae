# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import logger
from torchtitan.parallelisms.parallel_dims import ParallelDims


def parallelize_dino(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise, the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if job_config.experimental.enable_async_tensor_parallel and not job_config.training.compile:
            raise RuntimeError("Async TP requires --training.compile")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=job_config.float8.enable_float8_linear,
            enable_async_tp=job_config.experimental.enable_async_tensor_parallel,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        if job_config.model.norm_type == "fused_rmsnorm":
            raise NotImplementedError("fused_rmsnorm is not compatible with torch.compile yet. Please use rmsnorm or layernorm.")
        apply_compile(model)

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_shard_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh = world_mesh["dp_replicate", "dp_shard"]
            else:
                dp_mesh = world_mesh["dp"]

            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            )
            
            if parallel_dims.dp_replicate_enabled:
                logger.info("Applied HSDP to the model")
            else:
                logger.info("Applied FSDP to the model")
        else:
            if world_mesh.ndim > 1:
                raise RuntimeError("DDP has not supported > 1D parallelism")
            
            apply_ddp(
                model,
                world_mesh,
                enable_compile=job_config.training.compile,
                enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
            )


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8: bool,
    enable_async_tp: bool,
):
    """
    Apply tensor parallelism (sharding the hidden/channel dimension) to the FeatureDecoder ViT model.
    """
    # 1. Manually parallelize the Conv3d patch embedding
    # We shard the weight/bias on the out_channels dim (dim 0)
    try:
        conv_layer = model.segmentation_model[0].feature_model.patch_embed.proj
        
        weight_sharding = [Shard(0)]
        bias_sharding = [Shard(0)]

        conv_layer.weight = nn.Parameter(
            distribute_tensor(conv_layer.weight, tp_mesh, weight_sharding)
        )
        if conv_layer.bias is not None:
            conv_layer.bias = nn.Parameter(
                distribute_tensor(conv_layer.bias, tp_mesh, bias_sharding)
            )
        # The output of this layer is now sharded on the channel dim (B, C/TP, D, H, W)
        # which becomes (B, S, C/TP) after flattening.
    except AttributeError as e:
        logger.error(f"Failed to parallelize patch_embed.proj: {e}")
        raise

    # 2. Parallelize the transformer blocks
    if enable_float8:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Our data layout is (B, S, C). We shard on dim 2.
    HIDDEN_DIM_SHARD = 2

    for transformer_block in model.segmentation_model[0].feature_model.blocks:
        # This plan implements standard Megatron-style TP
        layer_plan = {
            "norm1": prepare_module_input(
                input_layouts=([Shard(HIDDEN_DIM_SHARD)],),
                desired_input_layouts=([Replicate()],),
            ),
            "attn.qkv": colwise_parallel(), # Out: [Shard(2)]
            "attn.proj": rowwise_parallel(
                output_layouts=[Shard(HIDDEN_DIM_SHARD)] # Out: [Shard(2)]
            ),
            
            "norm2": prepare_module_input(
                input_layouts=([Shard(HIDDEN_DIM_SHARD)],),
                desired_input_layouts=([Replicate()],),
            ),
            "mlp.w1": colwise_parallel(), # Out: [Shard(2)]
            "mlp.w2": colwise_parallel(), # Out: [Shard(2)]
            "mlp.w3": rowwise_parallel(
                output_layouts=[Shard(HIDDEN_DIM_SHARD)] # Out: [Shard(2)]
            ),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    # 3. Parallelize the final LayerNorm
    # Input is Shard(2) from blocks, output needs to be Replicate for the final Conv3d
    final_norm_plan = {
        "": prepare_module_input(
            input_layouts=([Shard(HIDDEN_DIM_SHARD)],),
            desired_input_layouts=([Replicate()],),
        )
    }
    parallelize_module(model.segmentation_model[0].feature_model.norm, tp_mesh, final_norm_plan)

    # 4. Manually parallelize the final Conv3d
    # Input is now Replicated from the final norm.
    # We shard the weight/bias on the out_channels dim (dim 0).
    try:
        final_conv = model.segmentation_model[1].conv
        
        weight_sharding = [Shard(0)]
        bias_sharding = [Shard(0)]

        final_conv.weight = nn.Parameter(
            distribute_tensor(final_conv.weight, tp_mesh, weight_sharding)
        )
        if final_conv.bias is not None:
            final_conv.bias = nn.Parameter(
                distribute_tensor(final_conv.bias, tp_mesh, bias_sharding)
            )
        
        # This manual sharding results in a channel-sharded output (Shard(1)).
        # This is equivalent to `loss_parallel=True`.
        if not loss_parallel:
            logger.warning(
                "`loss_parallel=False` is not supported for the final manually "
                "sharded Conv3d. The output will remain sharded."
            )
            
    except AttributeError as e:
        logger.error(f"Failed to parallelize segmentation_model.1.conv: {e}")
        raise

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''} Channel-Sharding Tensor Parallelism to the model")

# TODO: check TP implementation. Also NOTE: the plan here is specific to the DINO + linear head architecture 
# def apply_tp(
#     model: nn.Module,
#     tp_mesh: DeviceMesh,
#     loss_parallel: bool,
#     enable_float8: bool,
#     enable_async_tp: bool,
# ):
#     """Apply tensor parallelism to the FeatureDecoder ViT model."""
#     # 1. Parallelize the patch embedding and shard its outputs
#     # 2. Parallelize the root norm layer of the transformer over the sequence dim
#     # 3. Parallelize the final linear output layer (1x1 Conv)
#     # NOTE: FQN paths are used here for clarity and precision.
#     root_plan = {
#         # The Conv2d patch embedding is like the token embedding.
#         # Its weights are replicated, but its output is sharded along the channel dimension.
#         # This is analogous to sharding the embedding dimension in an LLM.
#         "segmentation_model.0.feature_model.patch_embed.proj": RowwiseParallel(
#             input_layouts=Replicate(),
#             output_layouts=Shard(1), # Shard the feature/channel dimension
#         ),
#         # Final norm of the transformer backbone
#         "segmentation_model.0.feature_model.norm": SequenceParallel(),
#         # The final 1x1 Conv head is equivalent to a linear layer.
#         # It takes replicated input and shards its weights column-wise.
#         "segmentation_model.1.conv": ColwiseParallel(
#             input_layouts=Replicate(),
#             output_layouts=Shard(-1) if loss_parallel else Replicate(),
#             use_local_output=not loss_parallel,
#         ),
#     }
#     parallelize_module(model, tp_mesh, root_plan)

#     # Parallel styles used for transformer block linear weights and their inputs may be different for float8 linears
#     if enable_float8:
#         # TODO(vkuzo): once float8 configuration supports delayed scaling,
#         # add a check here to enforce supported float8 all-gather configurations
#         # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
#         from torchao.float8.float8_tensor_parallel import Float8ColwiseParallel, Float8RowwiseParallel, PrepareFloat8ModuleInput

#         rowwise_parallel, colwise_parallel, prepare_module_input = Float8RowwiseParallel, Float8ColwiseParallel, PrepareFloat8ModuleInput
#     else:
#         rowwise_parallel, colwise_parallel, prepare_module_input = RowwiseParallel, ColwiseParallel, PrepareModuleInput

#     # Apply tensor + sequence parallelism to every transformer block
#     # Loop through the `ModuleList` of SelfAttentionBlocks
#     for transformer_block in model.segmentation_model[0].feature_model.blocks:
#         layer_plan = {
#             # First LayerNorm in the block
#             "norm1": SequenceParallel(),
#             # The QKV projection is a single linear layer
#             "attn.qkv": colwise_parallel(),
#             # The output projection
#             "attn.proj": rowwise_parallel(output_layouts=Shard(1)),
#             # Second LayerNorm in the block
#             "norm2": SequenceParallel(),
#             # Prepare the MLP input by all-gathering the sharded input from the residual connection
#             "mlp": prepare_module_input(
#                 input_layouts=(Shard(1),),
#                 desired_input_layouts=(Replicate(),),
#             ),
#             # In SwiGLUFFN, w1 and w2 are parallel up-projections
#             "mlp.w1": colwise_parallel(),
#             "mlp.w2": colwise_parallel(),
#             # w3 is the down-projection
#             "mlp.w3": rowwise_parallel(output_layouts=Shard(1)),
#         }

#         parallelize_module(
#             module=transformer_block,
#             device_mesh=tp_mesh,
#             parallelize_plan=layer_plan,
#         )

#     if enable_async_tp:
#         from torch.distributed._symmetric_memory import enable_symm_mem_for_group

#         torch._inductor.config._micro_pipeline_tp = True
#         enable_symm_mem_for_group(tp_mesh.get_group().group_name)

#     logger.info(f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''} Tensor Parallelism to the model")


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def _apply_ac_to_transformer_block(module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}")

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(f"Invalid selective AC option: {ac_config.selective_ac_option}. Valid options: 'op' or a positive int representing layer frequency")
    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.segmentation_model[0].feature_model.blocks.named_children():  # apply AC to SelfAttentionBlocks TODO: check this
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_config)
        model.segmentation_model[0].feature_model.blocks.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each SelfAttentionBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.segmentation_model[0].feature_model.blocks.named_children():  # apply torch.compile to SelfAttentionBlocks TODO: check this
        transformer_block = torch.compile(transformer_block, mode="default", fullgraph=True)
        model.segmentation_model[0].feature_model.blocks.register_module(layer_id, transformer_block)

    logger.info("Compiling each SelfAttentionBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for transformer_block in model.segmentation_model[0].feature_model.blocks:  # TODO: is this correct?
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=False)
        
    fully_shard(model, **fsdp_config, reshard_after_forward=False)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")