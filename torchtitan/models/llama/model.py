# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.norms import build_norm

import flash_attn_interface


@dataclass
class ModelArgs:
    dim: int = 2048
    decoder_dim: int = 512
    n_layers: int = 16
    decoder_n_layers: int = 4
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6
    rope_theta: float = 1_000_000
    max_seq_len: int = 2048
    mask_ratio: float = 0.95
    depth_init: bool = True  # if True, each transformer block init uses its layer ID; otherwise, each uses the total number of transformer blocks


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class PatchEmbedding(nn.Module):
    """Patch embedding layer"""
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 8,
        num_channels: int = 1,
        embed_dim: int = 2048
    ):
        super().__init__()
        img_size = (img_size,) * 3
        patch_size = (patch_size,) * 3
        assert img_size[0] % patch_size[0] == 0, "Image size must be divisible by patch size (dim=0)."
        assert img_size[1] % patch_size[1] == 0, "Image size must be divisible by patch size (dim=1)."
        assert img_size[2] % patch_size[2] == 0, "Image size must be divisible by patch size (dim=2)."
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2)
        x = torch.einsum("bcs->bsc", x)  # [B, D*H*W, dim]
        return x


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.
    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual local heads 
        # from sizes of xq, xk, and xv as TP may have sharded them after the above linear ops
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # FlashAttention-3
        output, _ = flash_attn_interface.flash_attn_func(xq, xk, xv, causal=False)
        output = output.contiguous().view(bs, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        n_layers (int): Number of layers in the model.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.n_layers = model_args.n_layers
        self.img_size = model_args.img_size
        self.patch_size = model_args.patch_size
        self.num_channels = model_args.num_channels

        self.patch_embedding = PatchEmbedding(model_args.img_size, model_args.patch_size, model_args.num_channels, model_args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.init_weights()

    def init_weights(self):
        """
        [Note on ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        if self.patch_embedding is not None:
            nn.init.normal_(self.patch_embedding.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()

    def random_mask(self, x: torch.Tensor, mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence, L is the total number of patches
        """
        B, L, dim = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1] TODO: check that this is correct

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep  # TODO: do we need all of these outputs?

    def forward(self, imgs: torch.Tensor, freqs_cis: torch.Tensor, mask_ratio: float):
        """
        Perform a forward pass through the TransformerEncoder model.
        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
        """
        x = self.patch_embedding(imgs)
        B, L, dim = x.shape

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_mask(x, mask_ratio)  # TODO: do we need all of these outputs?

        for layer in self.layers.values():
            x = layer(x, freqs_cis)

        x = self.norm(x) if self.norm else x
        return x, mask, ids_restore

    # do I still need the following method?
    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "TransformerEncoder":
        """
        Initialize a TransformerEncoder model from a ModelArgs object.
        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            TransformerEncoder: TransformerEncoder model.
        """
        return cls(model_args)


class TransformerDecoder(nn.Module):

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.decoder_n_layers = model_args.decoder_n_layers

        self.embedding = nn.Linear(model_args.dim, model_args.decoder_dim, bias=False)
    
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.decoder_n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.decoder_dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.decoder_dim, model_args.patch_size**3 * model_args.num_channels, bias=False)
        self.init_weights()

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
            if self.output.bias is not None:
                nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Perform a forward pass through the TransformerDecoder model.
        Args:
            x (torch.Tensor): tokens input to decoder.

        Returns:
            torch.Tensor: Output logits after applying the TransformerDecoder model.
        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.embedding(x)
        for layer in self.layers.values():
            h = layer(h, freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "TransformerDecoder":
        """
        Initialize a TransformerDecoder model from a ModelArgs object.
        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            TransformerDecoder: TransformerDecoder model.
        """
        return cls(model_args)


# TODO: consolidate all architectural parameters under model_args?
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args
        self.encoder = TransformerEncoder(model_args)
        self.decoder = TransformerDecoder(model_args)
        
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)  # precompute pos embeddings (this will register self.freqs_cis)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    def patchify(self, imgs: torch.Tensor):
        """
        Make patches from a batch of images. 

        TODO: Simplify this function by assuming all sides must be equal.
        
        Args:
            imgs: (B, C, D, H, W)

        Returns:    
            x: (B, L, patch_size**3 * C) where L is the number of patches
        """
        B, C, D, H, W = imgs.shape
        p = self.encoder.patch_size
        assert D == H == W and D % p == 0
        d = h = w = D // p

        x = imgs.reshape(shape=(B, C, d, p, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(B, d * h * w, p**3 * C))
        self.patch_info = (B, D, H, W, p, d, h, w)  # this is not the best way to do this, I think
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        Make a proper image from a collection of patches (the inverse of above).
        
        TODO: Simplify this function by assuming all sides must be equal.

        Args: 
            x: (B, L, patch_size**3 * C)

        Returns:    
            imgs: (B, C, D, H, W)
        """
        B, D, H, W, p, d, h, w = self.patch_info
        x = x.reshape(shape=(B, d, h, w, p, p, p, C))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(B, C, D, H, W))
        return imgs

    def forward(self, imgs: torch.Tensor):
        """
        Forward pass through the full MAE model
        """
        x, mask, ids_restore = self.encoder(imgs, self.freqs_cis, self.model_args.mask_ratio)
        preds = self.decoder(x, self.freqs_cis, ids_restore)
        targets = self.patchify(imgs)  # target patches
        loss = (preds - targets) ** 2  # MSE loss
        loss = loss.mean(dim=-1)  # (B, L) mean loss per patch
        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
