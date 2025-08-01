# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging import logger


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    tp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, tp = self.dp_replicate, self.dp_shard, self.tp
        for d in (dp_replicate, tp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // tp
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert tp >= 1, tp
        assert dp_replicate * dp_shard * tp == self.world_size, f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * tp({tp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip([self.dp_replicate, self.dp_shard, self.tp], ["dp_replicate", "dp_shard", "tp"]):
            if d > 1:
                dims.append(d)
                if (name == "dp_replicate" and self.dp_shard == 1) or (name == "dp_shard" and self.dp_replicate == 1):
                    names.append("dp")
                else:
                    names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are initialized
        if self.dp_replicate > 1 and self.dp_shard > 1:
            mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")
        
        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.tp