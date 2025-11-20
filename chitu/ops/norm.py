# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F

from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    LightweightModule
)
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.ops.utils import compatible_with_inplace

triton, has_triton = try_import_platform_dep("triton")
if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import rms_norm_triton
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
tbsgemm, has_tbsgemm = try_import_opt_dep("tbsgemm", "muxi_w8a8_kernels")

# Try to import ttnn for Tenstorrent device support
ttnn, has_ttnn = try_import_platform_dep("ttnn")


@compatible_with_inplace
def rms_norm_cpu(X: torch.Tensor, W: torch.Tensor, *, eps, compute_dtype: torch.dtype):
    if X.device.type != "cpu":
        raise ValueError(
            f"rms_norm input tensor must be on CPU, got device: {X.device}"
        )
    if W.device.type != "cpu":
        raise ValueError(
            f"rms_norm weight tensor must be on CPU, got device: {W.device}"
        )

    if not X.is_contiguous():
        X = X.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()

    hidden_size = X.shape[-1]
    batch_size = X.numel() // hidden_size
    output = torch.empty(X.shape, dtype=X.dtype, device="cpu").contiguous()

    config = cpuinfer.rmsnorm.RMSNormConfig(
        hidden_size,
        1024,  # Default max sequence length
        eps,
        W.data_ptr(),
        get_ggml_quant_type(X),
        get_ggml_quant_type(W),
        get_ggml_quant_type(output),
    )

    rms_norm = cpuinfer.rmsnorm.RMSNorm(config)

    cpu_infer = get_cpu_infer()
    cpu_infer.submit(rms_norm.forward(batch_size, X.data_ptr(), output.data_ptr()))
    cpu_infer.sync()

    return output


@compatible_with_inplace
def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    x = x.to(compute_dtype)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (y.to(weight.dtype) * weight).to(dtype)


@compatible_with_inplace
def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    return F.rms_norm(x.to(compute_dtype), (weight.numel(),), weight, eps).to(dtype)


@compatible_with_inplace
def rms_norm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    dtype = x.dtype
    return torch_npu.npu_rms_norm(x.to(weight.dtype), weight, epsilon=eps)[0].to(dtype)


def rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    eps,
    compute_dtype: torch.dtype,
):
    # Currently, this kernel always raise to float32 to compute
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if out is not None:
        out = out.view(-1, out.shape[-1])
    out = chitu_backend.cuda_rms_norm(x, weight, eps=eps, out=out)
    return out.view(x_shape)


@compatible_with_inplace
def rms_norm_muxi(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps,
    compute_dtype: torch.dtype,
):
    # Currently, this kernel always raise to float32 to compute
    assert eps == 1e-6
    assert x.dtype == torch.float16
    return tbsgemm.norm(x, weight)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    eps,
    compute_dtype: torch.dtype,
    impl: str = "auto",
):
    if impl == "auto":
        if has_cpuinfer and get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        elif out is not None and has_chitu_backend:
            impl = "cuda"
        elif has_tbsgemm and get_global_args().dtype == "float16" and eps == 1e-6:
            impl = "muxi_w8a8_kernels"
        elif has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "torch_npu"
        elif hasattr(F, "rms_norm"):
            impl = "torch"
        else:
            impl = "ref"

    if impl == "triton":
        return rms_norm_triton(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "cpu":
        return rms_norm_cpu(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "cuda":
        return rms_norm_cuda(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "muxi_w8a8_kernels":
        return rms_norm_muxi(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "torch_npu":
        return rms_norm_npu(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "torch":
        return rms_norm_torch(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    elif impl == "ref":
        return rms_norm_ref(x, weight, out=out, eps=eps, compute_dtype=compute_dtype)
    else:
        raise ValueError(f"Invalid RMSNorm implementation: {impl}")


# Tenstorrent (ttnn) specific RMSNorm class
# This is integrated from tt_qwen/models/rmsnorm.py
if has_ttnn:
    TILE = 32
    SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile

    class TTRMSNorm(LightweightModule):
        """
        RMSNorm supporting replication over a MeshDevice and sharding within devices.
        This is the Tenstorrent-specific implementation integrated from tt_qwen.

        This class implements a Root Mean Square Normalization (RMSNorm) that can be
        distributed across multiple devices and cores. If the `device` parameter is a
        MeshDevice, the weights and computations are replicated across all devices in
        the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

        Args:
            device: The device or MeshDevice on which to perform the computations.
            state_dict: The state dictionary containing the model parameters.
            dim: Input dimension (e.g. model hidden dimension size).
            layer_num: The layer number to determine the weight key in the state dictionary.
            weight_key: The key for retrieving the weight from the state dictionary.
            weight_cache_path: Optional path for caching the tilized weights.
            weight_memory_config: Configuration for the weight memory, default is DRAM_MEMORY_CONFIG.
            weight_dtype: The data type for the tensors, bfp8_b hits >0.999 PCC in the models we tested.
            model_config: Optional configuration dictionary for the model.
            eps (float): Small value to avoid division by zero in normalization, default is 1e-05.

        If model_config is provided, it must specify SHARDED_NORM_INPUT_MEMCFG, SHARDED_NORM_PRGM_CFG
        and SHARDED_NORM_OUTPUT_MEMCFG. If not provided, default configurations will be generated.
        """

        def __init__(
            self,
            device,
            dim,
            state_dict,
            weight_key,
            layer_num=None,
            state_dict_prefix=None,
            weight_cache_path=None,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weight_dtype=ttnn.bfloat16,
            is_distributed=None,
            eps: float = 1e-05,
            add_unit_offset=False,
            sharded_program_config=None,
            sharded_output_config=None,
            output_mem_config=None,
            ccl_topology=ttnn.Topology.Ring,
            tt_ccl=None,
        ):
            super().__init__()
            self.device = device
            self.eps = eps
            self.is_distributed = is_distributed
            self.ccl_topology = ccl_topology
            self.tt_ccl = tt_ccl

            if state_dict_prefix:
                weight_name = f"{state_dict_prefix}{weight_key}.weight"
            else:
                if layer_num is None:
                    weight_name = f"{weight_key}.weight"
                else:
                    weight_name = f"layers.{layer_num}.{weight_key}.weight"

            torch_weight = (
                state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
            )

            # Add offset before caching
            if add_unit_offset:
                torch_weight = torch_weight + 1.0

            # Compatibility with models that don't use mesh devices (e.g. single-chip Mistral-7b)
            is_mesh_device = device.__class__.__name__ == "MeshDevice"

            self.weight = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=None if weight_cache_path is None else weight_cache_path / weight_name,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )

            if self.is_distributed:
                self.weight_distributed = ttnn.as_tensor(
                    torch_weight,
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=weight_memory_config,
                    cache_file_name=(
                        None if weight_cache_path is None else weight_cache_path / (weight_name + "_distributed")
                    ),
                    mesh_mapper=(
                        ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                        if is_mesh_device
                        else None
                    ),
                )

            self.sharded_output_config = sharded_output_config
            self.sharded_program_config = sharded_program_config
            self.output_mem_config = output_mem_config

            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        def forward(self, x: ttnn.Tensor, mode, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
            # If input is sharded do sharded RMSNorm and optionally return sharded output
            program_config = self.sharded_program_config if in_sharded else None
            memory_config = self.sharded_output_config if out_sharded else None
            distributed = self.is_distributed and self.is_distributed(mode)
            norm = self._distributed_rmsnorm if distributed else ttnn.rms_norm
            weight = self.weight_distributed if distributed else self.weight

            if in_sharded:
                assert not distributed, "Distributed RMSNorm does not support sharded inputs"
            else:
                assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"

            x = norm(
                x,
                epsilon=self.eps,
                weight=weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

            if in_sharded and not out_sharded:
                return ttnn.sharded_to_interleaved(x)
            else:
                return x

        def _distributed_rmsnorm(
            self, inp, epsilon=None, weight=None, program_config=None, memory_config=None, compute_kernel_config=None
        ):
            assert program_config is None, "Distributed RMSNorm does not support sharded inputs"
            assert memory_config is None, "Distributed RMSNorm does not support sharded outputs"
            assert self.tt_ccl is not None, "Distributed RMSNorm requires tt_ccl"

            # Run distributed rmsnorm part 1
            tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
            # AllGather stats
            tt_stats = ttnn.experimental.all_gather_async(
                tt_stats,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            # Run distributed rmsnorm part 2
            tt_out = ttnn.rms_norm_post_all_gather(
                inp,
                tt_stats,
                epsilon=epsilon,
                weight=weight,
                compute_kernel_config=compute_kernel_config,
            )
            tt_stats.deallocate(True)

            return tt_out

    # Alias for backward compatibility
    RMSNorm = TTRMSNorm

    # DistributedNorm class integrated from tt_qwen/models/distributed_norm.py
    from chitu.utils import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm

    class DistributedNorm(LightweightModule):
        def __init__(self, norm, args, tt_ccl, TG=False):
            self.norm = norm
            self.args = args
            self.tt_ccl = tt_ccl

            if TG:
                core_grid_ln = (
                    min(4, args.dim // 4 // 32 // 8),
                    8,
                )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
                num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
                hidden_size_per_device_distributed_ln = args.dim // 4
                self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                    core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                    strategy=ttnn.ShardStrategy.WIDTH,
                )
                self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                    subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                    block_h=1,
                    block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                    inplace=False,
                )
                self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                    shape=[1, 1, 32, 32 * 4],
                    core_grid=ttnn.CoreGrid(y=1, x=1),
                    strategy=ttnn.ShardStrategy.WIDTH,
                )
                self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
            self.TG = TG

        def forward(self, x, mode):
            """Apply a norm, possibly gathering inputs if required."""
            if self.TG:
                if mode == "decode":
                    return tt_sharded_distributed_rmsnorm(
                        x,
                        epsilon=self.norm.eps,
                        gamma=self.norm.weight_distributed,
                        mesh_device=self.args.mesh_device,
                        tt_ccl=self.tt_ccl,
                        ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                        ln_sharded_progcfg=self.ln_prg_cfg,
                        ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                    )
                else:
                    return tt_distributed_rmsnorm(
                        x,
                        epsilon=self.norm.eps,
                        gamma=self.norm.weight_distributed,
                        mesh_device=self.args.mesh_device,
                        tt_ccl=self.tt_ccl,
                        compute_kernel_config=self.ln_cfg,
                    )

            input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

            # Distributed norm already performs a gather
            if self.args.is_multichip and not self.args.is_distributed_norm(mode):
                x = ttnn.experimental.all_gather_async(
                    x,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=self.args.ccl_topology(),
                    memory_config=input_mem_cfg,
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                x = ttnn.to_memory_config(x, input_mem_cfg)

            x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

            # Distributed norm requires a gather
            if self.args.is_distributed_norm(mode):
                x = ttnn.experimental.all_gather_async(
                    x,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=self.args.ccl_topology(),
                    memory_config=x.memory_config(),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

            return x
