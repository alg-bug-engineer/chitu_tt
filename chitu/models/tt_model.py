# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Integrated from tt_qwen/tt/model.py

import torch
from tqdm import tqdm
import sys
import os

from chitu.utils import try_import_platform_dep, LightweightModule
from chitu.ops.norm import TTRMSNorm as RMSNorm
from chitu.utils import TT_CCL
from chitu.utils import copy_host_to_device
from chitu.models.tt_decoder import TransformerBlock
from chitu.ops.norm import DistributedNorm
from chitu.ops.rotary import TTRotarySetup

import math
from chitu.utils import tt_all_reduce

ttnn, has_ttnn = try_import_platform_dep("ttnn")

if has_ttnn:
    from chitu.models.tt_model_config import TensorGroup


class Embedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        self.mesh_device = mesh_device

        # Get the prefix first
        prefix = args.get_state_dict_prefix("", None)
        base_name = prefix + "tok_embeddings.weight"
        
        # Try to get the weight, with fallback to alternative key names
        if base_name not in state_dict:
            # Try alternative key names (in case of key name conversion issues)
            alt_keys = [
                prefix + "tok_embeddings.weight",  # With prefix
                "tok_embeddings.weight",  # Without prefix
                prefix + "model.embed_tokens.weight",  # With prefix and model prefix
                "model.embed_tokens.weight",  # HF format
                prefix + "embed_tokens.weight",  # With prefix
                "embed_tokens.weight",  # Without prefix
            ]
            found = False
            for alt_key in alt_keys:
                if alt_key in state_dict:
                    base_name = alt_key
                    found = True
                    break
            
            if not found:
                # If still not found, try to find any key containing "embed" or "tok"
                for key in state_dict.keys():
                    if ("embed" in key.lower() or "tok" in key.lower()) and "weight" in key.lower():
                        base_name = key
                        found = True
                        break
            
            if not found:
                # Print available keys for debugging
                available_keys = [k for k in state_dict.keys() if "embed" in k.lower() or "tok" in k.lower() or "weight" in k.lower()]
                raise KeyError(
                    f"Could not find embedding weight in state_dict. "
                    f"Expected key: {prefix + 'tok_embeddings.weight'}, "
                    f"Available embedding-related keys: {available_keys[:20] if available_keys else 'None'}, "
                    f"Total state_dict keys: {len(state_dict.keys())}, "
                    f"First 10 keys: {list(state_dict.keys())[:10]}"
                )
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_name = None if args.dummy_weights else weight_cache_path / base_name
        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x


class ScaledEmbedding(Embedding):
    def __init__(self, mesh_device, args, weight_cache_path, state_dict, dtype, embed_scale: float = 1.0):
        super().__init__(mesh_device, args, weight_cache_path, state_dict, dtype)
        self.embed_scale = embed_scale

    def forward(self, x):
        e = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = ttnn.multiply(e, self.embed_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return s

class LMHead(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device,  # too many columns per device lead to L1 OOM
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices

        size_per_device = self.vocab_size // self.num_devices
        self.model_config = args.get_model_config()

        if args.is_galaxy:
            size_per_device = self.padded_vocab_size // self.num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        self.output_weights = []
        if args.is_galaxy:
            cache_file_name = (
                None if args.dummy_weights else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_0"
            )
            padded_lm_head = torch.zeros(1, 1, args.dim, self.padded_vocab_size)
            padded_lm_head[:, :, :, : self.vocab_size] = torch_output_weights

            memory_config = (
                ttnn.DRAM_MEMORY_CONFIG
                if args.dim == 2048
                else args.create_dram_sharded_mem_config(k=args.dim // 4, n=self.padded_vocab_size // 8)
            )
            self.output_weights.append(  # (2k, 16k) 128* 1024
                ttnn.as_tensor(
                    padded_lm_head,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=memory_config,
                    cache_file_name=cache_file_name,
                )
            )
        else:
            for i, split_size in enumerate(split_sizes):
                # Create a list to store the split tensors for each device
                device_splits = []
                for device in range(self.num_devices):
                    start = device * size_per_device + sum(split_sizes[:i])
                    end = start + split_size
                    device_splits.append(torch_output_weights[:, start:end])

                # Concatenate the splits from all devices
                combined_split = torch.cat(device_splits, dim=-1)

                cache_file_name = (
                    None
                    if args.dummy_weights
                    else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_{i}_{combined_split.shape[-1]}"
                )
                memory_config = args.create_dram_sharded_mem_config(
                    k=args.dim, n=math.ceil(combined_split.shape[-1] / self.num_devices)
                )
                self.output_weights.append(
                    ttnn.as_tensor(
                        combined_split,
                        device=mesh_device,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=dtype,
                        memory_config=memory_config,
                        cache_file_name=cache_file_name,
                    )
                )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        if args.is_galaxy:
            self.program_configs = [
                (
                    None
                    if args.dim == 2048
                    else args.dram_matmul_config(
                        args.tile_padded_batch_rows,  # (8k, 128k) -> (2k, 16k)
                        args.dim // 4,
                        16 * 1024,
                        args.lm_head_core_grid.num_cores,
                    )
                )
            ]

        else:
            self.program_configs = [
                args.dram_matmul_config(
                    args.tile_padded_batch_rows,
                    args.dim,
                    split_size,
                    args.lm_head_core_grid.num_cores,
                )
                for split_size in split_sizes
            ]

    def forward(self, x: ttnn.Tensor):
        outputs = []
        for weight, pc in zip(self.output_weights, self.program_configs):
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
            )
            outputs.append(
                ttnn.sharded_to_interleaved(
                    output, memory_config=self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
                )
            )

        # Concatenate the outputs
        output = ttnn.concat(
            outputs, dim=-1, memory_config=self.model_config.get("LM_HEAD_OUTPUT_MEMCFG", ttnn.L1_MEMORY_CONFIG)
        )

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
        )

        return output

class Transformer(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)

        self.tt_ccl = TT_CCL(self.mesh_device)

        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": args.weight_cache_path(dtype),
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,  # Row major layout requires bfloat16
        }
        if self.args.embed_scale is not None:
            embd_cls = ScaledEmbedding
            embd_kwargs["embed_scale"] = self.args.embed_scale
        else:
            embd_cls = Embedding
        self.embd = embd_cls(**embd_kwargs)

        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else TTRotarySetup
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )

        if args.rope_theta_local:
            self.rope_local_setup = TTRotarySetup(
                mesh_device,
                args.max_batch_size,
                args.head_dim,
                args.max_seq_len,
                args.rope_theta_local,
            )

        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                attention_class=attention_class,
            )
            for i in tqdm(range(self.n_layers))
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=self.model_config["LM_HEAD_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            self.tt_ccl,
            args.is_galaxy,
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            max_columns_per_device=self.args.max_columns_per_device_lm_head,
        )

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        assert tokens.dim() == 2, "tokens must be a 2D tensor"
        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if hasattr(self, "rope_local_setup"):
            tt_rot_mats_prefill_local = [
                self.rope_local_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
                self.rope_local_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
            ]
        else:
            tt_rot_mats_prefill_local = None

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_rot_mats_prefill_local, tt_page_table, tt_chunk_page_table

    def prepare_inputs_decode(self, *inputs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        return device_inputs

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        """
        B = tokens.shape[0]
        assert current_pos.shape[0] == B, "Batch size mismatch"
        assert B == self.args.max_batch_size, "Batch size must be equal to max_batch_size"

        # Necessary padding to be full tile sized when on device
        tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        rope_idxs_global = self.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)
        if hasattr(self, "rope_local_setup"):
            rope_idxs_local = self.rope_local_setup.get_rot_idxs(rot_current_pos, on_host=True)
        else:
            rope_idxs_local = None

        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 0) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
                    mesh_shape=self.args.cluster_shape,
                ),
            )
        return tokens, current_pos_tt, rope_idxs_global, rope_idxs_local, page_table

    def _transform_decode_inputs_device(self, tokens):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Embed tokens
        """
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        tt_tokens = ttnn.to_memory_config(
            tt_tokens,
            self.args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        return tt_tokens

    def concat_host_output(self, tt_out):
        """
        Concatenate the output of the devices into a single host tensor.
        """
        torch_out_tensors = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]
        if self.args.is_galaxy:
            row_dim, col_dim = (3, 1)
        else:
            row_dim, col_dim = (1, -1)

        rows, cols = self.args.cluster_shape
        mesh_shape = [torch_out_tensors[i : i + cols] for i in range(0, len(torch_out_tensors), cols)]
        row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]
        return torch.cat(row_concatenated, dim=row_dim)

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        return self.concat_host_output(tt_out.cpu())[0, 0, last_token_idx, : self.vocab_size]

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Input is ttnn host tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        if is_tokens:
            return self.concat_host_output(tt_out)[0, 0, :B, 0]

        if self.args.num_devices > 1:
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            tt_out = ttnn.to_torch(tt_out).float()
        tt_out = tt_out[:, :, :B, : self.vocab_size].view(B, S, -1)
        return tt_out

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos=None,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs_global, rot_mat_idxs_local):
        # ttnn.ne currently requires the input to be in TILE_LAYOUT
        current_pos_tiled = ttnn.to_layout(current_pos, layout=ttnn.TILE_LAYOUT)
        # Update only active positions (current_pos != -1)
        predicate = ttnn.ne(current_pos_tiled, -1)
        result = ttnn.where(
            predicate,
            ttnn.add(current_pos_tiled, 1),
            current_pos_tiled,
        )
        ttnn.copy(ttnn.to_layout(result, layout=ttnn.ROW_MAJOR_LAYOUT), current_pos)

        ttnn.plus_one(rot_mat_idxs_global)
        if rot_mat_idxs_local is not None:
            ttnn.plus_one(rot_mat_idxs_local)

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs_global=None,
        rot_mat_idxs_local=None,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs_global)
        rot_mats_local = (
            self.rope_local_setup.get_rot_mats(rot_mat_idxs_local) if rot_mat_idxs_local is not None else None
        )
        x_embed = self._transform_decode_inputs_device(x)

        tt_logits = self.forward(
            x_embed,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )

        # Gather the output across all devices and untilize the tensor (for argmax)
        if self.args.num_devices > 1:
            cluster_axis = 0 if self.args.is_galaxy else None
            num_links = 2 if self.args.is_galaxy else 1
            tt_logits = ttnn.experimental.all_gather_async(
                tt_logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=num_links,
                memory_config=tt_logits.memory_config(),
                cluster_axis=cluster_axis,
                topology=self.args.ccl_topology(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)

        if argmax_on_device:
            tt_logits = ttnn.argmax(tt_logits, dim=3, keepdim=True, use_multicore=True)

            # Update device tensors for the next iteration
            self._increment_decode_positions_device(current_pos, rot_mat_idxs_global, rot_mat_idxs_local)

            # Update input tokens with sampled tokens for the next iteration
            ttnn.copy(tt_logits.reshape(x.shape), x)
        elif not self.args.is_galaxy:
            # Send output logits to DRAM so L1 is not reserved for ttnn tracing and can be used by subsequent operations
            tt_logits = ttnn.to_memory_config(tt_logits, ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        for i, layer in enumerate(self.layers):
            # No-op if callers already provide the right memory config
            activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if mode == "decode" and not self.args.is_galaxy:
                x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"], activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode)

        if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        x = self.lm_head(x)

        if mode == "prefill":
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

