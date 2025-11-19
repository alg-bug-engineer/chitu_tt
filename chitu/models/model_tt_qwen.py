from __future__ import annotations

import os
import sys
import torch
from types import SimpleNamespace
from typing import Any, Optional, List

from chitu.models.registry import register_model, ModelType
from chitu.global_vars import get_global_args


class _NoopKVCacheManager:
    """
    A minimal KV cache manager that satisfies Executor/Backend calls but does nothing.
    Enhanced to manage Slot mapping for Batch inference.
    """

    def __init__(self, max_batch_size=32):
        self.block_size = 1
        self.num_layers = 0
        self.shape_per_token_dict = {}
        self.dtype_dict = {}
        # Compatibility with Scheduler
        self.num_additional_blocks_req_need = lambda req_id, num_blocks: 0
        self.num_used_blocks = 0
        
        # Slot management for TT Batching
        self.max_batch_size = max_batch_size
        self.req_to_slot = {}
        self.slot_to_req = {}
        # Use a stack for free slots (descending order to prefer lower slots)
        self.free_slots = list(range(max_batch_size))
        self.free_slots.sort(reverse=True)
        
        self.current_batch_req_ids = []

    # Prefill lifecycle
    def prepare_cache_prefill(self, req_ids, seq_lens):
        self.current_batch_req_ids = req_ids
        for rid in req_ids:
            if rid not in self.req_to_slot:
                if not self.free_slots:
                    # Try to recover slots if possible or fail gracefully
                    raise RuntimeError(f"No free slots for TT model (max_batch_size={self.max_batch_size}, active={len(self.req_to_slot)})")
                slot = self.free_slots.pop()
                self.req_to_slot[rid] = slot
                self.slot_to_req[slot] = rid

    def finalize_cache_all_prefill(self):
        return

    # Decode lifecycle
    def prepare_cache_decode(self, req_ids):
        self.current_batch_req_ids = req_ids

    def finalize_cache_single_decode(self, req_ids):
        return

    def finalize_cache_all_decode(self, req_id):
        # Recycle slot when request finishes
        if req_id in self.req_to_slot:
            slot = self.req_to_slot.pop(req_id)
            if slot in self.slot_to_req:
                del self.slot_to_req[slot]
            self.free_slots.append(slot)
            self.free_slots.sort(reverse=True) # Keep sorted

    # Optional API used in some flows
    def realloc(self, num_blocks: int):
        return

    def get_max_num_blocks(self):
        return 0

    def get_num_blocks(self):
        """Return the number of blocks (for compatibility with Scheduler)"""
        return 1000000 # Infinite for Noop

    @property
    def num_free_blocks(self):
        """Return number of free blocks (for compatibility with Scheduler)"""
        return 1000000


@register_model(ModelType.TT_QWEN)
class TTQwenModel:
    """
    TT-Qwen adapter supporting Batch Inference.
    """

    def __init__(
        self,
        args: Any,
        cache_manager: Optional[Any] = None,
        *extra_args,
        **extra_kwargs,
    ):
        # Lazy import
        from chitu.utils import try_import_platform_dep
        ttnn, has_ttnn = try_import_platform_dep("ttnn")
        if not has_ttnn:
            raise ImportError("ttnn is required for TT_QWEN model")
        
        from chitu.models.tt_common import create_tt_model
        from chitu.models.tt_generator import Generator
        from chitu.models.tt_model_config import DecodersPrecision

        self.cache_manager = cache_manager

        # Mesh device setup
        # all_device_ids = ttnn.get_device_ids()
        # if len(all_device_ids) < 1:
        #     raise RuntimeError(f"Tenstorrent device not found")
        
        # # Try to use 2 chips if available (as per original demo), otherwise 1
        # if len(all_device_ids) >= 2:
        #      device_ids = all_device_ids[:2]
        #      mesh_shape = (1, 2) # 1x2 mesh
        # else:
        #      device_ids = all_device_ids
        #      mesh_shape = (1, 1)

        # if len(all_device_ids) >= 4:
        #     # device_ids = [all_device_ids[2], all_device_ids[3]] # Original demo specific
        #     mesh_shape = (1, 2) 

        mesh_shape = (1, 1) 
        self._mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=24576,
            trace_region_size=70000000,
            num_command_queues=1,
        )

        # Get global args for infer config
        global_args = get_global_args()
        if hasattr(global_args, "infer"):
            infer_args = global_args.infer
        else:
            infer_args = SimpleNamespace(max_seq_len=256, max_reqs=32)
        
        # Configure max_batch_size from args
        self.max_batch_size = getattr(infer_args, "max_reqs", 32)
        # TT implementation often requires batch size to be 32 for tiling efficiency
        if self.max_batch_size < 32:
             print(f"Warning: Upgrading max_batch_size from {self.max_batch_size} to 32 for TT compatibility")
             self.max_batch_size = 32

        max_seq_len = getattr(infer_args, "max_seq_len", 256)
        
        print(f"Initializing TT-Qwen with max_batch_size={self.max_batch_size}, max_seq_len={max_seq_len}")

        self._model_args, self._tt_model, _, _ = create_tt_model(
            mesh_device=self._mesh_device,
            instruct=True,
            max_batch_size=self.max_batch_size, 
            optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            paged_attention_config=None,
            dtype=ttnn.bfloat16,
            state_dict=None,
            num_layers=None,
        )
        self._generator = Generator([self._tt_model], [self._model_args], self._mesh_device, tokenizer=self._model_args.tokenizer)

        self.vocab_size: int = int(self._model_args.vocab_size)

        # State for positions: [max_batch_size]
        self._current_pos_tensor = torch.zeros(self.max_batch_size, dtype=torch.int64)

    def prefill(
        self,
        tokens: torch.Tensor,                   # [B*T_total] (Packed)
        output_token_offsets: torch.Tensor,     # [B]
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch prefill.
        """
        req_ids = self.cache_manager.current_batch_req_ids
        batch_size = len(req_ids)
        assert batch_size <= self.max_batch_size
        
        # 1. Unpack tokens
        input_tokens_list = []
        prompt_lens = []
        start = 0
        for i in range(batch_size):
            end = output_token_offsets[i].item() + 1
            t = tokens[start:end]
            input_tokens_list.append(t)
            prompt_lens.append(len(t))
            start = end
            
        max_len = max(prompt_lens) if prompt_lens else 0
        
        # 2. Prepare padded inputs for TT (always max_batch_size)
        tt_input_tokens = torch.zeros((self.max_batch_size, max_len), dtype=torch.long)
        
        # FIX: Initialize prompt_lens with a safe non-zero value (e.g. 32) for inactive slots.
        # This prevents "End 0 must be greater than or equal to start" errors in ttnn.slice
        tt_prompt_lens = torch.full((self.max_batch_size,), 32, dtype=torch.int64)
        
        # Map active requests to their slots
        for i, rid in enumerate(req_ids):
            slot = self.cache_manager.req_to_slot[rid]
            seq_len = len(input_tokens_list[i])
            tt_input_tokens[slot, :seq_len] = input_tokens_list[i].cpu()
            tt_prompt_lens[slot] = seq_len
            
            # Update position state
            self._current_pos_tensor[slot] = seq_len

        # 3. Run Prefill
        logits = self._generator.prefill_forward_text(
            tt_input_tokens,
            prompt_lens=tt_prompt_lens,
        )
        # logits: [max_batch_size, 1, vocab]

        # 4. Gather results for active requests
        # Chitu expects [Batch, Vocab] in order of req_ids
        output_logits = torch.zeros((batch_size, self.vocab_size), dtype=logits.dtype, device=logits.device)
        
        for i, rid in enumerate(req_ids):
            slot = self.cache_manager.req_to_slot[rid]
            # logits is [B, 1, V] or [B, V] depending on impl
            if logits.dim() == 3:
                output_logits[i] = logits[slot, 0, :]
            else:
                output_logits[i] = logits[slot, :]
                
        return output_logits

    def decode(self, next_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Batch decode.
        """
        req_ids = self.cache_manager.current_batch_req_ids
        assert len(req_ids) == batch_size
        assert len(next_tokens) == batch_size
        
        # 1. Prepare inputs for TT (max_batch_size)
        tt_input_tokens = torch.zeros(self.max_batch_size, dtype=torch.long)
        
        for i, rid in enumerate(req_ids):
            slot = self.cache_manager.req_to_slot[rid]
            tt_input_tokens[slot] = next_tokens[i].cpu()
            
        # 2. Run Decode
        # Generator expects [B] tokens and [B] pos.
        logits = self._generator.decode_forward_text(
            tt_input_tokens.unsqueeze(1), # [B, 1]
            self._current_pos_tensor,      # [B]
            enable_trace=False, 
            page_table=None,
            kv_cache=None,
        )
        # logits: [max_batch_size, 1, vocab]
        
        # 3. Update positions for active requests
        for i, rid in enumerate(req_ids):
            slot = self.cache_manager.req_to_slot[rid]
            self._current_pos_tensor[slot] += 1
            
        # 4. Gather results
        output_logits = torch.zeros((batch_size, self.vocab_size), dtype=logits.dtype, device=logits.device)
        for i, rid in enumerate(req_ids):
            slot = self.cache_manager.req_to_slot[rid]
            if logits.dim() == 3:
                output_logits[i] = logits[slot, 0, :]
            else:
                output_logits[i] = logits[slot, :]
                
        return output_logits

    @staticmethod
    def build_noop_cache_manager() -> _NoopKVCacheManager:
        # Need to know max_reqs to initialize manager correctly
        global_args = get_global_args()
        max_reqs = getattr(global_args.infer, "max_reqs", 32)
        # Ensure alignment with model's padding logic (at least 32)
        if max_reqs < 32: 
            max_reqs = 32
        return _NoopKVCacheManager(max_batch_size=max_reqs)