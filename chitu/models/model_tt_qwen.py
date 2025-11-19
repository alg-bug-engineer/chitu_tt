from __future__ import annotations

import os
import sys
import torch
from types import SimpleNamespace
from typing import Any, Optional

from chitu.models.registry import register_model, ModelType
from chitu.global_vars import get_global_args


class _NoopKVCacheManager:
    """
    A minimal KV cache manager that satisfies Executor/Backend calls but does nothing.
    This is sufficient for single-request TT execution where TT runtime manages cache internally.
    """

    def __init__(self):
        self.block_size = 1
        self.num_layers = 0
        self.shape_per_token_dict = {}
        self.dtype_dict = {}
        # 兼容 Scheduler 需要的属性
        self.num_additional_blocks_req_need = 0
        self.num_used_blocks = 0

    # Prefill lifecycle
    def prepare_cache_prefill(self, req_ids, seq_lens):
        return

    def finalize_cache_all_prefill(self):
        return

    # Decode lifecycle
    def prepare_cache_decode(self, req_ids):
        return

    def finalize_cache_single_decode(self, req_ids):
        return

    def finalize_cache_all_decode(self, req_id):
        return

    # Optional API used in some flows
    def realloc(self, num_blocks: int):
        return

    def get_max_num_blocks(self):
        return 0

    def get_num_blocks(self):
        """Return the number of blocks (for compatibility with Scheduler)"""
        return 0

    @property
    def num_free_blocks(self):
        """Return number of free blocks (for compatibility with Scheduler)"""
        return 0


@register_model(ModelType.TT_QWEN)
class TTQwenModel:
    """
    Minimal TT-Qwen adapter exposing prefill/decode APIs compatible with chitu.Executor expectations.
    Constraints:
      - 单请求（batch=1）场景优先
      - 依赖 tt_qwen 的 Generator 和 create_tt_model
    """

    def __init__(
        self,
        args: Any,
        cache_manager: Optional[Any] = None,
        *extra_args,
        **extra_kwargs,
    ):
        # 延迟导入，避免环境缺失时报错影响其他后端
        from chitu.utils import try_import_platform_dep
        ttnn, has_ttnn = try_import_platform_dep("ttnn")
        if not has_ttnn:
            raise ImportError("ttnn is required for TT_QWEN model")
        
        from chitu.models.tt_common import create_tt_model
        from chitu.models.tt_generator import Generator
        from chitu.models.tt_model_config import DecodersPrecision

        # Mesh 设备选择：与 demo 一致，优先使用 2 张卡
        all_device_ids = ttnn.get_device_ids()
        if len(all_device_ids) < 2:
            raise RuntimeError(f"Tenstorrent 设备数量不足，需要至少2张，当前 {len(all_device_ids)}")
        if len(all_device_ids) >= 4:
            device_ids = [all_device_ids[2], all_device_ids[3]]
            mesh_shape = (1, 1)
        else:
            device_ids = all_device_ids
            mesh_shape = (1, 1)
        self._mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=24576,
            trace_region_size=70000000,
            num_command_queues=1,
        )

        # 从全局参数获取 infer 配置（因为传入的 args 是 args.models）
        global_args = get_global_args()
        if hasattr(global_args, "infer"):
            infer_args = global_args.infer
        else:
            # 兜底：如果全局参数没有 infer，使用默认值
            infer_args = SimpleNamespace(max_seq_len=256)
        
        # 创建 TT 模型与生成器（与 demo 保持一致的保守参数）
        max_seq_len = getattr(infer_args, "max_seq_len", 256)
        self._model_args, self._tt_model, _, _ = create_tt_model(
            mesh_device=self._mesh_device,
            instruct=True,
            max_batch_size=1,
            optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            paged_attention_config=None,
            dtype=ttnn.bfloat16,
            state_dict=None,
            num_layers=None,
        )
        self._generator = Generator([self._tt_model], [self._model_args], self._mesh_device, tokenizer=self._model_args.tokenizer)

        # 公开 vocab_size 给 executor 采样逻辑使用
        self.vocab_size: int = int(self._model_args.vocab_size)

        # 维护 decode 位置（单请求简化）
        self._current_pos: Optional[torch.Tensor] = None

    # --- chitu 期望的 API ---
    def prefill(
        self,
        tokens: torch.Tensor,                   # [B*T] 或 [B, T]；非 PP 模式 rank0 提供
        output_token_offsets: torch.Tensor,     # [B]，每个序列最后一个 token 的索引
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        返回 logits: [B, vocab_size]
        """
        if tokens.dim() == 1:
            # 转为 [1, T]
            tokens = tokens.view(1, -1)
        prompt_len = int(output_token_offsets[-1].item()) + 1
        # TT 侧 prefill（首次会触发编译）
        logits = self._generator.prefill_forward_text(
            tokens.to(dtype=torch.long, device="cpu"),
            prompt_lens=[prompt_len],
        )
        # 记录当前位置
        self._current_pos = torch.tensor([prompt_len], dtype=torch.int64)
        # 归一为 [B, vocab]
        if logits.dim() == 1:
            logits = logits.view(1, -1)
        else:
            logits = logits.view(logits.shape[0], -1)
        return logits

    def decode(self, next_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        单步 decode，输入 next_tokens: [B]，返回 logits: [B, vocab_size]
        """
        if next_tokens.dim() != 1:
            next_tokens = next_tokens.view(-1)
        if self._current_pos is None:
            # 安全兜底（理论上 prefill 后才会 decode）
            self._current_pos = torch.tensor([1], dtype=torch.int64)
        logits = self._generator.decode_forward_text(
            next_tokens.to(dtype=torch.long, device="cpu"),
            self._current_pos,
            enable_trace=False,
            page_table=None,
            kv_cache=None,
        )
        # 位置前移
        self._current_pos += 1
        # 归一 [B, vocab]
        if logits.dim() == 1:
            logits = logits.view(1, -1)
        else:
            logits = logits.view(logits.shape[0], -1)
        return logits

    # 供 Backend 特殊初始化时取用
    @staticmethod
    def build_noop_cache_manager() -> _NoopKVCacheManager:
        return _NoopKVCacheManager()


