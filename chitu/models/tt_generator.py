# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Integrated from tt_qwen/tt/generator.py

from dataclasses import dataclass

import torch
from loguru import logger

from chitu.utils import try_import_platform_dep
from chitu.utils import (
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)

ttnn, has_ttnn = try_import_platform_dep("ttnn")

if has_ttnn:
    @dataclass(frozen=True)
    class SamplingParams:
        """
        Used in Generator decode forward functions for greedy decoding / sampling on device.
        The same data class exists in vLLM at vllm/worker/tt_model_runner.py.
        """

        temperature: float
        top_k: int
        top_p: float


    class Generator:
        def __init__(self, model, model_args, mesh_device, tokenizer=None, formatter=None):
            """
            Creating a LlamaVision wrapper requires only a mesh_device and model_args.
            With model_args you have the checkpoint location, can specify max batch size
            and max seqlen, and other model specific parameters.

            LlamaVision is general to text and chat.

            For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

            """
            self.model = model
            self.model_args = model_args
            self.mesh_device = mesh_device
            self.tokenizer = tokenizer
            self.formatter = formatter
            self.data_parallel = len(self.model)
            self.prev_page_table = None

        # Note: This function is called by vLLM
        def prefill_forward_text(
            self,
            tokens: torch.Tensor,
            page_table=None,
            kv_cache=None,
            prompt_lens=None,
            empty_slots=None,
            **kwargs,
        ):
            if page_table is not None:
                assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"

            batch_size, batch_seq_len = tokens.shape
            max_batch_size_per_model = self.model_args[0].max_batch_size

            # Each model expected to run the same model, safe to use 1st vocab size
            output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
            prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

            if empty_slots is None:
                empty_slots = list(range(batch_size))

            out_list = []
            for idx, user_id in enumerate(empty_slots):
                model_id = user_id // max_batch_size_per_model
                group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
                seq_len = int(prompt_lens[idx])
                last_token_idx = seq_len - 1
                prefill_seq_len = get_padded_prefill_len(seq_len)
                local_kwargs = kwargs.copy()  # Avoid modifying original kwargs

                logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

                # Extracting data for the current user
                # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
                prefill_ids = torch.cat(
                    [tokens[idx : idx + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
                )
                page_table_user = (
                    self._get_prefill_user_page_table(page_table[idx : idx + 1], kv_cache[model_id], seq_len)
                    if page_table is not None
                    else None
                )
                model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

                # Check if 'pixel_values' exists and index it safely
                if local_kwargs.get("pixel_values", None) is not None:
                    local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                    if "image_grid_thw" in local_kwargs:
                        local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]

                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    **local_kwargs,
                )
                # if data parallel is greater than 1, we need to add logits to out_list and do the processing after all the prefill are done
                # otherwise, we can process the logits after prefill immediately
                if self.data_parallel > 1:
                    out_list.append(logits)
                else:
                    output_logits[idx] = self.model[model_id].process_output_prefill(
                        logits, last_token_idx=(last_token_idx % 32)
                    )
                    del logits

            # Process the logits after all the prefill are done in data parallel mode
            if self.data_parallel > 1:
                for idx, out in enumerate(out_list):
                    seq_len = int(prompt_lens[idx])
                    last_token_idx = seq_len - 1
                    user_id = empty_slots[idx]
                    model_id = user_id // max_batch_size_per_model

                    # Since we give unpadded_seq_len, only the tile containing the last token is returned
                    output_logits[idx] = self.model[model_id].process_output_prefill(
                        out, last_token_idx=(last_token_idx % 32)
                    )

            logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
            return output_logits

        def prefill_forward_single_user_text(
            self, tokens, page_table, user_id, last_token_idx, kv_cache=None, model_id=-1, **kwargs
        ):
            seq_len = tokens.shape[-1]
            use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
            if use_chunked_prefill:
                """
                Chunked prefill requires paged attention. There are some strange constraints which we must meet:
                 - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
                 checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
                 - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
                 - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
                 to keep it otherwise unaware that it is operating on a chunk.
                 - due to the above point, we must always set user_id to 0 for chunked prefill.
                """
                assert page_table is not None, "page_table must be provided for chunked prefill"
                assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
                assert (
                    last_token_idx is not None and last_token_idx < seq_len
                ), "last_token_idx must be provided and less than seq_len"
                chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
                block_size = get_block_size(kv_cache)
                last_token_idx_in_chunk = last_token_idx % chunk_size
                # Calculate which chunk contains the last_token_idx
                last_chunk_start = (last_token_idx // chunk_size) * chunk_size
                page_table_user = page_table[user_id : user_id + 1, :]
                # Pad page table to match number of blocks in seq_len
                num_padding_blocks = num_blocks_in_seq(seq_len, block_size) - page_table_user.shape[1]
                page_table_user_padded = torch.cat(
                    [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
                )
                CHUNK_USER_ID = 0

                for chunk_start in range(0, seq_len, chunk_size):
                    chunk_end = chunk_start + chunk_size
                    assert (
                        chunk_end <= seq_len
                    ), f"Chunk end should be less than seq_len, got chunk_end={chunk_end} and seq_len={seq_len}"
                    chunk_tokens = tokens[:, chunk_start:chunk_end]
                    chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                    (
                        chunk_prefill_input,
                        chunk_rot_mats_global_prefill,
                        chunk_rot_mats_local_prefill,
                        page_table_tt,
                        chunk_page_table_tt,
                    ) = self.model[model_id].prepare_inputs_prefill(
                        chunk_tokens,
                        start_pos=chunk_start,
                        page_table=page_table_user_padded,
                        chunk_page_table=chunk_page_table,
                        **kwargs,
                    )
                    tt_logits = self.model[model_id].ttnn_prefill_forward(
                        chunk_prefill_input,
                        rot_mats_global=chunk_rot_mats_global_prefill,
                        rot_mats_local=chunk_rot_mats_local_prefill,
                        user_id=CHUNK_USER_ID,
                        page_table=page_table_tt,
                        chunk_page_table=chunk_page_table_tt,
                        chunk_start_idx=chunk_start,
                        get_last_token=(last_token_idx_in_chunk // 32) * 32,
                        kv_cache=kv_cache,
                        **kwargs,
                    )

                    if chunk_start == last_chunk_start:
                        return tt_logits
                    else:
                        del tt_logits
            else:
                (
                    prefill_input,
                    rot_mats_global_prefill,
                    rot_mats_local_prefill,
                    page_table_tt,
                    _,
                ) = self.model[model_id].prepare_inputs_prefill(
                    tokens,
                    page_table=page_table,
                    **kwargs,
                )

                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    prefill_input,
                    rot_mats_global=rot_mats_global_prefill,
                    rot_mats_local=rot_mats_local_prefill,
                    user_id=user_id,
                    page_table=page_table_tt,
                    get_last_token=(last_token_idx // 32) * 32,
                    kv_cache=kv_cache,
                )
                return tt_logits

        # Note: This function is called by vLLM
        def decode_forward_text(
            self,
            tokens,
            start_pos,
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
        ):
            assert (
                sampling_params is None or sampling_params.temperature == 0
            ), "Currently only supporting greedy decoding (temperature=0) on device"
            argmax_on_device = sampling_params is not None and sampling_params.temperature == 0

            B = tokens.shape[0]
            tokens = torch.chunk(tokens, self.data_parallel, 0)
            start_pos = torch.chunk(start_pos, self.data_parallel, 0)
            page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None

            decode_kwargs = {
                "current_pos": start_pos,
                "tokens": tokens,
                "page_table": page_table,
                "kv_cache": kv_cache,
                "argmax_on_device": argmax_on_device,
            }
            if enable_trace:
                tt_decode_output = self._easy_trace_text(**decode_kwargs)
            else:
                tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)

            if read_from_device:
                to_host = self.read_decode_output(tt_decode_output)
                return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))

            return tt_decode_output

        def _decode_forward_no_trace_text(
            self,
            tokens,
            current_pos,
            page_table=None,
            kv_cache=None,
            argmax_on_device=False,
        ):
            """
            Performs text decode step.
            Returns tt_logits on device
            """
            tt_logits = []

            tt_tokens = []
            tt_current_pos = []
            tt_rot_mat_idxs_global = []
            tt_rot_mat_idxs_local = []
            tt_page_table = []

            for i in range(self.data_parallel):
                user_page_table = page_table[i] if page_table is not None else None
                model_i = self.model[i]
                (
                    tt_tokens_i,
                    tt_current_pos_i,
                    tt_rot_mat_idxs_global_i,
                    tt_rot_mat_idxs_local_i,
                    tt_page_table_i,
                ) = model_i.prepare_inputs_decode(tokens[i], current_pos[i], user_page_table)
                tt_tokens.append(tt_tokens_i)
                tt_current_pos.append(tt_current_pos_i)
                tt_rot_mat_idxs_global.append(tt_rot_mat_idxs_global_i)
                tt_rot_mat_idxs_local.append(tt_rot_mat_idxs_local_i)
                tt_page_table.append(tt_page_table_i)

            for i in range(self.data_parallel):
                user_kv_cache = kv_cache[i] if kv_cache is not None else None
                tt_logits_i = self.model[i].ttnn_decode_forward(
                    tt_tokens[i],
                    tt_current_pos[i],
                    rot_mat_idxs_global=tt_rot_mat_idxs_global[i],
                    rot_mat_idxs_local=tt_rot_mat_idxs_local[i],
                    page_table=tt_page_table[i],
                    kv_cache=user_kv_cache,
                    argmax_on_device=argmax_on_device,
                )
                tt_logits.append(tt_logits_i)

            return tt_logits

        def _capture_trace_text(
            self,
            tokens,
            current_pos,
            page_table=None,
            kv_cache=None,
            argmax_on_device=False,
        ):
            """
            Captures a trace for the decode_forward method.
            """

            # Compile run
            self._decode_forward_no_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, argmax_on_device=argmax_on_device
            )
            logger.info("Done Compiling Model")

            # Get inputs ready for trace run
            device_inputs = []
            tt_out_trace = []
            trace_ids = {}
            for i in range(self.data_parallel):
                user_page_table = page_table[i] if page_table is not None else None
                host_inputs = self.model[i].prepare_decode_inputs_host(
                    tokens[i], current_pos[i], page_table=user_page_table
                )

                device_inputs_i = copy_host_to_device(host_inputs, mesh_device=self.model_args[i].mesh_device)
                device_inputs.append(device_inputs_i)

            for i in range(self.data_parallel):
                trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
                trace_ids[i] = trace_id
                user_kv_cache = kv_cache[i] if kv_cache is not None else None
                tt_out_trace.append(
                    self.model[i].ttnn_decode_forward(
                        *device_inputs[i], kv_cache=user_kv_cache, argmax_on_device=argmax_on_device
                    )
                )
                ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
            logger.info("Done Capturing Decode Trace")
            return trace_ids, tt_out_trace, *device_inputs

        def _easy_trace_text(
            self,
            tokens,
            current_pos,
            page_table=None,
            kv_cache=None,
            argmax_on_device=False,
        ):
            """
            Tracing is easy! Just call this method and we'll handle tracing for you.
            """
            if not hasattr(self, "trace_ids_text"):
                trace_ids, tt_out_trace, *device_inputs = self._capture_trace_text(
                    tokens, current_pos, page_table=page_table, kv_cache=kv_cache, argmax_on_device=argmax_on_device
                )
                self.trace_ids_text = trace_ids
                self.trace_inputs_text = device_inputs
                self.trace_output_text = tt_out_trace

            reset_inputs = not argmax_on_device
            if self.prev_page_table is None or any(
                not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
            ):
                reset_inputs = True
                self.prev_page_table = page_table

            if reset_inputs:
                for i in range(self.data_parallel):
                    user_page_table = page_table[i] if page_table is not None else None
                    host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)

                    copy_host_to_device(
                        host_tensors=host_inputs_i,
                        device_tensors=self.trace_inputs_text[i],
                    )

            for i, trace_id in self.trace_ids_text.items():
                ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)

            return self.trace_output_text

        # Note: This function is called by vLLM
        def read_decode_output(self, tt_out, async_read=False):
            """
            Input tt_out is a list of ttnn device tensors
            """
            if not async_read:
                return [out.cpu() for out in tt_out]

            host_outputs = []
            read_events = []
            for i in range(self.data_parallel):
                host_outputs.append(tt_out[i].cpu(blocking=False))
                read_events.append(ttnn.record_event(self.model[i].mesh_device, 0))

            return host_outputs, read_events

        # Note: This function is called by vLLM
        def process_decode_output_host(self, tt_out, is_tokens=False):
            """
            Converts the input ttnn host tensors to a torch tensor.
            The input can be logits (if is_tokens=False) or tokens (if is_tokens=True).
            """
            max_batch_size_per_model = self.model_args[0].max_batch_size

            logits = []
            for i in range(self.data_parallel):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                logits.append(logits_i)

            return torch.cat(logits, 0)

        def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
            # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
            block_size = get_block_size(kv_cache)
            num_blocks = num_blocks_in_seq(prefill_len, block_size)
            return page_table[:, :num_blocks]

        ## Destructor

        def __del__(self):
            # Workaround for issue #19052
            if self.data_parallel > 1:
                for m in self.model:
                    ttnn.close_mesh_device(m.mesh_device)

            if hasattr(super(Generator, self), "__del__"):
                super().__del__()


    def create_submeshes(mesh_device, data_parallel):
        if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
            return [mesh_device]

        num_rows, num_cols = mesh_device.shape
        num_devices = num_rows * num_cols
        assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

        if num_rows == 8 and num_cols == 4 and num_cols % data_parallel == 0:
            submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows, num_cols // data_parallel))
            for submesh in submeshes:
                submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
            return submeshes

        return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))

