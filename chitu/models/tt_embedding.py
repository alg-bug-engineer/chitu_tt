# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Integrated from tt_qwen/models/embedding.py

from chitu.utils import try_import_platform_dep, LightweightModule

ttnn, has_ttnn = try_import_platform_dep("ttnn")

if has_ttnn:
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

        def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
            e = ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            s = ttnn.multiply(e, self.embed_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return s

