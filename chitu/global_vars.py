# SPDX-FileCopyrightText: 2022 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Megatron-LM
#
# This file has adaption of open-source code from the following sources:
# - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/global_vars.py

import operator
import time
import types
from functools import lru_cache, reduce
from logging import getLogger
from typing import Optional

import torch
from omegaconf import OmegaConf
import re

from chitu.schemas.serve_config import ServeConfig, StaticConfig

logger = getLogger(__name__)


_GLOBAL_ARGS: Optional[ServeConfig] = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_MEMORY_BUFFER = None
_GLOBAL_SLOT_HANDLE = None
_GLOBAL_DEBUG: bool = False


def get_global_memory_buffer():
    _ensure_var_is_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    return _GLOBAL_MEMORY_BUFFER


def get_slot_handle():
    # _ensure_var_is_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    return _GLOBAL_SLOT_HANDLE


def set_global_variables(global_args=None, debug=False):
    _set_debug(debug)
    set_global_args(global_args)
    _set_timers()
    if global_args is not None:
        _set_slot_handle(
            global_args.infer.max_reqs,
            global_args.infer.pp_size,
            global_args.infer.dp_size,
            global_args.infer.cache_type,
        )


def expand_layers(spec):
    layers = set()
    for item in spec:
        if isinstance(item, int):
            layers.add(item)
        elif isinstance(item, str) and "-" in item:
            lo, hi = item.split("-", 1)
            lo, hi = int(lo), int(hi)
            layers.update(range(lo, hi + 1))
        else:
            raise ValueError(f"Invalid layer spec: {item!r}")
    return sorted(layers)


def set_quant_variables(global_args=None):
    if global_args is None:
        return

    # Handle SimpleNamespace objects
    if hasattr(global_args, "models"):
        models = global_args.models
    elif isinstance(global_args, dict):
        models = global_args.get("models", {})
    else:
        return  # Skip for unsupported types

    # Handle SimpleNamespace or dict for models
    if hasattr(models, "name"):
        model_name = models.name
    elif isinstance(models, dict):
        model_name = models.get("name")
    else:
        return  # Skip if models is not accessible

    if model_name is None or not isinstance(model_name, str):
        return  # Skip if model_name is not valid

    model_name = model_name.lower()
    
    # For tt-qwen type, skip quant config setup
    if hasattr(models, "type") and models.type == "tt-qwen":
        return

    # Handle quant_config access
    if hasattr(models, "quant_config"):
        quant_config_attr = models.quant_config
    elif isinstance(models, dict):
        quant_config_attr = models.get("quant_config", None)
    else:
        quant_config_attr = None

    if quant_config_attr is None:
        # Try to set quant_config if models is OmegaConf
        try:
            OmegaConf.set_struct(models, False)
            if isinstance(models, dict):
                models["quant_config"] = {"rules": [], "type": None}
            else:
                models.quant_config = {"rules": [], "type": None}
            OmegaConf.set_struct(models, True)
        except:
            pass  # Skip if not OmegaConf
        return

    # Extract quant config values
    if hasattr(quant_config_attr, "get"):
        quant_type = quant_config_attr.get("type", None)
        quant_list = quant_config_attr.get("quant", [])
    elif hasattr(quant_config_attr, "type"):
        quant_type = getattr(quant_config_attr, "type", None)
        quant_list = getattr(quant_config_attr, "quant", [])
    else:
        quant_type = None
        quant_list = []

    quant_config = {"rules": [], "type": quant_type}

    for config in quant_list:
        if hasattr(config, "get"):
            pattern = config.get("model", "")
            rules = config.get("rules", [])
        else:
            pattern = getattr(config, "model", "")
            rules = getattr(config, "rules", [])
        
        if pattern != "":
            if re.match(pattern, model_name):
                quant_config["rules"] = []
                if rules and not quant_config["type"]:
                    first_rule = rules[0]
                    if hasattr(first_rule, "get"):
                        quant_config["type"] = first_rule.get("type")
                    else:
                        quant_config["type"] = getattr(first_rule, "type", None)
                for index, rule in enumerate(rules):
                    if hasattr(rule, "get"):
                        rule_type = rule.get("type", None)
                        layers_spec = rule.get("layers", [])
                    else:
                        rule_type = getattr(rule, "type", None)
                        layers_spec = getattr(rule, "layers", [])
                    if not rule_type:
                        rule_type = quant_config["type"]
                    layers = expand_layers(layers_spec)
                    try:
                        OmegaConf.set_struct(rule, False)
                        rule.type = rule_type
                        rule.layers = layers
                        OmegaConf.set_struct(rule, True)
                    except:
                        pass  # Skip if not OmegaConf
                    quant_config["rules"].append(rule)
                try:
                    if isinstance(models, dict):
                        models["quant_config"] = quant_config
                    else:
                        models.quant_config = quant_config
                except:
                    pass
                return
    try:
        if isinstance(models, dict):
            models["quant_config"] = quant_config
        else:
            models.quant_config = quant_config
    except:
        pass


def set_backend_variables(global_args=None):
    if global_args is None:
        return

    # Handle SimpleNamespace objects
    if hasattr(global_args, "models"):
        models = global_args.models
    elif isinstance(global_args, dict):
        models = global_args.get("models", {})
    else:
        return  # Skip for unsupported types

    # Handle SimpleNamespace or dict for models
    if hasattr(models, "name"):
        model_name = models.name
    elif isinstance(models, dict):
        model_name = models.get("name")
    else:
        return  # Skip if models is not accessible

    if model_name is None or not isinstance(model_name, str):
        return  # Skip if model_name is not valid

    model_name = model_name.lower()
    
    # For tt-qwen type, skip backend config setup
    if hasattr(models, "type") and models.type == "tt-qwen":
        return

    # Handle backend_config access
    if hasattr(models, "backend_config"):
        backend_config_attr = models.backend_config
    elif isinstance(models, dict):
        backend_config_attr = models.get("backend_config", None)
    else:
        backend_config_attr = None

    if backend_config_attr is None:
        # Try to set backend_config if models is OmegaConf
        try:
            OmegaConf.set_struct(models, False)
            if isinstance(models, dict):
                models["backend_config"] = {"rules": []}
            else:
                models.backend_config = {"rules": []}
            OmegaConf.set_struct(models, True)
        except:
            pass  # Skip if not OmegaConf
        return

    backend_config = {"rules": []}
    
    # Extract backend list
    if hasattr(backend_config_attr, "get"):
        backend_list = backend_config_attr.get("backend", [])
    elif hasattr(backend_config_attr, "backend"):
        backend_list = getattr(backend_config_attr, "backend", [])
    else:
        backend_list = []

    for config in backend_list:
        if hasattr(config, "get"):
            pattern = config.get("model", "")
            rules = config.get("rules", [])
        else:
            pattern = getattr(config, "model", "")
            rules = getattr(config, "rules", [])
        
        if pattern != "":
            if re.match(pattern, model_name):
                backend_config["rules"] = []
                for index, rule in enumerate(rules):
                    if hasattr(rule, "get"):
                        backend = rule.get("backend", "default")
                        layers_spec = rule.get("layers", [])
                    else:
                        backend = getattr(rule, "backend", "default")
                        layers_spec = getattr(rule, "layers", [])
                    layers = expand_layers(layers_spec)
                    try:
                        OmegaConf.set_struct(rule, False)
                        rule.backend = backend
                        rule.layers = layers
                        OmegaConf.set_struct(rule, True)
                    except:
                        pass  # Skip if not OmegaConf
                    backend_config["rules"].append(rule)
                try:
                    if isinstance(models, dict):
                        models["backend_config"] = backend_config
                    else:
                        models.backend_config = backend_config
                except:
                    pass
                return

    try:
        if isinstance(models, dict):
            models["backend_config"] = backend_config
        else:
            models.backend_config = backend_config
    except:
        pass


def _set_debug(debug: bool):
    global _GLOBAL_DEBUG
    _GLOBAL_DEBUG = debug


def get_debug():
    return _GLOBAL_DEBUG


def _set_slot_handle(max_reqs, pp_size, dp_size, cache_type):
    global _GLOBAL_SLOT_HANDLE
    # _ensure_var_is_not_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    if cache_type == "skew" and dp_size <= 1:
        _GLOBAL_SLOT_HANDLE = SlotHandle(max_reqs, pp_size)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if (
        hasattr(args, "tensorboard_dir")
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            logger.info("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            logger.warning(
                "TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


@lru_cache(maxsize=1)
def get_global_args():
    _ensure_var_is_initialized(_GLOBAL_ARGS, "global args")
    
    # Handle SimpleNamespace objects (for tt-qwen)
    if isinstance(_GLOBAL_ARGS, types.SimpleNamespace):
        return _GLOBAL_ARGS
    
    # Try to convert OmegaConf config to object
    try:
        cfg: ServeConfig = OmegaConf.to_object(_GLOBAL_ARGS)
    except (ValueError, TypeError):
        # If conversion fails, return as-is (might be SimpleNamespace or other type)
        return _GLOBAL_ARGS

    if isinstance(cfg, dict) and "models" in cfg and isinstance(cfg["models"], dict):
        cfg["models"] = StaticConfig(cfg["models"])
    elif hasattr(cfg, "models") and isinstance(cfg.models, dict):
        cfg.models = StaticConfig(cfg.models)
    if isinstance(cfg, dict):
        return StaticConfig(cfg)
    return cfg


def set_global_args(args, need_ensure=True):
    global _GLOBAL_ARGS
    if need_ensure == True:
        _ensure_var_is_not_initialized(_GLOBAL_ARGS, "global args")
    _GLOBAL_ARGS = args
    get_global_args.cache_clear()


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    _ensure_var_is_not_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.cnt = 0

    def start(self):
        """Start the timer."""
        if not get_debug():
            return
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        if not get_debug():
            return
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False
        self.cnt += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.cnt = 0

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + "-time", value, iteration)

    def log(self, names=[], normalizer=1.0, reset=True):
        """Log a group of timers."""
        if len(names) == 0:
            names = self.timers.keys()
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            cnt = self.timers[name].cnt
            if cnt == 0:
                continue
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f} {} {:.2f}".format(
                name, elapsed_time, cnt, elapsed_time / cnt
            )
        logger.info(string)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


class SlotHandle:
    """
    split max_reqs to micro_batch size
    max_req = 10, pp_size = 3, self.slots_size = [4, 3, 3]
    """

    def __init__(self, max_reqs, pp_size):
        self.slots_size = self.split_slots(max_reqs, pp_size)
        self.num_slots = len(self.slots_size)
        self.slot_idx = 0
        self.slot_start_idx = []
        self.slot_end_idx = []

        res = [0]
        for value in self.slots_size:
            res.append(res[-1] + value)
        self.slot_start_idx = res[:-1]
        self.slot_end_idx = res[1:]

    def split_slots(self, total, parts):
        result = [0] * min(total, parts)
        for i in range(total):
            result[i % parts] += 1
        return result

    def get_slot_size(self, idx):
        return self.slots_size[idx]

    def set_slot_idx(self, idx):
        self.slot_idx = idx

    def get_slot_idx(self):
        return self.slot_idx

    def get_slot_start_end_idx(self, idx):
        return self.slot_start_idx[idx], self.slot_end_idx[idx]

    def get_current_slot_start_end_idx(self):
        return self.get_slot_start_end_idx(self.slot_idx)
