"""LoRa engine class for managing LoRa weight."""
from typing import Dict, List, Tuple

from transformers import PretrainedConfig, OPTConfig, LlamaConfig

import torch
from vllm.config import LoRaConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

class LoRaWeight:

    def __init__(
        self,
        lora_engine,
        max_num: int,
        in_features: int,
        max_rank: int,
        out_features: int,
        dtype: torch.dtype,
        device: str):

        self.lora_engine = lora_engine
        if device == "cpu":
            self.lora_As = torch.zeros((max_num, in_features, max_rank),
                                dtype=dtype, device=device, pin_memory=True)
            self.lora_Bs = torch.zeros((max_num, max_rank, out_features),
                                dtype=dtype, device=device, pin_memory=True)
        else:
            self.lora_As = torch.zeros((max_num, in_features, max_rank),
                                dtype=dtype, device=device)
            self.lora_Bs = torch.zeros((max_num, max_rank, out_features),
                                dtype=dtype, device=device)
        self.active_lora_As = self.lora_As
        self.active_lora_Bs = self.lora_Bs
        self.active_idx = []
        self.merged_adapter = None

    def adjust_lora_adapter(self, active_idx: List[int]):
        self.active_idx = active_idx.copy()
        if len(active_idx) == 1:
            return
        if len(active_idx) == self.lora_engine.lora_config.gpu_capacity:
            self.active_lora_As = self.lora_As
            self.active_lora_Bs = self.lora_Bs
        else:
            self.active_lora_As = self.lora_As[active_idx]
            self.active_lora_Bs = self.lora_Bs[active_idx]

    def merge(self, weight: torch.Tensor, idx: int):
        if self.merged_adapter != None:
            self.unmerge(weight)
        self.merged_adapter = idx
        weight.addmm_(self.active_lora_Bs[idx].T, self.active_lora_As[idx].T)

    def unmerge(self, weight: torch.Tensor):
        if self.merged_adapter == None:
            return
        idx = self.merged_adapter
        weight.addmm_(self.active_lora_Bs[idx].T, self.active_lora_As[idx].T, alpha=-1.0)
        self.merged_adapter = None

    def copy_from_cpu(self, idx: int, cpu_A: torch.Tensor, cpu_B: torch.Tensor):
        self.lora_As[idx].copy_(cpu_A, non_blocking=True)
        self.lora_Bs[idx].copy_(cpu_B, non_blocking=True)



class LoRaEngine:
    lora_type = Tuple[LoRaWeight, LoRaWeight, LoRaWeight, LoRaWeight]
    lora_cnt = 4

    def __init__(
        self,
        config,
        model_config: ModelConfig,
        lora_config: LoRaConfig,
        parallel_config: ParallelConfig,
        device: torch.device
    ):

        self.model_config = model_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.device = device
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers // self.parallel_config.pipeline_parallel_size

        self.num_gpu_lora = 0
        self.gpu_lora_models = []
        self.active = []
        self.merged_adapter = None

        self.gpu_lora_weights: List[self.lora_type] = []
        self.cpu_lora_weights: List[self.lora_type] = []

        # LoRA swap tracking counters
        self.is_initialized = False  # Flag to distinguish init vs runtime swaps
        # Initialization swap counters
        self.init_swap_in_count = 0  # Models swapped during initialization
        self.init_swap_call_count = 0  # Swap operations during initialization
        # Runtime swap counters  
        self.runtime_swap_in_count = 0  # Models swapped during runtime
        self.runtime_swap_call_count = 0  # Swap operations during runtime

        # Initialize the stream for lora swap operations.
        self.lora_stream = torch.cuda.Stream()
        assert self.lora_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_hidden_layers)]

    def copy_into_gpu(self, gpu_idx: int, cpu_idx: int):
        for i in range(self.num_hidden_layers):
            for j in range(self.lora_cnt):
                self.gpu_lora_weights[i][j].copy_from_cpu(gpu_idx,
                    self.cpu_lora_weights[i][j].lora_As[cpu_idx], self.cpu_lora_weights[i][j].lora_Bs[cpu_idx])
            event = self.events[i]
            event.record(stream=self.lora_stream)

    def set_gpu_lora(self, gpu_lora_list: List[int]) -> bool:
        assert len(gpu_lora_list) <= self.lora_config.gpu_capacity
        if gpu_lora_list == self.gpu_lora_models:
            return False
        self.num_gpu_lora = len(gpu_lora_list)
        models_swapped = 0

        with torch.cuda.stream(self.lora_stream):
            for idx, model_id in enumerate(gpu_lora_list):
                if idx < len(self.gpu_lora_models) and self.gpu_lora_models[idx] == model_id:
                    continue
                models_swapped += 1
                self.copy_into_gpu(idx, model_id)

        if models_swapped > 0:
            if self.is_initialized:
                self.runtime_swap_call_count += 1
                self.runtime_swap_in_count += models_swapped
                logger.info(f"LoRA runtime swap #{self.runtime_swap_call_count}: {models_swapped} models swapped in, total runtime swaps: {self.runtime_swap_in_count}")
            else:
                self.init_swap_call_count += 1
                self.init_swap_in_count += models_swapped
                logger.info(f"LoRA init swap #{self.init_swap_call_count}: {models_swapped} models swapped in")

        self.gpu_lora_models = gpu_lora_list.copy()
        return models_swapped > 0
            
    def adjust_lora_adapter(self, gpu_model_list: List[int], active_model_list: List[int]) -> bool:
        swapped = self.set_gpu_lora(gpu_model_list)
        active_idx = []
        for model_id in active_model_list:
            idx = self.gpu_lora_models.index(model_id)
            active_idx.append(idx)

        for i in range(self.num_hidden_layers):
            for j in range(self.lora_cnt):
                self.gpu_lora_weights[i][j].adjust_lora_adapter(active_idx)

        return swapped
    
    def get_size(self):
        dtype_size = _get_dtype_size(self.model_config.dtype)
        return dtype_size * sum([self.lora_A_shapes[i][0] * self.lora_A_shapes[i][1] + self.lora_B_shapes[i][0] * self.lora_B_shapes[i][1] for i in range(self.lora_cnt)]) * self.num_hidden_layers

    def mark_initialized(self):
        """Mark initialization as complete. Future swaps will be tracked as runtime swaps.
        
        This is called once at the end of allocate_gpu_lora_weight().
        If called multiple times, only the first call takes effect.
        """
        if self.is_initialized:
            logger.warning("mark_initialized() called but already initialized, ignoring")
            return
        self.is_initialized = True
        logger.info(f"LoRA engine initialized with {self.init_swap_in_count} models loaded in {self.init_swap_call_count} operations")

    def get_swap_stats(self):
        """Returns swap statistics: (init_calls, init_swaps, runtime_calls, runtime_swaps)"""
        return (self.init_swap_call_count, self.init_swap_in_count, 
                self.runtime_swap_call_count, self.runtime_swap_in_count)
    
    def reset_swap_stats(self):
        """Resets only runtime swap counters (init stats preserved)."""
        self.runtime_swap_call_count = 0
        self.runtime_swap_in_count = 0


class OPTLoRaEngine(LoRaEngine):

    lora_type = Tuple[LoRaWeight, LoRaWeight, LoRaWeight, LoRaWeight] # qkv_proj, out_proj, fc1, fc2
    lora_cnt = 4

    def __init__(
        self,
        config: OPTConfig,
        model_config: ModelConfig,
        lora_config: LoRaConfig,
        parallel_config: ParallelConfig,
        device: torch.device
    ):
        LoRaEngine.__init__(self, config, model_config, lora_config, parallel_config, device)

        self.ffn_dim = config.ffn_dim

        self.lora_A_shapes = [(self.hidden_size, 3 * self.lora_config.max_r), (self.hidden_size // self.parallel_config.tensor_parallel_size, self.lora_config.max_r), (self.hidden_size, self.lora_config.max_r), (self.ffn_dim // self.parallel_config.tensor_parallel_size, self.lora_config.max_r)]
        self.lora_B_shapes = [(3 * self.lora_config.max_r, 3 * self.hidden_size // self.parallel_config.tensor_parallel_size), (self.lora_config.max_r, self.hidden_size), (self.lora_config.max_r, self.ffn_dim // self.parallel_config.tensor_parallel_size), (self.lora_config.max_r, self.hidden_size)]


        for i in range(self.num_hidden_layers):
            self.cpu_lora_weights.append((LoRaWeight(self, self.lora_config.num_models, self.hidden_size, 3 * self.lora_config.max_r,
                                        3 * self.hidden_size // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.hidden_size // self.parallel_config.tensor_parallel_size, 
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.hidden_size, self.lora_config.max_r,
                                        self.ffn_dim // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.ffn_dim // self.parallel_config.tensor_parallel_size,
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device='cpu')))

    def allocate_gpu_lora_weight(self, lora_types: List[int]):

        self.lora_config.gpu_capacity = len(lora_types)

        for i in range(self.num_hidden_layers):
            self.gpu_lora_weights.append((LoRaWeight(self, self.lora_config.gpu_capacity, self.hidden_size, 3 * self.lora_config.max_r,
                                        3 * self.hidden_size // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.hidden_size // self.parallel_config.tensor_parallel_size, 
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.hidden_size, self.lora_config.max_r,
                                        self.ffn_dim // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.ffn_dim // self.parallel_config.tensor_parallel_size,
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device=self.device)))
            
        self.adjust_lora_adapter(lora_types, lora_types)
        self.mark_initialized()

class LlamaLoRaEngine(LoRaEngine):
    lora_type = Tuple[LoRaWeight, LoRaWeight, LoRaWeight, LoRaWeight] # qkv_proj, o_proj, gate_up_proj, down_proj
    lora_cnt = 4

    def __init__(
        self,
        config: LlamaConfig,
        model_config: ModelConfig,
        lora_config: LoRaConfig,
        parallel_config: ParallelConfig,
        device: torch.device
    ):
        
        LoRaEngine.__init__(self, config, model_config, lora_config, parallel_config, device)

        self.intermediate_size = config.intermediate_size
        self.total_num_heads=config.num_attention_heads
        self.total_num_kv_heads=config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.qkv_dim = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim

        self.lora_A_shapes = [(self.hidden_size, 3 * self.lora_config.max_r), (self.total_num_heads * self.head_dim // self.parallel_config.tensor_parallel_size, self.lora_config.max_r), (self.hidden_size, self.lora_config.max_r), (self.intermediate_size // self.parallel_config.tensor_parallel_size, self.lora_config.max_r)]
        self.lora_B_shapes = [(3 * self.lora_config.max_r, self.qkv_dim // self.parallel_config.tensor_parallel_size), (self.lora_config.max_r, self.hidden_size), (self.lora_config.max_r, 2 * self.intermediate_size // self.parallel_config.tensor_parallel_size), (self.lora_config.max_r, self.hidden_size)]

        for i in range(self.num_hidden_layers):
            self.cpu_lora_weights.append((LoRaWeight(self, self.lora_config.num_models, self.hidden_size, 3 * self.lora_config.max_r,
                                        self.qkv_dim // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.total_num_heads * self.head_dim // self.parallel_config.tensor_parallel_size, 
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.hidden_size, self.lora_config.max_r,
                                        2 * self.intermediate_size // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device='cpu'),
                                        LoRaWeight(self, self.lora_config.num_models, self.intermediate_size // self.parallel_config.tensor_parallel_size,
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device='cpu')))
            
    def allocate_gpu_lora_weight(self, lora_types: List[int]):

        self.lora_config.gpu_capacity = len(lora_types)

        for i in range(self.num_hidden_layers):
            self.gpu_lora_weights.append((LoRaWeight(self, self.lora_config.gpu_capacity, self.hidden_size, 3 * self.lora_config.max_r,
                                        self.qkv_dim // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.total_num_heads * self.head_dim // self.parallel_config.tensor_parallel_size, 
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.hidden_size, self.lora_config.max_r,
                                        2 * self.intermediate_size // self.parallel_config.tensor_parallel_size, self.model_config.dtype, device=self.device),
                                        LoRaWeight(self, self.lora_config.gpu_capacity, self.intermediate_size // self.parallel_config.tensor_parallel_size,
                                        self.lora_config.max_r, self.hidden_size, self.model_config.dtype, device=self.device)))
            
        self.adjust_lora_adapter(lora_types, lora_types)
        self.mark_initialized()



_MODEL_LORA_MAPPING = {
    "opt": OPTLoRaEngine,
    "llama": LlamaLoRaEngine,
}


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
