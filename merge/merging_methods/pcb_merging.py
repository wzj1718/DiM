from merging_methods.utils import *
import torch
import sys
LLAVA_REPO = ""
sys.path.insert(0, LLAVA_REPO)
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def normalize(x, dim=0):
        min_values, _ = torch.min(x, dim=dim, keepdim=True)
        max_values, _ = torch.max(x, dim=dim, keepdim=True)
        return (x - min_values) / (max_values - min_values + 1e-12)

def clamp(x, min_ratio=0.0001, max_ratio=0.0001):
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min_val = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max_val = sorted_x[:, int(d * (1 - max_ratio) - 1)].unsqueeze(1)
        return torch.clamp(x, min_val, max_val)

def act(x):
        return torch.tanh(x)

def pcb_merge_logic( task_vectors, pcb_ratio=0.1):

        n, d = task_vectors.shape

        abs_vectors=task_vectors
        clamped_vectors=task_vectors

        self_pcb = normalize(abs_vectors, dim=1)**2
        self_pcb_act = torch.exp(n * self_pcb)

        sum_trend = torch.sum(task_vectors, dim=0)
        cross_pcb = task_vectors * sum_trend
        cross_pcb_act = act(cross_pcb)

        task_pcb = self_pcb_act * cross_pcb_act
        
        scale = normalize(clamp(task_pcb, 1 - pcb_ratio, 0), dim=1)
        
        merged_vector = torch.sum(clamped_vectors * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
        return merged_vector





class PCBMerger():

    def merge( BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path,  **kwargs):

        pcb_ratio = kwargs["pcb_ratio"]

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")
        vl_model   = LlavaLlamaForCausalLM.from_pretrained(VL_MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")
        emma_model = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH, torch_dtype=torch.float32, device_map="cpu")

        vl_model.get_model().get_vision_tower().load_model()
        print("Vision tower mean:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        for i in range(len(vl_model.model.layers)):
            vl_layer = vl_model.model.layers[i]
            base_layer = base_model.model.layers[i]
            emma_layer = emma_model.model.layers[i]

            base_params = dict(base_layer.named_parameters())
            emma_params = dict(emma_layer.named_parameters())

            for name, param in vl_layer.named_parameters():
                b_p = base_params[name].data
                e_p = emma_params[name].data
                v_p = param.data

                delta_vl = v_p - b_p
                delta_emma = e_p - b_p

                original_shape = delta_vl.shape
                # print("original_shape:",original_shape)
                combined_deltas = torch.stack([delta_vl.view(-1), delta_emma.view(-1)], dim=0)
                # print("combined_deltas:",combined_deltas.shape)
# 
                merged_delta = pcb_merge_logic(combined_deltas, pcb_ratio=pcb_ratio)

                new_weight = b_p + merged_delta.view(original_shape)
                param.data.copy_(new_weight)
            
            if i % 5 == 0: print(f"Layer {i} merged.")

        base_norm = base_model.model.norm.weight.data
        vl_norm = vl_model.model.norm.weight.data
        emma_norm = emma_model.model.norm.weight.data

        norm_deltas = torch.stack([
            (vl_norm - base_norm).view(-1),
            (emma_norm - base_norm).view(-1)
        ], dim=0)
        merged_norm = pcb_merge_logic(norm_deltas, pcb_ratio=pcb_ratio)
        vl_model.model.norm.weight.data.copy_(base_norm + merged_norm.view_as(vl_norm))

        base_embed = base_model.model.embed_tokens.weight.data
        vl_embed = vl_model.model.embed_tokens.weight.data
        emma_embed = emma_model.model.embed_tokens.weight.data

        embed_rows = min(vl_embed.shape[0], base_embed.shape[0], emma_embed.shape[0])
        embed_deltas = torch.stack([
            (vl_embed[:embed_rows] - base_embed[:embed_rows]).view(-1),
            (emma_embed[:embed_rows] - base_embed[:embed_rows]).view(-1)
        ], dim=0)
        merged_embed = pcb_merge_logic(embed_deltas, pcb_ratio=pcb_ratio)
        vl_model.model.embed_tokens.weight.data[:embed_rows].copy_(
            base_embed[:embed_rows] + merged_embed.view_as(vl_embed[:embed_rows])
        )

        base_head = base_model.lm_head.weight.data
        vl_head = vl_model.lm_head.weight.data
        emma_head = emma_model.lm_head.weight.data

        head_rows = min(vl_head.shape[0], base_head.shape[0], emma_head.shape[0])
        head_deltas = torch.stack([
            (vl_head[:head_rows] - base_head[:head_rows]).view(-1),
            (emma_head[:head_rows] - base_head[:head_rows]).view(-1)
        ], dim=0)
        merged_head = pcb_merge_logic(head_deltas, pcb_ratio=pcb_ratio)
        vl_model.lm_head.weight.data[:head_rows].copy_(
            base_head[:head_rows] + merged_head.view_as(vl_head[:head_rows])
        )

        vl_model.config.vocab_size = base_model.config.vocab_size

        for name, module in vl_model.named_modules():
            if "vision_tower" in name: continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

        print("Vision tower after half:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        print(f">>> Saving merged model to: {save_path}")
        vl_model.save_pretrained(save_path)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.save_pretrained(save_path)
        print(">>> PCB Merge Complete.")