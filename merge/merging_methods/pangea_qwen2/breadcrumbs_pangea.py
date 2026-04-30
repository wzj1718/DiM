import re
from merging_methods.utils import *
import torch
import sys
import os

LLAVA_REPO_CANDIDATES = [
    "LLaVA-NeXT",
    "multimodal",
]

for repo in LLAVA_REPO_CANDIDATES:
    llava_qwen_path = os.path.join(repo, "llava", "model", "language_model", "llava_qwen.py")
    if os.path.exists(llava_qwen_path):
        sys.path.insert(0, repo)
        break

from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM


class BreadcrumbsMergingPangeaQwen2:

    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path, **kwargs):

        param_density = kwargs.get("param_density", 0.7)
        param_value_mask_rate = kwargs.get("param_value_mask_rate", 0.09)
        scaling_coefficient = kwargs.get("scaling_coefficient", 1.0)
        lambda_1 = kwargs.get("lambda_1", 0.9)
        lambda_2 = kwargs.get("lambda_2", 0.1)

        # exclude_param_names_regex = kwargs.get("exclude_param_names_regex", [])

        assert isinstance(scaling_coefficient, float), "scaling_coefficient must be float"

        # --------------------------------------------------
        #  Load models
        # --------------------------------------------------
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype="auto")
        vl_model = LlavaQwenForCausalLM.from_pretrained(VL_MODEL_PATH, torch_dtype="auto")
        emma_model = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH, torch_dtype="auto")

        # Load vision tower explicitly
        vision_tower = getattr(vl_model.model, "vision_tower", None)
        if vision_tower is not None and hasattr(vision_tower, "load_model"):
            vision_tower.load_model()

        if vision_tower is not None:
            print("Vision tower mean before merging:",
                  next(vision_tower.parameters()).abs().mean())

        # --------------------------------------------------
        #  mask 
        # --------------------------------------------------
        def mask_smallest_largest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_density: float = 0.9, param_value_mask_rate: float = 0.8):

            num_mask_params = int(flattened_models_to_merge_param.shape[1] * (1 - param_density))
            num_mask_smallest_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
            num_mask_largest_params = num_mask_params - num_mask_smallest_params

            assert num_mask_smallest_params >= 0 and num_mask_largest_params >= 0

            kth_smallest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_smallest_params, dim=1, keepdim=True)

            smallest_mask = flattened_models_to_merge_param.abs() >= kth_smallest_values

            kth_largest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=flattened_models_to_merge_param.shape[1] - num_mask_largest_params, dim=1, keepdim=True)

            largest_mask = flattened_models_to_merge_param.abs() <= kth_largest_values


            mask = smallest_mask & largest_mask

            return flattened_models_to_merge_param * mask

        # --------------------------------------------------
        # --------------------------------------------------
        for i in range(len(vl_model.model.layers)):

            vl_layer = vl_model.model.layers[i]
            base_layer = base_model.model.layers[i]
            emma_layer = emma_model.model.layers[i]

            base_params = dict(base_layer.named_parameters())
            emma_params = dict(emma_layer.named_parameters())

            for name, vl_param in vl_layer.named_parameters():
                base_param = base_params[name]
                emma_param = emma_params[name]

                delta_vl = vl_param.data - base_param.data
                delta_emma = emma_param.data - base_param.data

                flat = torch.vstack([
                    delta_vl.flatten(),
                    delta_emma.flatten()
                ])
                # mask
                masked = mask_smallest_largest_magnitude_param_values(flat, param_density, param_value_mask_rate)
                # sum task vectors
                # merged_delta = masked.sum(dim=0).reshape(delta_vl.shape)
                masked_vl = masked[0]
                masked_emma = masked[1]
                merged_delta = (lambda_1 * masked_vl + lambda_2 * masked_emma).reshape(delta_vl.shape)
                # combine with base
                merged_param = base_param.data + scaling_coefficient * merged_delta
                vl_param.data.copy_(merged_param)

        # --------------------------------------------------
        # 4️ Final LayerNorm
        # --------------------------------------------------
        base_norm = base_model.model.norm.weight.data
        vl_norm = vl_model.model.norm.weight.data
        emma_norm = emma_model.model.norm.weight.data

        delta_vl = vl_norm - base_norm
        delta_emma = emma_norm - base_norm

        flat = torch.vstack([
            delta_vl.flatten(),
            delta_emma.flatten()
        ])

        masked = mask_smallest_largest_magnitude_param_values(flat, param_density, param_value_mask_rate)

        masked_vl = masked[0]
        masked_emma = masked[1]
        merged_delta = (lambda_1 * masked_vl + lambda_2 * masked_emma).reshape(base_norm.shape)

        vl_model.model.norm.weight.data.copy_(base_norm + scaling_coefficient * merged_delta)

        # --------------------------------------------------
        # 5️ VL embedding & lm_head
        # --------------------------------------------------
        # vl_model.model.embed_tokens = deepcopy(vl_model.model.embed_tokens)
        # vl_model.lm_head = deepcopy(vl_model.lm_head)

        base_embed = base_model.model.embed_tokens.weight.data
        vl_embed = vl_model.model.embed_tokens.weight.data
        emma_embed = emma_model.model.embed_tokens.weight.data

        delta_embed_vl = vl_embed - base_embed
        delta_embed_emma = emma_embed - base_embed

        flat = torch.vstack([
            delta_embed_vl.flatten(),
            delta_embed_emma.flatten()
        ])

        masked = mask_smallest_largest_magnitude_param_values(flat, param_density, param_value_mask_rate)

        masked_vl = masked[0]
        masked_emma = masked[1]
        merged_delta = (lambda_1 * masked_vl + lambda_2 * masked_emma).reshape(base_embed.shape)

        vl_model.model.embed_tokens.weight.data.copy_(base_embed + scaling_coefficient * merged_delta)

        # ==========================
        #  lm_head
        # ==========================
        base_head = base_model.lm_head.weight.data
        vl_head = vl_model.lm_head.weight.data
        emma_head = emma_model.lm_head.weight.data

        delta_head_vl = vl_head - base_head
        delta_head_emma = emma_head - base_head

        flat = torch.vstack([
            delta_head_vl.flatten(),
            delta_head_emma.flatten()
        ])

        masked = mask_smallest_largest_magnitude_param_values(flat, param_density, param_value_mask_rate)

        # merged_delta = masked.sum(dim=0).reshape(base_norm.shape)
        masked_vl = masked[0]
        masked_emma = masked[1]
        merged_delta = (lambda_1 * masked_vl + lambda_2 * masked_emma).reshape(base_head.shape)

        vl_model.lm_head.weight.data.copy_(base_head + scaling_coefficient * merged_delta)

        # ==============================================
        #  vocab_size
        vl_model.config.vocab_size = base_model.config.vocab_size

        # --------------------------------------------------
        # --------------------------------------------------
        for name, module in vl_model.named_modules():
            if "vision_tower" in name:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

        if vision_tower is not None:
            print("Vision tower mean after half:",
                  next(vision_tower.parameters()).abs().mean())

        # --------------------------------------------------
        # --------------------------------------------------
        print("Saving merged multimodal model to:", save_path)
        os.makedirs(save_path, exist_ok=True)
        vl_model.save_pretrained(save_path)

        try:
            processor = AutoProcessor.from_pretrained(VL_MODEL_PATH)
            processor.save_pretrained(save_path)
        except Exception as exc:
            print(f"AutoProcessor save skipped: {exc}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            tokenizer.save_pretrained(save_path)

        print("Breadcrumbs merging done.")
