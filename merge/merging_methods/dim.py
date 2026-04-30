import sys
import re
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAVA_REPO = ""
sys.path.insert(0, LLAVA_REPO)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


class DiM:

    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path, **kwargs):

        above_avg_ratio = kwargs.get("above_average_value_ratio", 1.0)
        calib_val       = kwargs.get("score_calibration_value", 1.0)

        print("Loading models...")
        base_model  = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH,  torch_dtype=torch.float32)
        vl_model    = LlavaLlamaForCausalLM.from_pretrained(VL_MODEL_PATH,   torch_dtype=torch.float32)
        emma_model  = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH,  torch_dtype=torch.float32)

        vl_model.get_model().get_vision_tower().load_model()
        print("Vision tower mean:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        LM_EXCLUDE = ("vision_tower", "mm_projector")

        def is_lm_param(name: str) -> bool:
            return not any(exc in name for exc in LM_EXCLUDE)

        def extract_params(model) -> dict:
            return {
                n: p.data.clone().float()
                for n, p in model.named_parameters()
                if is_lm_param(n)
            }

        pretrained_params = extract_params(base_model)
        vl_params         = extract_params(vl_model)
        emma_params       = extract_params(emma_model)


        task_vector_vl   = {n: vl_params[n]   - pretrained_params[n]
                            for n in pretrained_params if n in vl_params}
        task_vector_emma = {n: emma_params[n]  - pretrained_params[n]
                            for n in pretrained_params if n in emma_params}

        def transpose_embed(param_dict: dict):
            key = "model.embed_tokens.weight"
            if key in param_dict:
                param_dict[key] = param_dict[key].T          # (vocab, dim) → (dim, vocab)

        def compute_magnitude_direction(param_dict: dict):
            mag_dict, dir_dict = {}, {}
            for name, param in tqdm(param_dict.items(), desc="Computing magnitude/direction"):
                if param.dim() != 2:
                    continue
                mag           = torch.norm(param, p=2, dim=0)          # (in_dim,)
                direction     = param / (mag + 1e-8)                    # (out_dim, in_dim)
                mag_dict[name]  = mag
                dir_dict[name]  = direction
            return mag_dict, dir_dict

        def compute_diff(pre_mag, pre_dir, ft_mag, ft_dir):
            mag_diff, dir_diff = {}, {}
            for name in pre_mag:
                mag_diff[name] = torch.abs(ft_mag[name] - pre_mag[name])
                dir_diff[name] = 1.0 - torch.cosine_similarity(
                    ft_dir[name], pre_dir[name], dim=0)
            return mag_diff, dir_diff


        def rank_within_model(diff_tensor: torch.Tensor) -> torch.Tensor:
            num_models, in_dim = diff_tensor.shape
            sort_idx   = torch.argsort(diff_tensor, dim=1, stable=True)
            rank_vals  = (torch.arange(in_dim).float() / in_dim).repeat(num_models, 1)
            rank_result = torch.zeros_like(rank_vals)
            rank_result.scatter_(1, sort_idx, rank_vals)
            return rank_result

        def compute_importance(significance: torch.Tensor) -> torch.Tensor:
            scores = torch.softmax(significance, dim=0)
            avg    = significance.mean(dim=1, keepdim=True)
            mask   = significance > (avg * above_avg_ratio)
            scores[mask] = calib_val
            return scores

        def merge_2d(delta_stack, pretrained, mag_rank, dir_rank):
            mag_scores  = compute_importance(mag_rank)            # (num_models, in_dim)
            dir_scores  = compute_importance(dir_rank)            # (num_models, in_dim)
            weight      = 0.5 * (mag_scores + dir_scores)         # (num_models, in_dim)
            merged_delta = (delta_stack * weight.unsqueeze(1)).sum(0)  # (out_dim, in_dim)
            return pretrained + merged_delta



        def merge_1d(delta_stack, pretrained, param_stack):
            param_diff   = torch.abs(param_stack - pretrained)    # (num_models, dim)
            scores       = compute_importance(param_diff)          # (num_models, dim)
            merged_delta = (delta_stack * scores).sum(0)           # (dim,)
            return pretrained + merged_delta

        with torch.no_grad():

            # --- transpose embeddings ---
            for d in (pretrained_params, vl_params, emma_params,
                      task_vector_vl, task_vector_emma):
                transpose_embed(d)

            pre_mag, pre_dir   = compute_magnitude_direction(pretrained_params)
            vl_mag,  vl_dir    = compute_magnitude_direction(vl_params)
            emma_mag, emma_dir = compute_magnitude_direction(emma_params)

            vl_mag_diff,   vl_dir_diff   = compute_diff(pre_mag, pre_dir, vl_mag,   vl_dir)
            emma_mag_diff, emma_dir_diff = compute_diff(pre_mag, pre_dir, emma_mag, emma_dir)

            param_names_2d = set(vl_mag_diff.keys())  

            merged_params = {}
            for param_name in tqdm(pretrained_params.keys(), desc="Merging parameters"):
                if param_name not in vl_params or param_name not in emma_params:
                    continue

                pretrained = pretrained_params[param_name]

                if param_name in param_names_2d:
                    delta_stack   = torch.stack([task_vector_vl[param_name],
                                                 task_vector_emma[param_name]], dim=0)   # (2, out, in)
                    mag_diff_stack = torch.stack([vl_mag_diff[param_name],
                                                  emma_mag_diff[param_name]], dim=0)     # (2, in)
                    dir_diff_stack = torch.stack([vl_dir_diff[param_name],
                                                  emma_dir_diff[param_name]], dim=0)     # (2, in)

                    mag_rank = rank_within_model(mag_diff_stack)
                    dir_rank = rank_within_model(dir_diff_stack)

                    merged_params[param_name] = merge_2d(
                        delta_stack, pretrained, mag_rank, dir_rank)
                else:
                    delta_stack  = torch.stack([task_vector_vl[param_name],
                                                task_vector_emma[param_name]], dim=0)   # (2, dim)
                    param_stack  = torch.stack([vl_params[param_name],
                                                emma_params[param_name]], dim=0)        # (2, dim)

                    merged_params[param_name] = merge_1d(
                        delta_stack, pretrained, param_stack)

            key = "model.embed_tokens.weight"
            if key in merged_params:
                merged_params[key] = merged_params[key].T    # (dim, vocab) → (vocab, dim)

        print("Writing merged params back to vl_model...")
        vl_param_map = dict(vl_model.named_parameters())
        for param_name, merged_value in merged_params.items():
            if param_name in vl_param_map:
                vl_param_map[param_name].data.copy_(
                    merged_value.to(vl_param_map[param_name].dtype))
                
        vl_model.model.embed_tokens = deepcopy(vl_model.model.embed_tokens)
        vl_model.lm_head = deepcopy(vl_model.lm_head)

        vl_model.config.vocab_size = base_model.config.vocab_size

        for name, module in vl_model.named_modules():
            if "vision_tower" in name:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

        print("Vision tower after half:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        print("Saving merged multimodal model to:", save_path)
        vl_model.save_pretrained(save_path)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.save_pretrained(save_path)

        print("Merge done.")
