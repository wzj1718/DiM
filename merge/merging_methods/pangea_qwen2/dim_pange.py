import os
import sys
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


LLAVA_REPO_CANDIDATES = [
    "LLaVA-NeXT",
    "multimodal",
]

for repo in LLAVA_REPO_CANDIDATES:
    llava_qwen_path = os.path.join(repo, "llava", "model", "language_model", "llava_qwen.py")
    if os.path.exists(llava_qwen_path):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        break

from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM


class DiMPangeaQwen2:

    @staticmethod
    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path, **kwargs):
        above_avg_ratio = kwargs.get("above_average_value_ratio", 1.0)
        calib_val = kwargs.get("score_calibration_value", 1.0)

        print("Loading Pangea/Qwen2 models...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.float32
        )
        vl_model = LlavaQwenForCausalLM.from_pretrained(
            VL_MODEL_PATH, torch_dtype=torch.float32
        )
        emma_model = AutoModelForCausalLM.from_pretrained(
            EMMA_MODEL_PATH, torch_dtype=torch.float32
        )

        vision_tower = getattr(vl_model.model, "vision_tower", None)
        if vision_tower is not None and hasattr(vision_tower, "load_model"):
            vision_tower.load_model()
        if vision_tower is not None:
            print(
                "Vision tower mean:",
                next(vision_tower.parameters()).abs().mean(),
            )

        lm_exclude = ("vision_tower", "mm_projector", "vision_resampler")

        def is_lm_param(name: str) -> bool:
            return not any(exc in name for exc in lm_exclude)

        def extract_params(model) -> dict:
            return {
                name: param.data.clone().float()
                for name, param in model.named_parameters()
                if is_lm_param(name)
            }

        pretrained_params = extract_params(base_model)
        vl_params = extract_params(vl_model)
        emma_params = extract_params(emma_model)

        common_param_names = sorted(
            set(pretrained_params) & set(vl_params) & set(emma_params)
        )
        print(f"Common text params: {len(common_param_names)}")

        task_vector_vl = {
            name: vl_params[name] - pretrained_params[name] for name in common_param_names
        }
        task_vector_emma = {
            name: emma_params[name] - pretrained_params[name] for name in common_param_names
        }

        def transpose_embed(param_dict: dict):
            key = "model.embed_tokens.weight"
            if key in param_dict:
                param_dict[key] = param_dict[key].T

        def compute_magnitude_direction(param_dict: dict):
            mag_dict, dir_dict = {}, {}
            for name, param in tqdm(param_dict.items(), desc="Computing magnitude/direction"):
                if name not in common_param_names or param.dim() != 2:
                    continue
                mag = torch.norm(param, p=2, dim=0)
                direction = param / (mag + 1e-8)
                mag_dict[name] = mag
                dir_dict[name] = direction
            return mag_dict, dir_dict

        def compute_diff(pre_mag, pre_dir, ft_mag, ft_dir):
            mag_diff, dir_diff = {}, {}
            for name in pre_mag:
                if name not in ft_mag or name not in ft_dir:
                    continue
                mag_diff[name] = torch.abs(ft_mag[name] - pre_mag[name])
                dir_diff[name] = 1.0 - torch.cosine_similarity(
                    ft_dir[name], pre_dir[name], dim=0
                )
            return mag_diff, dir_diff

        def rank_within_model(diff_tensor: torch.Tensor) -> torch.Tensor:
            num_models, in_dim = diff_tensor.shape
            sort_idx = torch.argsort(diff_tensor, dim=1, stable=True)
            rank_vals = (torch.arange(in_dim, device=diff_tensor.device).float() / in_dim).repeat(
                num_models, 1
            )
            rank_result = torch.zeros_like(rank_vals)
            rank_result.scatter_(1, sort_idx, rank_vals)
            return rank_result

        def compute_importance(significance: torch.Tensor) -> torch.Tensor:
            scores = torch.softmax(significance, dim=0)
            avg = significance.mean(dim=1, keepdim=True)
            mask = significance > (avg * above_avg_ratio)
            scores[mask] = calib_val
            return scores

        def merge_2d(delta_stack, pretrained, mag_rank, dir_rank):
            mag_scores = compute_importance(mag_rank)
            dir_scores = compute_importance(dir_rank)
            weight = 0.5 * (mag_scores + dir_scores)
            merged_delta = (delta_stack * weight.unsqueeze(1)).sum(0)
            return pretrained + merged_delta

        def merge_1d(delta_stack, pretrained, param_stack):
            param_diff = torch.abs(param_stack - pretrained)
            scores = compute_importance(param_diff)
            merged_delta = (delta_stack * scores).sum(0)
            return pretrained + merged_delta

        with torch.no_grad():
            for param_dict in (
                pretrained_params,
                vl_params,
                emma_params,
                task_vector_vl,
                task_vector_emma,
            ):
                transpose_embed(param_dict)

            pre_mag, pre_dir = compute_magnitude_direction(pretrained_params)
            vl_mag, vl_dir = compute_magnitude_direction(vl_params)
            emma_mag, emma_dir = compute_magnitude_direction(emma_params)

            vl_mag_diff, vl_dir_diff = compute_diff(pre_mag, pre_dir, vl_mag, vl_dir)
            emma_mag_diff, emma_dir_diff = compute_diff(pre_mag, pre_dir, emma_mag, emma_dir)

            param_names_2d = set(vl_mag_diff) & set(emma_mag_diff)
            merged_params = {}

            for param_name in tqdm(common_param_names, desc="Merging parameters"):
                pretrained = pretrained_params[param_name]

                if (
                    vl_params[param_name].shape != pretrained.shape
                    or emma_params[param_name].shape != pretrained.shape
                ):
                    print(f"Skipping shape mismatch: {param_name}")
                    continue

                if param_name in param_names_2d:
                    delta_stack = torch.stack(
                        [task_vector_vl[param_name], task_vector_emma[param_name]], dim=0
                    )
                    mag_diff_stack = torch.stack(
                        [vl_mag_diff[param_name], emma_mag_diff[param_name]], dim=0
                    )
                    dir_diff_stack = torch.stack(
                        [vl_dir_diff[param_name], emma_dir_diff[param_name]], dim=0
                    )
                    mag_rank = rank_within_model(mag_diff_stack)
                    dir_rank = rank_within_model(dir_diff_stack)
                    merged_params[param_name] = merge_2d(
                        delta_stack, pretrained, mag_rank, dir_rank
                    )
                else:
                    delta_stack = torch.stack(
                        [task_vector_vl[param_name], task_vector_emma[param_name]], dim=0
                    )
                    param_stack = torch.stack(
                        [vl_params[param_name], emma_params[param_name]], dim=0
                    )
                    merged_params[param_name] = merge_1d(
                        delta_stack, pretrained, param_stack
                    )

            embed_key = "model.embed_tokens.weight"
            if embed_key in merged_params:
                merged_params[embed_key] = merged_params[embed_key].T

        print("Writing merged params back to Pangea model...")
        vl_param_map = dict(vl_model.named_parameters())
        copied = 0
        for param_name, merged_value in merged_params.items():
            if param_name not in vl_param_map:
                continue
            vl_param_map[param_name].data.copy_(
                merged_value.to(vl_param_map[param_name].dtype)
            )
            copied += 1
        print(f"Copied merged params: {copied}")

        input_embeddings = vl_model.get_input_embeddings()
        if input_embeddings is not None:
            vl_model.model.embed_tokens = deepcopy(input_embeddings)

        output_embeddings = vl_model.get_output_embeddings()
        if output_embeddings is not None:
            vl_model.lm_head = deepcopy(output_embeddings)

        vl_model.config.vocab_size = base_model.config.vocab_size

        for name, module in vl_model.named_modules():
            if any(tag in name for tag in ("vision_tower", "mm_projector", "vision_resampler")):
                continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

        if vision_tower is not None:
            print(
                "Vision tower after half:",
                next(vision_tower.parameters()).abs().mean(),
            )

        print("Saving merged multimodal model to:", save_path)
        os.makedirs(save_path, exist_ok=True)
        vl_model.save_pretrained(save_path)

        processor_saved = False
        try:
            processor = AutoProcessor.from_pretrained(VL_MODEL_PATH)
            processor.save_pretrained(save_path)
            processor_saved = True
        except Exception as exc:
            print(f"AutoProcessor save skipped: {exc}")

        if not processor_saved:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            tokenizer.save_pretrained(save_path)

        print("Merge done.")
