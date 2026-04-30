from merging_methods.utils import merge_matrix
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


class SSFTaskMergePangeaQwen2():

    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path, **kwargs):

        lambda_1 = kwargs["lambda_vl"]
        lambda_2 = kwargs["lambda_emma"]
        scale = kwargs["scale"]

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype="auto")
        vl_model = LlavaQwenForCausalLM.from_pretrained(VL_MODEL_PATH, torch_dtype="auto")
        emma_model = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH, torch_dtype="auto")

        # load vision tower
        vision_tower = getattr(vl_model.model, "vision_tower", None)
        if vision_tower is not None and hasattr(vision_tower, "load_model"):
            vision_tower.load_model()
        if vision_tower is not None:
            print("Vision tower mean:",
                  next(vision_tower.parameters()).abs().mean())

        base_layers = base_model.model.layers
        vl_layers = vl_model.model.layers
        emma_layers = emma_model.model.layers

        for i in range(len(vl_layers)):
            vl_layer = vl_layers[i]
            base_layer = base_layers[i]
            emma_layer = emma_layers[i]

            base_params = dict(base_layer.named_parameters())
            emma_params = dict(emma_layer.named_parameters())

            for name, param in vl_layer.named_parameters():
                base_param = base_params[name]
                emma_param = emma_params[name]

                # --------------------------
                # 1D parameter (LayerNorm)
                # --------------------------
                if param.data.ndim == 1:
                    delta_vl = param.data - base_param.data
                    delta_emma = emma_param.data - base_param.data
                    merged = (
                        base_param.data
                        + lambda_1 * delta_vl
                        + lambda_2 * delta_emma
                    )
                    param.data.copy_(merged)

                # --------------------------
                # 2D parameter (Linear)
                # --------------------------
                elif param.data.ndim == 2:
                    delta_vl = param.data - base_param.data
                    delta_emma = emma_param.data - base_param.data

                    param_mat = torch.stack(
                        [delta_vl.float(), delta_emma.float()], dim=0
                    )

                    merged_delta = merge_matrix(param_mat)
                    merged = base_param.data.float() + merged_delta * scale

                    param.data.copy_(merged.to(param.dtype))

                else:
                    print("Skip parameter:", name)

        base_norm = base_model.model.norm.weight.data
        vl_norm = vl_model.model.norm.weight.data
        emma_norm = emma_model.model.norm.weight.data

        delta_vl = vl_norm - base_norm
        delta_emma = emma_norm - base_norm

        vl_model.model.norm.weight.data.copy_(
            base_norm
            + lambda_1 * delta_vl
            + lambda_2 * delta_emma
        )

        base_embed = base_model.model.embed_tokens.weight.data
        vl_embed = vl_model.model.embed_tokens.weight.data
        emma_embed = emma_model.model.embed_tokens.weight.data

        merged_embed = (
            base_embed
            + lambda_1 * (vl_embed - base_embed)
            + lambda_2 * (emma_embed - base_embed)
        )

        vl_model.model.embed_tokens.weight.data.copy_(merged_embed)

        base_head = base_model.lm_head.weight.data
        vl_head = vl_model.lm_head.weight.data
        emma_head = emma_model.lm_head.weight.data

        merged_head = (
            base_head
            + lambda_1 * (vl_head - base_head)
            + lambda_2 * (emma_head - base_head)
        )

        vl_model.lm_head.weight.data.copy_(merged_head)

        vl_model.config.vocab_size = base_model.config.vocab_size

        for name, module in vl_model.named_modules():
            if "vision_tower" in name:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

        if vision_tower is not None:
            print("Vision tower after half:",
                  next(vision_tower.parameters()).abs().mean())

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

        print("Merge done.")
