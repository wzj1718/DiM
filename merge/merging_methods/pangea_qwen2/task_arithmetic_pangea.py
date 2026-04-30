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


class TaskArithmeticPangeaQwen2():

    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path, **kwargs):

        base_path = BASE_MODEL_PATH
        llava_path = VL_MODEL_PATH
        emma_path = EMMA_MODEL_PATH
        lambda_1 = kwargs["lambda_1"]
        lambda_2 = kwargs["lambda_2"]

        base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
        vl_model = LlavaQwenForCausalLM.from_pretrained(llava_path, torch_dtype="auto")
        emma_model = AutoModelForCausalLM.from_pretrained(emma_path, torch_dtype="auto")

        vision_tower = getattr(vl_model.model, "vision_tower", None)
        if vision_tower is not None and hasattr(vision_tower, "load_model"):
            vision_tower.load_model()
        if vision_tower is not None:
            print("Vision tower mean:",
                  next(vision_tower.parameters()).abs().mean())

        for i in range(len(vl_model.model.layers)):

            vl_layer = vl_model.model.layers[i]
            base_layer = base_model.model.layers[i]
            emma_layer = emma_model.model.layers[i]

            base_params = dict(base_layer.named_parameters())
            emma_params = dict(emma_layer.named_parameters())

            for name, param in vl_layer.named_parameters():

                base_param = base_params[name]
                emma_param = emma_params[name]

                base_data = base_param.data.float()
                vl_data = param.data.float()
                emma_data = emma_param.data.float()

                delta_vl = vl_data - base_data
                delta_emma = emma_data - base_data

                merged = (
                    base_data
                    + lambda_1 * delta_vl
                    + lambda_2 * delta_emma
                )

                param.data.copy_(merged.to(param.dtype))


        base_norm = base_model.model.norm.weight.data.float()
        vl_norm = vl_model.model.norm.weight.data.float()
        emma_norm = emma_model.model.norm.weight.data.float()

        delta_vl = vl_norm - base_norm
        delta_emma = emma_norm - base_norm

        vl_model.model.norm.weight.data.copy_(
            (
                base_norm
                + lambda_1 * delta_vl
                + lambda_2 * delta_emma
            ).to(vl_model.model.norm.weight.dtype)
        )


        base_embed = base_model.model.embed_tokens.weight.data.float()
        vl_embed = vl_model.model.embed_tokens.weight.data.float()
        emma_embed = emma_model.model.embed_tokens.weight.data.float()

        merged_embed = (
            base_embed
            + lambda_1 * (vl_embed - base_embed)
            + lambda_2 * (emma_embed - base_embed)
        )

        vl_model.model.embed_tokens.weight.data.copy_(
            merged_embed.to(vl_model.model.embed_tokens.weight.dtype)
        )


        base_head = base_model.lm_head.weight.data.float()
        vl_head = vl_model.lm_head.weight.data.float()
        emma_head = emma_model.lm_head.weight.data.float()

        merged_head = (
            base_head
            + lambda_1 * (vl_head - base_head)
            + lambda_2 * (emma_head - base_head)
        )

        vl_model.lm_head.weight.data.copy_(
            merged_head.to(vl_model.lm_head.weight.dtype)
        )

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
            processor = AutoProcessor.from_pretrained(llava_path)
            processor.save_pretrained(save_path)
        except Exception as exc:
            print(f"AutoProcessor save skipped: {exc}")
            tokenizer = AutoTokenizer.from_pretrained(base_path)
            tokenizer.save_pretrained(save_path)

        print("Merge done.")
