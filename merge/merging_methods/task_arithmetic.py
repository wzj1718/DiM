from merging_methods.utils import *
import torch
import sys
LLAVA_REPO = ""
sys.path.insert(0, LLAVA_REPO)
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

class TaskArithmetic():

    def merge(BASE_MODEL_PATH,VL_MODEL_PATH,EMMA_MODEL_PATH,save_path, **kwargs):

        base_path= BASE_MODEL_PATH
        llava_path = VL_MODEL_PATH
        emma_path = EMMA_MODEL_PATH
        lambda_1 = kwargs["lambda_1"]
        lambda_2 = kwargs["lambda_2"]

        base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto")
        vl_model   = LlavaLlamaForCausalLM.from_pretrained(llava_path, torch_dtype="auto")
        emma_model = AutoModelForCausalLM.from_pretrained(emma_path, torch_dtype="auto")

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

                base_param = base_params[name]
                emma_param = emma_params[name]

                delta_vl = param.data - base_param.data
                delta_emma = emma_param.data - base_param.data

                merged = (
                    base_param.data
                    + lambda_1 * delta_vl
                    + lambda_2 * delta_emma
                )

                param.data.copy_(merged)

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

        print("Vision tower after half:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())


        print("Saving merged multimodal model to:", save_path)
        vl_model.save_pretrained(save_path)

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        tokenizer.save_pretrained(save_path)

        print("Merge done.")