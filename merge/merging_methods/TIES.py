from merging_methods.ties_merging_utils import *
from merging_methods.utils import *
import torch
import sys
LLAVA_REPO = ""
sys.path.insert(0, LLAVA_REPO)
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

class TIES():

    def merge(BASE_MODEL_PATH,VL_MODEL_PATH,EMMA_MODEL_PATH,save_path, **kwargs):

        scaling_coef = kwargs["scaling_coef"]
        K = kwargs["K"]
        merge_func = kwargs["merge_func"]
        lambdas=kwargs["lambdas"]

        def merge_weight_like_norm(vl_weight, base_weight, emma_weight):
            delta_vl = vl_weight - base_weight
            delta_emma = emma_weight - base_weight

            stacked = torch.stack([
                delta_vl.view(-1),
                delta_emma.view(-1)
            ])

            merged_delta = ties_merging(
                stacked,
                lambdas=lambdas,
                reset_thresh=K,
                merge_func=merge_func
            )

            merged_delta = scaling_coef * merged_delta
            return base_weight + merged_delta.view_as(vl_weight)

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype="auto")
        vl_model = LlavaLlamaForCausalLM.from_pretrained(VL_MODEL_PATH, torch_dtype="auto")
        emma_model = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH, torch_dtype="auto")

        vl_model.get_model().get_vision_tower().load_model()
        print("Vision tower mean:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        with torch.no_grad():
            for i in range(len(vl_model.model.layers)):

                vl_layer = vl_model.model.layers[i]
                base_layer = base_model.model.layers[i]
                emma_layer = emma_model.model.layers[i]

                base_params = dict(base_layer.named_parameters())
                emma_params = dict(emma_layer.named_parameters())

                for name, param in vl_layer.named_parameters():

                    if name not in base_params:
                        continue

                    base_param = base_params[name]
                    emma_param = emma_params[name]

                    delta_vl = param.data - base_param.data
                    delta_emma = emma_param.data - base_param.data

                    stacked = torch.stack([
                        delta_vl.view(-1),
                        delta_emma.view(-1)
                    ])

                    merged_delta = ties_merging(
                        stacked,
                        lambdas=lambdas,
                        reset_thresh=K,
                        merge_func=merge_func
                    )

                    merged_delta = scaling_coef * merged_delta
                    merged_delta = merged_delta.view_as(param.data)

                    new_weight = base_param.data + merged_delta
                    param.data.copy_(new_weight)

        base_norm = base_model.model.norm.weight.data
        vl_norm = vl_model.model.norm.weight.data
        emma_norm = emma_model.model.norm.weight.data

        vl_model.model.norm.weight.data.copy_(
            merge_weight_like_norm(vl_norm, base_norm, emma_norm)
        )


        vl_embed = vl_model.model.embed_tokens.weight.data
        base_embed = base_model.model.embed_tokens.weight.data
        emma_embed = emma_model.model.embed_tokens.weight.data

        embed_merge_rows = min(vl_embed.shape[0], base_embed.shape[0], emma_embed.shape[0])
        vl_embed[:embed_merge_rows].copy_(
            merge_weight_like_norm(
                vl_embed[:embed_merge_rows],
                base_embed[:embed_merge_rows],
                emma_embed[:embed_merge_rows]
            )
        )

        vl_lm_head = vl_model.lm_head.weight.data
        base_lm_head = base_model.lm_head.weight.data
        emma_lm_head = emma_model.lm_head.weight.data

        lm_head_merge_rows = min(vl_lm_head.shape[0], base_lm_head.shape[0], emma_lm_head.shape[0])
        vl_lm_head[:lm_head_merge_rows].copy_(
            merge_weight_like_norm(
                vl_lm_head[:lm_head_merge_rows],
                base_lm_head[:lm_head_merge_rows],
                emma_lm_head[:lm_head_merge_rows]
            )
        )

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

        print("TIES merge done.")


