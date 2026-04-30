import torch
import sys
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

LLAVA_REPO = ""
sys.path.insert(0, LLAVA_REPO)


class NeuroTaskArithmetic:

    def resolve_sign(Tensor):
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
        majority_sign = torch.sign(sign_to_mult.sum())
        sign_to_mult[sign_to_mult == 0] = majority_sign
        return sign_to_mult

    def disjoint_merge(Tensor, sign_to_mult):
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )

        selected_entries = Tensor * rows_to_keep

        non_zero_counts = (selected_entries != 0).sum(dim=0).float()

        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )

        return disjoint_aggs

    def ties_kernel(pt_w, ft_ws):

        original_shape = pt_w.shape

        ft_ws_flat = torch.stack([ft_w.view(-1) for ft_w in ft_ws], dim=0)

        final_signs = NeuroTaskArithmetic.resolve_sign(ft_ws_flat)

        merged_tv = NeuroTaskArithmetic.disjoint_merge(
            ft_ws_flat, final_signs
        )

        return merged_tv.view(original_shape)

    def inner_ties_kernel(scales):

        sign = torch.sign(scales.sum(dim=0))
        majority_sign = torch.sign(sign.sum())

        sign[sign == 0] = majority_sign

        keep_pos = torch.where(
            sign.unsqueeze(0) > 0,
            scales > 0,
            scales < 0
        )

        scales = scales * keep_pos

        non_zero_counts = (scales != 0).sum(dim=0).float()

        scales = scales.sum(dim=0) / torch.clamp(
            non_zero_counts,
            min=1
        )

        return scales



    def ours_kernel(pt_w, ft_ws, key, cum_threshold=1):
        if pt_w.ndim == 2:
            pt_w_ = pt_w.unsqueeze(0)
        elif pt_w.ndim == 1:
            pt_w_ = pt_w.unsqueeze(0).unsqueeze(0)
            ft_ws = [ft_w.unsqueeze(0) for ft_w in ft_ws]
        else:
            raise KeyError(f"Unexpected shape {pt_w.shape}")

        ft_ws = torch.stack(ft_ws, dim=0)
        dot_num = torch.sum(ft_ws * pt_w_, dim=2)
        dot_den = torch.sum(pt_w_ * pt_w_, dim=2)
        scale = dot_num / dot_den

        proj = scale.unsqueeze(2) * pt_w_
        perp = ft_ws - proj

        # projection merge
        scale = NeuroTaskArithmetic.inner_ties_kernel(scale)
        nor_vec = pt_w_ / torch.norm(pt_w_, dim=2, keepdim=True)
        proj = scale.unsqueeze(1) * nor_vec.squeeze(0)
        # proj = torch.zeros_like(proj) # Type2

        # perpendicular merge
        perp = perp.permute(1,0,2)
        _, S, VT = torch.linalg.svd(perp, full_matrices=False)

        S_squared = S ** 2
        cum_var = torch.cumsum(S_squared, dim=1) / torch.sum(
            S_squared,dim=1,keepdim=True)

        rank = torch.minimum(
            (cum_var < cum_threshold).sum(dim=1) + 1,
            torch.tensor(VT.shape[-2], device=perp.device)
        )

        V = VT.permute(0,2,1)
        scales = torch.matmul(perp, V)
        scales_ = []

        for m in range(scales.size(0)):
            scales_.append(
                NeuroTaskArithmetic.inner_ties_kernel(scales[m])
            )

        scales_ = torch.stack(scales_, dim=0)

        rep_vecs = []

        for m in torch.arange(scales_.size(0)):
            rep_vecs.append(
                (scales_[m,:rank[m]] *
                 V[m,:,:rank[m]]).sum(dim=1)
            )

        rep_vecs = torch.stack(rep_vecs, dim=0)

        if pt_w.ndim == 2:
            merged = proj + rep_vecs
        else:
            merged = proj.squeeze(0) + rep_vecs.squeeze(0)

        return merged



    def merge(BASE_MODEL_PATH, VL_MODEL_PATH, EMMA_MODEL_PATH, save_path):

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
        vl_model = LlavaLlamaForCausalLM.from_pretrained(VL_MODEL_PATH)
        emma_model = AutoModelForCausalLM.from_pretrained(EMMA_MODEL_PATH)

        # load vision tower
        vl_model.get_model().get_vision_tower().load_model()
        print("Vision tower mean:",
              next(vl_model.get_model().get_vision_tower().parameters()).abs().mean())

        # merge language backbone

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

                # ==========================
                # operator routing
                # ==========================
                if param.data.ndim == 2:
                    # Linear weights
                    merged_delta = NeuroTaskArithmetic.ours_kernel(
                        base_param.data,
                        [delta_vl, delta_emma],
                        name
                    )

                else:
                    # LayerNorm / bias / 1D params
                    merged_delta = NeuroTaskArithmetic.ties_kernel(
                        base_param.data,
                        [delta_vl, delta_emma]
                    )

                param.data.copy_(base_param.data +  merged_delta)
                

        base_norm = base_model.model.norm.weight.data
        vl_norm = vl_model.model.norm.weight.data
        emma_norm = emma_model.model.norm.weight.data


        delta_vl = vl_norm - base_norm
        delta_emma = emma_norm - base_norm

        vl_model.model.norm.weight.data.copy_(
            base_norm
            + 0.9 * delta_vl
            + 0.1 * delta_emma
        )


        base_embed = base_model.model.embed_tokens.weight.data
        vl_embed = vl_model.model.embed_tokens.weight.data
        emma_embed = emma_model.model.embed_tokens.weight.data
        lambda_1 = 0.9
        lambda_2 = 0.1

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

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        tokenizer.save_pretrained(save_path)

        print("Merge done.")