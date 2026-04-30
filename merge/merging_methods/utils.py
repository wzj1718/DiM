import torch

def flatten_ckpt_into_vec(ckpt):
    vec = []
    for param in ckpt.values():
        vec.append(param.flatten())
    return torch.cat(vec)

def select_trainable_params(model):
    params = {}

    for n, p in model.named_parameters():
        if 'embed' not in n and 'Embedding' not in n:
            params[n] = p
                    
    return params

def get_task_vector(ft_model, base_model):
    ft_model.to('cpu')
    base_model.to('cpu')

    ft_params = select_trainable_params(ft_model)
    base_params = select_trainable_params(base_model)

    ft_vec = flatten_ckpt_into_vec(ft_params)
    base_vec = flatten_ckpt_into_vec(base_params)

    return ft_vec - base_vec

def vector_to_state_dict(vec, pretrained_model, return_dict=False):
    i = 0
    vec.to('cpu')
    pretrained_model.to('cpu')
    for k, v in pretrained_model.state_dict().items():
        if 'embed' not in k.lower() and 'lm_head' not in k:
            if torch.nonzero(v).size(0) == 0:
                continue
            vec[i:i+v.numel()].reshape(v.shape).to(pretrained_model.device)
            pretrained_model.state_dict()[k] += vec[i:i+v.numel()].reshape(v.shape)
            i += v.numel()

    if return_dict:
        return pretrained_model.state_dict()
    else:
        return pretrained_model
    


#  DRM
def get_chained_attributes(obj, attributes):
    """Getattr that allows chained attributes."""
    for attribute in attributes.split("."):
        obj = getattr(obj, attribute)
    return obj


def get_inner_most_object_from_chained_attributes(obj: object, attributes: str):
    attributes = attributes.split(".")
    for attribute in attributes[:-1]:
        obj = getattr(obj, attribute)
    return obj

# STF

import copy
import torch
from collections import OrderedDict

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask
    return M * final_mask


def merge_matrix(param_mat):
    U, S, Vh = torch.linalg.svd(param_mat, full_matrices=False)
    U_merge = U.permute(1, 0, 2).reshape(U.shape[1], -1)
    S_merge = S.reshape(-1)
    Vh_merge = Vh.reshape(-1, Vh.shape[2])
    # print(S_merge)
    
    matrix_left = U_merge.T @ U_merge
    matrix_right = Vh_merge @ Vh_merge.T
    equation_coef = matrix_left * matrix_right * S_merge.reshape(1, -1)
    equation_output = S_merge
    try:
        solution = torch.linalg.solve(equation_coef, equation_output)
    except:
        print('Using lstsq...')
        solution = torch.linalg.lstsq(equation_coef.cpu(), equation_output.cpu(), driver='gelsy')[0]
        
        # solution = torch.linalg.lstsq(equation_coef, equation_output)[0]
    merge_tv = U_merge @ ((solution * S_merge).reshape(-1, 1) * Vh_merge)
    return merge_tv.float()