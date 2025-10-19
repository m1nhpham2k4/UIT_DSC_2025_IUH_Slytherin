from typing import Tuple, List, Dict, Any
import torch


def build_number_token_mapping(tokenizer, num_classes: int = 3) -> List[int]:
    """
    Encode các số 0..num_classes thành token ID từ tokenizer.
    Trả về list các token ID tương ứng.
    """
    number_token_ids = []
    for i in range(0, num_classes + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        if not tokens:
            raise ValueError(f"Tokenization for '{i}' failed.")
        number_token_ids.append(tokens[0])
    return number_token_ids


def trim_lm_head_to_numbers(model: Any, number_token_ids: List[int]) -> Dict[str, Any]:
    """
    Cắt lớp lm_head của model để chỉ giữ lại các trọng số ứng với các token số (0..num_classes).
    Trả về metadata cần thiết để restore lại sau khi train.
    """
    old_shape = model.lm_head.weight.shape
    old_size = old_shape[0]

    trimmed_weight = model.lm_head.weight[number_token_ids, :].clone().detach()
    model.lm_head.weight = torch.nn.Parameter(trimmed_weight)

    reverse_map = {value: idx for idx, value in enumerate(number_token_ids)}

    return {
        "old_shape": old_shape,
        "old_size": old_size,
        "number_token_ids": number_token_ids,
        "reverse_map": reverse_map,
    }


def restore_full_lm_head(model: Any, old_size: int, number_token_ids: List[int]) -> None:
    """
    Khôi phục lại lm_head về kích thước ban đầu sau khi fine-tune.
    Giữ lại trọng số cho các token số và set bias thấp cho token còn lại.
    """
    trimmed_lm_head = model.lm_head.weight.data.clone()
    trimmed_lm_head_bias = (
        model.lm_head.bias.data.clone()
        if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None
        else torch.zeros(len(number_token_ids), device=trimmed_lm_head.device)
    )

    hidden_dim = trimmed_lm_head.shape[1]
    new_lm_head = torch.full(
        (old_size, hidden_dim),
        0.0,
        dtype=trimmed_lm_head.dtype,
        device=trimmed_lm_head.device,
    )
    new_lm_head_bias = torch.full(
        (old_size,),
        -1000.0,
        dtype=trimmed_lm_head_bias.dtype,
        device=trimmed_lm_head_bias.device,
    )

    # Copy lại phần trọng số cho các token số
    for new_idx, orig_token_id in enumerate(number_token_ids):
        new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
        new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]

    with torch.no_grad():
        new_lm_head_module = torch.nn.Linear(hidden_dim, old_size, bias=True, device=model.device)
        new_lm_head_module.weight.data.copy_(new_lm_head)
        new_lm_head_module.bias.data.copy_(new_lm_head_bias)
        model.lm_head.modules_to_save["default"] = new_lm_head_module

