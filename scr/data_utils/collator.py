from typing import List, Dict, Any, Union
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    """
    Custom collator:
    - Chỉ tính loss tại token cuối cùng (phần nhãn ở cuối chuỗi)
    - Map nhãn từ id gốc sang index trong lm_head đã cắt gọn.
    """

    def __init__(
        self,
        *args,
        reverse_map: Dict[int, int],
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.reverse_map = reverse_map

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        labels = batch["labels"]
        for i in range(labels.size(0)):
            valid_tokens = (labels[i] != self.ignore_index).nonzero(as_tuple=True)[0]
            if len(valid_tokens) == 0:
                continue  # skip nếu sample toàn padding

            last_token_idx = valid_tokens[-1].item()

            # Chỉ giữ lại token cuối cùng
            labels[i, :last_token_idx] = self.ignore_index

            # Ánh xạ nhãn cuối cùng về index mới trong lm_head
            original_id = labels[i, last_token_idx].item()
            if original_id in self.reverse_map:
                labels[i, last_token_idx] = self.reverse_map[original_id]
            else:
                labels[i, last_token_idx] = self.ignore_index  # fallback an toàn

        batch["labels"] = labels
        return batch
