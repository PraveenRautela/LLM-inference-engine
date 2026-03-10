
import torch

class KVCacheEngine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    @torch.no_grad()
    def forward_step(self, input_ids, past_key_values=None):

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        return logits, past_key_values
