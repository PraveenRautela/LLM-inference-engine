
import torch

class BatchDecoder:

    def __init__(self, model, tokenizer, kv_engine, sampler):

        self.model = model
        self.tokenizer = tokenizer
        self.kv_engine = kv_engine
        self.sampler = sampler
        self.device = model.device

    def generate(self, prompts, max_new_tokens=100):

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        input_ids = inputs.input_ids
        generated = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):

            logits, past_key_values = self.kv_engine.forward_step(
                input_ids,
                past_key_values
            )

            next_token = self.sampler.sample(logits)

            generated = torch.cat([generated, next_token], dim=-1)
            input_ids = next_token

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
