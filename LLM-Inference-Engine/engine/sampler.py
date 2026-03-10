
import torch

class Sampler:

    def __init__(self, temperature=0.8):
        self.temperature = temperature

    def sample(self, logits):

        probs = torch.softmax(logits / self.temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token
