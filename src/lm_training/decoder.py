import torch

from src.tokenizer.tokenizer import Tokenizer
from src.transformer.transformer_lm import TransformerLM
from src.utils import softmax


class Decoder:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def decode(self, prompt: str, max_num_tokens: int, temperature: float, top_p: float):
        ids = self.tokenizer.encode(prompt)

        while len(ids) < max_num_tokens:
            logit = self.model(torch.tensor(ids[-self.model.context_length:], device=self.model.device))[-1]
            probs, prob_ids = softmax(logit / temperature, -1).sort(descending=True)

            prefix_probs = probs.cumsum(-1)
            top_p_id = min(torch.searchsorted(prefix_probs, top_p), len(prefix_probs) - 1)

            scaled_probs = probs[:(top_p_id + 1)] / prefix_probs[top_p_id]
            next_id = int(prob_ids[torch.multinomial(scaled_probs, 1)])

            if self.tokenizer.is_end_of_text(next_id):
                break

            ids.append(next_id)

        return self.tokenizer.decode(ids)
