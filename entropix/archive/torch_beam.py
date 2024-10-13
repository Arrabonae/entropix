import torch
import torch.nn.functional as F
from typing import Tuple, List, Callable, Any
from entropix.torch_model import xfmr
from entropix.config import LLAMA_1B_PARAMS

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def sparkling_beam_search(
    initial_logits: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: Any,
    xfmr_weights: Any,
    beam_width: int = 5,
    max_steps: int = 3,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    def sample_with_temperature(logits: torch.Tensor, temp: float, k: int, p: float) -> torch.Tensor:
        probs = F.softmax(logits / temp, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=min(k, probs.shape[-1]))
        cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        top_k_probs.masked_fill_(mask, 0.0)
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        sample = torch.multinomial(top_k_probs, num_samples=1)
        return torch.gather(top_k_indices, -1, sample)

    # Initialize beam with top beam_width tokens
    beam_scores, beam_tokens = torch.topk(F.log_softmax(initial_logits[:, -1], dim=-1), k=beam_width, dim=-1)
    beam_tokens = beam_tokens.unsqueeze(-1)

    for step in range(max_steps - 1):
        all_candidates = []
        for score, tokens in zip(beam_scores[0], beam_tokens[0]):
            # Generate next logits using xfmr
            next_token = tokens[-1].unsqueeze(0).unsqueeze(0)
            next_pos = cur_pos + tokens.size(0)
            next_logits, next_kvcache, scores, stats= xfmr(xfmr_weights, LLAMA_1B_PARAMS, tokens=next_token, cur_pos=next_pos, freqs_cis=freqs_cis, kvcache=kvcache)
            
            next_tokens = sample_with_temperature(next_logits[:, -1], temperature, top_k, top_p)
            next_scores = F.log_softmax(next_logits[:, -1], dim=-1)
            for token, token_score in zip(next_tokens[0], next_scores[0, next_tokens[0]]):
                candidate_score = score + token_score
                candidate_tokens = torch.cat([tokens, token.unsqueeze(0)])
                all_candidates.append((candidate_score, candidate_tokens))

        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beam_scores = torch.tensor([c[0] for c in all_candidates[:beam_width]], device=initial_logits.device).unsqueeze(0)
        beam_tokens = torch.stack([c[1] for c in all_candidates[:beam_width]]).unsqueeze(0)

    # Return the highest scoring sequence
    best_sequence = beam_tokens[0, 0, -1].unsqueeze(0).unsqueeze(0)
    return best_sequence, next_kvcache

def beam_search_wrapper(
    logits: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: Any,
    xfmr_weights: Any,
    beam_width: int = 5,
    max_steps: int = 3,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    next_token, next_kvcache = sparkling_beam_search(
        logits,
        cur_pos,
        freqs_cis,
        kvcache,
        xfmr_weights,
        beam_width=beam_width,
        max_steps=max_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return next_token, next_kvcache