import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(attention_probs.clamp(min=1e-10)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)

    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }


class SamplerConfig:
    """
    Configuration for the sampling strategy, including threshold values for various metrics
    and adaptive sampling parameters.
    """

    # Sampling Hyperparameters
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_probability: float = 0.03  # Minimum probability threshold for token selection

    # Logits Entropy Thresholds
    low_logits_entropy_threshold: float = 0.01
    medium_logits_entropy_threshold: float = 0.7
    high_logits_entropy_threshold: float = 2.1

    # Logits Varentropy Thresholds
    low_logits_varentropy_threshold: float = 0.05
    medium_logits_varentropy_threshold: float = 2.0
    high_logits_varentropy_threshold: float = 5.8

    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 11.915
    medium_attention_entropy_threshold: float = 11.921
    high_attention_entropy_threshold: float = 11.926

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.001
    medium_attention_varentropy_threshold: float = 0.0045
    high_attention_varentropy_threshold: float = 0.009

    # Agreement Thresholds
    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    # Offsets and Coefficients for Adjusting Sampling Parameters
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2

    low_entropy_interaction_strength_offset: float = 1.2
    low_entropy_interaction_strength_coefficient: float = 0.3

    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5

    # Adaptive Sampling Parameters
    number_of_adaptive_samples: int = 5

    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2
    adaptive_temperature_agreement_coefficient: float = 0.2
    adaptive_top_p_coefficient: float = 0.1
    adaptive_top_k_interaction_coefficient: float = 0.3
    adaptive_top_k_agreement_coefficient: float = 0.2
    adaptive_min_p_coefficient: float = 0.5
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4
    adaptive_score_agreement_coefficient: float = 0.5
    adaptive_score_interaction_strength_coefficient: float = 0.6


def get_color_for_metric(metrics: Dict[str, float], config) -> Tuple[int, int, int]:
    """Get color for metrics based on their values."""
    ent = metrics["logits_entropy"]
    temporal_ent = metrics["logits_varentropy"]

    # Normalize entropy and temporal entropy to a 0-1 range
    max_ent = config.high_logits_entropy_threshold
    max_vent = config.high_logits_varentropy_threshold

    normalized_ent = min(ent / max_ent, 1.0)
    normalized_vent = min(temporal_ent / max_vent, 1.0)
    combined_value = (normalized_ent + normalized_vent) / 2

    # Interpolate between green (0, 255, 0) and red (255, 0, 0)
    red = int(255 * combined_value)
    green = int(255 * (1 - combined_value))

    return (red, green, 0)

def _vick_rotation(logits: torch.Tensor, entropy, varentropy, k=0.3):

    norm_entropy = torch.clamp(entropy / 2.1, 0, 1)
    norm_varentropy = torch.clamp(varentropy / 5.8, 0, 1)
    tau = 1j * norm_varentropy * norm_entropy
    
    # Perform the rotation
    rotated_logits = logits * torch.exp(torch.pi * k * tau)

    return torch.real(rotated_logits)

def calculate_varentropy_logsoftmax(logits: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=dim) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=dim)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    # Use torch.rand instead of Exponential distribution
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, *, temperature: float | torch.Tensor, top_p: float | torch.Tensor, 
            top_k: int | torch.Tensor, min_p: float | torch.Tensor, 
            generator = torch.Generator(device=device)) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
    probs_sort = top_k_probs.flip(dims=[-1])
    probs_idx = top_k_indices.flip(dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Apply top-p sampling
    mask = torch.where(probs_sum - probs_sort > top_p, torch.ones_like(probs_sort), torch.zeros_like(probs_sort))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)

    next_token = multinomial_sample_one(probs_sort, generator)
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor, cfg: SamplerConfig,
           temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0) -> Tuple[torch.Tensor, Any]:
    
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    color = get_color_for_metric(metrics, cfg)
    clarifying_question_token = 2564
    #print(f'{metrics=}')

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32), color

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength < cfg.low_interaction_strength_threshold):
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:, -1], clarifying_question_token).any():
            return torch.tensor([[clarifying_question_token]], device=logits.device, dtype=torch.int32), color
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent
            return _sample(
                logits,
                temperature=min(1.5, temperature * temp_adj),
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            ), color

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (ent < cfg.high_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength > cfg.low_interaction_strength_threshold):
        #temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        logits = _vick_rotation(logits, ent, vent)
        return _sample(
            logits,
            temperature=1,   #min(1.5, cfg.temperature * temp_adj),
            top_p=top_p,
            top_k=top_k_adj,
            min_p=min_p,
        ), color

    # High Entropy, High Varentropy: "resampling in the mist"
    elif (ent > cfg.medium_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement > cfg.high_agreement_threshold and
          interaction_strength > cfg.high_interaction_strength_threshold):
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, top_p - cfg.high_entropy_attention_coefficient * attn_ent)  # Decrease top_p when attention entropy is high
        logits = _vick_rotation(logits, ent, vent)
        return _sample(
            logits,
            temperature=1,   #max(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=top_k,
            min_p=min_p
        ), color

    # Middle ground: use adaptive sampling
    else:
        temperature = cfg.temperature * (
        1 +
        cfg.adaptive_temperature_logits_coefficient * ent +
        cfg.adaptive_temperature_attention_coefficient * attn_ent -
        cfg.adaptive_temperature_agreement_coefficient * agreement
    )
    top_p = torch.clamp(cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent), 0.1, 1.0)
    top_k = int(torch.clamp(
        torch.round(torch.tensor(cfg.top_k) 
                    * (1 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() - 
                        cfg.adaptive_top_k_agreement_coefficient * agreement.item())),
        min=1,
        max=100
    ).item())
    min_p = torch.clamp(cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient* vent), 0.01, 0.5)

    samples = []
    logits = _vick_rotation(logits, ent, vent)
    for _ in range(cfg.number_of_adaptive_samples):
        sample = _sample(logits, temperature=1, top_p=top_p, top_k=top_k, min_p=min_p)
        samples.append(sample)

    def score_sample(sample):
        # Flatten the sample tensor and convert to long (int64)
        sample_flat = sample.flatten().to(torch.long)

        # Create one-hot encoding
        one_hot = F.one_hot(sample_flat, logits.shape[-1])

        # Reshape log_softmax output to match one_hot
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])

        # Calculate log probability
        log_prob = torch.sum(log_probs * one_hot)

        confidence_score = (
            (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
            (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
            (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
            (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
            (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
            (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
        )
        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    return samples[best_sample_idx], color