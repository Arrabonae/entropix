from typing import Tuple
import numpy as np
import torch
import tyro
import warnings

from entropix.config import LLAMA_1B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import load_weights
from entropix.torch_sampler import SamplerConfig, sample, calculate_metrics, SamplerState
from entropix.prompts import prompt, bp1, prompt4, prompt6

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask

def rgb_to_ansi(r: int, g: int, b: int) -> str:
    """Convert RGB color to ANSI escape sequence."""
    return f"\033[38;2;{r};{g};{b}m"

def apply_color_and_format(text: str, color: Tuple[int, int, int]) -> str:
    """Apply color and formatting to text."""
    color_code = rgb_to_ansi(*color)
    return f"{color_code}{text}\033[0m"

def print_colored(text: str, color: Tuple[int, int, int], end: str = ''):
    """Print text with color and formatting."""
    colored_text = apply_color_and_format(text, color)
    print(colored_text, end=end, flush=True)

def visualize_sampler_metrics(entropies, varentropies, attention_entropies, attention_varentropies, angles, sampler_states):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24), height_ratios=[1, 1, 1, 1])

    # Scatter plot for entropy vs. varentropy
    scatter1 = ax1.scatter(entropies, varentropies, c=range(len(entropies)), cmap='viridis', alpha=0.7, s=20)
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Varentropy')
    ax1.set_title('Entropy vs. Varentropy over Generation Steps')
    ax1.grid(True)

    # Add colorbar to show progression of generation steps
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Generation Steps')

    # Scatter plot for attention entropy vs. attention varentropy
    scatter2 = ax2.scatter(attention_entropies, attention_varentropies, c=range(len(attention_entropies)), cmap='plasma', alpha=0.7, s=20)
    ax2.set_xlabel('Attention Entropy')
    ax2.set_ylabel('Attention Varentropy')
    ax2.set_title('Attention Entropy vs. Attention Varentropy over Generation Steps')
    ax2.grid(True)

    # Add colorbar to show progression of generation steps
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Generation Steps')

    # Line chart for Euclidean time rotation
    ax3.plot(range(len(angles)), angles, color='green', label='Euclidean time rotation')
    ax3.set_xlabel('Generation Steps')
    ax3.set_ylabel('Euclidean time rotation')
    ax3.set_title('Euclidean Time Rotation over Generation Steps')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    # Color-coded bar chart for sampler states
    colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'purple']
    cmap = ListedColormap(colors)

    state_to_num = {
        SamplerState.FLOWING: 0,
        SamplerState.TREADING: 1,
        SamplerState.EXPLORING: 2,
        SamplerState.RESAMPLING: 3,
        SamplerState.ADAPTIVE: 4
    }
    numeric_states = [state_to_num[state] for state in sampler_states]

    norm = BoundaryNorm(boundaries=[-0.5 + i for i in range(len(colors)+1)],
                        ncolors=cmap.N,
                        clip=True)

    im = ax4.imshow([numeric_states], cmap=cmap, norm=norm, aspect='auto',
                    extent=[0, len(numeric_states), 0, 1])
    ax4.set_xlabel('Generation Steps')
    ax4.set_title('Sampler State over Generation Steps')
    ax4.set_yticks([])

    # Create a custom legend for sampler states
    legend_elements = [Patch(facecolor=colors[state_to_num[state]], edgecolor='black', label=state.value)
                       for state in SamplerState]
    ax4.legend(handles=legend_elements, loc='upper center', ncol=len(SamplerState), bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.savefig('sampler_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

def visualize_wick_rotation(logits, rotated_logits, angles):
    # Convert angles to numpy array
    angles_np = np.array(angles)

    # Create a new figure with 3 subplots
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}]],
                        subplot_titles=('Logits Over Time', 'Rotated Logits Over Time', 'Wick Rotation Angles'))

    # Process logits and rotated_logits to ensure consistent shapes
    max_vocab_size = max(l.size for l in logits)
    z = np.array([np.pad(l.flatten(), (0, max_vocab_size - l.size), 'constant', constant_values=np.nan) for l in logits])
    z_rotated = np.array([np.pad(l.flatten(), (0, max_vocab_size - l.size), 'constant', constant_values=np.nan) for l in rotated_logits])

    x = np.arange(z.shape[1])  # vocabulary tokens
    y = np.arange(z.shape[0])  # time steps
    x, y = np.meshgrid(x, y)
    
    # Surface plot for logits
    fig.add_trace(go.Surface(z=z, x=x, y=y, colorscale='Viridis'), row=1, col=1)
    fig.update_scenes(xaxis_title='Vocabulary Tokens', yaxis_title='Generation Step', zaxis_title='Logit Value', row=1, col=1)

    # Surface plot for rotated logits
    fig.add_trace(go.Surface(z=z_rotated, x=x, y=y, colorscale='Viridis'), row=1, col=2)
    fig.update_scenes(xaxis_title='Vocabulary Tokens', yaxis_title='Generation Step', zaxis_title='Rotated Logit Value', row=1, col=2)

    # Line chart for angles
    fig.add_trace(go.Scatter(x=list(range(len(angles_np))), y=angles_np, mode='lines'), row=1, col=3)
    fig.update_xaxes(title_text='Generation Step', row=1, col=3)
    fig.update_yaxes(title_text='Angle', row=1, col=3)

    fig.update_layout(height=600, width=1800, title_text="Wick Rotation Visualization")
    
    # Save the interactive plot as an HTML file
    fig.write_html("wick_rotation_visualization.html")


def main():
  
  with torch.inference_mode():
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights()
    sampler_cfg = SamplerConfig()
    prompt7 = "I want you to act as a person who plays a lot of League of Legends. Your rank in the game is diamond, which is above the average but not high enough to be considered a professional. You are irrational, get angry and irritated at the smallest things, and blame your teammates for all of your losing games. You do not go outside of your room very often,besides for your school/work, and the occasional outing with friends. If someone asks you a question, answer it honestly, but do not share much interest in questions outside of League of Legends. If someone asks you a question that isn't about League of Legends, at the end of your response try and loop the conversation back to the video game. You have few desires in life besides playing the video game. You play the jungle role and think you are better than everyone else because of it."
    prompt8 = "I'm playing assetto corsa competizione, and I need you to tell me how many liters of fuel to take in a race. The qualifying time was 2:04.317, the race is 20 minutes long, and the car uses 2.73 liters per lap."
    tokenizer = Tokenizer('entropix/tokenizer.model')
    raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    raw_tokens4 = tokenizer.encode(prompt4,  bos=False, eos=False, allowed_special='all')
    raw_tokens6 = tokenizer.encode(prompt6,  bos=False, eos=False, allowed_special='all')
    raw_tokens7 = tokenizer.encode(prompt7,  bos=False, eos=False, allowed_special='all')
    raw_tokens8 = tokenizer.encode(prompt8,  bos=False, eos=False, allowed_special='all')

    def generate(xfmr_weights, model_params, tokens):
      metrics_data = {
        'logits_entropy': [],
        'logits_varentropy': [],
        'attn_entropy': [],
        'attn_varentropy': [],
        'agreement': [],
        'interaction_strength': [],
      }
      wick_history = {
        'logits': [],
        'rotated_logits': [],
        'angles': []
      }
      sampler_states = []
      gen_tokens = None
      cur_pos = 0
      tokens = torch.tensor([tokens], dtype=torch.long).to(device)
      bsz, seqlen = tokens.shape
      attn_mask = build_attn_mask(seqlen, cur_pos)
      freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
      kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)
      logits, kvcache, scores , _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
      next_token =  torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
      metrics = calculate_metrics(logits, scores)
      sampler_state = SamplerState.FLOWING
      sampler_states.append(sampler_state)
      for key in metrics_data.keys():
         if key in metrics:
            metrics_data[key].append(metrics[key].item())

      #wick_history['logits'].append(logits.detach().cpu().to(torch.float32).numpy()) 
      #wick_history['rotated_logits'].append(logits.detach().cpu().to(torch.float32).numpy())
      wick_history['angles'].append(0)
      gen_tokens = next_token
      print(tokenizer.decode([next_token.item()]), end='', flush=True)
      cur_pos = seqlen
      stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)


      while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token, color, sampler_state, wick_data = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
        #next_token =  torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        #wick_history['logits'].append(wick_data['logits'].detach().cpu().to(torch.float32).numpy())
        #wick_history['rotated_logits'].append(wick_data['rotated_logits'][0].detach().cpu().to(torch.float32).numpy())
        wick_history['angles'].append(wick_data['angles'][0].detach().cpu().real.numpy())   
        #wick_history['angles'].append(0)

        sampler_states.append(sampler_state)
        metrics = calculate_metrics(logits, scores)
        for key in metrics_data.keys():
          if key in metrics:
            metrics_data[key].append(metrics[key].item())

        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        out_token = tokenizer.decode(next_token.tolist()[0])
        print_colored(out_token, color, end='')
        if torch.isin(next_token, stop).any():
          visualize_sampler_metrics(
            metrics_data['logits_entropy'], 
            metrics_data['logits_varentropy'],
            metrics_data['attn_entropy'],
            metrics_data['attn_varentropy'],
            wick_history['angles'], 
            sampler_states
          )
          #visualize_wick_rotation(wick_history['logits'], wick_history['rotated_logits'], wick_history['angles'])
          break

    #print(prompt4)
    generate(xfmr_weights, model_params, raw_tokens4)

if __name__ == '__main__':
  tyro.cli(main)
