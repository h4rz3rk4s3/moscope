"""
analysis/feature_analysis.py

Mechanistic interpretability analysis tools for both the Top-K SAE
and the Cross-Layer Transcoder.

Tools provided
--------------
1.  MaxActivatingExamples   — find tokens/sequences that maximally activate a feature
2.  FeatureSteering         — patch feature activations and observe downstream effects
3.  FeatureEvolution        — track how a CLT feature "evolves" across layers
4.  ExpertRoutingAnalysis   — correlate CLT features with MoE expert choices
5.  FeatureDashboard        — aggregate all of the above into a single HTML report
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.topk_sae import TopKSAE
from ..models.cross_layer_transcoder import CrossLayerTranscoder
from ..hooks.model_hooks import ActivationExtractor


# ---------------------------------------------------------------------------
# 1. Max-Activating Examples
# ---------------------------------------------------------------------------

@dataclass
class MaxActivatingExamples:
    """
    For a given feature (layer, feature_idx), collect the text snippets
    and token positions that produce the highest activation values.

    Works for both SAE and CLT.
    """
    texts: List[str] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    activations: List[float] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)  # decoded token at position


@torch.no_grad()
def find_max_activating_examples(
    model: nn.Module,
    tokenizer,
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    # encoder_fn(x: [B*T, D]) -> [B*T, N] activation matrix
    feature_idx: int,
    layer_idx: int,
    corpus: List[str],
    top_n: int = 20,
    batch_size: int = 16,
    max_seq_len: int = 512,
    device: str = "cpu",
) -> MaxActivatingExamples:
    """
    Scan a corpus for the `top_n` token positions that maximally
    activate `feature_idx` at `layer_idx`.

    Parameters
    ----------
    encoder_fn : a callable that maps flat activations [B*T, D] to feature
                 activations [B*T, N].  For SAE: lambda x: sae(x).z_topk.
                 For CLT: lambda x: clt_encoder(x, layer_idx).
    """
    top_acts: List[Tuple[float, str, int, str]] = []
    model.eval()

    extractor = ActivationExtractor(model, [layer_idx], ("residual",), detach=True)

    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus[i: i + batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_seq_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with extractor:
            model(**enc)

        flat_acts = extractor.get_flat("residual")[layer_idx].to(device)  # [B*T, D]
        feature_acts = encoder_fn(flat_acts)  # [B*T, N]
        feat_col = feature_acts[:, feature_idx]  # [B*T]

        B = len(batch_texts)
        T = enc["input_ids"].shape[1]
        mask = enc["attention_mask"].reshape(-1).bool()
        ids_flat = enc["input_ids"].reshape(-1)

        for pos in range(B * T):
            if not mask[pos]:
                continue
            val = feat_col[pos].item()
            seq_idx = pos // T
            tok_pos = pos % T
            tok_str = tokenizer.decode([ids_flat[pos].item()])
            entry = (val, batch_texts[seq_idx], tok_pos, tok_str)
            top_acts.append(entry)

        extractor.clear()

    top_acts.sort(key=lambda x: -x[0])
    top_acts = top_acts[:top_n]

    return MaxActivatingExamples(
        activations=[e[0] for e in top_acts],
        texts=[e[1] for e in top_acts],
        positions=[e[2] for e in top_acts],
        tokens=[e[3] for e in top_acts],
    )


# ---------------------------------------------------------------------------
# 2. Feature Steering
# ---------------------------------------------------------------------------

@dataclass
class SteeringResult:
    original_logits: torch.Tensor
    steered_logits: torch.Tensor
    kl_divergence: float
    top_token_change: List[Tuple[str, float, float]]  # (token, orig_prob, new_prob)


def steer_feature(
    model: nn.Module,
    tokenizer,
    text: str,
    layer_idx: int,
    feature_idx: int,
    steer_magnitude: float,
    sae: Optional[TopKSAE] = None,
    device: str = "cpu",
) -> SteeringResult:
    """
    Add a fixed multiple of feature `feature_idx`'s decoder direction
    to the residual stream at `layer_idx` and compare output logits.

    This implements the standard SAE steering vector technique.
    """
    model.eval()
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # Get decoder direction for this feature
    if sae is None:
        raise ValueError("Must pass a TopKSAE to extract the decoder direction.")
    direction = sae.W_dec[:, feature_idx].detach()  # [d_model]
    steering_vec = steer_magnitude * direction  # [d_model]

    # Store original logits
    with torch.no_grad():
        orig_out = model(**enc)
        orig_logits = orig_out.logits[:, -1, :].softmax(-1)  # [1, vocab]

    # Hook to add the steering vector
    added = [False]
    def _steer_hook(module, args, output):
        if not added[0]:
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, :, :] += steering_vec.unsqueeze(0).unsqueeze(0)
            added[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

    layers = ActivationExtractor._find_layers(model)
    layer = layers[layer_idx]
    mlp = ActivationExtractor._find_mlp(layer)
    hook = mlp.register_forward_hook(_steer_hook)

    with torch.no_grad():
        steered_out = model(**enc)
        steered_logits = steered_out.logits[:, -1, :].softmax(-1)

    hook.remove()

    # KL divergence
    kl = F.kl_div(
        steered_logits.log(), orig_logits, reduction="sum"
    ).item()

    # Top token changes
    top_k = 10
    orig_top = orig_logits[0].topk(top_k)
    steered_top = steered_logits[0].topk(top_k)
    all_ids = set(orig_top.indices.tolist() + steered_top.indices.tolist())
    changes = []
    for tid in all_ids:
        op = orig_logits[0, tid].item()
        sp = steered_logits[0, tid].item()
        changes.append((tokenizer.decode([tid]), op, sp))
    changes.sort(key=lambda x: -abs(x[2] - x[1]))

    return SteeringResult(orig_logits, steered_logits, kl, changes[:top_k])


# ---------------------------------------------------------------------------
# 3. Feature Evolution (CLT-specific)
# ---------------------------------------------------------------------------

@dataclass
class FeatureEvolutionResult:
    feature_idx: int
    source_layer: int
    target_contributions: Dict[int, float]  # {target_layer: contribution magnitude}
    decoder_norm_by_target: Dict[int, float]


@torch.no_grad()
def analyse_feature_evolution(
    clt: CrossLayerTranscoder,
    feature_idx: int,
    source_layer: int,
) -> FeatureEvolutionResult:
    """
    For a given CLT feature (source_layer, feature_idx), compute how strongly
    that feature contributes to each downstream layer's MLP reconstruction.

    Returns the L2 norm of each decoder vector W_dec_{source→target}[:, feature_idx].
    This tells you *where* in the network a feature has its largest effect.
    """
    L = clt.cfg.n_layers
    target_contributions = {}
    decoder_norm_by_target = {}

    for l_tgt in range(source_layer, L):
        W = clt.get_decoder(source_layer, l_tgt)  # [d_model, n_features]
        feat_col = W[:, feature_idx]               # [d_model]
        norm = feat_col.norm().item()
        target_contributions[l_tgt] = norm
        decoder_norm_by_target[l_tgt] = norm

    return FeatureEvolutionResult(
        feature_idx=feature_idx,
        source_layer=source_layer,
        target_contributions=target_contributions,
        decoder_norm_by_target=decoder_norm_by_target,
    )


@torch.no_grad()
def compute_feature_similarity_matrix(
    clt: CrossLayerTranscoder,
    layer_a: int,
    layer_b: int,
    top_n: int = 50,
) -> torch.Tensor:
    """
    Compute cosine similarity between decoder directions of features at
    `layer_a` and `layer_b` in a common target layer.

    Returns a [top_n, top_n] matrix — useful for visualising feature
    clustering across layers.
    """
    L = clt.cfg.n_layers
    # Use the last layer as common target
    l_tgt = L - 1
    W_a = clt.get_decoder(layer_a, l_tgt)[:, :top_n]  # [d_model, top_n]
    W_b = clt.get_decoder(layer_b, l_tgt)[:, :top_n]  # [d_model, top_n]

    W_a_n = F.normalize(W_a, dim=0)
    W_b_n = F.normalize(W_b, dim=0)
    return W_a_n.T @ W_b_n  # [top_n, top_n]


# ---------------------------------------------------------------------------
# 4. MoE Expert Routing Analysis
# ---------------------------------------------------------------------------

@dataclass
class ExpertRoutingCorrelation:
    """
    Stores the correlation between CLT/SAE feature activations and
    MoE expert routing decisions.
    """
    feature_idx: int
    layer_idx: int
    # expert_idx → Pearson r between feature activation and expert gate probability
    correlations: Dict[int, float]
    # experts most strongly correlated with this feature
    top_experts: List[Tuple[int, float]]  # (expert_idx, correlation)


@torch.no_grad()
def analyse_expert_routing(
    model: nn.Module,
    tokenizer,
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    feature_idx: int,
    layer_idx: int,
    corpus: List[str],
    n_experts: int,
    top_k_experts: int = 4,
    batch_size: int = 16,
    max_seq_len: int = 512,
    device: str = "cpu",
) -> ExpertRoutingCorrelation:
    """
    Correlate activations of `feature_idx` at `layer_idx` with the expert
    gate probabilities of the MoE block at that layer.

    Requires the model to expose expert gate logits.  We look for them at:
        model.model.layers[layer_idx].mlp.gate  (ModernBERT MoE convention)
    The gate's output should be accessible via a forward hook.

    Parameters
    ----------
    encoder_fn    : Maps [B*T, d_model] → [B*T, N] feature activations.
    n_experts     : Total number of experts at this layer.
    top_k_experts : How many experts are selected per token (for gate sparsity).
    """
    model.eval()

    # Locate gate module
    layers = ActivationExtractor._find_layers(model)
    moe_block = ActivationExtractor._find_mlp(layers[layer_idx])
    gate = getattr(moe_block, "gate", None)
    if gate is None:
        raise ValueError(
            f"Cannot find `gate` sub-module in MoE block at layer {layer_idx}. "
            "Ensure your MoE architecture exposes `moe_block.gate`."
        )

    # Collect feature activations and gate probabilities in parallel
    all_feat_acts: List[float] = []
    all_gate_probs: List[List[float]] = [[] for _ in range(n_experts)]

    # Hook to capture gate outputs
    gate_output_buf: List[torch.Tensor] = []
    def _gate_hook(module, args, output):
        probs = output.softmax(-1) if output.dim() == 2 else output
        gate_output_buf.append(probs.detach().cpu())

    gate_hook = gate.register_forward_hook(_gate_hook)
    act_extractor = ActivationExtractor(model, [layer_idx], ("residual",), detach=True)

    for i in range(0, min(len(corpus), 500), batch_size):
        batch_texts = corpus[i: i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=max_seq_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        gate_output_buf.clear()

        with act_extractor:
            model(**enc)

        flat_res = act_extractor.get_flat("residual")[layer_idx].to(device)
        feat_acts = encoder_fn(flat_res)[:, feature_idx]  # [B*T]
        mask = enc["attention_mask"].reshape(-1).bool().cpu()

        if gate_output_buf:
            gate_probs = gate_output_buf[0].reshape(-1, n_experts)  # [B*T, n_exp]
            for pos in range(flat_res.shape[0]):
                if not mask[pos]:
                    continue
                all_feat_acts.append(feat_acts[pos].item())
                for e in range(n_experts):
                    all_gate_probs[e].append(gate_probs[pos, e].item())

        act_extractor.clear()

    gate_hook.remove()

    if len(all_feat_acts) < 10:
        return ExpertRoutingCorrelation(feature_idx, layer_idx, {}, [])

    import statistics
    feat_t = all_feat_acts
    correlations = {}
    for e in range(n_experts):
        gate_t = all_gate_probs[e]
        # Pearson r
        try:
            r = _pearson_r(feat_t, gate_t)
        except Exception:
            r = 0.0
        correlations[e] = r

    top_experts = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:5]
    return ExpertRoutingCorrelation(feature_idx, layer_idx, correlations, top_experts)


def _pearson_r(x: List[float], y: List[float]) -> float:
    import math
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# 5. Feature Dashboard (aggregated HTML report)
# ---------------------------------------------------------------------------

def build_feature_dashboard(
    feature_idx: int,
    layer_idx: int,
    max_examples: MaxActivatingExamples,
    evolution: Optional[FeatureEvolutionResult] = None,
    routing: Optional[ExpertRoutingCorrelation] = None,
    output_path: str = "feature_dashboard.html",
) -> str:
    """
    Generate a minimal HTML dashboard for a single feature.
    Returns the HTML string and optionally writes to `output_path`.
    """
    # Build max-activating examples table
    example_rows = ""
    for act, text, pos, tok in zip(
        max_examples.activations,
        max_examples.texts,
        max_examples.positions,
        max_examples.tokens,
    ):
        snippet = text[:120].replace("<", "&lt;").replace(">", "&gt;")
        tok_str = tok.replace("<", "&lt;")
        example_rows += (
            f"<tr><td>{act:.3f}</td><td><b>{tok_str}</b></td>"
            f"<td>{pos}</td><td>{snippet}</td></tr>\n"
        )

    # Feature evolution section
    evo_rows = ""
    if evolution is not None:
        for l_tgt, norm in evolution.decoder_norm_by_target.items():
            evo_rows += f"<tr><td>{l_tgt}</td><td>{norm:.4f}</td></tr>\n"

    # Expert routing section
    routing_rows = ""
    if routing is not None:
        for exp_idx, corr in routing.top_experts:
            routing_rows += f"<tr><td>{exp_idx}</td><td>{corr:.4f}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Feature {feature_idx} @ Layer {layer_idx}</title>
<style>
  body {{ font-family: monospace; max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
  h2 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .section {{ margin-top: 2em; }}
</style>
</head>
<body>
<h1>Feature {feature_idx} — Layer {layer_idx}</h1>

<div class="section">
<h2>Top Activating Tokens</h2>
<table>
<tr><th>Activation</th><th>Token</th><th>Position</th><th>Context</th></tr>
{example_rows}
</table>
</div>

<div class="section">
<h2>Cross-Layer Decoder Norms (Feature Evolution)</h2>
{"<p><em>Not available</em></p>" if not evo_rows else f"""
<table>
<tr><th>Target Layer</th><th>||W_dec[:, f]||</th></tr>
{evo_rows}
</table>"""}
</div>

<div class="section">
<h2>MoE Expert Routing Correlations</h2>
{"<p><em>Not available</em></p>" if not routing_rows else f"""
<table>
<tr><th>Expert</th><th>Pearson r</th></tr>
{routing_rows}
</table>"""}
</div>

</body>
</html>"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)

    return html
