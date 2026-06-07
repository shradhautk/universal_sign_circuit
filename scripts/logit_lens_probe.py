"""
Logit Lens Probe — Universal Sign Decision Circuit
====================================================
Probes the sign decision mechanism in Qwen2.5-72B (80 layers) using the logit lens.

For each sign token (+/-) in a model-generated mathematical derivation,
we extract the hidden state h[p-1] at every layer, project it through the
final LayerNorm + lm_head, and record P(+) and P(-) at each layer.

This reveals WHEN each layer "knows" what sign will be produced.

Usage:
    Provide your own `records` list (see format below) and model path.
    Run on a GPU machine with sufficient VRAM (Qwen2.5-72B requires ~150GB).

Requirements:
    transformers, torch, json, re
"""

import json
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_PATH = "/path/to/Qwen2.5-72B-Instruct"   # set your local model path
OUTPUT_FILE = "logit_lens_results.json"

# System prompt used during inference (must match the prompt used to generate responses)
SYSTEM_PROMPT = (
    "You are a precise mathematical assistant. Show all computation steps clearly. "
    "Always put your final integer answer inside \\boxed{}."
)

# ---------------------------------------------------------------------------
# DATA FORMAT
# ---------------------------------------------------------------------------
# Provide records in this format:
#
# records = [
#     {
#         "id": "example_001",
#         "subcat": "rank",           # "rank" or "nullity" (affects problem preamble)
#         "problem_latex": r"A = \begin{bmatrix}...\end{bmatrix}",
#         "response": "<full model response text>",
#     },
#     ...
# ]
#
# The script will:
#   1. Reconstruct the full chat-formatted input (system + user + response prefix)
#   2. Tokenize it
#   3. Find all sign tokens (+/-) in the response portion
#   4. Run a single forward pass with output_hidden_states=True
#   5. For each sign position p, probe h[p-1] at all 80 layers

records = []   # <-- populate this with your data

# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

PROBLEM_TEXT = {
    "rank":    "Find the rank of the 5×5 matrix A.",
    "nullity": "Find the nullity of the 5×5 matrix A.",
}

def build_prompt(subcat, problem_latex, response):
    """Reconstruct the full teacher-forced input string."""
    task_text = PROBLEM_TEXT.get(subcat.lower(), "Solve the following problem.")
    user_content = (
        "Solve this linear algebra problem. Show your work and give the final answer.\n\n"
        f"{task_text}\n\n{problem_latex}\n\n"
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}"
    )

# ---------------------------------------------------------------------------
# SIGN TOKEN DETECTION
# ---------------------------------------------------------------------------

# Row-operation sign pattern — matches e.g. "R_2 ← R_2 − 5/3 R_1"
ROW_OP_SIGN_RE = re.compile(
    r'R_?\d+\s*\\?leftarrow\s*R_?\d+\s*([+\-])',
    re.UNICODE
)

def find_sign_positions(tokenizer, token_ids, full_text, response_start_idx):
    """
    Find token indices of +/- signs that appear in row operations,
    within the response portion of the token sequence.
    Returns list of dicts with token_idx, sign, context.
    """
    # Decode each token to find sign positions
    decoded_tokens = [tokenizer.decode([t]) for t in token_ids]

    # Find response start in token space
    prefix_text = full_text[:response_start_idx]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    resp_token_start = len(prefix_ids)

    sign_infos = []
    for i in range(resp_token_start, len(decoded_tokens)):
        tok = decoded_tokens[i]
        if tok.strip() not in ('+', '-'):
            continue
        # Check surrounding context for row operation pattern
        context_start = max(0, i - 30)
        context = ''.join(decoded_tokens[context_start:i+5])
        if ROW_OP_SIGN_RE.search(context):
            sign_infos.append({
                "token_idx": i,
                "sign": tok.strip(),
                "context": context[-80:],
                "token_str": tok,
            })
    return sign_infos

# ---------------------------------------------------------------------------
# SIGN TOKEN IDS
# ---------------------------------------------------------------------------

def get_sign_token_ids(tokenizer, model_vocab_size):
    """
    Collect all token IDs that represent + or - signs,
    including space-prefixed variants (e.g. ' +', ' -').
    Critical: Qwen tokenizes ' +' differently from '+'.
    """
    def collect(surfaces):
        seen, ids = set(), []
        for s in surfaces:
            for tid in tokenizer.encode(s, add_special_tokens=False):
                if tid < model_vocab_size and tid not in seen:
                    seen.add(tid); ids.append(tid)
        return ids

    plus_ids  = collect(['+', ' +', '+ ', '++'])
    minus_ids = collect(['-', ' -', '- ', '--'])
    return plus_ids, minus_ids

# ---------------------------------------------------------------------------
# LOGIT LENS CORE
# ---------------------------------------------------------------------------

def run_logit_lens(model, tokenizer, token_ids, sign_infos, n_layers,
                   plus_ids, minus_ids):
    """
    Single forward pass with output_hidden_states=True.
    For each sign position, probe h[p-1] at every layer.
    Returns per-sign results with P(+) and P(-) at each layer.
    """
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(model.device)

    with torch.inference_mode():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

    norm    = model.model.norm
    lm_head = model.lm_head
    norm_device = next(norm.parameters()).device

    results = []
    for si in sign_infos:
        probe_p = si["token_idx"] - 1  # h[p-1] predicts the token at p
        per_layer_plus  = []
        per_layer_minus = []
        peak_p_minus = 0.0
        peak_layer   = 1

        for l_idx in range(1, n_layers + 1):
            h = hidden_states[l_idx][0, probe_p, :].unsqueeze(0).unsqueeze(0).to(norm_device)
            h_normed = norm(h)
            logits = lm_head(h_normed)[0, 0, :]
            probs  = F.softmax(logits.float(), dim=-1)

            p_plus  = sum(probs[t].item() for t in plus_ids  if t < probs.shape[0])
            p_minus = sum(probs[t].item() for t in minus_ids if t < probs.shape[0])

            per_layer_plus.append(round(p_plus,  6))
            per_layer_minus.append(round(p_minus, 6))

            if p_minus > peak_p_minus:
                peak_p_minus = p_minus
                peak_layer   = l_idx

        # Top-3 layers by P(-)
        indexed = sorted(enumerate(per_layer_minus, 1), key=lambda x: x[1], reverse=True)
        top3 = [l for l, _ in indexed[:3]]

        results.append({
            **si,
            "peak_layer_1idx":  peak_layer,
            "peak_P_minus":     round(peak_p_minus, 6),
            "peak_P_plus":      round(per_layer_plus[peak_layer - 1], 6),
            "final_P_plus":     round(per_layer_plus[-1],  6),
            "final_P_minus":    round(per_layer_minus[-1], 6),
            "top3_peak_layers": top3,
            "per_layer_P_plus":  per_layer_plus,
            "per_layer_P_minus": per_layer_minus,
        })

    return results

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers  # 80 for Qwen2.5-72B
    plus_ids, minus_ids = get_sign_token_ids(tokenizer, model.config.vocab_size)

    print(f"Model loaded. Layers: {n_layers}")
    print(f"Plus token IDs:  {plus_ids}")
    print(f"Minus token IDs: {minus_ids}")

    all_results = {}

    for rec in records:
        record_id = rec["id"]
        print(f"\nProcessing: {record_id}")

        full_text = build_prompt(rec["subcat"], rec["problem_latex"], rec["response"])
        response_start_idx = full_text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")

        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        sign_infos = find_sign_positions(tokenizer, token_ids, full_text, response_start_idx)

        print(f"  Sequence length: {len(token_ids)} tokens")
        print(f"  Sign positions found: {len(sign_infos)}")

        if not sign_infos:
            print("  No row-operation signs found — skipping.")
            continue

        probe_results = run_logit_lens(
            model, tokenizer, token_ids, sign_infos, n_layers, plus_ids, minus_ids
        )

        all_results[record_id] = {
            "subcat":         rec["subcat"],
            "seq_len":        len(token_ids),
            "n_probes":       len(probe_results),
            "probe_summaries": probe_results,
        }

        # Print summary per probe
        for p in probe_results:
            print(f"    [{p['sign']}] @ token {p['token_idx']} | "
                  f"peak_layer={p['peak_layer_1idx']} | "
                  f"P(-)_peak={p['peak_P_minus']:.4f} | "
                  f"final P(+)={p['final_P_plus']:.4f} P(-)={p['final_P_minus']:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
