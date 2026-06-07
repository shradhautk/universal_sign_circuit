# Universal Sign Decision Circuit in Large Language Models

**Research area:** Mechanistic interpretability — mathematical reasoning  
**Models:** Qwen2.5-72B, LLaMA-3.3-70B  
**Method:** Teacher forcing + attention weight analysis + logit lens  

---

## Key Finding

> **Layers 76–80 in Qwen2.5-72B (L77–79 in LLaMA-3.3-70B) form a universal sign decision circuit for mathematical formula generation.** This circuit fires in the same narrow layer window across four distinct mathematical domains, two model families, and two analysis methods — independent of recursion depth, problem size, or task structure.

Sign errors in LLM mathematical reasoning are not random. They originate from a single localized mechanism that either correctly activates the `−` pathway or defaults to `+` via pretraining frequency bias.

---

## Experiments

| Experiment | Model | Domain | Method | Decision Layer |
|---|---|---|---|---|
| EXP1 | LLaMA-3.3-70B | 5×5 cofactor expansion | Attention weights | L78–79 |
| EXP1b | LLaMA-3.3-70B | Integration by parts (12 recursion levels) | Attention weights | L77–79 |
| EXP2 | Qwen2.5-72B | 5×5 cofactor expansion | Attention weights | L78–79 |
| EXP3 | Qwen2.5-72B | 4×4 cofactor expansion | Attention weights | L77–78 |
| EXP3-Gauss | Qwen2.5-72B | 5×5 Gaussian elimination (RREF) | Logit lens | L76–80 |

Four domains. Two models. Two analysis methods. Same layer window every time.

---

## Methodology

### Step 1 — Teacher Forcing

Model responses to linear algebra problems are collected. For each response, the correct solution steps are injected via teacher forcing: the model is fed the correct prefix and its hidden states are captured at every sign token position (`+` or `−`).

This isolates the sign decision mechanism from earlier reasoning errors — the question is not whether the model gets the answer right, but *where and how it commits to a sign token*.

### Step 2 — Attention Weight Analysis (EXP1–EXP3)

At each sign token position `p`, attention weights from all heads at all layers are extracted. The analysis identifies which layers show the sharpest attention shifts correlated with sign polarity. This revealed L76–L79 as the consistent decision window.

### Step 3 — Logit Lens (EXP3-Gauss)

For Gaussian elimination sign errors, the logit lens was applied: at each layer `l`, the hidden state `h_l[p−1]` (the token just before the sign) is projected through the final LayerNorm and lm_head to obtain per-layer token probabilities P(`+`) and P(`−`).

This directly reveals *when* each layer "knows" what sign will be produced. Before L66: both probabilities are essentially zero. At L76–L80: the decision crystallizes, reaching P → 1.0.

---

## Sign Error Taxonomy

Two mechanistically distinct failure modes were identified, both localized to L76–L80:

### Type A — Total Activation Failure (majority of errors)
- P(`−`) never rises above 1e−5 at any of the 80 layers
- P(`+`) ramps from ~0 at L1–L66 to **0.9999+** at L76–L80
- The circuit commits confidently to `+` without ever considering `−`
- The matrix context failed to activate the subtraction pathway entirely

### Type B — Weak Activation + Override (minority of errors)
- The circuit briefly prefers `−` at L76 (P(`−`) ≈ 0.515, P(`+`) ≈ 0.484)
- But L77–L80 overrides this back to P(`+`) ≈ 0.78
- The subtraction signal fires weakly but loses to competing signals within the same window

In both cases, the error is made *at the same layer window as correct decisions*. This rules out a downstream correction failure — the circuit IS the only sign decision mechanism.

---

## Why `+` Wins by Default

Pretraining frequency: `+` appears more often than `−` in mathematical text. When the circuit does not strongly activate the `−` pathway, `+` wins by default. The circuit's function is to push against this prior when subtraction is required. When it fails (Type A) or fires weakly (Type B), the prior dominates.

---

## Depth-Invariance

The circuit is **task-agnostic**:
- Cofactor expansion has 3 recursion levels
- Integration by parts has 12 recursion levels  
- Gaussian elimination is iterative with no fixed depth

The sign decision always crystallizes in the same layer window regardless of the surrounding computation structure. This confirms the circuit encodes a **general mathematical sign-token prior**, not a task-specific mechanism.

---

## Repository Structure

```
universal_sign_circuit/
├── README.md                          # This file
├── findings/
│   └── GAUSS_LOGIT_LENS_FINDINGS.md   # Detailed findings for the Gaussian elimination domain
├── scripts/
│   └── logit_lens_probe.py            # Logit lens methodology (Qwen2.5-72B, 80-layer probe)
└── visualizations/
    └── (layer-by-layer P(+)/P(-) curves — available on request)
```

---

## Dataset

Model responses were generated using a curated linear algebra benchmark covering determinant, rank, nullity, and matrix operation problems across multiple matrix sizes and two model families.

The benchmark and full dataset are part of ongoing research under separate publication. Raw model outputs and problem data are available for review upon request.
