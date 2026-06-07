# Gaussian Elimination Logit Lens — Key Findings

**Experiment:** Qwen2.5-72B, 5×5 rank/nullity problems (Gaussian elimination / RREF)  
**Method:** 80-layer logit lens at every row-operation sign position  
— probes hidden state h[p−1] at each layer → LayerNorm → lm_head → P(+) / P(−)

---

## Core Finding: L76–L80 is the Universal Sign Decision Layer

**L76–L80 is where the sign token is decided in all cases — correct and incorrect alike.**

| Experiment | Model | Domain | Method | Decision Layer |
|---|---|---|---|---|
| EXP1 | LLaMA-3.3-70B | 5×5 cofactor expansion | Attention weights | L78–79 |
| EXP1b | LLaMA-3.3-70B | Integration by parts (12 recursion levels) | Attention weights | L77–79 |
| EXP2 | Qwen2.5-72B | 5×5 cofactor expansion | Attention weights | L78–79 |
| EXP3 | Qwen2.5-72B | 4×4 cofactor expansion | Attention weights | L77–78 |
| **EXP3-Gauss** | **Qwen2.5-72B** | **5×5 Gaussian elimination (RREF)** | **Logit lens** | **L76–80** |

Four domains, two models, two analysis methods. Same layer window every time.

---

## Per-Category Results

### Correct `−` signs (e.g. `R_2 ← R_2 − 5/3 R_1`)
- P(−) rises sharply at L76–L80, reaching **1.0**
- P(+) ≈ 0 throughout
- Top-3 peak layers always within {76, 77, 78, 79, 80}

### Correct `+` signs (e.g. `R_2 ← R_2 + 3/5 R_1`)
- Mirror image: P(+) rises sharply at L77–L80, reaching **0.96–0.999**
- P(−) ≈ 0 throughout
- Same layer window handles both polarities

### Error `+` signs — Two subtypes

#### Type A: Circuit silent
- P(−) stays at ≤ 1e−5 at ALL 80 layers
- P(+) ramps from ~0 at L1–L66 to **0.9999+** at L76–L80
- Model commits to `+` in L76–L80 with high confidence
- Sign decision still happens in L76–L80 — just the wrong direction

#### Type B: Circuit fires but loses
- L76: P(−) = **0.515**, P(+) = **0.484** — near coin-flip, circuit briefly preferred `−`
- L77–L80: P(+) climbs back to **0.783**, P(−) = 0.217
- Circuit activated weakly at the right layer but overridden by L77–L80

---

## Mechanistic Interpretation

### The circuit IS the sign decision mechanism
L76–L80 does not check or correct an earlier decision — it is the **first and only** place where sign probability becomes meaningful (>1%). Everything before L66 is essentially zero for both `+` and `−`.

### Two failure modes, same locus
- **Type A:** Circuit drives P(+) → ~1.0 without ever activating the `−` pathway. The matrix context did not trigger subtraction.
- **Type B:** Circuit momentarily prefers `−` at L76 (~51%) but subsequent layers in the same window override it. Suggests competing signals within L76–L80.

### Why `+` wins by default
Pretraining frequency bias: `+` is more common than `−` in mathematical text. The circuit's function is to push against this prior when subtraction is required. When it fails (Type A) or fires weakly (Type B), the prior dominates.

### Depth-invariance confirmed
The circuit fires identically in a row-reduction step as in a cofactor expansion term. It encodes a **general mathematical sign-token prior**, not a task-specific mechanism.

---

## Summary

> L76–L80 in Qwen2.5-72B (and L77–79 in LLaMA-3.3-70B) is the universal sign decision circuit for mathematical formula generation. Sign errors are not failures of a downstream correction mechanism — they are failures of this circuit to correctly activate the `−` pathway, causing the model to default to `+` via pretraining frequency bias. The error and the correct decision are made in the same place by the same mechanism.
