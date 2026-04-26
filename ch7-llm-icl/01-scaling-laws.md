# 01. Scaling Laws (Kaplan 2020 / Hoffmann 2022)

## 🎯 핵심 질문

- Kaplan 2020 의 scaling laws — model size $N$, data $D$, compute $C$ 의 power-law 관계?
- $L(N, D, C)$ 의 정확한 form 과 fit 한 exponent 의 의미?
- Chinchilla (Hoffmann 2022) 가 GPT-3 의 training recipe 를 어떻게 뒤집었는가?
- Compute-optimal: $N \propto C^{0.5}, D \propto C^{0.5}$ 의 도출?
- 2024+ 의 추가 발견 — over-training 과 inference-aware scaling?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Scaling Laws 는 **modern LLM training 의 과학적 토대**:

1. **Predictable scaling** — 작은 실험으로 큰 모델 예측
2. **Compute budget 분배** — params vs data 의 optimal balance
3. **Emergent capabilities** — scale 이 capability 결정
4. **Frontier model 의 design** — GPT-4, LLaMA-3 의 size 결정

이 문서는 scaling laws 의 **수학, derivation, modern updates** 를 다룹니다.

---

## 📐 수학적 선행 조건

- 통계학: power-law fit, log-log regression
- Chapter 6: GPT-3 의 scale (Ch6-02)

---

## 📖 직관적 이해

### Kaplan 2020 의 핵심 발견

```
Loss L 이 N (params), D (tokens), C (compute) 에 power-law:

L(N) ∝ N^(-α)        with α ≈ 0.07
L(D) ∝ D^(-β)        with β ≈ 0.07
L(C) ∝ C^(-γ)        with γ ≈ 0.05

→ 100× scale 시 loss 의 예측 가능 감소
```

### Compute Frontier

```
For fixed compute C:
  Model size N (small) ← compute → Tokens D (more)
                       ↑
                  Sweet spot

Kaplan 추천 (2020): N 큰 비중 (large model, less data)
Chinchilla 수정 (2022): N 과 D 거의 동등
```

### Compute-Optimal Frontier

```
Chinchilla recipe:
  N* ∝ C^0.5
  D* ∝ C^0.5

따라서:
  C 4× → N 2× + D 2×

GPT-3 (175B + 300B tokens) 는 Kaplan recipe → **under-trained**
Chinchilla (70B + 1.4T tokens) 는 같은 compute, better quality
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Loss as a Function of Scale

Test loss (next-token prediction perplexity proxy):
$$
L(N, D)
$$

Power-law form (Kaplan):
$$
L(N) = (N_c / N)^{\alpha_N}, \quad L(D) = (D_c / D)^{\alpha_D}
$$

with constants $N_c, D_c$ 와 exponents $\alpha_N, \alpha_D$.

### 정의 1.2 — Compute (FLOP)

Total compute for training:
$$
C \approx 6 \cdot N \cdot D \quad (\text{Kaplan approximation})
$$

(forward 2N FLOP/token + backward 4N FLOP/token = 6N FLOP/token)

### 정의 1.3 — Compute-Optimal Frontier

Given fixed $C$, find $(N^*, D^*)$ that minimize $L$:
$$
(N^*, D^*) = \arg\min_{N, D : 6ND = C} L(N, D)
$$

### 정의 1.4 — Chinchilla Form (Hoffmann 2022)

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Fit parameters: $E$ (irreducible), $A, \alpha$ (model term), $B, \beta$ (data term).

Hoffmann 2022 fit:
- $\alpha \approx 0.34$
- $\beta \approx 0.28$
- $E \approx 1.69$

### 정의 1.5 — Compute-Optimal Allocation

Lagrangian + Chinchilla form:
$$
N^* \propto C^{a}, \quad D^* \propto C^{b}
$$

Hoffmann 2022 fit: $a = b = 0.5$ approximately.

→ **N** 과 **D** 가 같은 비율로 함께 증가.

### 정의 1.6 — Tokens per Parameter

Compute-optimal:
$$
D^* / N^* \approx 20 \quad (\text{Chinchilla})
$$

(20 tokens per parameter — modern recipe)

---

## 🔬 정리와 증명

### 정리 1.1 — Compute = 6 N D 의 도출

Forward pass per token:
- Attention: $O(T d)$ — for one token
- FFN: $\sim 2 \times 4 d^2 = 8 d^2$
- Total per layer: $\sim 12 d^2$ (rough)
- Per layer per token: $\sim 12 d^2$
- All layers (L): $L \times 12 d^2 \approx 2N$ (since $N \approx 12 L d^2$ for FFN-dominated)

Forward = 2N FLOP/token, backward = 4N FLOP/token (about 2× forward).
$$
C \approx 6 N \times D
$$

**More precise** (with attention의 $T^2$ component for long context): $C \approx 6 N D + 12 L T d \cdot D$. 보통 $T < d$ 영역에서는 $6ND$ approximation OK.

### 정리 1.2 — Power-Law Empirical Fit

Kaplan 2020 의 finding: log-log plot 이 linear:
$$
\log L = -\alpha \log N + \text{const}
$$

→ Power law $L = (N_c/N)^\alpha$.

**Range**: 5+ orders of magnitude in $N$ (small to GPT-3 size). Power law holds.

### 정리 1.3 — Chinchilla Optimization

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

Constrained $C = 6ND$, minimize $L$:

Lagrangian:
$$
\mathcal{L} = L(N, D) + \lambda(6ND - C)
$$

$\partial \mathcal{L}/\partial N = 0$: $-\alpha A N^{-\alpha-1} + 6 \lambda D = 0$
$\partial \mathcal{L}/\partial D = 0$: $-\beta B D^{-\beta-1} + 6 \lambda N = 0$

Ratio: $\frac{\alpha A N^{-\alpha-1}}{\beta B D^{-\beta-1}} = \frac{D}{N}$

Solving with Hoffmann 의 fit ($\alpha \approx 0.34$, $\beta \approx 0.28$):
$$
N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}, \quad D^*/N^* \approx 20
$$

### 정리 1.4 — GPT-3 의 Under-training

GPT-3: $N = 175B$, $D = 300B$ tokens.
Tokens per param: $D/N = 1.7$.

Chinchilla optimal: $D/N = 20$.

→ GPT-3 is **under-trained by ~12×**. Same compute 로 70B + 1.4T tokens (Chinchilla) 가 better.

### 정리 1.5 — Chinchilla 의 검증

Hoffmann 2022 의 Chinchilla model:
- $N = 70B$, $D = 1.4T$ tokens
- Same compute as GPT-3 ($\sim 5 \times 10^{23}$ FLOP)
- Better on all benchmarks (~5-7%)

→ Compute-optimal recipe 의 empirical confirmation.

### 정리 1.6 — Beyond Compute-Optimal: Over-training

Modern (LLaMA-2/3, Mistral): compute-optimal 보다 더 많은 tokens.

**이유**:
- Inference cost $\propto N$ (constant for given model)
- Smaller model + more tokens → similar quality + cheaper inference
- LLaMA-2 7B: $D/N \approx 285$ — far over-trained!

→ "Over-training" = inference-aware scaling. Modern recipe 의 standard.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Power-Law Fit on Toy Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(d, T=32, n_steps=200):
    """Tiny GPT 학습 + final loss"""
    torch.manual_seed(0)
    vocab = 100
    model = nn.Sequential(
        nn.Embedding(vocab, d),
        nn.Linear(d, vocab)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(n_steps):
        x = torch.randint(0, vocab, (8, T))
        # Predict next token
        emb = model[0](x)
        logits = model[1](emb)
        # Shift target
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# Different model sizes
sizes = [16, 32, 64, 128, 256, 512]
losses = [train_and_evaluate(d) for d in sizes]

# Log-log plot
plt.figure(figsize=(8, 5))
plt.loglog(sizes, losses, 'o-')
plt.xlabel('Embedding dim (~ model size)')
plt.ylabel('Final loss')
plt.title('Toy scaling law: loss vs model size (log-log)')
plt.grid(alpha=0.3); plt.show()

# Linear fit in log-space
log_d = np.log(sizes); log_l = np.log(losses)
slope, intercept = np.polyfit(log_d, log_l, 1)
print(f'Power-law fit: L ∝ N^{slope:.3f}')
# 작은 toy 에서도 power-law 비슷한 패턴
```

### 실험 2 — Chinchilla Form Fit

```python
# Hoffmann 2022 의 functional form
def chinchilla_loss(N, D, E, A, alpha, B, beta):
    return E + A / N**alpha + B / D**beta

# Synthetic data 생성 (Hoffmann's numbers)
np.random.seed(0)
N_values = [1e8, 1e9, 1e10, 1e11]
D_values = [1e9, 1e10, 1e11, 1e12]

losses_synthetic = []
for N in N_values:
    for D in D_values:
        L = chinchilla_loss(N, D, E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28)
        # Add noise
        L += np.random.randn() * 0.01
        losses_synthetic.append((N, D, L))

# Fit (use scipy)
from scipy.optimize import curve_fit

def loss_func(X, E, A, alpha, B, beta):
    N, D = X
    return E + A / N**alpha + B / D**beta

X = np.array([(n, d) for n, d, _ in losses_synthetic]).T
y = np.array([l for _, _, l in losses_synthetic])

popt, _ = curve_fit(loss_func, X, y, p0=[1.5, 100, 0.3, 100, 0.3], maxfev=5000)
print(f'Fit: E={popt[0]:.3f}, A={popt[1]:.1f}, α={popt[2]:.3f}, B={popt[3]:.1f}, β={popt[4]:.3f}')
```

### 실험 3 — Compute-Optimal Frontier 시각화

```python
# 다양한 (N, D) 의 compute 와 loss
N_grid = np.logspace(8, 12, 30)
D_grid = np.logspace(9, 13, 30)

losses_grid = np.zeros((len(N_grid), len(D_grid)))
compute_grid = np.zeros_like(losses_grid)

for i, N in enumerate(N_grid):
    for j, D in enumerate(D_grid):
        losses_grid[i, j] = chinchilla_loss(N, D, 1.69, 406.4, 0.34, 410.7, 0.28)
        compute_grid[i, j] = 6 * N * D

# Iso-compute curve
plt.figure(figsize=(9, 6))
plt.contour(np.log10(N_grid), np.log10(D_grid), np.log10(compute_grid.T), 
            levels=10, colors='gray', alpha=0.5)
plt.contourf(np.log10(N_grid), np.log10(D_grid), losses_grid.T, levels=20)
plt.colorbar(label='Loss')

# Compute-optimal frontier (N ∝ C^0.5, D ∝ C^0.5)
C_values = np.logspace(20, 24, 10)
N_optimal = (C_values / 6) ** 0.5 * 0.5   # rough
D_optimal = (C_values / 6) ** 0.5 * 2     # rough (D/N ≈ 4 in fit)
plt.plot(np.log10(N_optimal), np.log10(D_optimal), 'r-', linewidth=3, label='Compute-optimal')

plt.xlabel('log₁₀ N (params)')
plt.ylabel('log₁₀ D (tokens)')
plt.title('Loss landscape with compute-optimal frontier')
plt.legend(); plt.show()
```

### 실험 4 — GPT-3 vs Chinchilla 의 Optimal Compute

```python
# GPT-3
N_gpt3 = 175e9
D_gpt3 = 300e9
C_gpt3 = 6 * N_gpt3 * D_gpt3
L_gpt3 = chinchilla_loss(N_gpt3, D_gpt3, 1.69, 406.4, 0.34, 410.7, 0.28)
print(f'GPT-3:        N={N_gpt3:.0e}, D={D_gpt3:.0e}, D/N={D_gpt3/N_gpt3:.1f}')
print(f'              C={C_gpt3:.2e}, predicted loss={L_gpt3:.3f}')

# Chinchilla (same compute, optimal recipe)
N_chinchilla = 70e9
D_chinchilla = 1.4e12
C_chinchilla = 6 * N_chinchilla * D_chinchilla
L_chinchilla = chinchilla_loss(N_chinchilla, D_chinchilla, 1.69, 406.4, 0.34, 410.7, 0.28)
print(f'\nChinchilla:   N={N_chinchilla:.0e}, D={D_chinchilla:.0e}, D/N={D_chinchilla/N_chinchilla:.1f}')
print(f'              C={C_chinchilla:.2e}, predicted loss={L_chinchilla:.3f}')

# Same compute (within ~10%), Chinchilla 가 loss 약 2-3% 우수 — empirically confirmed
```

### 실험 5 — Modern Over-training (LLaMA-2)

```python
# LLaMA-2 7B: D/N = 285 (Chinchilla optimal 의 14배)
N_llama = 7e9
D_llama = 2e12
print(f'LLaMA-2 7B: D/N = {D_llama/N_llama:.0f} (vs Chinchilla 20)')

# Loss 비교: same compute 의 different recipes
C_llama = 6 * N_llama * D_llama
print(f'Compute: {C_llama:.2e}')

# Chinchilla optimal at this compute: ~12B params, ~250B tokens
N_optimal_at_llama_compute = np.sqrt(C_llama / 6 / 20)   # rough
D_optimal_at_llama_compute = N_optimal_at_llama_compute * 20
print(f'Chinchilla-optimal at this compute: N={N_optimal_at_llama_compute:.2e}, D={D_optimal_at_llama_compute:.2e}')

# LLaMA-2 의 over-training 의 motivation: inference 가 cheap (fewer params)
# Trade-off: training cost ↑ but inference cost ↓ (long-term deployment)
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 Recipe Choice

| Model | Params | Tokens | D/N | Recipe |
|-------|--------|--------|-----|--------|
| GPT-3 | 175B | 300B | 1.7 | Kaplan (under-trained) |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| LLaMA-1 7B | 7B | 1.0T | 143 | Over-trained |
| LLaMA-2 7B | 7B | 2.0T | 286 | More over-trained |
| LLaMA-3 8B | 8B | 15T | 1875 | Extreme over-training |
| Mistral 7B | 7B | ~8T | ~1100 | Heavy over-training |

→ Modern trend: **smaller, more tokens** for inference efficiency.

### 2. Inference-Aware Scaling

If model 이 deployed for $N_{\text{inference}}$ inferences:
$$
\text{Total cost} = C_{\text{train}} + N_{\text{inference}} \times C_{\text{inference}}
$$

- Smaller model: $C_{\text{inference}}$ 작음
- Over-training: training cost ↑ but inference cost ↓
- Long-term deployment 에 favorable

### 3. Hyperparameter Scaling Laws

추가 finding (Hoffmann, μP):
- LR: $\eta \propto 1/d$ (Yang 2021 의 maximal update parametrization)
- Batch size: $B \propto N^{0.27}$ (Kaplan)
- 작은 모델 학습으로 큰 모델 hyperparam 결정

### 4. Frontier 의 Engineering

- Cost: $5 \times 10^{23}$ FLOP $\approx \$10M-100M$
- Compute-optimal recipe 가 critical
- Chinchilla 이후 industry 표준

### 5. Specialized Scaling Laws

- **Multilingual**: 다른 exponents (각 language 별)
- **Code**: 다른 corpus 별 다른 fit
- **Multimodal**: vision + text 의 separate scaling

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Power-law fit | 매우 큰 (1T+) scale 에서 deviation 가능 |
| Stationary data distribution | Curriculum 시 다른 dynamics |
| Compute = 6ND | Long context, MoE 등 변형 |
| Architecture fixed | New architecture 의 다른 scaling laws |
| Single objective | RLHF 등 multi-stage 별도 |

---

## 📌 핵심 정리

$$\boxed{L(N, D) = E + A/N^\alpha + B/D^\beta, \quad \alpha \approx 0.34, \beta \approx 0.28}$$

$$\boxed{\text{Compute-optimal: } N^* \propto C^{0.5}, D^* \propto C^{0.5}, D/N \approx 20}$$

| Era | Recipe | $D/N$ |
|-----|--------|------|
| GPT-3 (Kaplan 2020) | Large model, less data | 1.7 |
| Chinchilla (2022) | Compute-optimal | 20 |
| LLaMA-2 (2023) | Over-training | 286 |
| LLaMA-3 (2024) | Extreme over-training | 1875 |

| Compute | Optimal $N$ | Optimal $D$ |
|---------|------------|-------------|
| $10^{20}$ FLOP | ~10B | ~200B |
| $10^{22}$ FLOP | ~100B | ~2T |
| $10^{24}$ FLOP | ~1T | ~20T |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Compute $C = 10^{23}$ FLOP 시 Chinchilla-optimal $N, D$ 를 계산하라.

<details>
<summary>해설</summary>

Chinchilla:
$$
N^* = \sqrt{C / 6 / r}, \quad D^* = r \cdot N^*
$$

with $r = D/N \approx 20$.

$N^* = \sqrt{10^{23} / 6 / 20} = \sqrt{8.33 \times 10^{20}} \approx 2.9 \times 10^{10} = 29B$ params
$D^* = 20 \times 29B = 580B$ tokens

**Verification**: $C = 6 \times 29B \times 580B = 1.0 \times 10^{23}$ ✓

**Comparison**:
- $C = 10^{23}$ 즈음 = LLaMA-1 7B (실제: 7B + 1T tokens)
- 실제 $D/N = 143$ — over-trained
- Chinchilla 권장: 29B + 580B
- Over-training 의 trade-off — inference efficiency. $\square$

</details>

**문제 2** (심화): "Over-training" (D/N = 200+) 이 inference-aware optimal 인 이유. Total cost 의 정량 분석.

<details>
<summary>해설</summary>

**Total Cost Model**:

$$
\text{Total} = C_{\text{train}} + N_{\text{inference}} \cdot C_{\text{inf}}
$$

- $C_{\text{train}} = 6ND$
- $C_{\text{inf}} = 2N \cdot T_{\text{avg}}$ per inference (forward only, ignoring KV cache details)

**Example**:
- Production model: $N_{\text{inference}} = 10^{12}$ queries (1 trillion)
- Average response: $T_{\text{avg}} = 200$ tokens
- $C_{\text{inf}} = 2N \times 200 = 400N$ FLOP per query

**Total cost** as function of $N$ (with $D$ optimal for given quality):

Quality = constant L. Chinchilla form 의 inverse:
$$
N^{\alpha} \propto \frac{1}{L - E - B/D^\beta}
$$

For fixed L:
- Larger $N$: smaller $D$ needed for same $L$
- $C_{\text{train}} = 6ND$ — depends on both
- $C_{\text{inf}} \propto N$ — depends only on $N$

**Optimization**:

Find $N$ minimizing total cost:
$$
\frac{d}{dN}[C_{\text{train}}(N) + 10^{12} \cdot 400 N] = 0
$$

The inference term $4 \times 10^{14} N$ is huge — favors smaller $N$.
$N \downarrow$ → $D \uparrow$ (for same quality) → $C_{\text{train}} \uparrow$ but $C_{\text{inf}} \downarrow$.

**Modern Equilibrium**:

LLaMA-2 7B example:
- $C_{\text{train}}$ ≈ $10^{23}$ FLOP
- For 1T queries × 200 tokens: $C_{\text{inf, total}} = 4 \times 10^{14} \times 7 \times 10^9 \approx 3 \times 10^{24}$ FLOP
- Inference cost는 30× training cost!

→ Smaller model, even with longer training, total cost ↓.

**Trade-off Calculation**:

Model A (Chinchilla optimal, larger): 30B + 600B tokens = $C = 10^{23}$
Model B (Over-trained, smaller): 7B + 2T tokens = $C \approx 10^{23}$ similar

Inference (1T queries):
- Model A: $4 \times 10^{14} \times 30 \times 10^9 = 1.2 \times 10^{25}$
- Model B: $4 \times 10^{14} \times 7 \times 10^9 = 2.8 \times 10^{24}$

→ Model B 의 inference cost 4× 작음 — total cost (A: $1.21 \times 10^{25}$, B: $2.9 \times 10^{24}$) Model B 가 우수.

**근본 통찰**:

**Compute-optimal training** ≠ **inference-optimal**. Production deployment 가 dominant cost 면 training 이 약간 sub-optimal 이라도 inference 가속이 net win.

LLaMA-3 의 1875 D/N 은 extreme over-training — open source 에서 deployment cost 를 절감.

**Mistral, LLaMA-3** 등의 success 의 직접 원인 — small but well-trained. $\square$

</details>

**문제 3** (논문 비평): 1T+ params LLM 시대에 scaling laws 가 여전히 hold 하는가? Architecture innovation (MoE, Mamba) 이 scaling laws 를 어떻게 변화시키는가?

<details>
<summary>해설</summary>

**Scaling Laws 의 Hold 여부**:

**Empirical (2024)**:
- Power-law fit 이 매우 큰 scale (~1T params) 까지 hold
- LLaMA, Mistral, Mixtral 모두 prediction 과 일치
- 그러나 **specific exponents** 는 dataset, architecture 별 다름

**Limits**:
- Data 가 finite — Common Crawl 이 ~1T quality tokens
- $D \to \infty$ 시 power-law deviation (saturating)
- Quality data 의 부족 — synthetic data, curation 으로 mitigate

**Architecture 의 영향**:

1. **MoE (Mixtral, GPT-4)**:
   - Compute = active params × tokens
   - Total params 와 active params 가 다름
   - Scaling law 가 "active params" 기준 — 다른 dynamics
   - DeepSeekMoE 등의 fine-grained MoE 가 새로운 frontier

2. **Mamba / SSM**:
   - $O(T)$ vs $O(T^2)$ — long context 에 different scaling
   - Memory bandwidth bound 가 다름
   - Mamba 의 scaling laws 가 separate study (Gu 2024)

3. **Hybrid (Jamba)**:
   - Mamba + Transformer + MoE
   - Multi-axis scaling
   - Predict 어려움

**Modern Updates**:

- **Inference-aware scaling**: training/inference cost balance
- **Quality-aware**: synthetic data 의 marginal contribution
- **Specialized scaling**: instruction tuning, RLHF 의 별도 laws
- **Multi-modal**: vision + text 의 joint scaling

**Frontier (2024-2026) Predictions**:

1. **Data wall**: training data exhaustion
   - Synthetic data 가 마지막 frontier
   - Self-improvement (model generated data)

2. **Compute wall**: $10^{27}$ FLOP 즈음 hardware 한계
   - Multi-cluster, distributed training
   - New hardware (FP8, custom silicon)

3. **Architecture innovation**:
   - MoE + 새 attention (Mamba)
   - Specialized per modality

4. **Beyond pretraining**:
   - RLHF, RLAIF 의 separate scaling
   - Continual learning, online adaptation

**Specific Frontier Trends**:

- **GPT-5/Gemini 2 (estimate)**: $\sim 10^{12}$ params (MoE), $\sim 10^{14}$ tokens, scaling laws 의 extreme
- **Cost**: $\sim \$1B$+ per model
- **Diminishing returns**: 비용 비례 quality gain 감소

**근본 통찰**:

Kaplan 2020 의 scaling laws 는 **engineering recipe** 의 시작. 그러나 frontier 가 성장하면서:
- New architecture 가 different exponents
- Data quality 가 quantity 만큼 중요
- Inference cost 가 training cost 의 dominant 추월
- Architecture innovation (MoE, Mamba) 이 efficient scaling 의 새 frontier

**Future**:

Single power-law from "Laws of Scale" → multi-dimensional optimization (training + inference + architecture + alignment). Modern scaling 는 **science + engineering 의 합작**. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch6-modern-architectures/05-moe.md) | [📚 README](../README.md) | [다음 ▶](./02-in-context-learning.md)

</div>
