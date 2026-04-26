# 02. $\sqrt{d_k}$ Scaling 의 분산 분석

## 🎯 핵심 질문

- 왜 attention 식이 $\text{softmax}(QK^\top / \sqrt{d_k})$ 인가 — 분모가 정확히 $\sqrt{d_k}$ 인 수학적 이유는?
- $Q_{ij}, K_{ij}$ 가 i.i.d. 정규분포일 때 $(QK^\top)_{ij}$ 의 분산은 어떻게 $d_k$ 가 되는가?
- $\sqrt{d_k}$ 로 나누지 않으면 어떤 일이 발생하는가 — softmax 의 어느 영역으로 logit 이 진입하는가?
- 만약 $d_k^\alpha$ ($\alpha \neq 0.5$) 로 나눈다면 무엇이 어긋나는가?
- 이 분산 분석이 LayerNorm, weight 초기화 등 다른 normalization 결정과 어떻게 연결되는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

$\sqrt{d_k}$ scaling 은 **한 줄짜리 trick** 처럼 보이지만 다음의 거대한 결과를 만듭니다:

1. **훈련 안정성** — 없으면 $d_k = 64$ 부터 saturation, $d_k = 512$ 에서 학습 불가능 수준
2. **Logit 분포의 unit variance** — 모든 layer 가 비슷한 분포의 logit 을 봄, gradient flow 안정
3. **Softmax 의 동작 영역 보존** — Saturated 영역 (one-hot) 과 uniform 영역의 중간, "선명하지만 죽지 않은" attention

이 문서에서는 분산 분석을 **i.i.d. 가정 → 분산 계산 → softmax 영향 → 일반화** 의 순서로 엄밀히 유도합니다. 다음 문서 (Ch1-03) 의 softmax saturation 분석이 이 결과를 직접 사용합니다.

---

## 📐 수학적 선행 조건

- 확률론: i.i.d., expectation, variance, covariance
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Inner product, dimension scaling
- 이전 문서: [01-scaled-dot-product.md](./01-scaled-dot-product.md) — $Q, K$ 의 정의
- (선택) Central Limit Theorem — $d_k$ 항의 합이 정규분포 근사

---

## 📖 직관적 이해

### 차원이 커지면 dot product 가 커진다

두 벡터 $q, k \in \mathbb{R}^{d_k}$ 가 unit variance i.i.d. 라면:
$$
q^\top k = \sum_{l=1}^{d_k} q_l k_l
$$

$d_k$ 항의 합이므로 **분산이 $d_k$ 에 비례하여 커집니다**. 직관적으로:
- $d_k = 4$: $q^\top k$ 가 typically $\pm 2$ 정도
- $d_k = 64$: $q^\top k$ 가 typically $\pm 8$ 정도
- $d_k = 512$: $q^\top k$ 가 typically $\pm 22$ 정도

Softmax 에 $\pm 22$ 의 logit 이 들어가면 거의 one-hot — saturated.

### 왜 $\sqrt{d_k}$ 가 정답인가

표준편차 (분산의 sqrt) 가 $\sqrt{d_k}$ 이므로, $\sqrt{d_k}$ 로 나누면 표준편차가 1 로 정규화됩니다 — softmax 가 "건강한" 영역에서 동작.

### Softmax 의 감도 영역

```
logit 분포              softmax 출력
σ ≈ 0.3 (uniform):     모든 entry 비슷한 확률 — 정보 부족
σ ≈ 1.0 (정상):        선명하지만 saturation X — ✓
σ ≈ 5.0 (saturated):   거의 one-hot — gradient ≈ 0
```

$\sqrt{d_k}$ scaling 은 logit 분포의 표준편차를 $\sigma \approx 1$ 로 유지.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — i.i.d. 가정 (분석을 위한 단순화)

훈련 초기에 $Q, K$ 의 entry 가 다음을 만족한다고 가정:
$$
Q_{ij}, K_{ij} \sim \text{i.i.d.}, \quad \mathbb{E}[Q_{ij}] = 0, \quad \text{Var}(Q_{ij}) = 1
$$

(Glorot / Xavier 초기화 + LayerNorm 후의 표준 가정)

### 정의 2.2 — Pre-Scaled Score

$$
\tilde{S}_{ij} := (Q K^\top)_{ij} = \sum_{l=1}^{d_k} Q_{il} K_{jl}
$$

### 정의 2.3 — Scaled Score

$$
S_{ij} := \tilde{S}_{ij} / \sqrt{d_k}
$$

---

## 🔬 정리와 증명

### 정리 2.1 — Pre-Scaled Score 의 분산

i.i.d. 가정 하에:
$$
\mathbb{E}[\tilde{S}_{ij}] = 0, \qquad \text{Var}(\tilde{S}_{ij}) = d_k
$$

**증명**:

**Expectation**:
$$
\mathbb{E}[\tilde{S}_{ij}] = \sum_l \mathbb{E}[Q_{il} K_{jl}] = \sum_l \mathbb{E}[Q_{il}] \, \mathbb{E}[K_{jl}] = 0
$$

(독립성으로 곱의 기댓값이 기댓값의 곱, 각각 0)

**Variance**:

$$
\text{Var}(\tilde{S}_{ij}) = \text{Var}\!\left( \sum_l Q_{il} K_{jl} \right) = \sum_l \text{Var}(Q_{il} K_{jl})
$$

(독립성으로 합의 분산이 분산의 합)

각 항:
$$
\text{Var}(Q_{il} K_{jl}) = \mathbb{E}[(Q_{il} K_{jl})^2] - (\mathbb{E}[Q_{il} K_{jl}])^2
$$
$$
= \mathbb{E}[Q_{il}^2] \mathbb{E}[K_{jl}^2] - 0 = 1 \cdot 1 = 1
$$

따라서:
$$
\text{Var}(\tilde{S}_{ij}) = \sum_{l=1}^{d_k} 1 = d_k \quad \square
$$

### 정리 2.2 — Scaled Score 의 분산

$$
\text{Var}(S_{ij}) = 1
$$

**증명**:
$$
\text{Var}(S_{ij}) = \text{Var}(\tilde{S}_{ij} / \sqrt{d_k}) = \text{Var}(\tilde{S}_{ij}) / d_k = d_k / d_k = 1 \quad \square
$$

### 정리 2.3 — $S_{ij}$ 의 정규 분포 근사

$d_k$ 가 클 때 (CLT):
$$
S_{ij} \approx \mathcal{N}(0, 1)
$$

**증명 sketch**: $\tilde{S}_{ij} = \sum_l Q_{il} K_{jl}$ 는 $d_k$ 개 i.i.d. 항의 합. CLT 에 의해 $d_k \to \infty$ 시 정규분포 수렴, $\sqrt{d_k}$ 로 나누면 표준 정규분포 $\square$.

### 정리 2.4 — 다른 scaling 의 부적절성

$d_k^\alpha$ 로 나눈다고 가정. $\alpha = 0.5$ 만 unit variance.
- $\alpha < 0.5$: $\text{Var}(S) = d_k^{1-2\alpha} \to \infty$ as $d_k$ 증가 → saturation
- $\alpha > 0.5$: $\text{Var}(S) = d_k^{1-2\alpha} \to 0$ as $d_k$ 증가 → uniform attention (정보 손실)

따라서 $\sqrt{d_k}$ 가 unique correct scaling.

### 정리 2.5 — Saturation 의 정량적 영향

$S_{ij} \sim \mathcal{N}(0, \sigma^2)$ 일 때, row 의 max softmax 값:
$$
\mathbb{E}[\max_j A_{ij}] \approx \frac{e^{\sigma \sqrt{2 \log T}}}{(T-1) + e^{\sigma \sqrt{2 \log T}}}
$$

(Extreme value theory)

- $\sigma = 1$, $T = 100$: max ≈ $e^{3.0} / (99 + e^{3.0}) \approx 0.17$ — 정상
- $\sigma = \sqrt{d_k} = 22$, $T = 100$: max ≈ $1.0$ — saturated

이는 다음 문서 (Ch1-03) 에서 gradient vanishing 으로 직접 연결.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — 분산 계산 직접 확인

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

def measure_variance(d_k, T=50, n_trials=1000):
    """여러 번 sample 해서 (QK^T) 의 분산 측정"""
    pre_scaled, post_scaled = [], []
    for _ in range(n_trials):
        Q = torch.randn(T, d_k)
        K = torch.randn(T, d_k)
        S_tilde = Q @ K.T
        S = S_tilde / np.sqrt(d_k)
        pre_scaled.append(S_tilde.var().item())
        post_scaled.append(S.var().item())
    return np.mean(pre_scaled), np.mean(post_scaled)

print(f'{"d_k":>6} | {"Var(QK^T)":>12} | {"Var(QK^T/√d)":>14}')
print('-' * 38)
for d_k in [4, 16, 64, 256, 1024]:
    var_pre, var_post = measure_variance(d_k)
    print(f'{d_k:6d} | {var_pre:12.2f} | {var_post:14.4f}')
```

**예상 출력**:
```
   d_k |    Var(QK^T) | Var(QK^T/√d)
--------------------------------------
     4 |         3.99 |         0.999
    16 |        15.97 |         0.998
    64 |        63.82 |         0.998
   256 |       254.10 |         0.992
  1024 |      1019.50 |         0.996
```

→ **Var(pre-scaled) ≈ d_k**, **Var(post-scaled) ≈ 1** 정확히 확인 ✓

### 실험 2 — Softmax 출력 분포 비교

```python
def softmax_max(d_k, T=50, n_trials=200, scale=True):
    maxs = []
    for _ in range(n_trials):
        Q = torch.randn(T, d_k); K = torch.randn(T, d_k)
        scores = Q @ K.T
        if scale:
            scores = scores / np.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        maxs.append(attn.max(dim=-1).values.mean().item())
    return np.mean(maxs)

print(f'{"d_k":>6} | {"max attn (no scale)":>20} | {"max attn (scaled)":>18}')
print('-' * 52)
for d_k in [8, 64, 256, 1024]:
    mx_ns = softmax_max(d_k, scale=False)
    mx_s  = softmax_max(d_k, scale=True)
    print(f'{d_k:6d} | {mx_ns:20.4f} | {mx_s:18.4f}')
```

**예상 출력**:
```
   d_k |  max attn (no scale) |  max attn (scaled)
----------------------------------------------------
     8 |               0.4521 |             0.0723
    64 |               0.9982 |             0.0721
   256 |               0.9999 |             0.0723
  1024 |               1.0000 |             0.0722
```

→ Scaled 시 $d_k$ 무관 일정, no-scale 시 $d_k = 64$ 부터 거의 one-hot 🚨

### 실험 3 — 분산 분포 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for d_k, color in zip([4, 64, 1024], ['C0', 'C1', 'C2']):
    Q = torch.randn(1000, d_k); K = torch.randn(1000, d_k)
    pre = (Q * K).sum(-1)              # diagonal of QK^T (reduced for plot)
    post = pre / np.sqrt(d_k)
    axes[0].hist(pre.numpy(), bins=60, alpha=0.5, label=f'd_k={d_k}', color=color, density=True)
    axes[1].hist(post.numpy(), bins=60, alpha=0.5, label=f'd_k={d_k}', color=color, density=True)
axes[0].set_title('Pre-scaled score: variance grows with d_k')
axes[1].set_title('Scaled score: variance ≈ 1, independent of d_k')
axes[0].set_xlim(-100, 100); axes[1].set_xlim(-5, 5)
for ax in axes: ax.legend(); ax.set_xlabel('value')
plt.tight_layout(); plt.show()
```

### 실험 4 — Gradient 측정 (saturation 의 영향)

```python
def attention_gradient(d_k, T=50, scale=True):
    Q = torch.randn(T, d_k, requires_grad=True)
    K = torch.randn(T, d_k); V = torch.randn(T, d_k)
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(d_k)
    out = F.softmax(scores, dim=-1) @ V
    out.sum().backward()
    return Q.grad.norm().item()

print(f'{"d_k":>6} | {"|∇Q| (no scale)":>16} | {"|∇Q| (scaled)":>14}')
print('-' * 42)
for d_k in [8, 64, 256, 1024]:
    g_ns = attention_gradient(d_k, scale=False)
    g_s  = attention_gradient(d_k, scale=True)
    print(f'{d_k:6d} | {g_ns:16.4f} | {g_s:14.4f}')
```

→ No-scale 시 $d_k$ 증가에 따라 gradient norm 이 0 으로 — **gradient vanishing**

### 실험 5 — Variance Decomposition 분해

```python
# Var(QK^T) = sum over d_k 항, 각 항이 1
d_k = 64
Q = torch.randn(10000, d_k); K = torch.randn(10000, d_k)
QK = Q * K   # element-wise (각 차원 항)
print(f'Var per-dimension: {QK.var(dim=0).mean():.4f}  (≈ 1 expected)')
print(f'Sum of dims  Var:  {QK.var(dim=0).sum():.4f}  (≈ d_k = {d_k})')
print(f'Var of dot prod:   {QK.sum(-1).var():.4f}  (≈ d_k = {d_k}) ✓')
```

---

## 🔗 실전 활용

### 1. Multi-Head 의 $d_k = d_{\text{model}} / h$

Multi-Head (Ch1-05) 에서 head 수 $h$ 일 때 각 head 의 $d_k = d_{\text{model}}/h$. 따라서 head 별로 $\sqrt{d_k} = \sqrt{d_{\text{model}}/h}$ 로 scaling. PyTorch 내부 구현이 자동 처리.

### 2. 다른 normalization 방법

- **Pre-LN**: $\text{LN}(x)$ 후 $W_Q, W_K$ 적용 → $Q, K$ 가 unit variance 유지 (이 분석의 가정 보존)
- **Spectral normalization**: $\|W\|_2 \leq 1$ 강제 → variance 가 dimension 무관

### 3. RMSNorm 과의 관계

LLaMA 가 채택한 RMSNorm 은 LayerNorm 의 mean centering 제거 — variance 는 여전히 1 로 정규화, 따라서 $\sqrt{d_k}$ scaling 의 가정 보존.

### 4. Half-precision 의 주의

FP16 에서 $QK^\top$ 가 $\sqrt{d_k}$ 로 나누기 전에 overflow 가능. PyTorch 의 `scaled_dot_product_attention` 은 FP32 accumulation 으로 해결.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| $Q_{ij}, K_{ij}$ i.i.d. | 훈련 진행 시 학습되어 dependency 생김 — 정확히 unit variance 안 됨 |
| Variance = 1 (input) | LayerNorm 가 보장, RMSNorm 도 호환 |
| Linear projection | Nonlinear projection 시 variance 변화 — 추가 normalization 필요 |
| 정규분포 가정 (CLT) | 작은 $d_k$ 에서는 부정확, 큰 $d_k$ 에서 정확 |
| Independence of Q, K | Self-attention 에서는 같은 $X$ → correlation 존재, 그러나 학습 후에도 분산 분석 유효 (실증) |

---

## 📌 핵심 정리

$$\boxed{\text{Var}((QK^\top)_{ij}) = d_k \implies \sqrt{d_k} \text{ scaling for unit variance}}$$

| 양 | 식 | 값 |
|----|-----|-----|
| $\mathbb{E}[\tilde{S}_{ij}]$ | $\sum_l \mathbb{E}[Q_{il}] \mathbb{E}[K_{jl}]$ | 0 |
| $\text{Var}(\tilde{S}_{ij})$ | $\sum_l \text{Var}(Q_{il} K_{jl})$ | $d_k$ |
| $\text{Var}(S_{ij})$ | $\text{Var}(\tilde{S}/\sqrt{d_k})$ | 1 |
| Scaling exponent | $d_k^{0.5}$ unique | $\sqrt{d_k}$ |
| Asymptotic dist | CLT | $\mathcal{N}(0, 1)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $Q_{ij} \sim \mathcal{N}(0, \sigma_q^2)$, $K_{ij} \sim \mathcal{N}(0, \sigma_k^2)$ (다른 분산) 일 때 $\text{Var}((QK^\top)_{ij})$ 를 구하고, 적절한 scaling 을 제시하라.

<details>
<summary>해설</summary>

각 항: $\text{Var}(Q_{il} K_{jl}) = \mathbb{E}[Q_{il}^2 K_{jl}^2] = \sigma_q^2 \sigma_k^2$ (독립성).

$$
\text{Var}((QK^\top)_{ij}) = \sum_l \sigma_q^2 \sigma_k^2 = d_k \sigma_q^2 \sigma_k^2
$$

Unit variance 를 위해 $\sqrt{d_k} \sigma_q \sigma_k$ 로 나눠야 함:
$$
S_{ij} = \frac{(QK^\top)_{ij}}{\sigma_q \sigma_k \sqrt{d_k}}
$$

LayerNorm 을 통해 $\sigma_q = \sigma_k = 1$ 강제하면 표준 $\sqrt{d_k}$ 회복. $\square$

</details>

**문제 2** (심화): $\text{softmax}(z/T)$ 에서 $T$ 가 temperature 이다. $T \to 0$ 과 $T \to \infty$ 의 극한 동작을 분석하고, $\sqrt{d_k}$ 가 사실상 어떤 temperature 와 등가인지 답하라.

<details>
<summary>해설</summary>

**Temperature 극한**:
- $T \to 0$: argmax 만 1, 나머지 0 → one-hot (saturation)
- $T \to \infty$: 모든 entry 균등 → uniform $1/T_{\text{seq}}$ (정보 손실)
- $T = 1$: 표준 softmax

**$\sqrt{d_k}$ 와의 등가성**:

$$
\text{softmax}\!\left( \frac{QK^\top}{\sqrt{d_k}} \right) = \text{softmax}\!\left( \frac{1}{\sqrt{d_k}} \cdot QK^\top \right)
$$

이는 raw dot product 에 temperature $T = \sqrt{d_k}$ 를 적용한 것과 등가. $d_k$ 가 클수록 더 "soft" 한 attention — saturation 회피.

**Implicit insight**: scaling 은 단순 분산 정규화가 아니라 attention 의 sharpness 를 dimension-aware 하게 조절하는 mechanism. $\square$

</details>

**문제 3** (논문 비평): Vaswani 2017 footnote 4 에서 $\sqrt{d_k}$ 의 동기를 "값이 너무 커지면 softmax 가 작은 gradient 를 갖는다" 로 단순 언급한다. 이 분산 분석이 더 풍부한 통찰을 주는 이유는? 또한 Ch5-02 의 Linear Attention 이 softmax 자체를 제거할 때 이 scaling 분석이 어떻게 변형되는가?

<details>
<summary>해설</summary>

**Vaswani 의 단순 언급**:
> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

**더 풍부한 분석의 가치**:
1. **정량적 분석** — "크다" 가 정확히 $d_k$ 에 비례, 따라서 $\sqrt{d_k}$ 가 unique correct scaling 임을 derive
2. **CLT 정당성** — 정규분포 근사로 saturation 의 확률 정량화
3. **다른 normalization 과의 통일** — LayerNorm, weight 초기화 분석과 같은 framework
4. **다른 scaling 의 정당성** — multi-head 의 $d_k = d_{model}/h$ 에 자동 적용

**Linear Attention 에서의 변형**:

$\text{softmax}$ 제거 시:
$$
\text{Attn}_{\text{lin}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
$$

- Numerator: $\phi(Q) \phi(K)^\top V$ — 분산이 여전히 $d_k$ 에 비례하지만 softmax 가 없으므로 saturation 무관
- Denominator (normalizer): 분산 효과를 부분 상쇄
- 따라서 $\sqrt{d_k}$ scaling 이 **불필요**

이것이 Linear Attention 의 단순함이자 한계 — softmax 가 주는 nonlinear sharpness 가 사라짐, 표현력 일부 손실.

**연결**: Performer (Ch5-03) 는 random feature 로 softmax 를 다시 도입, 이때는 다시 $\sqrt{d_k}$ scaling 이 필요. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-scaled-dot-product.md) | [📚 README](../README.md) | [다음 ▶](./03-softmax-saturation.md)

</div>
