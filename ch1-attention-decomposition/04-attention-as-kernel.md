# 04. Attention as Kernel Method

## 🎯 핵심 질문

- $\text{softmax}(QK^\top)_{ij}$ 가 어떤 kernel function $\kappa(q_i, k_j)$ 의 row-normalized 형태인가?
- Attention 이 Nadaraya-Watson estimator 의 학습 가능 일반화로 해석되는 이유는?
- Exponential kernel $\exp(q^\top k / \sqrt{d_k})$ 이 RBF kernel 과 어떤 관계인가?
- 이 kernel 관점이 Linear Attention (Ch5-02), Performer (Ch5-03) 에 어떻게 직접 활용되는가?
- Attention 의 표현력이 RKHS (Reproducing Kernel Hilbert Space) 위계에서 어디에 위치하는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Attention 을 kernel 로 해석하는 것은 단순한 수학적 재포장이 아닙니다. 다음의 거대한 결과를 만듭니다:

1. **Nadaraya-Watson 일반화** — Attention 이 학습 가능한 nonparametric regression
2. **Linear Attention 의 직접 동기** — $\exp(q^\top k) \approx \phi(q)^\top \phi(k)$ 분해 시 $O(T^2) \to O(T)$
3. **Performer 의 random features** — Kernel Methods 레포의 random feature 가 직접 적용
4. **표현력 분석의 framework** — RKHS 위계로 attention 의 한계 분석

이 문서는 attention 을 **kernel method 의 학습 가능 변형** 으로 재해석하고, Ch5 의 효율화 기법들의 이론적 토대를 다집니다.

---

## 📐 수학적 선행 조건

- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): Kernel function, RBF, RKHS, random features
- 이전 문서: [01-scaled-dot-product.md](./01-scaled-dot-product.md), [02-sqrt-dk-scaling.md](./02-sqrt-dk-scaling.md)
- 통계학: Nonparametric regression, kernel density estimation

---

## 📖 직관적 이해

### Nadaraya-Watson Estimator

Nonparametric regression 의 고전:
$$
\hat{f}(x) = \frac{\sum_i \kappa(x, x_i) y_i}{\sum_i \kappa(x, x_i)}
$$

각 학습 데이터 $(x_i, y_i)$ 에 대해 kernel similarity 가중합. **Attention 식과 같은 구조**:
$$
\text{Attn}_i = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)} \quad \text{where} \quad \kappa = \exp(\cdot / \sqrt{d_k})
$$

### Attention 은 학습 가능한 Nadaraya-Watson

차이점:
- Nadaraya-Watson: $x_i$ 가 고정 데이터, kernel 은 hand-designed
- Attention: $k_j = X W_K$ 가 학습되는 representation, kernel 은 fixed exponential

Attention = **데이터 점을 학습하는 nonparametric regression**. 각 token 이 prediction time 에 다른 token 들에 query.

### Exponential Kernel = "Soft Hashing"

```
Hard hash:    q_i hits exactly one bucket (key)
Soft hash:    q_i has weights to all buckets, weight ∝ exp(similarity)
```

Attention 은 differentiable soft hash, exp 가 sharpness 결정.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Kernel Function

$\kappa: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 가 **kernel** 이려면:
- Symmetric: $\kappa(x, y) = \kappa(y, x)$
- Positive semi-definite: 임의 $x_1, \ldots, x_n$ 에 대해 Gram matrix $K_{ij} = \kappa(x_i, x_j) \succeq 0$

### 정의 4.2 — Exponential Inner Product Kernel

$$
\kappa_{\exp}(q, k) := \exp\!\left( \frac{q^\top k}{\sqrt{d_k}} \right)
$$

이는 PSD 임. (Mercer's theorem 적용 가능)

### 정의 4.3 — RBF (Gaussian) Kernel

$$
\kappa_{\text{RBF}}(q, k) := \exp\!\left( -\frac{\|q - k\|^2}{2\sigma^2} \right)
$$

### 정의 4.4 — Attention as Kernel Smoother

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \kappa_{\exp}(q_i, k_j) v_j}{\sum_j \kappa_{\exp}(q_i, k_j)}
$$

(분모가 row-normalization)

### 정의 4.5 — Feature Map (Mercer)

Kernel $\kappa(x, y) = \langle \phi(x), \phi(y) \rangle$ 의 분해. $\phi: \mathcal{X} \to \mathcal{H}$ (Hilbert space).

---

## 🔬 정리와 증명

### 정리 4.1 — Attention = Row-normalized Kernel Smoother

$$
\text{softmax}(QK^\top / \sqrt{d_k})_{ij} = \frac{\kappa_{\exp}(q_i, k_j)}{\sum_l \kappa_{\exp}(q_i, k_l)}
$$

**증명**: 정의 적용:
$$
\text{softmax}(S)_{ij} = \frac{e^{S_{ij}}}{\sum_l e^{S_{il}}} = \frac{e^{q_i^\top k_j / \sqrt{d_k}}}{\sum_l e^{q_i^\top k_l / \sqrt{d_k}}} = \frac{\kappa_{\exp}(q_i, k_j)}{\sum_l \kappa_{\exp}(q_i, k_l)} \quad \square
$$

### 정리 4.2 — Exp Inner Product 와 RBF 의 관계

$$
\kappa_{\exp}(q, k) = \exp\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \exp\!\left(\frac{\|q\|^2 + \|k\|^2 - \|q - k\|^2}{2 \sqrt{d_k}}\right)
$$

(**Identity**: $q^\top k = (\|q\|^2 + \|k\|^2 - \|q - k\|^2)/2$)

만약 $\|q\|, \|k\|$ 가 normalize 되어 있다면:
$$
\kappa_{\exp}(q, k) \propto \exp\!\left(-\frac{\|q-k\|^2}{2\sqrt{d_k}}\right)
$$

→ **RBF kernel** 과 본질적으로 같음 $\square$

### 정리 4.3 — Mercer Decomposition (분해 가능성)

$\kappa_{\exp}(q, k) = \exp(q^\top k)$ 는 무한 차원 feature map 으로 분해 가능 (Taylor):
$$
\exp(q^\top k) = \sum_{n=0}^\infty \frac{(q^\top k)^n}{n!}
$$

각 항 $(q^\top k)^n$ 은 차수 $n$ polynomial kernel — feature 차원 $\binom{d_k + n - 1}{n}$.

따라서 **exp kernel = 무한 차원 polynomial kernel 의 weighted sum** $\square$

### 정리 4.4 — Random Feature Approximation (Performer 의 토대)

**Bochner's theorem** + Rahimi-Recht 2007:
$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k), \qquad \phi(x) = e^{\|x\|^2 / 2} \cdot [\cos(\omega^\top x), \sin(\omega^\top x)]
$$

with $\omega \sim \mathcal{N}(0, I)$. $D$ 개 random sample 으로 $D$-차원 feature map 근사.

**Performer FAVOR+** 는 이 근사를 positive feature 와 orthogonal RF 로 개선 — Ch5-03에서 자세히.

**의미**: $\text{softmax}(QK^\top)$ 를 $\phi(Q) \phi(K)^\top$ 로 근사할 수 있다면 **Linear Attention 과 같은 결합 순서 변경 가능** → $O(T^2) \to O(T)$.

### 정리 4.5 — Linear Attention 의 Kernel 해석

Linear Attention (Katharopoulos 2020):
$$
\text{Attn}_{\text{lin}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
$$

는 임의 feature map $\phi$ 에 대한 kernel smoother:
$$
\kappa_{\phi}(q, k) = \phi(q)^\top \phi(k)
$$

$\phi(x) = \text{ELU}(x) + 1$ 같은 단순 선택은 exp kernel 의 정확한 근사 아니지만 (sharp peak 잃음), 결합 순서 변경의 이점이 trade-off 가치 있음.

### 정리 4.6 — Attention 의 표현력 (RKHS 관점)

$\kappa_{\exp}$ 가 universal kernel (대부분의 연속 함수 RKHS 로 근사 가능) → attention 의 single layer 가 충분히 표현력 풍부.

그러나 **finite sample regime** 에서 representation $K = X W_K$ 가 데이터 의존적, deep stacking + FFN 이 표현력에 추가 기여.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Attention = Kernel Smoother 등가성

```python
import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
T, d_k = 6, 8
Q = torch.randn(T, d_k); K = torch.randn(T, d_k); V = torch.randn(T, d_k)

# 표준 attention
scores = Q @ K.T / np.sqrt(d_k)
attn_std = F.softmax(scores, dim=-1) @ V

# Kernel smoother 형태
def exp_kernel(q, k, d_k):
    return torch.exp((q @ k) / np.sqrt(d_k))

kernel_matrix = exp_kernel(Q, K.T, d_k)   # (T, T) — 각 (i,j) 가 κ(q_i, k_j)
kernel_norm = kernel_matrix / kernel_matrix.sum(dim=-1, keepdim=True)
attn_kernel = kernel_norm @ V

print(f'Max difference: {(attn_std - attn_kernel).abs().max():.2e}')   # ≈ 0
```

### 실험 2 — RBF Kernel 과의 관계

```python
# Q, K 를 unit norm 으로 정규화
Q_n = Q / Q.norm(dim=-1, keepdim=True)
K_n = K / K.norm(dim=-1, keepdim=True)

# Exp inner product
exp_inner = torch.exp(Q_n @ K_n.T / np.sqrt(d_k))

# RBF (with appropriate sigma)
sq_dist = ((Q_n[:, None] - K_n[None, :]) ** 2).sum(-1)
exp_rbf = torch.exp(-sq_dist / (2 * np.sqrt(d_k)))

# Unit norm 시: exp(q^T k / √d) ∝ exp(-||q-k||² / (2√d)) × exp(2/(2√d))
ratio = exp_inner / exp_rbf
print(f'Ratio (should be ~constant): {ratio.std() / ratio.mean():.4e}')   # 매우 작음
```

### 실험 3 — Random Feature Approximation

```python
def random_features(x, D, seed=0):
    """Rahimi-Recht 2007 의 random feature for exp(q^T k)"""
    torch.manual_seed(seed)
    d = x.size(-1)
    omega = torch.randn(d, D)
    proj = x @ omega
    return torch.exp(0.5 * (x ** 2).sum(-1, keepdim=True)) * \
           torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / np.sqrt(D)

D = 256
phi_Q = random_features(Q, D)
phi_K = random_features(K, D)

kernel_approx = phi_Q @ phi_K.T   # (T, T)
kernel_true = torch.exp(Q @ K.T)   # without √d for simplicity

print(f'Approx vs true (relative): {(kernel_approx - kernel_true).abs().mean() / kernel_true.abs().mean():.4f}')
# D 가 클수록 작아짐
```

### 실험 4 — Linear Attention as Kernel Smoother

```python
# Linear attention with ELU+1 feature
def phi_linear(x):
    return F.elu(x) + 1

phi_Q_lin = phi_linear(Q)
phi_K_lin = phi_linear(K)

# Linear attention
KV = phi_K_lin.T @ V               # (d_k, d_k)
K_sum = phi_K_lin.sum(0)           # (d_k,)
num = phi_Q_lin @ KV               # (T, d_k)
denom = (phi_Q_lin @ K_sum)[:, None] + 1e-6
attn_lin = num / denom

# 표준 attention 과 비교
print(f'Standard: {attn_std[0, :3]}')
print(f'Linear:   {attn_lin[0, :3]}')
# 다름 (sharp 한 exp kernel 을 잃음), 그러나 같은 kernel smoother 형식
```

### 실험 5 — Mercer Decomposition 시각화

```python
import matplotlib.pyplot as plt

# 작은 차원에서 polynomial kernel terms 의 contribution
def poly_kernel_terms(q, k, max_order=8):
    """exp(q^T k) ≈ Σ (q^T k)^n / n!"""
    inner = q @ k.T
    terms = []
    for n in range(max_order):
        terms.append((inner ** n) / np.math.factorial(n))
    return torch.stack(terms)   # (max_order, T, T)

terms = poly_kernel_terms(Q, K)
exp_true = torch.exp(Q @ K.T)
exp_approx = terms.sum(0)

print(f'Truncated 8 terms vs full exp: {(exp_approx - exp_true).abs().max():.4f}')
# 충분한 order 에서 정확한 근사
```

---

## 🔗 실전 활용

### 1. Performer 의 FAVOR+ (Ch5-03)

Random feature 를 **positive** & **orthogonal** 로 개선:
- Positive: $\phi(x) = \exp(-\|x\|^2/2) [\exp(\omega^\top x); \exp(-\omega^\top x)]$ — 음수 entry 없음 → variance 감소
- Orthogonal: $\omega$ 들이 mutually orthogonal — variance 추가 감소

### 2. Kernelizable Attention 의 일반 framework

Tsai 2019 "Transformer Dissection" — 다양한 kernel 비교:
- Polynomial: $(q^\top k + c)^n$ — 표현력 제한
- RBF: $\exp(-\|q-k\|^2/\sigma^2)$ — exponential 변형
- Cosine: $q^\top k / (\|q\|\|k\|)$ — magnitude 무시

### 3. Inductive Bias 분석

Attention 의 kernel 이 inductive bias 결정:
- Exp kernel: smooth, all-pairs interaction
- Local kernel ($\kappa = 0$ if $|i-j| > w$): sparse attention 의 동기 (Ch5-04)
- Multi-head: 여러 kernel 의 동시 적용

### 4. Hopfield Network 와의 연결

Ramsauer 2021 "Hopfield Networks is All You Need" — modern Hopfield 가 정확히 attention 과 등가, kernel-based memory retrieval 로 해석.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Exp kernel 이 fixed | 학습 가능한 kernel 변형 가능 (kernelized attention) |
| Single kernel | Multi-head 가 multiple kernels (Ch1-05) |
| Symmetric kernel 가정 | Attention 은 비대칭 (Q, K 다름) — formal kernel theory 와 약간 차이 |
| Finite sample | RKHS 의 universal approximation 은 무한 sample 가정 |
| Inner-product kernel | Distance-based kernel 도 가능 (RBF) — 등가 변환 |

---

## 📌 핵심 정리

$$\boxed{\text{Attn}(Q,K,V)_i = \frac{\sum_j \kappa_{\exp}(q_i, k_j) v_j}{\sum_j \kappa_{\exp}(q_i, k_j)} \quad \text{— learnable Nadaraya-Watson}}$$

| Kernel | 형태 | Attention 변형 |
|--------|------|---------------|
| **Exp inner product** | $\exp(q^\top k / \sqrt{d_k})$ | Standard (Vaswani 2017) |
| **RBF** | $\exp(-\|q-k\|^2/\sigma^2)$ | 등가 (unit norm 시) |
| **Polynomial** | $(q^\top k)^n / n!$ | Mercer 분해의 항 |
| **Random feature** | $\phi(q)^\top \phi(k)$ | Performer (Ch5-03) |
| **Identity feature** | $\phi(x) = \text{ELU}+1$ | Linear Attention (Ch5-02) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 두 token $q_i, k_j \in \mathbb{R}^4$ 에 대해 $\kappa_{\exp}$ 와 $\kappa_{\text{RBF}}$ ($\sigma = 1$) 의 값을 손으로 계산하고, unit norm 가정 하에 비례 관계를 확인하라. ($q = (1, 0, 0, 0)$, $k = (\cos\theta, \sin\theta, 0, 0)$ 에 대해)

<details>
<summary>해설</summary>

$q^\top k = \cos\theta$, $\|q-k\|^2 = 2 - 2\cos\theta$.

- $\kappa_{\exp}(q, k) = \exp(\cos\theta / \sqrt{4}) = \exp(\cos\theta/2)$
- $\kappa_{\text{RBF}}(q, k) = \exp(-(2 - 2\cos\theta)/2) = \exp(\cos\theta - 1)$

비율:
$$
\frac{\kappa_{\exp}}{\kappa_{\text{RBF}}} = \exp(\cos\theta/2 - \cos\theta + 1) = \exp(1 - \cos\theta/2)
$$

$\theta$ 에만 의존, $|q|, |k|$ 가 같은 unit norm 일 때 두 kernel 은 ranking-equivalent (같은 순서). Softmax 후 attention 분포는 같음 (rescale 으로 흡수). $\square$

</details>

**문제 2** (심화): Linear Attention 의 feature map $\phi(x) = \text{ELU}(x) + 1$ 이 어떤 kernel 을 implicit 하게 정의하는가? Exp kernel 과의 차이를 함수 그래프로 비교 분석하라.

<details>
<summary>해설</summary>

**ELU+1 의 kernel**:
$$
\kappa_{\text{lin}}(q, k) = \phi(q)^\top \phi(k) = \sum_l (\text{ELU}(q_l) + 1)(\text{ELU}(k_l) + 1)
$$

$x \geq 0$: $\phi(x) = x + 1$ (linear + offset)
$x < 0$: $\phi(x) = e^x$ (smooth saturation to 0)

따라서 $\kappa_{\text{lin}}$ 은:
- 양수 input 에 대해 polynomial-like (linear in each dim)
- 음수 input 에 대해 saturated

**Exp kernel 과의 차이**:
- Exp: dot product 의 비선형 함수, sharp peak
- Linear (ELU+1): 각 차원의 선형 결합 + 양수성 보장 — 차원 간 상호작용 약함

**그래프 비교**: $q = k = x \mathbf{1}$ 에 대해
- $\kappa_{\exp} = e^{x^2 d / \sqrt{d}}$ — exponential growth
- $\kappa_{\text{lin}} = d \cdot \phi(x)^2$ — quadratic for $x > 0$

→ Linear attention 은 exp 의 "sharpness" 를 잃음, peak 가 부드러움. 그러나 결합 순서로 $O(T)$ 가능.

**Trade-off**: 표현력 (sharp focus) vs 효율 (linear scaling). 짧은 sequence 에서는 standard 가 우월, 긴 sequence 에서는 linear 가 필수. $\square$

</details>

**문제 3** (논문 비평): Tsai et al. 2019 "Transformer Dissection" 은 attention 을 unified kernel framework 로 분석한다. Vaswani 2017 의 exp inner product 가 다른 kernel (polynomial, cosine) 보다 더 좋은 이유는? 또한 이 분석이 Performer (Ch5-03) 의 random feature 선택을 어떻게 이끄는가?

<details>
<summary>해설</summary>

**Exp inner product 의 우수성**:
1. **무한 표현력** — Mercer 분해로 무한 차원 polynomial 의 weighted sum, universal approximator
2. **자연스러운 sharpness** — exp 가 큰 dot product 에 더 큰 weight, "선명한" attention
3. **Softmax 와의 자연스러운 결합** — log-bilinear form 이 softmax cross-entropy 와 일관
4. **Magnitude-aware** — cosine 과 달리 $\|q\|$ 정보 보존, 학습 자유도

**다른 kernel 의 한계**:
- Polynomial $(q^\top k)^n$: 표현력 차수 제한, 큰 $n$ 시 oscillation
- Cosine $q^\top k / (\|q\| \|k\|)$: magnitude 정보 손실, 학습 어려움
- Identity $q^\top k$: linear, sharpness 없음

**Performer 의 random feature 선택 동기**:

Tsai 의 분석이 보여주는 것: exp kernel 의 우수성을 보존하면서 결합 순서 변경 → random feature.

Performer FAVOR+ 의 핵심 결정:
1. **Positive RF** — $\phi(x) = \exp(-\|x\|^2/2)[\exp(\omega^\top x); \exp(-\omega^\top x)]$ 가 양수성 보장 → variance 감소
2. **Orthogonal RF** — $\omega$ 들이 mutually orthogonal (Givens rotation) → variance 더 감소
3. **Causal masking 호환** — Linear Attention 의 RNN-form 이 causal generation 에 적합

따라서 Performer 는 "exp kernel 을 효율적으로 보존" 의 직접적 instantiation. Linear Attention 보다 표현력 우수, Standard 보다 빠름. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-softmax-saturation.md) | [📚 README](../README.md) | [다음 ▶](./05-multi-head.md)

</div>
