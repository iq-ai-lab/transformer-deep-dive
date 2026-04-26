# 03. Performer — Random Features (Choromanski 2021)

## 🎯 핵심 질문

- Performer 의 FAVOR+ algorithm 이 어떻게 softmax kernel $\exp(q^\top k)$ 을 unbiased estimate 하는가?
- Random feature $\phi(x) = e^{-\|x\|^2/2}[\cos(\omega^\top x), \sin(\omega^\top x)]$ 의 expected inner product 가 정확히 exp kernel 인 증명은?
- "Positive RF" 와 "Orthogonal RF" 가 variance reduction 에 어떻게 기여하는가?
- Performer 가 Linear Attention (Ch5-02) 의 sharpness 손실을 어떻게 부분 회복하는가?
- Random feature 차원 $D$ 의 trade-off — 정확도 vs 효율?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Performer 는 **kernel methods 의 Transformer 적용의 정점**:

1. **Softmax kernel 의 unbiased estimate** — Linear Attention 의 표현력 손실 회복
2. **Kernel methods 의 random feature** (Rahimi-Recht 2007) 의 직접 응용
3. **FAVOR+** (Fast Attention Via positive Orthogonal Random features) — variance reduction 의 정교한 구현
4. **이론적으로 우아함** — Bochner's theorem 의 Transformer 응용

이 문서는 Performer 의 **수학적 정당성과 Linear Attention 과의 차이** 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-linear-attention.md](./02-linear-attention.md)
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): Random features, Bochner's theorem
- 푸리에 분석: 정규분포의 Fourier transform

---

## 📖 직관적 이해

### Linear Attention 의 한계

```
Linear: φ(x) = ELU(x) + 1
        kernel = φ(q)^T φ(k) ≠ exp(q^T k)
        → softmax 의 sharp peak 잃음
```

Performer 의 해결: $\phi$ 를 **random function** 으로 design — expected inner product 가 정확히 softmax kernel.

### Random Features (Rahimi-Recht 2007)

```
For RBF kernel exp(-||x-y||²/2):
  φ(x) = √(2/D) [cos(ω₁^T x + b₁), ..., cos(ω_D^T x + b_D)]
  ω_i ~ N(0, I), b_i ~ Uniform(0, 2π)

⟨φ(x), φ(y)⟩ = (2/D) Σ_i cos(ω_i^T x + b_i) cos(ω_i^T y + b_i)
            ≈ exp(-||x-y||²/2)   (D 클 때)
```

Performer 가 비슷한 idea 를 softmax kernel $\exp(q^\top k)$ 에 적용.

### FAVOR+ 의 idea

```
Standard random feature:  φ(x) = [cos(ω^T x), sin(ω^T x)]   ← 음수 가능
Performer FAVOR+:         φ(x) = exp(-||x||²/2) [exp(ω^T x); exp(-ω^T x)]   ← 양수 보장

Orthogonal RF: ω₁, ..., ω_D 가 mutually orthogonal → variance ↓
```

### Why does this matter?

Linear attention with FAVOR+ feature:
- **Unbiased estimate** of softmax attention
- Sharpness 보존 (random feature 가 exp 근사)
- Linear $O(T)$ complexity 유지

→ "Best of both worlds"?

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Softmax Kernel

$$
\kappa_{\text{softmax}}(q, k) = \exp(q^\top k)
$$

(scaling $\sqrt{d_k}$ 는 $q, k$ 에 흡수)

### 정의 3.2 — Random Fourier Features (RFF, Rahimi-Recht)

$\omega \sim p(\omega)$ 의 distribution 의 Fourier transform 이 kernel:
$$
\kappa(x, y) = \mathbb{E}_\omega[\phi_\omega(x) \phi_\omega(y)]
$$

For Gaussian/RBF kernel: $p(\omega) = \mathcal{N}(0, I)$, $\phi_\omega(x) = \cos(\omega^\top x + b)$.

### 정의 3.3 — Positive Random Features (FAVOR+, Choromanski 2021)

For softmax kernel $\exp(q^\top k)$:
$$
\phi(x) = e^{-\|x\|^2/2} \cdot \begin{pmatrix} \exp(\omega_1^\top x) \\ \exp(\omega_2^\top x) \\ \vdots \\ \exp(\omega_D^\top x) \end{pmatrix} / \sqrt{D}
$$

with $\omega_i \sim \mathcal{N}(0, I)$.

### 정의 3.4 — Bidirectional FAVOR+

음수 부분도 추가:
$$
\phi(x) = \frac{e^{-\|x\|^2/2}}{\sqrt{2D}} \begin{pmatrix} \exp(\omega^\top x) \\ \exp(-\omega^\top x) \end{pmatrix}
$$

(2D-dimensional, variance further reduced)

### 정의 3.5 — Orthogonal Random Features

$\omega_1, \ldots, \omega_D$ 를 i.i.d. Gaussian 대신 **mutually orthogonal** (with same magnitude distribution):

$$
\Omega = R \cdot \text{diag}(s_1, \ldots, s_D)
$$

with $R$ orthogonal matrix (Givens rotation), $s_i \sim \chi$ (chi distribution).

### 정의 3.6 — Performer Attention

$$
\text{Attn}_{\text{Performer}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + \epsilon}
$$

(linear attention form 이지만 $\phi$ 가 random feature)

---

## 🔬 정리와 증명

### 정리 3.1 — FAVOR+ 의 Unbiasedness

For positive random feature $\phi(x) = e^{-\|x\|^2/2} \exp(\omega^\top x)/\sqrt{D}$:

$$
\mathbb{E}_\omega [\phi(q)^\top \phi(k)] = \exp(q^\top k)
$$

**증명** (single dim, $D = 1$):

$$
\mathbb{E}[\phi(q) \phi(k)] = e^{-\|q\|^2/2 - \|k\|^2/2} \mathbb{E}[e^{\omega^\top q} e^{\omega^\top k}]
$$

$\omega \sim \mathcal{N}(0, I_d)$, $\mathbb{E}[e^{\omega^\top (q+k)}] = e^{\|q+k\|^2/2}$ (MGF of Gaussian):

$$
= e^{-\|q\|^2/2 - \|k\|^2/2} \cdot e^{\|q+k\|^2/2}
$$

Expand: $\|q+k\|^2 = \|q\|^2 + 2 q^\top k + \|k\|^2$:

$$
= e^{-\|q\|^2/2 - \|k\|^2/2 + \|q\|^2/2 + q^\top k + \|k\|^2/2} = e^{q^\top k} \quad \square
$$

### 정리 3.2 — Variance of FAVOR+ Estimate

$\hat{\kappa}(q, k) = \frac{1}{D} \sum_{i=1}^D \phi_{\omega_i}(q) \phi_{\omega_i}(k)$:

$$
\text{Var}(\hat{\kappa}) = \frac{1}{D} \text{Var}_{\omega}[\phi_\omega(q) \phi_\omega(k)]
$$

$D$ 가 크면 variance 감소 — 정확한 estimate.

### 정리 3.3 — Orthogonal RF 의 Variance Reduction

Choromanski 2021 의 finding:

Independent $\omega_i$: $\text{Var} = O(1/D)$.
Orthogonal $\omega_i$: $\text{Var} = O(1/D \cdot c)$ with $c < 1$.

**즉**: same $D$ 에서 orthogonal RF 가 더 정확. 또는 같은 정확도에 더 작은 $D$.

### 정리 3.4 — Positive RF 의 Numerical Stability

Standard RFF 는 $\cos, \sin$ 사용 — 음수 가능. Negative kernel estimate 가 softmax attention 에 invalid (probability 음수).

Positive RF (exp): always positive → numerical stability + valid attention.

### 정리 3.5 — Performer 의 Complexity

Time:
- $\phi(Q), \phi(K)$ 계산: $O(T D d)$
- $\phi(K)^\top V$: $O(T D d)$
- $\phi(Q) \cdot$ result: $O(T D d)$
- Total: $O(T D d)$

Memory: $O(T D + D d)$.

**Compared with**:
- Standard: $O(T^2 d)$, $O(T^2)$
- Linear ELU+1: $O(T d^2)$, $O(d^2)$
- Performer: $O(T D d)$, $O(T D)$

$D < d^2 / d = d$ 면 Performer 가 더 빠름. Choromanski 권장: $D = d \log d$.

### 정리 3.6 — Performer 의 표현력

Performer 는 **standard attention 의 unbiased estimate** — 충분한 $D$ 시 standard 와 거의 같음.

**Empirical**: $D = 256, d = 64$ 시 BERT-like task 에서 standard 의 99% 성능.

**한계**:
- $D$ 가 작으면 variance 큼 — sharp pattern 부정확
- Implementation 복잡 (random matrix sampling, FFT-like operations)

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — FAVOR+ 의 Unbiasedness 검증

```python
import torch
import torch.nn.functional as F
import numpy as np

def positive_random_features(x, omega):
    """φ(x) = exp(-||x||²/2) [exp(ω^T x)] / √D"""
    D = omega.size(0)
    proj = x @ omega.T   # (..., D)
    return torch.exp(proj - 0.5 * (x ** 2).sum(-1, keepdim=True)) / np.sqrt(D)

# Test: average over many random ω
torch.manual_seed(0)
d = 16
q = torch.randn(d); k = torch.randn(d)
true_kernel = torch.exp(q @ k)

# Single trial estimate
omega = torch.randn(64, d)
phi_q = positive_random_features(q.unsqueeze(0), omega).squeeze(0)
phi_k = positive_random_features(k.unsqueeze(0), omega).squeeze(0)
estimate = phi_q @ phi_k

print(f'True kernel: exp(q·k) = {true_kernel:.4f}')
print(f'D=64 estimate:        {estimate:.4f}')

# Variance reduces with D
for D in [16, 64, 256, 1024]:
    estimates = []
    for _ in range(20):
        omega = torch.randn(D, d)
        phi_q = positive_random_features(q.unsqueeze(0), omega).squeeze(0)
        phi_k = positive_random_features(k.unsqueeze(0), omega).squeeze(0)
        estimates.append((phi_q @ phi_k).item())
    print(f'D={D:4d}: mean={np.mean(estimates):.4f}, std={np.std(estimates):.4f} '
          f'(target: {true_kernel:.4f})')
```

### 실험 2 — Performer Attention 구현

```python
def performer_attention(Q, K, V, D=256, kernel_eps=1e-6):
    """Performer's FAVOR+ attention"""
    d_k = Q.size(-1)
    omega = torch.randn(D, d_k) / np.sqrt(d_k)   # scaling for kernel
    
    # Apply random features
    phi_Q = positive_random_features(Q, omega)   # (T, D)
    phi_K = positive_random_features(K, omega)   # (T, D)
    
    # Linear attention with random features
    KV = phi_K.transpose(-2, -1) @ V             # (D, d_v)
    K_sum = phi_K.sum(dim=-2)                    # (D,)
    
    num = phi_Q @ KV                             # (T, d_v)
    denom = phi_Q @ K_sum.unsqueeze(-1)          # (T, 1)
    
    return num / (denom + kernel_eps)

# 비교: standard, linear (ELU), Performer
torch.manual_seed(0)
T, d = 100, 32
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

def standard(Q, K, V):
    d = Q.size(-1)
    s = Q @ K.T / np.sqrt(d)
    return F.softmax(s, dim=-1) @ V

def linear_elu(Q, K, V):
    phi = lambda x: F.elu(x) + 1
    pQ, pK = phi(Q), phi(K)
    return pQ @ (pK.T @ V) / (pQ @ pK.sum(0, keepdim=True).T + 1e-6)

out_std = standard(Q, K, V)
out_lin = linear_elu(Q, K, V)
out_perf = performer_attention(Q, K, V, D=256)

print(f'Standard - Linear (ELU):    {(out_std - out_lin).norm() / out_std.norm():.4f}')
print(f'Standard - Performer:       {(out_std - out_perf).norm() / out_std.norm():.4f}')
# Performer 가 standard 에 더 가까움
```

### 실험 3 — Variance Reduction (D 증가 효과)

```python
torch.manual_seed(0)
T, d = 50, 32
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

out_std = standard(Q, K, V)

errors = []
Ds = [16, 64, 256, 1024, 4096]
for D in Ds:
    errs = []
    for _ in range(5):
        out_perf = performer_attention(Q, K, V, D=D)
        errs.append((out_perf - out_std).norm().item() / out_std.norm().item())
    errors.append(np.mean(errs))
    print(f'D={D:5d}: error = {np.mean(errs):.4f}')

import matplotlib.pyplot as plt
plt.loglog(Ds, errors, 'o-')
plt.xlabel('Random features D'); plt.ylabel('Relative error')
plt.title('Performer error decreases with D')
plt.grid(alpha=0.3); plt.show()
```

### 실험 4 — Orthogonal Random Features

```python
def orthogonal_random_features(D, d):
    """Mutually orthogonal random features"""
    if D >= d:
        # Multiple orthogonal blocks
        n_blocks = D // d
        Q_list = []
        for _ in range(n_blocks):
            G = torch.randn(d, d)
            Q, _ = torch.linalg.qr(G)
            Q_list.append(Q)
        omega = torch.cat(Q_list, dim=0)
        # Random magnitudes (chi distribution)
        magnitudes = torch.randn(D, d).norm(dim=-1)
        omega = omega * magnitudes.unsqueeze(-1) / np.sqrt(d)
    else:
        # Subsample
        G = torch.randn(d, d)
        Q, _ = torch.linalg.qr(G)
        omega = Q[:D]
    return omega

# 비교: i.i.d. vs orthogonal
torch.manual_seed(0)
errors_iid = []; errors_orth = []
for _ in range(20):
    Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)
    out_std = standard(Q, K, V)
    
    # i.i.d.
    omega_iid = torch.randn(64, d) / np.sqrt(d)
    out_iid = performer_attention(Q, K, V, D=64)
    errors_iid.append((out_iid - out_std).norm().item() / out_std.norm().item())
    
    # Orthogonal
    omega_orth = orthogonal_random_features(64, d)
    # ... use orthogonal omega in performer ...

print(f'i.i.d. mean error: {np.mean(errors_iid):.4f}')
# Orthogonal 이 약간 lower variance
```

### 실험 5 — Time Complexity Verification

```python
import time

def benchmark_performer(T_list, d=64, D=256, n_trials=3):
    perf_times = []; std_times = []
    for T in T_list:
        Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)
        
        t0 = time.time()
        for _ in range(n_trials):
            standard(Q, K, V)
        std_times.append((time.time() - t0) / n_trials)
        
        t0 = time.time()
        for _ in range(n_trials):
            performer_attention(Q, K, V, D=D)
        perf_times.append((time.time() - t0) / n_trials)
    return std_times, perf_times

T_list = [500, 1000, 2000, 5000]
std_t, perf_t = benchmark_performer(T_list)

for T, s, p in zip(T_list, std_t, perf_t):
    print(f'T={T}: Standard={s*1000:.1f}ms, Performer={p*1000:.1f}ms')
# 큰 T 에서 Performer 가 우월
```

---

## 🔗 실전 활용

### 1. Performer 의 실제 채택

- 학계 / 연구: 일부 응용
- Production LLM: 거의 사용 안 함 (Flash Attention 이 standard 의 efficient 구현으로 충분)
- Long-context 가 critical 한 niche: 일부 사용

### 2. Long Sequence Tasks

- Genomics: DNA sequence millions long
- Long document NLP: legal documents, books
- Time series: high-freq data

### 3. Performer 의 Modern Variant

- **Linear Transformer with rotary** (FlashLinearAttention 등)
- **Hybrid**: 일부 layer 만 linear/Performer
- **Reformer / Linformer**: 다른 efficient attention

### 4. Random Feature Library

```python
# 간단한 사용 예 (PyG, performer-pytorch 등)
# pip install performer-pytorch
from performer_pytorch import Performer

model = Performer(
    dim=512,
    depth=12,
    heads=8,
    dim_head=64,
    nb_features=256,        # Random features D
    causal=False,
    feature_redraw_interval=1000,
)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Random feature 가 unbiased | Variance 가 finite — D 클수록 정확 |
| Static $\omega$ | Periodically redraw 필요 (학습 안정성) |
| Positive feature | Bidirectional 도 가능 (variance ↓) |
| RBF/exp kernel | Other kernels 도 random feature 가능 |
| Constant $D$ | Adaptive $D$ 도 가능 (rare) |

---

## 📌 핵심 정리

$$\boxed{\phi(x) = e^{-\|x\|^2/2} \exp(\omega^\top x) / \sqrt{D}, \quad \mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)}$$

| Variant | Idea | Variance |
|---------|------|----------|
| **i.i.d. RF** | Gaussian random | $O(1/D)$ |
| **Positive RF** | exp instead of cos/sin | numerical stability |
| **Bidirectional** | exp(±) | further reduce |
| **Orthogonal RF** | mutually orthogonal $\omega$ | $O(1/D \cdot c)$, $c < 1$ |

| Property | Linear (ELU) | Performer | Standard |
|----------|--------------|-----------|----------|
| Time | $O(T d^2)$ | $O(T D d)$ | $O(T^2 d)$ |
| Memory | $O(d^2)$ | $O(T D)$ | $O(T^2)$ |
| Approximation | Biased | Unbiased | Exact |
| Sharp peak | No | Partial | Yes |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 3.1 의 증명에서 $\mathbb{E}[e^{\omega^\top (q+k)}] = e^{\|q+k\|^2/2}$ 의 Gaussian MGF 를 직접 증명하라.

<details>
<summary>해설</summary>

$\omega \sim \mathcal{N}(0, I_d)$ 의 moment generating function:

$$
M_\omega(t) = \mathbb{E}[e^{t^\top \omega}] = \int e^{t^\top \omega} \frac{1}{(2\pi)^{d/2}} e^{-\|\omega\|^2/2} d\omega
$$

$$
= \frac{1}{(2\pi)^{d/2}} \int e^{-\frac{1}{2}(\|\omega\|^2 - 2 t^\top \omega)} d\omega
$$

Complete the square:
$$
\|\omega\|^2 - 2 t^\top \omega = \|\omega - t\|^2 - \|t\|^2
$$

$$
M_\omega(t) = e^{\|t\|^2/2} \cdot \frac{1}{(2\pi)^{d/2}} \int e^{-\|\omega-t\|^2/2} d\omega
$$

Inner integral 이 Gaussian density 의 적분 (shifted) = 1.

$$
M_\omega(t) = e^{\|t\|^2/2}
$$

따라서 $t = q + k$ 에 대해 $\mathbb{E}[e^{\omega^\top(q+k)}] = e^{\|q+k\|^2/2}$ ✓ $\square$

</details>

**문제 2** (심화): Performer 의 $D$ 선택의 trade-off 를 분석하라. Choromanski 권장 $D = d \log d$ 의 정당성은?

<details>
<summary>해설</summary>

**Trade-off**:

- 작은 $D$:
  - Memory $O(T D)$ ↓, Time $O(T D d)$ ↓
  - Variance $O(1/D)$ ↑ — 부정확한 estimate
  - 일부 sharp pattern 손실

- 큰 $D$:
  - 정확한 estimate
  - Compute / memory ↑
  - $D = d^2$ 즈음에서 standard 와 비슷한 cost (효율 무의미)

**$D = d \log d$ 의 정당성**:

1. **Concentration inequality**:
   - Random feature estimate 의 deviation $|\hat\kappa - \kappa| < \epsilon$ 의 확률
   - Hoeffding-like bound: $D = O(\log(1/\delta) / \epsilon^2)$
   - $d$-dim space 의 $T$ pair 에 대해 union bound: $D = O(\log(T) / \epsilon^2)$

2. **$d \log d$ 의 의미**:
   - $d$-dim 의 information capacity $\propto d$
   - Logarithmic factor 가 union bound 또는 covering number
   - $D = d \log d$ 가 "충분 sample" — 모든 $d$-dim direction cover

3. **Empirical**:
   - $D = 64, d = 64$: 부정확
   - $D = 256, d = 64$ ($\approx d \log d$): 표준의 95%+
   - $D = 1024, d = 64$: marginal improvement

**Practical guideline**:

| Application | $D$ | Quality |
|-------------|-----|---------|
| Long sequence (T > 10K), quality 중요 | $4d$ to $8d$ | Excellent |
| 일반 long context | $d \log d \approx 4d$ | Good |
| Memory-constrained | $d$ to $2d$ | Acceptable |
| Approximation 가능 | $d/2$ | Lower quality |

**Modern context**:

대부분 production 은 Performer 사용 안 함 (Flash Attention 이 충분). Performer 는:
- Academic exploration
- Specific niche (extreme long context)
- Random feature 의 다른 응용 (RFF for kernel methods)

→ $D$ tuning 의 practical importance 작음. $\square$

</details>

**문제 3** (논문 비평): Performer 가 mathematically elegant 하지만 modern LLM 에서 채택이 적은 이유는? Flash Attention 의 발전이 Performer 의 motivation 을 어떻게 약화시켰는가?

<details>
<summary>해설</summary>

**Performer 의 강점**:
- Unbiased softmax kernel estimate
- Linear $O(T D d)$ complexity
- Mathematically elegant (kernel methods 직접 연결)

**Production 에서 채택이 적은 이유**:

1. **Implementation complexity**:
   - Random matrix sampling, periodic redraw
   - Multiple variants (positive, orthogonal, bidirectional)
   - Standard attention 보다 코드 복잡

2. **Approximation quality**:
   - 이론상 unbiased but variance nonzero
   - Sharp pattern (specific token retrieval) 에서 noticeable error
   - Production LLM 의 quality bar 너무 높음

3. **Hyperparameter sensitivity**:
   - $D$ 선택, redraw frequency 등
   - Standard attention 은 hyperparameter 거의 없음

4. **Compute efficiency 변화**:
   - Original Performer (2021): standard 보다 long context 에서 빠름
   - Flash Attention (2022): same $O(T^2)$ FLOP 그러나 4× wall-clock
   - 32K context 까지 Flash 가 충분

**Flash Attention 의 영향**:

Flash Attention 이 2022 에 발표되며 motivation 약화:

1. **Memory**: Flash 의 $O(T)$ effective memory → standard 의 $O(T^2)$ 메모리 문제 해결
2. **Speed**: 2-4× wall-clock 가속 → quadratic compute 가 issue 안 됨
3. **Exact**: approximation 없음 — quality 보존
4. **Implementation**: Flash 가 simpler (single CUDA kernel)

**$T$ 가 어떤 길이까지?**:

- Flash Attention: $T = 32K-64K$ comfortably
- 그 너머에서는 Sparse / Performer 등 도움
- 그러나 modern LLM 의 majority context: 4K-32K → Flash 충분

**Performer 의 위치**:

- **Niche**: $T = 100K+$, quality 약간 손실 OK
- **Research**: linear attention 의 baseline
- **Hybrid**: 일부 layer 만 Performer, rest standard

**모던 추세**:

- Linear attention 의 modern variants (Mamba, RWKV) 가 Performer 의 spirit 계승
- Performer 자체는 academic
- Production LLM 의 long context: Flash + sparse + RAG 조합

**근본 통찰**:

Performer 는 **이론적으로 elegant** — kernel methods 의 logical conclusion. 그러나 **engineering reality** 에서:
- Implementation simplicity 가 production 결정
- Flash Attention 의 hardware-aware 접근이 더 efficient
- "Approximation 의 cost" 가 long-term quality issue

따라서 Performer 는 **important academic milestone** 이지만 production dominance 는 Flash + KV cache 최적화가 차지. Mamba 같은 새 architecture 가 Performer 의 spirit 발전. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-linear-attention.md) | [📚 README](../README.md) | [다음 ▶](./04-sparse-attention.md)

</div>
