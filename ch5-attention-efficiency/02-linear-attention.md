# 02. Linear Attention (Katharopoulos 2020)

## 🎯 핵심 질문

- Katharopoulos 2020 의 kernel trick: $\text{softmax}(QK^\top)V \approx \phi(Q)(\phi(K)^\top V)$ — 결합 순서 변경의 정당성?
- $O(T^2 d) \to O(T d^2)$ 의 정확한 derivation?
- Feature map $\phi$ 의 선택 — ELU+1 이 왜 합리적이고 어떤 한계가 있는가?
- Linear attention 이 왜 RNN-form 으로 incremental computation 가능한가?
- "Transformers are RNNs" 의 의미와 Performer (Ch5-03) 와의 관계?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Linear Attention 은 **첫 번째 fundamental complexity reduction**:

1. **$O(T^2) \to O(T d^2)$** — sequence length 에서 quadratic → linear
2. **결합 순서 변경의 trick** — kernel methods 와 직접 연결
3. **RNN-form** — autoregressive generation 시 $O(d^2)$/step
4. **Performer 의 기반** — Ch5-03 의 random feature 가 더 정교한 변형

이 문서는 linear attention 의 **수학적 정당성과 표현력 trade-off** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [04-attention-as-kernel.md](../ch1-attention-decomposition/04-attention-as-kernel.md) — Kernel 해석
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive)
- 이전 문서: [01-quadratic-bottleneck.md](./01-quadratic-bottleneck.md)

---

## 📖 직관적 이해

### Standard vs Linear

```
Standard Attention:
  scores = Q K^T  ∈ ℝ^{T×T}     ← O(T²) memory
  attn = softmax(scores) V       ← O(T²) compute

Linear Attention:
  φ(Q) (φ(K)^T V)
       └────────┘
        ∈ ℝ^{d×d}                  ← O(d²) memory! (T 무관)
  
  T 가 커도 KV product 가 fixed size
```

### 결합 순서 변경의 핵심

```
A · (B · C)   vs   (A · B) · C

Matrix size:
  Q ∈ ℝ^{T×d}, K ∈ ℝ^{T×d}, V ∈ ℝ^{T×d}
  
Standard: (Q · K^T) · V
  Q · K^T  ∈ ℝ^{T×T}    → O(T²d)
  · V     ∈ ℝ^{T×d}    → O(T²d)
  
Linear:   Q · (K^T · V)
  K^T · V ∈ ℝ^{d×d}    → O(Td²)
  Q · ()  ∈ ℝ^{T×d}    → O(Td²)
```

→ **$T$ 와 $d$ 의 위치 swap** — $T$ 가 크면 linear 가 우월.

### Softmax 의 Replacement

Softmax 는 결합 가능 안 함 ($\text{softmax}(QK^\top)$ 는 row-normalized — 분리 불가).

**Trick**: $\text{softmax}(QK^\top)$ 를 $\phi(Q) \phi(K)^\top$ 로 근사 (positive feature map). 이때:

$$
\text{softmax}(QK^\top) V \approx \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
$$

(분모가 row normalization)

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Generalized Attention

Kernel function $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$ 에 대해:
$$
\text{Attn}(Q, K, V)_i = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)}
$$

(softmax + dot product 의 일반화 — Ch1-04)

### 정의 2.2 — Linear Attention via Feature Map

Feature map $\phi: \mathbb{R}^d \to \mathbb{R}^{d'}_{+}$ (양수 valued):
$$
\kappa(q, k) = \phi(q)^\top \phi(k)
$$

Substituting:
$$
\text{Attn}_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

### 정의 2.3 — Linear Attention 의 Matrix Form

Numerator: $\phi(Q) (\phi(K)^\top V)$ where $\phi(K)^\top V \in \mathbb{R}^{d' \times d}$.
Denominator: $\phi(Q) (\phi(K)^\top \mathbf{1}) = \phi(Q) \cdot s$ where $s = \sum_j \phi(k_j) \in \mathbb{R}^{d'}$.

$$
\text{LinAttn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) s + \epsilon}
$$

(epsilon for numerical stability)

### 정의 2.4 — ELU + 1 Feature Map (Katharopoulos)

$$
\phi(x) = \text{ELU}(x) + 1
$$

(elementwise positive)

- $x \geq 0$: $\phi(x) = x + 1$
- $x < 0$: $\phi(x) = e^x$ — saturates to $0^+$

**Properties**: positive, smooth, $\phi(0) = 1$.

### 정의 2.5 — Causal Linear Attention (RNN form)

Causal mask 적용 시:
$$
S_i = \sum_{j \leq i} \phi(k_j) v_j^\top \in \mathbb{R}^{d \times d}
$$

$$
z_i = \sum_{j \leq i} \phi(k_j) \in \mathbb{R}^d
$$

$$
\text{Attn}_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i + \epsilon}
$$

**Recurrent**:
$$
S_i = S_{i-1} + \phi(k_i) v_i^\top, \quad z_i = z_{i-1} + \phi(k_i)
$$

→ **Per-step $O(d^2)$**, total $O(T d^2)$.

---

## 🔬 정리와 증명

### 정리 2.1 — Linear Attention 의 Complexity

Time:
- $\phi(K)^\top V$: $O(T d^2)$
- $\phi(Q) \cdot$ result: $O(T d^2)$
- Total: $O(T d^2)$

Memory:
- $\phi(K)^\top V \in \mathbb{R}^{d \times d}$: $O(d^2)$
- $Q, K, V$: $O(T d)$
- Total: $O(T d + d^2)$

**비교** with standard $O(T^2 d)$ time, $O(T^2)$ memory:
- $T \gg d$ 시 linear 가 우월
- $T = d$ 즈음에서 break-even

### 정리 2.2 — Linear Attention 의 RNN 등가성

Causal linear attention 의 incremental computation:
- $S_i = S_{i-1} + \phi(k_i) v_i^\top$ — recurrent state update
- 각 step: $O(d^2)$ for matrix-vector outer product

→ **Linear attention = RNN with state $S \in \mathbb{R}^{d \times d}$**.

**의미**: Generation 시 매 token 당 $O(d^2)$ — KV cache (standard) 의 $O(T d)$ 보다 $T$ 무관.

### 정리 2.3 — Approximation Error

Linear attention 은 softmax attention 의 approximation:
$$
\text{softmax}(q^\top k / \sqrt{d}) \neq \phi(q)^\top \phi(k) \quad \text{generally}
$$

**Error sources**:
1. **Softmax 의 nonlinear sharpness 손실**: Linear kernel 이 dot product 의 polynomial expansion 인 반면 softmax 가 exponential
2. **Normalization 의 차이**: softmax 가 exact row-stochastic, linear approximation 은 numerator/denominator separately 계산

### 정리 2.4 — Performer 의 정확한 Approximation

Random feature map $\phi(x) = e^{\|x\|^2/2} [\cos(\omega^\top x), \sin(\omega^\top x)]$ (Rahimi-Recht):
$$
\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)
$$

→ **Unbiased estimate of softmax kernel**. Performer (Ch5-03) 가 이 idea 활용.

### 정리 2.5 — Sharpness 손실의 영향

Standard softmax attention 의 selectivity:
- Sharp peak: 한 token 에 대부분 weight
- Linear: peak 부드러움 — 모든 token 비슷한 weight

**Empirical**: Linear attention 은 some task 에서 standard 보다 worse:
- Long-range fine retrieval
- Specific token matching
- Code, math reasoning

**OK tasks**: smooth, distributional patterns
- Translation
- Image-like sequence

### 정리 2.6 — Causal Form 의 Implementation

Standard causal attention: full $T \times T$ matrix masking — $O(T^2)$.

Linear causal: incremental — $O(T d^2)$, but per-token $O(d^2)$.

**Generation 시 advantage**: 
- Standard: $O(T)$ for each new token (KV cache)
- Linear: $O(d^2)$ for each new token (state update) — $T$ 무관

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Linear Attention 구현

```python
import torch
import torch.nn.functional as F
import numpy as np
import time

def standard_attention(Q, K, V):
    """O(T² d)"""
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    return attn @ V

def linear_attention(Q, K, V, phi=lambda x: F.elu(x) + 1):
    """O(T d²)"""
    phi_Q = phi(Q)   # (T, d)
    phi_K = phi(K)   # (T, d)
    
    # KV product (d, d)
    KV = phi_K.transpose(-2, -1) @ V                 # (d, d)
    K_sum = phi_K.sum(dim=-2)                         # (d,)
    
    # Numerator and denominator
    num = phi_Q @ KV                                  # (T, d)
    denom = (phi_Q @ K_sum.unsqueeze(-1)).squeeze(-1)  # (T,)
    
    return num / (denom.unsqueeze(-1) + 1e-6)

torch.manual_seed(0)
T, d = 100, 32
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

out_std = standard_attention(Q, K, V)
out_lin = linear_attention(Q, K, V)

print(f'Standard:  {out_std[0, :5]}')
print(f'Linear:    {out_lin[0, :5]}')
print(f'Difference: {(out_std - out_lin).abs().mean():.4f}')
# 다름 — approximation
```

### 실험 2 — Speed Comparison

```python
def benchmark(T_list, d=64, n_trials=5):
    std_times = []; lin_times = []
    for T in T_list:
        Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)
        
        # Standard
        t0 = time.time()
        for _ in range(n_trials):
            standard_attention(Q, K, V)
        std_times.append((time.time() - t0) / n_trials)
        
        # Linear
        t0 = time.time()
        for _ in range(n_trials):
            linear_attention(Q, K, V)
        lin_times.append((time.time() - t0) / n_trials)
    return std_times, lin_times

T_list = [100, 500, 1000, 2000, 5000]
std_t, lin_t = benchmark(T_list)

import matplotlib.pyplot as plt
plt.plot(T_list, [t*1000 for t in std_t], 'o-', label='Standard O(T²)')
plt.plot(T_list, [t*1000 for t in lin_t], 's-', label='Linear O(T)')
plt.xlabel('T'); plt.ylabel('time (ms)'); plt.legend()
plt.title('Linear vs Standard attention'); plt.show()

for T, std, lin in zip(T_list, std_t, lin_t):
    print(f'T={T:5d}: Standard={std*1000:6.2f}ms, Linear={lin*1000:6.2f}ms, '
          f'Linear/Standard={lin/std:.2f}')
```

### 실험 3 — Causal Linear Attention (RNN form)

```python
def causal_linear_attention_recurrent(Q, K, V, phi=lambda x: F.elu(x) + 1):
    """RNN-form: per-step O(d²)"""
    T, d = Q.size()
    S = torch.zeros(d, d)              # state matrix
    z = torch.zeros(d)                 # state vector
    out = []
    
    phi_Q, phi_K = phi(Q), phi(K)
    for t in range(T):
        # Update state
        S = S + phi_K[t].unsqueeze(-1) @ V[t].unsqueeze(0)   # outer product
        z = z + phi_K[t]
        
        # Output for token t
        num_t = phi_Q[t] @ S
        denom_t = phi_Q[t] @ z + 1e-6
        out.append(num_t / denom_t)
    
    return torch.stack(out)

# 검증: causal version 과 일관성
torch.manual_seed(0)
T, d = 50, 16
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

out_recurrent = causal_linear_attention_recurrent(Q, K, V)
print(f'Recurrent linear attention: {out_recurrent.shape}')
print(f'First 3 values: {out_recurrent[0, :3]}')

# 비교: standard causal attention with linear features
def causal_linear_full(Q, K, V, phi=lambda x: F.elu(x) + 1):
    """Full computation 그러나 causal mask 적용"""
    T, d = Q.size()
    phi_Q, phi_K = phi(Q), phi(K)
    scores = phi_Q @ phi_K.T   # (T, T)
    causal = torch.tril(torch.ones(T, T))
    scores = scores * causal
    
    z = scores.sum(dim=-1, keepdim=True) + 1e-6
    return scores / z @ V

out_full = causal_linear_full(Q, K, V)
print(f'Full version match recurrent: {(out_recurrent - out_full).abs().max():.6f}')
```

### 실험 4 — 표현력 비교 (Sharp 패턴)

```python
# Standard 가 잘하는 task: sharp attention pattern (one-hot 비슷)
torch.manual_seed(0)
T = 30
# Pattern: token 5 가 token 25 와 강하게 align
Q_sharp = torch.randn(T, 16)
K_sharp = torch.randn(T, 16)
V_sharp = torch.randn(T, 16)

# Token 5 의 query 와 token 25 의 key 가 매우 align (강조)
K_sharp[25] = Q_sharp[5] * 5   # 매우 큰 dot product

out_std = standard_attention(Q_sharp, K_sharp, V_sharp)
out_lin = linear_attention(Q_sharp, K_sharp, V_sharp)

# Standard 의 token 5 에서 V[25] 의 contribution 큼?
print(f'Standard token 5 output: {out_std[5][:3]}')
print(f'V[25]: {V_sharp[25][:3]}')
print(f'Linear token 5 output:   {out_lin[5][:3]}')

# Standard 가 V[25] 와 더 비슷, linear 는 부드러운 평균
sim_std = F.cosine_similarity(out_std[5].unsqueeze(0), V_sharp[25].unsqueeze(0))
sim_lin = F.cosine_similarity(out_lin[5].unsqueeze(0), V_sharp[25].unsqueeze(0))
print(f'Standard sim(out, target): {sim_std.item():.4f}')
print(f'Linear   sim(out, target): {sim_lin.item():.4f}')
# Standard 가 더 sharp (V[25] 와 가까움), linear 는 평균
```

### 실험 5 — Approximation Error 측정

```python
# 다양한 T 에서 linear vs standard 의 차이
errors = []
for T in [50, 100, 500, 1000]:
    diffs = []
    for _ in range(5):
        Q = torch.randn(T, 32); K = torch.randn(T, 32); V = torch.randn(T, 32)
        out_std = standard_attention(Q, K, V)
        out_lin = linear_attention(Q, K, V)
        diff = (out_std - out_lin).norm() / out_std.norm()
        diffs.append(diff.item())
    errors.append(np.mean(diffs))
    print(f'T={T}: relative error = {np.mean(diffs):.4f}')

# 큰 T 에서도 일정 — 본질적 approximation gap
```

---

## 🔗 실전 활용

### 1. RWKV (Peng 2023)

**Reinventing RNNs for Transformer Era**:
- Linear attention 의 modern 구현
- RNN-form 으로 incremental generation
- 14B-parameter model 학습 성공
- Transformer 와 competitive on some benchmarks

### 2. RetNet (Sun 2023)

**Retentive Network**:
- Linear attention + exponential decay (multi-scale)
- Parallel training + recurrent inference
- 7B model 으로 Transformer 와 비교

### 3. Mamba (Gu & Dao 2023)

State Space Model — 정확히 linear attention 의 변형:
- Selective state update (data-dependent)
- $O(T)$ both training and inference
- Linear attention 의 전혀 다른 접근

### 4. Limitations 의 인정

대부분 frontier LLM 은 여전히 standard attention + Flash. Linear attention 은:
- **Niche application**: long context with relaxed quality
- **Research**: alternative architecture exploration
- **Hybrid**: 일부 layer 만 linear, 나머지 standard

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Positive feature map | Negative 시 정의 안 됨 — ELU+1, exp 등 |
| Softmax 근사 | Sharpness 손실 — Performer 가 일부 회복 |
| $d^2$ memory state | Multi-head 시 $h \times d^2$ |
| Linear in $T$ | $d^2$ 가 큰 $d$ 에서 dominant 가능 |
| Causal generation | Bidirectional 도 가능 |

---

## 📌 핵심 정리

$$\boxed{\text{LinAttn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + \epsilon}}$$

| Property | Standard | Linear |
|----------|----------|--------|
| Time | $O(T^2 d)$ | $O(T d^2)$ |
| Memory | $O(T^2)$ | $O(T d + d^2)$ |
| Generation step | $O(T d)$ (KV cache) | $O(d^2)$ |
| Exact? | Yes | No (approximation) |
| Sharp peak | Yes | No (smoothed) |
| RNN form? | No | Yes |

| Feature map | Form | Property |
|-------------|------|----------|
| ELU+1 | $\text{ELU}(x) + 1$ | Simple, positive |
| Performer FAVOR+ | Random features | Unbiased softmax estimate |
| RWKV | Time-decay weighted | Modern variant |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 1000$, $d = 64$ 인 attention 의 standard vs linear 의 FLOP 와 memory 를 정확히 계산하라.

<details>
<summary>해설</summary>

**Standard**:
- $QK^\top$: $T \times T \times d = 1000 \times 1000 \times 64 = 64M$ FLOP
- Softmax: $T \times T = 1M$ ops (negligible)
- $A V$: $T \times T \times d = 64M$ FLOP
- **Total: ~128M FLOP**

Memory: $T \times T \times 4 = 4MB$ (FP32, attn matrix)

**Linear**:
- $\phi(K)^\top V$: $d \times T \times d = 64 \times 1000 \times 64 = 4M$ FLOP
- $\phi(Q) \cdot ()$: $T \times d \times d = 1000 \times 64 \times 64 = 4M$ FLOP
- **Total: ~8M FLOP**

Memory: $d \times d \times 4 = 16KB$ (KV product)

**Ratio**: Standard 가 Linear 보다 **$16\times$ 더 많은 FLOP**, **$256\times$ 더 큰 메모리**.

→ $T = 1000, d = 64$ 에서 linear 가 huge gain. $\square$

</details>

**문제 2** (심화): Linear attention 의 RNN form 에서 state $S_t \in \mathbb{R}^{d \times d}$ 가 무엇을 나타내는가? Standard attention 의 KV cache 와의 information 차이는?

<details>
<summary>해설</summary>

**Linear attention 의 state $S_t$**:

$$
S_t = \sum_{j \leq t} \phi(k_j) v_j^\top
$$

이는 **모든 이전 (key, value) pair 의 outer product 합** — fixed size $d \times d$ regardless of $t$.

**Standard attention 의 KV cache**:

$$
\text{Cache}_t = \{(k_1, v_1), (k_2, v_2), \ldots, (k_t, v_t)\}
$$

이는 **개별 (key, value) 의 list** — size $O(t \times d)$ — grows with $t$.

**Information 차이**:

- **KV cache**: 모든 정보 보존 — 임의 query 가 임의 key 에 attend 가능
- **Linear state**: information compression — 모든 (k, v) 를 단일 $d \times d$ matrix 로 합침

**의미**:
- Linear attention 은 **lossy compression** of KV cache
- 표현력 limit: rank $\min(T, d)$ — 즉 $T > d$ 시 일부 information 손실
- Standard 는 모든 token 정보 직접 access — full expressivity

**Trade-off**:

| | Linear State | KV Cache |
|---|--------------|----------|
| Memory | $d^2$ (fixed) | $T d$ (growing) |
| Information | Compressed | Full |
| Sharp retrieval | Poor | Excellent |
| Generation step | $O(d^2)$ | $O(T d)$ |

**RNN 과의 유사성**:

$S_t$ 가 fixed-size hidden state — RNN 의 hidden state $h_t$ 와 같은 역할. Linear attention 이 RNN 의 modern reformulation.

**Mamba / RWKV 의 idea**:

- Linear attention 의 state 를 더 expressive 하게:
  - Time-varying decay
  - Selective updates (input-dependent)
  - Multi-head state

→ Linear attention 의 limitation 을 mitigate, RNN-Transformer hybrid. $\square$

</details>

**문제 3** (논문 비평): "Transformers are RNNs" — Katharopoulos 2020 의 claim 이 왜 important 한가? Mamba/RWKV 같은 modern non-Transformer architecture 가 이 idea 를 어떻게 발전시켰는가?

<details>
<summary>해설</summary>

**"Transformers are RNNs" 의 의미**:

Katharopoulos 2020 의 주요 contribution: Linear attention 의 causal version 이 **정확히 RNN 형태** 임을 증명. 이는 다음을 함의:

1. **Conceptual unification**: Transformer 와 RNN 이 별개 architecture 가 아닌 — kernel choice 가 다른 같은 framework
2. **Generation 효율**: $O(d^2)$ per step, $T$ 무관 — autoregressive generation 의 ideal property
3. **Bidirectional / Unidirectional 의 자연스러운 처리**: same model, different temporal direction

**Importance**:

- **Mental model 변화**: "Transformer 가 attention 으로 token 간 정보 직접 교환" → "fixed-size hidden state 의 sequential update + nonlinear projection"
- **Architecture exploration 의 새 방향**: Linear attention 의 다양한 변형 가능
- **Hardware 의미**: RNN-style sequential 보다 parallel training 가능 (linear matrix form)

**Mamba 의 발전**:

Gu & Dao 2023 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces":

- **State Space Model** (SSM): linear attention 의 generalization
- **Selective**: state update 가 input-dependent ($A, B$ matrices 가 input 에 의존)
- **Hardware-aware**: scan algorithm 으로 GPU 효율적

**Mamba 의 advantage over Linear Attention**:

1. **Selective forgetting**: state update 가 input 에 따라 다른 retention
2. **Wider expressivity**: state dim 이 더 클 수 있음
3. **Empirical**: small/medium models 에서 Transformer competitive

**RWKV 의 발전**:

Peng 2023 "Reinventing RNNs for the Transformer Era":

- Linear attention + time-decay
- Multi-channel state (vs Mamba 의 selective)
- 14B parameters with Transformer-like performance

**모든 modern non-Transformer 의 공통점**:

- Fixed-size state (not growing KV cache)
- Linear-time training (parallel scan)
- $O(d^2)$ inference per step
- Linear attention 의 idea 의 evolution

**미래 (2026+)**:

- Hybrid: Transformer + Mamba layers
- Specialized for long context (1M+)
- Edge deployment (low memory)

**근본 통찰**:

"Transformers are RNNs" 는 단순한 mathematical observation 이 아니라 **architecture design 의 paradigm shift** 의 출발점. Linear attention → Mamba/RWKV → 미래의 새 architecture.

Standard attention 의 표현력 vs RNN-style efficiency 의 trade-off 가 architectural 핵심 axis. **Transformer 는 한 점**, modern alternatives 가 다른 점들. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-quadratic-bottleneck.md) | [📚 README](../README.md) | [다음 ▶](./03-performer.md)

</div>
