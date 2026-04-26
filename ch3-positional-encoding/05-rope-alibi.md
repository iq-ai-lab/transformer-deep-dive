# 05. RoPE 와 ALiBi (현대 대세)

## 🎯 핵심 질문

- RoPE (Su 2021) 의 회전 행렬 $R(t)$ 이 어떻게 자동으로 relative position 을 인코딩하는가?
- $\langle R(i) q, R(j) k \rangle = q^\top R(j-i) k$ 의 증명과 의미는?
- ALiBi (Press 2021) 의 단순한 linear bias $-m_h |i-j|$ 가 왜 가장 강한 extrapolation 을 보이는가?
- RoPE 의 frequency 와 sinusoidal PE 의 frequency 의 관계는?
- LLaMA 가 RoPE, BLOOM 이 ALiBi 를 선택한 이유는?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

RoPE 와 ALiBi 는 **modern LLM 의 사실상 표준 PE** 입니다:

1. **RoPE**: GPT-NeoX, LLaMA-1/2/3, Mistral, Qwen — modern decoder-only 표준
2. **ALiBi**: BLOOM, MPT — extrapolation 강조 모델
3. **수학적 elegance**: 회전 또는 단순 linear bias — implementation 매우 simple
4. **Long context enabler**: 32K-128K context 의 직접 토대

이 문서는 두 PE 의 **수학적 정의, 증명, 비교** 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-sinusoidal-pe.md](./02-sinusoidal-pe.md) (회전 행렬 토대), [04-relative-pe.md](./04-relative-pe.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 회전 행렬, complex numbers
- 복소수: $e^{i\theta} = \cos\theta + i\sin\theta$

---

## 📖 직관적 이해

### RoPE 의 핵심 idea

```
Standard Q, K:        q, k ∈ ℝ^d_k
RoPE Q, K:           R(pos) · q,  R(pos) · k

Inner product:
  <R(i) q, R(j) k> = q^T R(i)^T R(j) k = q^T R(j-i) k
                                          ↑
                                     자동 relative!
```

**핵심**: Q, K 에 position-dependent 회전 적용 → inner product 가 distance $j-i$ 만 의존.

### ALiBi 의 핵심 idea

```
Attention score:  q^T k - m_h |i - j|
                            ↑
                  단순 linear penalty (head-specific slope)

→ 거리가 멀수록 attention score 작아짐 (linearly)
→ 매우 단순, 매우 효과적
```

### 두 방법의 차이

```
RoPE:    Position 을 Q, K 에 회전으로 주입 (multiplicative)
         → 표현력 풍부, frequency 다양

ALiBi:   Attention score 에 distance penalty (additive)
         → 매우 단순, extrapolation 최강
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — RoPE Rotation

차원 $d_k$ (짝수), pair $(2i, 2i+1)$ 에 대해 frequency:
$$
\theta_i = 10000^{-2i/d_k}
$$

(sinusoidal PE 와 같은 frequency)

Position $t$ 의 회전:
$$
R(t)_i = \begin{pmatrix} \cos(t \theta_i) & -\sin(t \theta_i) \\ \sin(t \theta_i) & \cos(t \theta_i) \end{pmatrix}
$$

전체 $R(t) \in \mathbb{R}^{d_k \times d_k}$ 는 $d_k/2$ 개 2×2 회전의 block-diagonal.

### 정의 5.2 — RoPE Application

Position $i$ 의 query 에 회전:
$$
\tilde{q}_i = R(i) q_i
$$

Same for key:
$$
\tilde{k}_j = R(j) k_j
$$

Attention score:
$$
S_{ij} = \frac{\tilde{q}_i^\top \tilde{k}_j}{\sqrt{d_k}} = \frac{q_i^\top R(j-i) k_j}{\sqrt{d_k}}
$$

### 정의 5.3 — RoPE 의 복소수 form

$d_k$-dim vector 를 $d_k/2$ complex 로:
$$
q^{\text{complex}} = q_{2i} + i \, q_{2i+1}
$$

회전:
$$
\tilde{q}^{\text{complex}}_i = e^{i \theta_i \cdot pos} \cdot q^{\text{complex}}_i
$$

(complex multiplication = 2D 회전)

### 정의 5.4 — ALiBi

Head $h$ 의 slope $m_h$:
$$
m_h = 2^{-8h/H} \quad \text{for } h = 1, \ldots, H
$$

(geometric series, Press 2021)

Attention score:
$$
S_{ij}^{(h)} = \frac{q_i^{(h) \top} k_j^{(h)}}{\sqrt{d_k}} - m_h |i - j|
$$

(causal mask 에서는 $i \geq j$, $|i-j| = i-j$)

### 정의 5.5 — ALiBi 의 의미

각 head 의 slope 가 다름:
- $h = 1$: $m_1 = 2^{-8/H}$ — 작음 (멀리 까지 attend)
- $h = H$: $m_H = 2^{-8}$ — 큼 (local 만 attend)

→ 각 head 가 다른 distance scale specialize.

---

## 🔬 정리와 증명

### 정리 5.1 — RoPE 의 Auto-Relative

$$
\langle R(i) q, R(j) k \rangle = q^\top R(j-i) k
$$

(distance $j-i$ 에만 의존)

**증명**:

$R(t)$ 가 직교 (rotation): $R(t)^\top R(t) = I$, $R(t)^\top = R(-t)$.

$R$ 의 합성: $R(a) R(b) = R(a+b)$ (회전의 추가).

$$
\langle R(i) q, R(j) k \rangle = (R(i) q)^\top (R(j) k) = q^\top R(i)^\top R(j) k = q^\top R(-i) R(j) k = q^\top R(j-i) k \quad \square
$$

### 정리 5.2 — RoPE 의 Per-pair Decomposition

각 pair $(2i, 2i+1)$ 별로 독립:
$$
\langle R(i)_p q_p, R(j)_p k_p \rangle = q_p^\top R_p(j-i) k_p
$$

(각 frequency $\theta_p$ 별 독립 회전)

**의미**: RoPE 는 각 pair 가 independent 의 frequency channel — multi-scale relative encoding.

### 정리 5.3 — Sinusoidal 과의 관계

Sinusoidal PE 의 **inner product** 가 relative (정리 2.3) 였지만 attention score 의 다른 항이 깸.

RoPE 는 sinusoidal frequency 를 **rotation 으로 적용** — inner product 의 relative 성질을 attention 전체에 baked-in.

→ **RoPE = sinusoidal 의 logical conclusion**.

### 정리 5.4 — ALiBi 의 Linear Bias

Attention $A_{ij} \propto \exp(q_i^\top k_j / \sqrt{d_k} - m |i-j|)$.

$|i-j|$ 가 클수록 $\exp(-m|i-j|) = e^{-m \cdot \text{dist}}$ — **exponential decay** (in raw weight, before softmax).

**Effect**: distant token 의 weight 가 자연스럽게 작음 → "soft sliding window" effect.

### 정리 5.5 — ALiBi 의 Extrapolation 우수성

학습 시 max distance $T$ 에 대해 ALiBi 가 train 한 attention pattern:
- Local attention (small $|i-j|$): 거의 영향 없음
- Distant attention: $-m |i-j|$ penalty

Inference 시 더 긴 distance:
- 학습한 pattern 을 자연스럽게 extend (linear extrapolation)
- $\exp(-m \cdot \text{larger dist})$ 가 자연스럽게 작아짐 → 효과적 sliding window
- → train length 의 4× 이상까지 robust

**RoPE 와의 비교**: RoPE 는 frequency 가 train length 에 fit, 그 너머에서 sinusoidal-like 한계. ALiBi 는 simple linear penalty 가 임의 distance 처리.

### 정리 5.6 — Per-head Slope $m_h$ 의 동기

Geometric series $m_h = 2^{-8h/H}$:
- 다양한 distance scale specialize
- 일부 head 는 local (큰 $m$), 일부는 global (작은 $m$)
- 자연스러운 multi-scale processing

### 정리 5.7 — Implementation 비교

**RoPE**:
- Q, K 에 회전 적용 (per-token cosine/sine multiply)
- Per-layer: $O(T d_k)$ extra
- Attention computation 자체는 unchanged

**ALiBi**:
- Attention score 에 bias matrix 추가 (precomputed)
- Per-attention: $O(T^2)$ 한 번 계산 후 cache
- Even simpler

**둘 다** parameter 거의 없음 (RoPE: 0, ALiBi: $H$ slope).

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — RoPE 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def precompute_rope_freqs(d_k, max_seq_len, base=10000.0):
    """precompute cos and sin for RoPE"""
    theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    positions = torch.arange(max_seq_len).float()
    freqs = positions[:, None] * theta[None, :]                       # (T, d_k/2)
    cos = freqs.cos(); sin = freqs.sin()
    return cos, sin

def apply_rope(x, cos, sin):
    """x: (..., T, d_k), cos/sin: (T, d_k/2)"""
    T, d_k = x.size(-2), x.size(-1)
    x_pair1 = x[..., 0::2]   # even indices
    x_pair2 = x[..., 1::2]   # odd indices
    
    # Rotate: (cos*p1 - sin*p2, sin*p1 + cos*p2)
    cos_t = cos[:T].view(*([1] * (x.dim() - 2)), T, d_k // 2)
    sin_t = sin[:T].view(*([1] * (x.dim() - 2)), T, d_k // 2)
    
    rotated_p1 = x_pair1 * cos_t - x_pair2 * sin_t
    rotated_p2 = x_pair1 * sin_t + x_pair2 * cos_t
    
    # Interleave back
    rotated = torch.stack([rotated_p1, rotated_p2], dim=-1).flatten(-2)
    return rotated

# 테스트
d_k = 64; T = 20
cos, sin = precompute_rope_freqs(d_k, T)
q = torch.randn(1, T, d_k)
q_rotated = apply_rope(q, cos, sin)
print(f'RoPE-rotated Q: {q_rotated.shape}')

# Inner product test: <R(i) q, R(j) k> = q^T R(j-i) k
torch.manual_seed(0)
q_vec = torch.randn(d_k); k_vec = torch.randn(d_k)
i, j = 5, 12

# 직접 계산
q_at_i = apply_rope(q_vec.unsqueeze(0).expand(T, d_k), cos, sin)[i]
k_at_j = apply_rope(k_vec.unsqueeze(0).expand(T, d_k), cos, sin)[j]
ip_direct = q_at_i @ k_at_j

# Relative formula: q^T R(j-i) k
k_at_dist = apply_rope(k_vec.unsqueeze(0).expand(T, d_k), cos, sin)[j-i]
ip_relative = q_vec @ k_at_dist
print(f'<R(i)q, R(j)k> = {ip_direct:.4f}')
print(f'q^T R(j-i) k  = {ip_relative:.4f}')
# 같음 → auto-relative ✓
```

### 실험 2 — RoPE 의 Attention 통합

```python
class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=512):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        cos, sin = precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer('cos', cos); self.register_buffer('sin', sin)
    
    def forward(self, x):
        B, T, d = x.size()
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        
        # RoPE 적용
        Q = apply_rope(Q, self.cos, self.sin)
        K = apply_rope(K, self.cos, self.sin)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)

torch.manual_seed(0)
attn = RoPEAttention(64, 8, max_seq_len=128)
x = torch.randn(1, 50, 64)
y = attn(x)
print(f'RoPE attention output: {y.shape}')
```

### 실험 3 — ALiBi 구현

```python
def get_alibi_slopes(num_heads):
    """Press 2021 의 geometric sequence of slopes"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(np.log2(n) - 3))
        ratio = start
        return [start * ratio**i for i in range(n)]
    
    if num_heads & (num_heads - 1) == 0:   # power of 2
        return get_slopes_power_of_2(num_heads)
    else:
        # Closest power of 2 below
        closest_p2 = 2 ** int(np.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_p2)
        if num_heads > closest_p2:
            extra = get_slopes_power_of_2(2 * closest_p2)[0::2][:num_heads - closest_p2]
            slopes.extend(extra)
        return slopes

class ALiBiAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes))
    
    def get_alibi_bias(self, T):
        """ALiBi bias matrix: (h, T, T)"""
        positions = torch.arange(T)
        rel = -torch.abs(positions[None, :] - positions[:, None]).float()   # (T, T): -|i-j|
        return rel.unsqueeze(0) * self.slopes[:, None, None]                # (h, T, T)
    
    def forward(self, x):
        B, T, d = x.size()
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        scores = scores + self.get_alibi_bias(T).to(x.device).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)

torch.manual_seed(0)
attn = ALiBiAttention(64, 8)
x = torch.randn(1, 50, 64)
y = attn(x)
print(f'ALiBi attention output: {y.shape}')

# Slope 시각화
print(f'\nALiBi slopes: {get_alibi_slopes(8)}')
```

### 실험 4 — RoPE vs ALiBi 의 Attention Pattern

```python
torch.manual_seed(0)
T = 30; d_k = 32

# RoPE
cos, sin = precompute_rope_freqs(d_k, T)
q = torch.randn(1, T, d_k); k = torch.randn(1, T, d_k)
q_rope = apply_rope(q, cos, sin)
k_rope = apply_rope(k, cos, sin)
scores_rope = (q_rope @ k_rope.transpose(-2, -1)) / np.sqrt(d_k)
attn_rope = F.softmax(scores_rope, dim=-1)

# ALiBi
slopes = torch.tensor([0.25])   # single head for visualization
positions = torch.arange(T)
alibi_bias = -torch.abs(positions[None, :] - positions[:, None]).float() * slopes[0]
scores_alibi = (q @ k.transpose(-2, -1)) / np.sqrt(d_k) + alibi_bias
attn_alibi = F.softmax(scores_alibi, dim=-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(attn_rope[0].detach().numpy(), cmap='Blues')
axes[0].set_title('RoPE attention')
axes[1].imshow(attn_alibi[0].detach().numpy(), cmap='Blues')
axes[1].set_title('ALiBi attention')
plt.tight_layout(); plt.show()
# ALiBi 가 명확한 diagonal-locality, RoPE 는 frequency-based pattern
```

### 실험 5 — Extrapolation 실험

```python
# Train length 20, test length 60
# RoPE 와 ALiBi 의 long context 동작 비교

T_train, T_test = 20, 60
d = 32
torch.manual_seed(0)

# Random Q, K vectors
q_test = torch.randn(1, T_test, d)
k_test = torch.randn(1, T_test, d)

# RoPE: train 시 max_seq_len=20 으로 fit, test 60 까지 사용
cos_train, sin_train = precompute_rope_freqs(d, T_train)
cos_test,  sin_test  = precompute_rope_freqs(d, T_test)   # extend

q_rope_test = apply_rope(q_test, cos_test, sin_test)
k_rope_test = apply_rope(k_test, cos_test, sin_test)
attn_rope_test = F.softmax((q_rope_test @ k_rope_test.transpose(-2, -1)) / np.sqrt(d), dim=-1)

# ALiBi: 자연스럽게 임의 길이
alibi_bias_test = -torch.abs(torch.arange(T_test)[None, :] - torch.arange(T_test)[:, None]).float() * 0.25
attn_alibi_test = F.softmax((q_test @ k_test.transpose(-2, -1)) / np.sqrt(d) + alibi_bias_test, dim=-1)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(attn_rope_test[0].detach().numpy(), cmap='Blues')
axes[0].set_title(f'RoPE attention (T_test={T_test} > T_train={T_train})')
axes[1].imshow(attn_alibi_test[0].detach().numpy(), cmap='Blues')
axes[1].set_title('ALiBi attention (extrapolated)')
plt.tight_layout(); plt.show()
# RoPE 는 train 한 frequency 에 fit, far position 은 다른 분포
# ALiBi 는 일관된 distance-based pattern (다 같은 모양)
```

---

## 🔗 실전 활용

### 1. LLaMA / Mistral 의 RoPE

LLaMA-1: 2K context, RoPE
LLaMA-2: 4K (RoPE) → 32K (NTK-aware RoPE)
LLaMA-3: 8K → 128K (RoPE + position interpolation)

RoPE 의 frequency 조정 (NTK-aware):
- Original: $\theta_i = 10000^{-2i/d}$
- NTK-aware: $\theta_i = (10000 \cdot k)^{-2i/d}$ for context extension factor $k$
- 기존 학습한 frequency 를 effective long context 로 extend

### 2. BLOOM 의 ALiBi

BLOOM (2022): 176B params, ALiBi
- Train length 2048 → inference 4× 까지 robust
- Extrapolation 강조

### 3. NTK-aware RoPE / YaRN

Long context extension:
- **NTK-aware** (Reddit user "u/bloc97"): high-freq 보존, low-freq scale
- **YaRN** (Peng 2023): NTK + temperature 조정

### 4. xPos (Sun 2023)

RoPE + ALiBi 의 합성:
- 회전 + decay (each pair 의 magnitude 도 감소)
- Long context 에서 RoPE 보다 더 robust

### 5. 학습 hyperparameter

- **RoPE base**: 10000 (default), context extension 시 더 큰 값
- **ALiBi slopes**: geometric series, head 수에 따라

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 1D position | 2D/3D 는 별도 (RoPE-2D 가능) |
| Per-pair frequency | 다른 frequency scheme 도 가능 |
| RoPE: full attention | Sparse 와 결합 시 추가 고려 |
| ALiBi: linear penalty | Long-range 에서 너무 aggressive 일 수 있음 |
| Static slopes | Learned slopes 도 시도 |

---

## 📌 핵심 정리

$$\boxed{\text{RoPE: } \langle R(i) q, R(j) k \rangle = q^\top R(j-i) k \quad \text{(automatic relative)}}$$

$$\boxed{\text{ALiBi: } S_{ij}^{(h)} = q^\top k - m_h |i-j| \quad \text{(linear distance penalty)}}$$

| Method | Param | Mechanism | Pros | Cons |
|--------|-------|-----------|------|------|
| **RoPE** | 0 | Q, K 회전 | Auto-relative, 풍부한 표현 | NTK 조정 필요 (long ctx) |
| **ALiBi** | $H$ slopes | Score bias | 최강 extrapolation | Less expressive |
| **Sinusoidal** | 0 | Add to embedding | Simple | Limited extrapolation |
| **Learned** | $T_{\max} \times d$ | Lookup | Data-driven | No extrapolation |

| Modern LLM | PE 선택 |
|------------|--------|
| LLaMA-1/2/3, Mistral, Qwen | RoPE |
| BLOOM, MPT | ALiBi |
| GPT-4, Claude (estimate) | RoPE variants |

---

## 🤔 생각해볼 문제

**문제 1** (기초): RoPE 의 회전 행렬 $R(t)$ 를 $d_k = 4$ 인 경우에 명시적으로 적어라. $\theta_0 = 1, \theta_1 = 0.01$ 가정.

<details>
<summary>해설</summary>

$d_k = 4$ → 2 pairs.

$$
R(t) = \begin{pmatrix}
\cos t & -\sin t & 0 & 0 \\
\sin t & \cos t & 0 & 0 \\
0 & 0 & \cos(0.01 t) & -\sin(0.01 t) \\
0 & 0 & \sin(0.01 t) & \cos(0.01 t)
\end{pmatrix}
$$

(block-diagonal of 2×2 회전, frequency $\theta_0 = 1, \theta_1 = 0.01$)

**$t = 0$**: $R(0) = I$ (no rotation).
**$t = 5$**: 첫 pair 회전 5 rad (대략 286°), 둘째 pair 회전 0.05 rad (작음).

Multi-scale: 첫 pair 가 짧은 거리 빠르게 변화, 둘째 pair 가 긴 거리 천천히 변화. $\square$

</details>

**문제 2** (심화): RoPE 의 inner product $\langle R(i) q, R(j) k \rangle$ 의 첫 pair (frequency $\theta_0$) 의 contribution 을 분석하라. 두 pair 의 contribution 이 어떻게 합쳐져 multi-scale relative 표현이 만들어지는가?

<details>
<summary>해설</summary>

**Per-pair contribution**:

각 pair $p$ 의 inner product:
$$
\langle R(i)_p q_p, R(j)_p k_p \rangle = q_p^\top R_p(j-i) k_p
$$

$R_p$ 가 2×2 회전:
$$
R_p(d) = \begin{pmatrix} \cos(\theta_p d) & -\sin(\theta_p d) \\ \sin(\theta_p d) & \cos(\theta_p d) \end{pmatrix}
$$

with $d = j - i$.

Pair $p$ 의 inner product 를 $q_p = (a, b)$, $k_p = (c, d)$ 로 표기:
$$
q_p^\top R_p(d) k_p = a c \cos(\theta_p d) + a d (-\sin(\theta_p d)) + b c \sin(\theta_p d) + b d \cos(\theta_p d)
$$
$$
= (ac + bd) \cos(\theta_p d) + (bc - ad) \sin(\theta_p d)
$$

**전체 inner product**:

$$
\langle R(i) q, R(j) k \rangle = \sum_{p=0}^{d_k/2 - 1} \big[ (a_p c_p + b_p d_p) \cos(\theta_p (j-i)) + (b_p c_p - a_p d_p) \sin(\theta_p (j-i)) \big]
$$

**의미**:
- 각 pair 가 다른 frequency $\theta_p$ 의 oscillation
- 짧은 거리 (high $\theta_p$): 빠르게 oscillate, 정확한 거리 구분
- 긴 거리 (low $\theta_p$): 느리게 oscillate, 큰 거리 차이만 구분
- **Multi-scale Fourier-like decomposition** 의 attention score

**왜 이것이 좋은가**:
- 자연어의 다양한 distance scale (word, sentence, paragraph) 모두 다룸
- Sinusoidal PE 와 같은 inductive bias 그러나 attention 자체에 baked-in
- 학습이 각 frequency 의 contribution 조정 (Q, K 의 학습된 분포로)

→ **RoPE = sinusoidal PE 의 attention-aware 변형**, 더 자연스러운 통합. $\square$

</details>

**문제 3** (논문 비평): ALiBi 가 RoPE 보다 simpler 하지만 일부 task 에서는 RoPE 가 더 좋다. 어떤 task 에 어떤 PE 가 적합한지, 그리고 LLaMA 가 RoPE, BLOOM 이 ALiBi 를 선택한 design 결정의 차이를 분석하라.

<details>
<summary>해설</summary>

**RoPE vs ALiBi Trade-off**:

| 측면 | RoPE | ALiBi |
|------|------|-------|
| Expressivity | 풍부 (multi-frequency) | 단순 (linear penalty) |
| Implementation | 약간 복잡 (회전) | 매우 단순 (bias) |
| Extrapolation | NTK 조정 필요 | 자연스럽게 robust |
| Long-range deps | 잘 학습 | distance penalty 로 약화 |
| Local pattern | OK | 명시적으로 강조 |

**Task별 적합성**:

1. **Long-range dependency 필요** (예: long-document understanding, retrieval):
   - RoPE 가 적합 — distance penalty 가 long-range attention 을 약화시키지 않음
   - ALiBi 의 linear penalty 가 critical info 차단 가능

2. **Local pattern dominant** (예: code, tabular data):
   - ALiBi 자연스러움 — local 강조
   - RoPE 도 OK 그러나 ALiBi 가 simpler

3. **Extreme extrapolation** (train 의 5×+):
   - ALiBi 우수 — 자연스러운 길이 일반화
   - RoPE 는 NTK/YaRN 같은 추가 기법 필요

**LLaMA 의 RoPE 선택**:

- General-purpose LLM: 다양한 task — long-range 도 처리해야
- Quality 우선 (chat, reasoning) → expressive RoPE
- Long context 는 NTK-aware 로 확장 가능 — 충분히 robust
- Open source community 의 RoPE 생태계 (GPT-NeoX, Falcon 등 모두)

**BLOOM 의 ALiBi 선택**:

- Multilingual + general
- Computational simplicity — 큰 모델 (176B) 에서 매 layer 의 simplicity 중요
- Length generalization 강조 — 학습 시 짧게, 사용 시 길게
- 개발 시기 (2022): ALiBi 가 막 등장, RoPE 의 long context 기법 (NTK) 미발달

**Modern 추세**:

- 대부분 모델 RoPE 채택 (LLaMA 생태계 dominance)
- ALiBi 는 minority but specific niche
- Hybrid (xPos) 는 academic, production 채택 적음

**근본적 차이**:

- **RoPE**: "frequency-based, multi-scale" — 정교한 representation
- **ALiBi**: "distance-penalty, simple" — 직관적이고 robust

각 design philosophy 가 다른 strength. "What works in practice" 가 ranking 보다 task-specific.

**예측**: Modern LLM 의 dominant choice 는 RoPE + long context extension techniques (NTK, YaRN, position interpolation). ALiBi 는 simplicity 가 critical 한 niche 에서 유지. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-relative-pe.md) | [📚 README](../README.md) | [다음 ▶](../ch4-training-math/01-warmup.md)

</div>
