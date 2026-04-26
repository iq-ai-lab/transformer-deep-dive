# 04. Relative Positional Encoding (Shaw 2018)

## 🎯 핵심 질문

- Relative PE 의 핵심 idea — absolute position 대신 token 쌍의 distance $i-j$ 를 사용?
- Shaw 2018 의 식 $e_{ij} = (x_i W_Q)(x_j W_K + a_{ij}^K)^\top$ 의 의미와 absolute PE 와의 차이?
- "Clip distance" 가 왜 필요하고 어떤 효과를 갖는가?
- T5 의 bucketed relative bias 가 어떤 변형인가?
- Transformer-XL 의 segment-level recurrence 가 relative PE 와 어떻게 상호작용?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Relative PE 는 absolute PE 의 한계를 직접 해결합니다:

1. **Translation invariance** — 같은 거리는 항상 같은 representation
2. **Better generalization** — train 분포에 있는 거리만 학습하면 됨
3. **Long context 적합** — clip distance 로 임의 길이 처리
4. **RoPE/ALiBi 의 토대** — relative idea 가 modern PE 로 발전

이 문서는 Shaw 2018 의 original 정식과 T5/Transformer-XL 의 변형을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-sinusoidal-pe.md](./02-sinusoidal-pe.md) (relative encoding 의 sinusoidal 해석), [03-learned-pe.md](./03-learned-pe.md)
- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md) — attention score 분해

---

## 📖 직관적 이해

### Absolute vs Relative

```
Absolute PE:                        Relative PE:
"The cat sat" (pos 0,1,2)          "The cat sat" (rel 0, 1, 2 from cat)
"... The cat sat" (pos 5,6,7)      Same: rel = (0, 1, 2) from cat
                                    같은 패턴 → 같은 representation!
```

**핵심**: 두 token 의 relative 거리만 중요 — 절대 위치 무관.

### Shaw 2018 의 직관

Attention score:
$$
e_{ij} = (q_i)(k_j + a_{ij})^\top = q_i k_j^\top + q_i a_{ij}^\top
$$

추가 항 $q_i a_{ij}^\top$:
- $a_{ij}$ 가 **relative distance $i - j$ 의 함수**
- Query 가 자신과의 거리에 따라 다른 attention bias 부여

### Clip Distance

```
Distance:    -5  -4  -3  -2  -1   0   1   2   3   4   5
Embedding:   a₋₃ a₋₃ a₋₃ a₋₂ a₋₁ a₀  a₁  a₂  a₃  a₃  a₃
                ↑           clip ↑                    clip
        max_dist 너머는 같은 embedding
```

먼 거리 차이는 무시 — 모델이 long-range 패턴은 학습 못 하지만 short-range 가 dominant.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Shaw 2018 의 Relative PE

Token pair $(i, j)$ 에 대해 학습 가능 vector:
$$
a_{ij}^K, a_{ij}^V \in \mathbb{R}^{d_k}
$$

distance $i - j$ 에 의존:
$$
a_{ij}^K = a_{\text{clip}(i-j, k_{\max})}^K
$$

(clip 적용 후 lookup)

### 정의 4.2 — Modified Attention Score

$$
e_{ij} = \frac{(x_i W_Q)(x_j W_K + a_{ij}^K)^\top}{\sqrt{d_k}}
$$

$$
A_{ij} = \text{softmax}_j(e_{ij})
$$

### 정의 4.3 — Modified Output

$$
\text{Attn}_i = \sum_j A_{ij} (x_j W_V + a_{ij}^V)
$$

(Value 에도 relative term 추가)

### 정의 4.4 — Clip Function

$$
\text{clip}(x, k) = \max(-k, \min(k, x))
$$

거리 $|i-j| > k$ 시 모두 $\pm k$ 로 압축. 학습 가능 vector 수: $2k + 1$.

### 정의 4.5 — T5 의 Bucketed Relative Bias

Shaw 의 단순화: $W_K, W_V$ 와 separate 한 vector 안 쓰고 **scalar bias** 만:
$$
e_{ij} = \frac{(x_i W_Q)(x_j W_K)^\top}{\sqrt{d_k}} + b_{i-j}^h
$$

각 head $h$ 별 학습 가능 scalar $b$. Bucket: 가까운 거리는 fine, 먼 거리는 logarithmic bucketing.

---

## 🔬 정리와 증명

### 정리 4.1 — Translation Invariance

Shaw 2018 의 attention score:
$$
e_{i+s, j+s} = e_{ij} \quad \forall s
$$

(같은 $i-j$ 라면 absolute position 무관)

**증명**:
- $W_Q$ output 은 token-only $x_{i+s} W_Q$ — token 자체는 같지만 position 정보 없으면 same
- 추가: relative term $a_{(i+s)-(j+s)}^K = a_{i-j}^K$ — distance 보존
- 따라서 token + relative = invariant under translation $\square$

(단, token embedding 만 position 무관 시 — input 에 absolute PE 가 안 들어가야)

### 정리 4.2 — Clip 의 효과

$|i-j| > k_{\max}$ 인 모든 token pair 가 same relative embedding 사용. 결과:
- Long-distance attention 의 표현력 ↓ (모두 같은 bias)
- Param 수 한정 ($2k_{\max} + 1$)
- **그러나** softmax + clip 으로 long-distance attention 이 자연스럽게 작아짐 (saturate)

### 정리 4.3 — Param 수 비교

Sinusoidal: 0 (학습 X)
Learned absolute: $T_{\max} \times d$
Shaw relative: $(2k_{\max} + 1) \times d_k$ (per head, K, V) → $2 h \times (2k_{\max}+1) \times d_k$
T5 relative bias: $h \times \text{num_buckets}$ (보통 32) — 매우 작음

T5 가 가장 efficient.

### 정리 4.4 — Sinusoidal Relative 와의 등가성

Sinusoidal 의 inner product 가 relative (정리 2.3) 였지만, attention 의 modify (W_Q W_K^\top) 로 깨짐.

Shaw 가 명시적으로 relative term 추가 → architecture 에 baked-in. RoPE 는 더 elegant 한 통합.

### 정리 4.5 — Shaw vs T5 의 표현력

- **Shaw**: relative term 이 $d_k$-dim vector → expressive
- **T5**: scalar bias only → less expressive but simple

**실증**: T5 의 simplicity 가 충분 — large scale 에서 bias 만으로 OK.

### 정리 4.6 — Combined with Absolute

일부 모델: absolute + relative 함께 사용. 표현력 max 이지만 redundancy 위험.

Modern 추세: relative-only (RoPE/ALiBi) 가 dominant.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Shaw 2018 의 Relative Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShawRelativeAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_relative_distance=10):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.max_dist = max_relative_distance
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # Relative position embeddings (learned)
        # 2 * max_dist + 1 unique distances (-max..-1, 0, 1..max)
        self.rel_emb_K = nn.Parameter(torch.randn(2*max_relative_distance+1, self.d_k) * 0.02)
        self.rel_emb_V = nn.Parameter(torch.randn(2*max_relative_distance+1, self.d_k) * 0.02)
    
    def get_relative_indices(self, T):
        """Position differences clipped to [-max_dist, max_dist]"""
        positions = torch.arange(T)
        rel = positions[:, None] - positions[None, :]   # (T, T): i-j
        rel_clipped = rel.clamp(-self.max_dist, self.max_dist)
        return rel_clipped + self.max_dist   # shift to [0, 2*max_dist]
    
    def forward(self, x):
        B, T, d = x.size()
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)   # (B,h,T,d_k)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        
        # Standard scores
        scores_std = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Relative scores
        rel_idx = self.get_relative_indices(T)         # (T, T)
        rel_K = self.rel_emb_K[rel_idx]                 # (T, T, d_k)
        # Q @ rel_K^T per (i, j): einsum
        scores_rel = torch.einsum('bhid,ijd->bhij', Q, rel_K) / np.sqrt(self.d_k)
        
        scores = scores_std + scores_rel
        attn = F.softmax(scores, dim=-1)
        
        # Standard output
        out_std = attn @ V
        
        # Relative output
        rel_V = self.rel_emb_V[rel_idx]                 # (T, T, d_k)
        out_rel = torch.einsum('bhij,ijd->bhid', attn, rel_V)
        
        out = (out_std + out_rel).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)

# 테스트
torch.manual_seed(0)
attn = ShawRelativeAttention(64, 8, max_relative_distance=10)
x = torch.randn(2, 20, 64)
y = attn(x)
print(f'Output: {y.shape}')   # (2, 20, 64)
```

### 실험 2 — Translation Invariance 검증

```python
torch.manual_seed(0)
attn = ShawRelativeAttention(32, 4, max_relative_distance=5)
attn.eval()

# 짧은 sequence
x_short = torch.randn(1, 5, 32)
y_short = attn(x_short)

# 같은 sequence 를 longer batch 안에 padding
x_long = torch.cat([torch.zeros(1, 10, 32), x_short, torch.zeros(1, 10, 32)], dim=1)
y_long = attn(x_long)

# y_short 이 y_long 의 [10:15] 와 같아야 함 (translation invariance)
diff = (y_short - y_long[:, 10:15]).abs().max()
print(f'Translation invariance: max diff = {diff:.6f}')
# Note: zero padding 이 attention 에 영향을 줘서 정확히 0 안 됨,
# 그러나 attention mask 적용 시 정확히 invariant
```

### 실험 3 — T5-style Bucketed Relative Bias

```python
def t5_relative_bucket(rel_pos, num_buckets=32, max_distance=128):
    """T5 의 logarithmic bucketing"""
    ret = 0
    n = -rel_pos
    num_buckets //= 2
    ret += (n < 0).to(torch.long) * num_buckets
    n = n.abs()
    
    max_exact = num_buckets // 2
    is_small = n < max_exact
    
    # Logarithmic bucketing for large values
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = val_if_large.clamp(max=num_buckets - 1)
    
    ret += torch.where(is_small, n, val_if_large)
    return ret

# Visualize bucketing
positions = torch.arange(-100, 101)
buckets = t5_relative_bucket(positions, num_buckets=32, max_distance=128)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 4))
plt.plot(positions.numpy(), buckets.numpy())
plt.xlabel('Relative position'); plt.ylabel('Bucket index')
plt.title('T5 relative bucket — logarithmic for far distances')
plt.grid(alpha=0.3); plt.show()
```

### 실험 4 — Relative Bias 만 사용 (T5)

```python
class T5RelativeBias(nn.Module):
    def __init__(self, num_heads, num_buckets=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.bias_table = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.bias_table.weight, std=0.02)
    
    def forward(self, T):
        positions = torch.arange(T)
        rel = positions[None, :] - positions[:, None]   # (T, T)
        bucket = t5_relative_bucket(rel, self.num_buckets)
        return self.bias_table(bucket).permute(2, 0, 1)   # (h, T, T)

bias = T5RelativeBias(num_heads=8)
b = bias(T=10)
print(f'T5 relative bias shape: {b.shape}')   # (8, 10, 10)

# Heatmap
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(b[0].detach().numpy(), cmap='RdBu')
ax.set_title('T5 relative bias (head 0)')
plt.show()
```

### 실험 5 — Relative vs Absolute 의 Generalization 비교

```python
# Toy experiment: 같은 sequence pattern 의 다른 위치
torch.manual_seed(0)

# Pattern: "A B C" 가 어디 있든 같은 처리
T = 20
x_pattern_pos5 = torch.randn(1, T, 32)   # pattern at pos 5-7
x_pattern_pos15 = x_pattern_pos5.clone()
# 단순화: 그냥 같은 input

# Absolute PE (sinusoidal init)
pe_abs = sinusoidal_init(T, 32)
y_abs_5 = (x_pattern_pos5 + pe_abs.unsqueeze(0))
y_abs_15 = (x_pattern_pos15 + pe_abs.unsqueeze(0))

# Relative attention (Shaw)
attn_rel = ShawRelativeAttention(32, 4, max_relative_distance=5)
y_rel = attn_rel(x_pattern_pos5)

# Same input → same output for relative (since no abs PE)
y_rel_2 = attn_rel(x_pattern_pos5.flip(1))   # reversed
print(f'Relative-only (same input): same output = {(y_rel - attn_rel(x_pattern_pos5)).abs().max():.6f}')
```

---

## 🔗 실전 활용

### 1. T5 의 채택

T5 (Raffel 2020): bucketed relative bias 만, no absolute PE.
- 32 buckets per head
- Param 수 매우 적음 (수천 정도)
- Long context (4K) 에서도 잘 작동

### 2. Transformer-XL (Dai 2019)

Segment-level recurrence + relative PE:
- Long context 처리 위해 이전 segment 의 hidden 재사용
- Absolute PE 시 segment 마다 position 재시작 → confusion
- Relative PE 가 자연스러움 — distance 기준

### 3. DeBERTa (He 2021)

Disentangled attention:
- Content $\to$ Content score (token-token)
- Content $\to$ Position score (token-relative position)
- Position $\to$ Content score
- 세 score 의 합

→ Relative PE 의 가장 정교한 변형.

### 4. Modern LLM 에서의 의미

LLaMA / Mistral 등 RoPE 가 dominant. Shaw-style relative 는 legacy 인접 기법.

T5 의 simplicity 가 여전히 일부 모델 (Flan-T5, ByT5) 에서 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Distance only | Edge type / dependency 같은 다른 정보 제외 |
| Clip distance | 먼 거리 정확히 다루지 못함 |
| Per-head bias (T5) | Less expressive but simple |
| Bidirectional | Causal mask 와 별개 |
| Discrete distance | Continuous 도 가능 (interpolation) |

---

## 📌 핵심 정리

$$\boxed{\text{Shaw 2018: } e_{ij} = \frac{(x_i W_Q)(x_j W_K + a_{i-j}^K)^\top}{\sqrt{d_k}}}$$

$$\boxed{\text{T5: } e_{ij} = \frac{(x_i W_Q)(x_j W_K)^\top}{\sqrt{d_k}} + b_{\text{bucket}(i-j)}^h}$$

| 변형 | Param | Pros | Cons |
|------|-------|------|------|
| Shaw 2018 | $(2k+1) \times d_k$ per K, V | Expressive | Clip 한계 |
| T5 | $h \times \text{buckets}$ scalar | 매우 적음 | Less expressive |
| Transformer-XL | Sinusoidal-derived | Segment-level | 복잡 |
| DeBERTa | Disentangled triple | 정교 | 복잡 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Shaw 2018 에서 $\text{max_dist} = 5$, $T = 10$ 인 경우, distance lookup index matrix $\text{rel_idx}_{ij}$ 의 가운데 영역 (i, j ∈ [3,6]) 을 손으로 그려라.

<details>
<summary>해설</summary>

$\text{rel}_{ij} = i - j$, clip to $[-5, 5]$, shift by $+5$:

```
       j=3  j=4  j=5  j=6
i=3 |  0+5=5  -1+5=4  -2+5=3  -3+5=2
i=4 |  1+5=6  0+5=5   -1+5=4  -2+5=3
i=5 |  2+5=7  1+5=6   0+5=5   -1+5=4
i=6 |  3+5=8  2+5=7   1+5=6   0+5=5
```

대각선이 5 (relative=0), 위는 작아짐 (j > i, negative), 아래는 커짐.

먼 거리에서는 모두 0 (clip 의 -5 후 +5) 또는 10 (clip 의 +5 후 +5).

→ Lookup table 의 11 entry ($2 \times 5 + 1$) 사용. $\square$

</details>

**문제 2** (심화): T5 의 logarithmic bucketing 의 동기는? Linear bucketing 보다 왜 더 효율적인가?

<details>
<summary>해설</summary>

**Logarithmic bucketing 의 동기**:

Natural language 의 distance distribution:
- 짧은 거리 (1-10 token): 매우 흔함, syntactic dependency
- 중간 거리 (10-100): 흔함, sentence-level
- 긴 거리 (100+): 점점 드물어짐, document-level

**Linear bucketing 의 단점**:
- 모든 거리를 동일하게 다룸 → 짧은 거리에 더 많은 resolution 필요한데 단일 bucket
- 긴 거리에 over-resolution (수많은 bucket but 거의 사용 안 됨)

**Logarithmic bucketing 의 advantage**:
- 짧은 거리에 fine-grained bucket (1, 2, 3, ..., $k_{\max}/2$)
- 긴 거리는 logarithmic — 같은 범위에 fewer bucket
- 거리 distribution 의 데이터 분포와 자연스럽게 align

**T5 specific**:
- 32 buckets total (16 positive + 16 negative)
- 처음 16 은 exact distances (1 to 16)
- 그 후는 log spacing (16 to 128)

**효율 vs 표현력**:

Linear with 32 buckets → distance up to 32 만 정확히
Logarithmic with 32 buckets → distance up to 128 cover (with appropriate granularity)

→ 같은 param 수에서 더 긴 context 처리 가능. **Linear 의 expressivity 와 long-range 의 trade-off** 를 logarithmic 이 자연스럽게 해결.

**Modern variants**:
- ALiBi: 단순 linear bias — log bucketing 없음, 그러나 매우 효과적
- RoPE: continuous frequency — bucketing 자체 제거

따라서 T5 의 logarithmic 은 absolute → relative → continuous (RoPE) 의 evolution 의 한 단계. $\square$

</details>

**문제 3** (논문 비평): Shaw 2018 의 relative PE 에서 RoPE 로 진화한 이유를 architecture 와 inductive bias 측면에서 분석하라. 두 방법의 inner product 처리 방식의 본질적 차이는?

<details>
<summary>해설</summary>

**Shaw 2018 의 처리**:

$$
e_{ij} = \frac{(x_i W_Q)(x_j W_K + a_{i-j}^K)^\top}{\sqrt{d_k}}
$$

- Token + relative 가 **add** (sum 으로 결합)
- Attention score 안에 relative 항 명시적 추가
- Implementation: 별도 lookup + 별도 matmul

**RoPE 의 처리**:

$$
\langle R(i) q_i, R(j) k_j \rangle = q_i^\top R(j-i) k_j
$$

- Token 과 position 이 **multiply** (회전)
- Inner product 가 자동으로 relative
- Implementation: Q, K 에 회전 직접 적용 (단순)

**본질적 차이 — Architectural Integration**:

1. **Shaw**: 추가 component (relative embedding) 를 architecture 에 plug-in. Attention computation 에 두 항 (standard + relative).

2. **RoPE**: Architecture 자체가 relative-aware. 단일 attention computation, position 이 inner product metric 에 baked-in.

**Inductive Bias**:

- **Shaw**: "relative information 은 token information 과 합쳐 attend" — additive prior
- **RoPE**: "attention metric 이 inherently relative" — multiplicative prior

**왜 RoPE 가 우수?**

1. **Mathematical elegance**: 회전 함수가 distance 의 자연스러운 표현
2. **Implementation simple**: 추가 lookup table 없음, Q/K 에 elementwise 회전
3. **No clip**: 임의 distance 처리 (sinusoidal-like extrapolation)
4. **Matrix-free**: 매 layer 마다 회전 계산이 cheap
5. **Translation invariance baked-in**: clip 없는 자연스러운 invariance

**Shaw 의 한계**:
- Clip distance hyperparameter
- Param 수 증가 (relative embedding lookup)
- Long context 에서 clip 으로 정보 손실
- Implementation 복잡 (separate computation paths)

**진화 계보**:

```
Sinusoidal absolute PE (Vaswani 2017)
       ↓ "relative 가 더 자연스럽다"
Shaw 2018 — explicit relative embedding
       ↓ "더 elegant 한 통합"
T5 (2020) — bucketed scalar bias (simpler)
       ↓ "회전이 핵심"
RoPE (Su 2021) — rotation = automatic relative
       ↓ "더 단순한 bias 도 충분"
ALiBi (Press 2021) — linear bias 만
```

각 step 이 simplicity + effectiveness 동시 개선. Modern (2024+) LLM 은 RoPE/ALiBi dominant. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-learned-pe.md) | [📚 README](../README.md) | [다음 ▶](./05-rope-alibi.md)

</div>
