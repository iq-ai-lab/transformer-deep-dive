# 06. Multi-Query / Grouped-Query Attention

## 🎯 핵심 질문

- MQA (Shazeer 2019) 의 핵심 — $h$ Q-head 가 단일 K, V head 를 공유하는 의미는?
- GQA (Ainslie 2023) 가 MQA 와 MHA 사이의 어떤 trade-off 를 제공하는가?
- KV cache 의 $h$-fold 절약이 inference latency 에 미치는 영향?
- LLaMA-2 70B 의 $h_q = 64, h_{kv} = 8$ (GQA-8) 채택 이유는?
- MQA/GQA 의 표현력 손실은 얼마나? 학습 vs from-MHA conversion?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

MQA / GQA 는 **inference 효율의 game changer**:

1. **KV cache 의 $h$-fold 절약** — long-context generation 의 메모리 bottleneck 해결
2. **Inference 가속** — memory bandwidth 가 generation 의 bottleneck
3. **표현력 손실 작음** — Q-head 다양성 보존
4. **Modern LLM 의 표준** — LLaMA-2/3, Mistral 모두 GQA

이 문서는 MQA/GQA 의 **수학적 정의, 표현력 trade-off, KV cache 분석** 을 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [05-multi-head.md](../ch1-attention-decomposition/05-multi-head.md) — MHA
- 이전 문서: [01-quadratic-bottleneck.md](./01-quadratic-bottleneck.md) — KV cache 의 bottleneck

---

## 📖 직관적 이해

### MHA vs MQA

```
MHA (Multi-Head Attention):
  h Q-heads, h K-heads, h V-heads (모두 다름)
  KV cache size = 2 × T × h × d_k = 2 × T × d

MQA (Multi-Query Attention):
  h Q-heads, 1 K-head, 1 V-head (KV 공유)
  KV cache size = 2 × T × 1 × d_k = 2 × T × d_k (h 배 작음!)
```

```
   Q heads:  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇  Q₈  (8개)
   K head:           K (1개, 모든 Q 가 share)
   V head:           V (1개)
```

### GQA — MQA 와 MHA 사이

```
GQA-2 (group=2):
  Q heads:  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇  Q₈
  K heads:  K₁ K₁  K₂ K₂  K₃ K₃  K₄ K₄  (4개, group 별 share)
  V heads: 동일

g groups: KV cache = 2 × T × g × d_k
```

LLaMA-2 70B: 64 Q-head, 8 KV-group → **8× KV cache 절약**.

### Why does this work?

- 학습 시 모든 Q-head 가 다양하게 (independent W_Q)
- KV head 가 fewer 하지만 multiple Q 가 share — "여러 관점에서 같은 K, V 에 attend"
- 표현력 손실 작음 (실증적으로 1-2% accuracy)

### Inference 의 KV Cache Bottleneck

```
Generation step t:
  - Load entire KV cache (K, V for all previous tokens)
  - Compute attention with new Q
  - Append new K, V to cache

Memory bandwidth: KV cache size 가 dominant for long context
  T = 32K, h = 64, d = 8192:
    MHA: 32K × 64 × 128 × 2 (K+V) × 2bytes = 1GB per layer per sample
    GQA-8: 1/8 size = 128MB
```

→ **8× faster** memory access → **8× higher throughput** for batch generation.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Multi-Head Attention (재확인)

$h$ heads, each with $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$:
$$
\text{head}_i = \text{Attn}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)})
$$

KV size: $h \cdot d_k = d_{\text{model}}$.

### 정의 6.2 — Multi-Query Attention (Shazeer 2019)

$h$ Q-heads, **single** K-head, **single** V-head:
$$
W_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k} \text{ for } i = 1, \ldots, h
$$
$$
W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k} \text{ (single)}
$$

각 head:
$$
\text{head}_i = \text{Attn}(X W_Q^{(i)}, X W_K, X W_V)
$$

KV size: $1 \cdot d_k = d_k$ — $h$ 배 작음.

### 정의 6.3 — Grouped-Query Attention (Ainslie 2023)

$g$ groups, each group 이 $h/g$ Q-heads 와 1 KV-head:
$$
W_K^{(j)}, W_V^{(j)} \in \mathbb{R}^{d_{\text{model}} \times d_k} \text{ for } j = 1, \ldots, g
$$

Q-head $i$ 가 group $\lceil i \cdot g / h \rceil$ 의 KV-head 사용.

KV size: $g \cdot d_k$.

- $g = 1$: MQA
- $g = h$: MHA
- $g = 8$ (LLaMA-2 70B with $h = 64$): KV size $h/g = 8\times$ 절약

### 정의 6.4 — KV Cache 의 일반 식

Cache size for $T$ tokens, $L$ layers, $d_{\text{model}}$:
$$
\text{KV cache} = 2 \cdot L \cdot T \cdot \frac{d_{\text{model}} \cdot g}{h}
$$

(MHA: $g=h$, full $d_{\text{model}}$; MQA: $g=1$, $d_k = d_{\text{model}}/h$)

### 정의 6.5 — MQA-from-MHA Conversion (Ainslie 2023)

기존 MHA 모델을 MQA 로 변환:
1. KV head 를 mean-pool (또는 다른 aggregation)
2. 작은 fine-tuning (~5% original training compute)
3. Recover 거의 모든 성능

---

## 🔬 정리와 증명

### 정리 6.1 — KV Cache Memory Savings

MHA: $h \cdot d_k = d_{\text{model}}$
MQA: $1 \cdot d_k = d_{\text{model}}/h$
GQA: $g \cdot d_k = g \cdot d_{\text{model}}/h$

**Savings ratio**: $h/g$.
- MQA: $h\times$
- GQA-8 with $h=64$: $8\times$

### 정리 6.2 — Attention Computation 의 Equivalence

각 Q-head 의 attention computation:

MHA: $\text{Attn}(Q^{(i)}, K^{(i)}, V^{(i)})$
MQA: $\text{Attn}(Q^{(i)}, K, V)$ — 모든 head 가 same K, V

**Mathematical**:
$$
\text{out}_i^{MQA} = \text{softmax}(Q^{(i)} K^\top / \sqrt{d_k}) V
$$

각 Q-head 가 same KV space 에서 different perspective 로 query.

### 정리 6.3 — 표현력 분석

**Claim**: MQA 의 표현력 ≤ MHA, but 손실 작음.

**증명 sketch**:

MHA 가 $h$-fold KV diversity. MQA 는 single KV.

그러나:
- 각 Q-head 가 independent $W_Q^{(i)}$ → 다른 query subspace
- Same K, V 에서 다른 perspective: 여전히 다양한 attention pattern 가능
- 학습이 이 한계를 부분 보상

**실증** (Shazeer 2019):
- MHA → MQA: ~1-2% performance drop in NMT
- Acceptable for inference 가속

### 정리 6.4 — Inference Bottleneck Analysis

Generation 의 per-step cost:
- Compute: small ($O(T d)$ for new token)
- Memory bandwidth: dominant ($O(T d_{KV})$ for cache load)

$d_{KV}$ 가 dominant — MQA/GQA 가 $d_{KV}$ 를 $h \to 1$ 또는 $h \to g$ 로 감소.

**Concrete (LLaMA-2 70B, $T = 8K$)**:
- MHA: 5GB cache load per layer per token
- GQA-8: 0.6GB
- → 8× higher token/s for generation

### 정리 6.5 — Training vs Inference Trade-off

- **Training**: full computation, MQA/GQA 의 cost saving 작음 (compute-bound)
- **Inference**: memory bandwidth bound, MQA/GQA 의 절약 huge

→ MQA/GQA 의 설계 motivation 이 **inference 가속**.

### 정리 6.6 — GQA 의 Sweet Spot

- $g = h$ (MHA): full diversity, full cost
- $g = 1$ (MQA): minimal cost, possible quality drop
- $g = h/8$ (GQA-8): 거의 MHA quality, 8× cost saving

LLaMA-2 ablation: $g = 8$ 이 best trade-off.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — MQA 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)        # h Q-heads
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)        # 1 K-head
        self.W_V = nn.Linear(d_model, self.d_k, bias=False)        # 1 V-head
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)   # (B, h, T, d_k)
        K = self.W_K(x).unsqueeze(1)                                    # (B, 1, T, d_k)
        V = self.W_V(x).unsqueeze(1)                                    # (B, 1, T, d_k)
        
        # Broadcast K, V to all heads
        K = K.expand(-1, self.h, -1, -1)
        V = V.expand(-1, self.h, -1, -1)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)

# Test
torch.manual_seed(0)
mqa = MultiQueryAttention(d_model=64, num_heads=8)
x = torch.randn(1, 10, 64)
y = mqa(x)
print(f'MQA output: {y.shape}')

# Param 비교
mha_params = 4 * 64 * 64   # MHA: 4 d² (Q, K, V, O)
mqa_params = 64 * 64 + 64 * 8 + 64 * 8 + 64 * 64
print(f'MHA params: {mha_params}')
print(f'MQA params: {mqa_params}  (saving: {mha_params - mqa_params})')
```

### 실험 2 — GQA 구현

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.h_q = num_q_heads
        self.h_kv = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads
        
        self.W_Q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(num_q_heads * self.d_k, d_model, bias=False)
    
    def forward(self, x):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.h_q, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)
        
        # Repeat KV to match Q heads
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)

# Test
torch.manual_seed(0)
gqa = GroupedQueryAttention(d_model=64, num_q_heads=8, num_kv_heads=2)   # 4× KV save
x = torch.randn(1, 10, 64)
y = gqa(x)
print(f'GQA output: {y.shape}')

# KV cache size
print(f'\nKV cache analysis (T=8K, single layer, FP16):')
T = 8192; d_k = 8
print(f'MHA  (h_kv=8):  {2 * T * 8 * d_k * 2 / 1e6:.1f} MB')
print(f'GQA-2 (h_kv=2): {2 * T * 2 * d_k * 2 / 1e6:.1f} MB  (4x saving)')
print(f'MQA  (h_kv=1):  {2 * T * 1 * d_k * 2 / 1e6:.1f} MB  (8x saving)')
```

### 실험 3 — Inference Speed Comparison

```python
import time

# Generation step 시뮬레이션
def generation_step(attn_module, kv_cache, x_new):
    """한 token 생성 시 cost"""
    # Update cache (simplified)
    return attn_module(x_new)

torch.manual_seed(0)
T_max = 1024
d = 64
mha = nn.MultiheadAttention(d, 8, batch_first=True)
mqa = MultiQueryAttention(d, 8)

# 비교: T 가 클 때 generation 의 KV cache load cost 시뮬레이션
# (Real implementation 은 KV cache 의 explicit management 필요)
print('Conceptual KV cache load cost:')
for T in [128, 1024, 8192]:
    # MHA: load h × T × d_k × 2 (K, V) bytes
    mha_cache = 8 * T * 8 * 2 * 2   # h=8, d_k=8, FP16
    mqa_cache = 1 * T * 8 * 2 * 2   # h=1
    print(f'T={T:5d}: MHA cache={mha_cache} bytes, MQA={mqa_cache} bytes ({mha_cache/mqa_cache:.0f}x)')
```

### 실험 4 — MQA-from-MHA Conversion

```python
def convert_mha_to_mqa(mha_weights, num_heads, d_k):
    """기존 MHA 의 K, V weights 를 mean-pool 해서 MQA 로"""
    # mha_weights['W_K']: (d_model, h*d_k)
    h = num_heads
    W_K_mha = mha_weights['W_K']   # (d_model, h*d_k)
    W_V_mha = mha_weights['W_V']
    
    # Reshape and mean across heads
    d_model = W_K_mha.size(0)
    W_K_mha = W_K_mha.view(d_model, h, d_k)
    W_V_mha = W_V_mha.view(d_model, h, d_k)
    
    # Mean across heads
    W_K_mqa = W_K_mha.mean(dim=1)   # (d_model, d_k)
    W_V_mqa = W_V_mha.mean(dim=1)
    
    return W_K_mqa, W_V_mqa

# 시뮬레이션
torch.manual_seed(0)
W_K_mha = torch.randn(64, 8 * 8)   # d=64, h=8, d_k=8
W_V_mha = torch.randn(64, 8 * 8)

W_K_mqa, W_V_mqa = convert_mha_to_mqa({'W_K': W_K_mha, 'W_V': W_V_mha}, 8, 8)
print(f'Converted MQA W_K: {W_K_mqa.shape}')

# Then need fine-tuning (~5% original training) to recover quality
```

### 실험 5 — Param 수 비교

```python
def count_params(d_model, num_q_heads, num_kv_heads):
    d_k = d_model // num_q_heads
    Q_params = d_model * num_q_heads * d_k
    KV_params = 2 * d_model * num_kv_heads * d_k
    O_params = num_q_heads * d_k * d_model
    return Q_params + KV_params + O_params

d_model = 4096
h = 32
print(f'd_model={d_model}, num_q_heads={h}')
for variant_h_kv in [h, h//4, h//8, 1]:
    name = f'GQA-{variant_h_kv}' if variant_h_kv > 1 else 'MQA'
    if variant_h_kv == h: name = 'MHA'
    p = count_params(d_model, h, variant_h_kv)
    print(f'{name:6s} (h_kv={variant_h_kv:2d}): {p/1e6:.1f}M params')

# h=32 → MHA: 67M, GQA-8: 50M, GQA-4: 47M, MQA: 41M
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 채택

| Model | h_q | h_kv | KV savings |
|-------|-----|------|------------|
| LLaMA-2 7B | 32 | 32 (MHA) | 1× |
| LLaMA-2 70B | 64 | 8 (GQA-8) | 8× |
| Mistral 7B | 32 | 8 (GQA-4) | 4× |
| Mixtral 8x7B | 32 | 8 (GQA-4) | 4× |
| LLaMA-3 70B | 64 | 8 (GQA-8) | 8× |
| GPT-4 (estimate) | ? | ? (likely GQA) | - |

### 2. vLLM, TGI 의 inference 최적화

KV cache 가 inference 의 dominant memory:
- vLLM 의 PagedAttention: KV cache 의 page 단위 관리
- GQA 와 결합 시 더 큰 batch size 가능

### 3. Flash Attention 과 결합

- Flash Attention 2/3 가 GQA 지원
- 같은 algorithm, fewer KV head 만 처리
- 추가 가속

### 4. Long Context Inference

100K context with 70B GQA-8:
- KV cache: ~5GB (manageable on single A100/H100)
- MHA 였다면 40GB (single GPU 어려움)

→ GQA 가 long context production 의 직접 enabler.

### 5. Continued Pre-training 으로 GQA 추가

기존 MHA 모델에 GQA 추가:
1. K, V head 를 mean (또는 first) → fewer heads
2. Continued pretraining ~5% original compute
3. Quality 거의 보존

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 표현력 손실 작음 | Edge case 에서 일부 task quality ↓ |
| Q-head diversity 보존 | $h_q$ 도 줄이면 큰 손실 |
| Equal group size | Variable group size 도 가능 (rare) |
| Static grouping | Learned grouping 가능 |
| Attention 만 | FFN 의 grouped variants 도 있음 |

---

## 📌 핵심 정리

$$\boxed{\text{GQA: } h_q \text{ Q-heads share } g \text{ KV-groups, KV cache } \times h/g \text{ saving}}$$

| Variant | $h_q$ | $h_{kv}$ | KV cache | 표현력 손실 |
|---------|-------|----------|----------|------------|
| **MHA** | $h$ | $h$ | $h \cdot d_k$ | 0 (baseline) |
| **GQA-8** (with h=64) | 64 | 8 | $8 d_k$ | ~0.5% |
| **GQA-4** (with h=32) | 32 | 8 | $8 d_k$ | ~0.5% |
| **MQA** | $h$ | 1 | $d_k$ | ~1-2% |

| Use case | Choice |
|----------|--------|
| Inference 가속 critical | GQA-8 / MQA |
| Quality 절대적 | MHA |
| Mid-sized | GQA-4 |
| Production LLM | GQA-8 (LLaMA-2/3 표준) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LLaMA-2 70B 의 attention layer 의 parameter 수를 MHA, GQA-8 로 각각 계산하라. ($d=8192, h=64, d_k = 128$)

<details>
<summary>해설</summary>

**MHA 가정** ($h_{kv} = h_q = 64$):
- $W_Q$: $8192 \times 8192 = 67M$
- $W_K$: $8192 \times 8192 = 67M$
- $W_V$: $8192 \times 8192 = 67M$
- $W_O$: $8192 \times 8192 = 67M$
- Total: $268M$ per attention layer

**GQA-8** ($h_{kv} = 8$):
- $W_Q$: $8192 \times 8192 = 67M$
- $W_K$: $8192 \times (8 \times 128) = 8192 \times 1024 = 8.4M$ ← **8× smaller**
- $W_V$: $8.4M$
- $W_O$: $8192 \times 8192 = 67M$
- Total: $151M$ per attention layer

**Savings**: $268M \to 151M$ — about 44% per attention layer.

**전체 70B model** ($L = 80$ layer):
- MHA attention: $80 \times 268M = 21.4B$
- GQA attention: $80 \times 151M = 12.1B$
- 9.3B parameter savings → FFN, embedding 에 더 사용 가능

**KV cache** ($T = 8K$, BF16):
- MHA: $80 \times 8K \times 64 \times 128 \times 2 \times 2 = 21GB$
- GQA-8: $80 \times 8K \times 8 \times 128 \times 2 \times 2 = 2.6GB$ → **8× saving** ✓ $\square$

</details>

**문제 2** (심화): MQA 의 표현력 손실이 작은 이유를 정확히 분석하라. Q-head 의 다양성이 보존되는 mechanism 은? 어떤 task 에서 손실이 더 큰가?

<details>
<summary>해설</summary>

**MQA 의 표현력 분석**:

각 Q-head: $q^{(i)} = x W_Q^{(i)}$ — independent learnable projection.
Single K, V: $k = x W_K$, $v = x W_V$.

각 head 의 attention:
$$
\text{head}_i = \text{softmax}(q^{(i)} k^\top / \sqrt{d_k}) v
$$

**다양성 보존 mechanism**:

1. **Q-head 의 independent subspace**: 각 head 가 다른 $W_Q^{(i)}$ → 다른 query subspace
2. **Same KV space, different perspective**: 같은 K 에 대해 다른 query 가 다른 attention pattern
3. **W_O 의 mixing**: 각 head 출력이 concat 후 $W_O$ 로 fuse — "각 head 가 다른 정보 추출" 의 효과 보존

**예시**:

$h$ Q-heads with same K:
- Q-head 1: syntactic pattern (local dependency)
- Q-head 2: semantic pattern (long-range)
- ... 각각 같은 K 에 대해 다른 query 로 다른 attention 형성

**MHA 의 K 의 추가 자유도**:

각 head 가 own K → "different concept of similarity"
- Head 1: syntactic similarity
- Head 2: semantic similarity

MQA 는 single K → single similarity space.

**Loss 가 큰 task**:

1. **Multi-modal alignment**:
   - 다른 modality 사이 다른 alignment metric 필요
   - Single K 가 충분하지 않을 가능성

2. **Fine-grained retrieval**:
   - 매우 specific 한 pattern matching
   - Multiple "search keys" 필요

3. **Mathematical reasoning**:
   - 다양한 symbolic relationship
   - Q-head 다양성만으로 부족할 가능성

**Loss 가 작은 task**:

1. **Language modeling**: smooth, distributional patterns — Q-head 다양성으로 충분
2. **Translation**: local + semantic — well-served
3. **Code**: structural patterns

**실증**:

- BERT-large fine-tuning: MHA → MQA 시 GLUE 평균 1.5% 손실
- T5: 1.0% 손실
- LLaMA-style: <0.5% 손실 (large scale 이 손실 보상)

**Scale matters**:

- 작은 모델 (1B 이하): MQA loss 더 명확
- 큰 모델 (70B+): scale 의 implicit regularization 으로 손실 minimal
- → LLaMA-2 70B 의 GQA-8 채택 이 정당화

**GQA 의 sweet spot**:

GQA-$g$ with $g$ groups:
- $g = 1$ (MQA): 표현력 손실 max
- $g = h$ (MHA): 손실 0
- $g = 8$ (LLaMA-2 표준): "거의 MHA quality, 8x cost saving"

따라서 MQA 의 표현력 손실은 task-dependent + scale-dependent. Modern 큰 LLM 은 GQA-8 이 sweet spot. $\square$

</details>

**문제 3** (논문 비평): MQA 가 inference 가속에 huge gain 을 주지만 training 에서는 minimal gain 이다. Why? Inference 가 memory-bandwidth bound 인 본질을 분석하라.

<details>
<summary>해설</summary>

**Training vs Inference 의 fundamental 차이**:

**Training**:
- Forward + Backward (BP)
- Batch processing — 많은 sample 동시
- Compute-bound: matmul 이 dominant
- Memory: 큰 batch + activation checkpointing 으로 GPU 활용 ↑

**Inference (Generation)**:
- Forward only, sequential token generation
- Per-token: small Q (single token), full K, V cache
- Memory-bound: KV cache 의 load 가 dominant
- GPU underutilized (small new Q, sequential)

**Per-step cost 분석**:

Generation step at $T$ tokens:
- New Q: $1 \times d$ — small
- KV cache load: $T \times h_{kv} \times d_k$
- Attention: $1 \times T \times h_q \times d_k$ — small compute
- FFN: $1 \times d \times 4d$ — small

→ **KV cache load 가 critical path**. GPU 의 memory bandwidth 가 bottleneck.

**MQA/GQA 의 Memory Bandwidth Saving**:

A100 의 HBM bandwidth: 1.5 TB/s.

Per generation step (per layer):
- MHA: $T \cdot h \cdot d_k \cdot 2 = T \cdot d$ bytes for K + V load
- MQA: $T \cdot 1 \cdot d_k \cdot 2 = T \cdot d/h$ bytes
- GQA-8: $T \cdot 8 \cdot d_k \cdot 2 = T \cdot d \cdot 8/h$ bytes

For $T = 8K$, $d = 8K$ (LLaMA-2 70B):
- MHA: 64MB load per layer
- GQA-8: 8MB

Time per layer:
- MHA: $64MB / 1.5 TB/s \approx 40 \mu s$
- GQA-8: $5 \mu s$

80 layers: $40 \times 80 = 3.2 ms$ (MHA) vs $0.4 ms$ (GQA-8) per token.

→ **8× faster** generation!

**Training 에서는 왜 gain 이 작은가**:

Training 의 forward:
- Batch processing: $B$ samples concurrently
- $h$ Q-head 가 동시에 K 와 attend — compute-bound
- KV cache 가 forward 안에서만 사용 (no cumulative load)

Backward:
- 더 큰 compute (gradient computation)
- Memory 는 activation checkpointing 으로 관리

→ Training 의 bottleneck 은 **compute**, MQA/GQA 가 compute 절약 적음.

**Real-world Implications**:

1. **Inference Service**:
   - vLLM 같은 framework 가 GQA 와 결합 시 batch size 대폭 증가
   - Cost per token 절감

2. **Long Context**:
   - 100K+ context 가 GQA 없이 GPU 메모리 한계
   - GQA 가 production long-context 의 enabler

3. **Edge Deployment**:
   - Phone, embedded 에서 generation
   - MQA 의 cache 절약이 critical

4. **Training 의 minor gain**:
   - Param 수 감소 (~10-20%) — small
   - Forward / backward time 거의 같음
   - 그러나 long-context training (gradient checkpointing 과 결합) 에서 약간 도움

**미래**:

- Modern LLM 의 표준이 GQA
- Inference 의 latency, throughput 모두 개선
- 학습-추론 paradigm: inference-friendly architecture 가 점점 중요

**근본 통찰**:

**Hardware bottleneck** 이 algorithm choice 직접 결정. Training 과 inference 의 다른 bottleneck 이 다른 architecture optimization 요구. MQA/GQA 는 **inference-aware design** 의 prime example.

미래 architecture (Mamba, RWKV 등) 도 같은 lens — "inference 의 memory bandwidth bottleneck 을 어떻게 해결?" 의 답. $\square$

</details>

---

<div align="center">

[◀ 이전](./05-flash-attention.md) | [📚 README](../README.md) | [다음 ▶](../ch6-modern-architectures/01-bert.md)

</div>
