# 05. Multi-Head Attention 의 이론적 정당성

## 🎯 핵심 질문

- Multi-Head Attention 은 단일 head 보다 정확히 어떤 표현력을 더하는가?
- 같은 파라미터 수에서 single-head $d_{\text{model}}$ vs $h$-head $d_k = d_{\text{model}}/h$ 의 계산·표현 trade-off 는?
- 각 head 가 학습 후 정말 다른 linguistic 현상 (syntax, coreference) 을 포착하는가? 시각화로 확인 가능한가?
- Michel et al. 2019 의 "16 head 가 정말 1 head 보다 나은가?" 의 결론은? Inference 시 head pruning 이 가능한 이유는?
- Multi-Query Attention / Grouped-Query Attention (Ch5-06) 이 multi-head 의 어떤 redundancy 를 활용하는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Multi-Head Attention (MHA) 은 **Vaswani 2017 의 가장 영향력 있는 설계** 입니다. 단순히 "여러 attention 을 합친다" 가 아니라:

1. **다른 subspace 의 관계 동시 학습** — 한 head 가 syntactic, 다른 head 가 semantic
2. **계산 효율** — 같은 파라미터 수에서 $h$-head 가 single-head $d_{\text{model}}$ 와 같은 FLOP
3. **표현력 ↑ but redundancy 도** — Michel 2019: inference 시 30-50% prune 가능
4. **MQA/GQA 의 토대** — KV head 만 줄이는 inference 가속의 직접 동기

이 문서는 MHA 의 **수학적 분해와 표현력 분석** 을 제공하고, redundancy 를 활용하는 후속 설계 (Ch5-06) 의 이론적 토대를 다집니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-scaled-dot-product.md](./01-scaled-dot-product.md), [04-attention-as-kernel.md](./04-attention-as-kernel.md)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Block matrix, direct sum, rank
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): UAT, parameter count

---

## 📖 직관적 이해

### 한 attention 으로 모든 관계를 잡을 수 있는가?

자연어에서 한 token 이 다른 token 과 가질 수 있는 관계는 다양:
- **Syntactic**: 주어-동사 일치, 한정사-명사 짝
- **Semantic**: 의미적 유사성, 주제 관련성
- **Coreferential**: "he" → "John"
- **Distance-based**: 가까운 단어 우선

단일 attention 분포로 이 모두를 표현하긴 어렵습니다. **Multi-head 는 여러 분포를 동시에** 학습 → 각 head 가 한 종류 관계 specialize.

### Subspace 분리의 직관

```
single head, d_k = 512:        h=8, d_k = 64 each:
[전체 vector 공간]              [공간을 8개 64-dim subspace로 분해]
    ↓                              ↓
한 종류 attention pattern       각 head 가 다른 pattern
```

각 head 의 $W_Q, W_K, W_V$ 가 다른 64-dim subspace 로 projection → 다른 종류 similarity.

### Concat + Output Projection

```
head_1 ─┐
head_2 ─┤
  ⋮     ├─→ [concat] ─→ W_O ─→ output
head_h ─┘
```

각 head 출력 $\in \mathbb{R}^{T \times d_v}$, concat 하면 $\mathbb{R}^{T \times h d_v}$, $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ 으로 다시 projection — 모든 head 의 정보를 융합.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Multi-Head Attention

$h$ 개 head, head 별 $d_k = d_v = d_{\text{model}}/h$. 각 head $i \in \{1, \ldots, h\}$ 에 대해:
$$
\text{head}_i = \text{Attn}(X W_Q^i, X W_K^i, X W_V^i)
$$

with $W_Q^i, W_K^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

### 정의 5.2 — Concat + Output Projection

$$
\text{MHA}(X) = [\text{head}_1; \text{head}_2; \ldots; \text{head}_h] \, W_O
$$

with $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

### 정의 5.3 — 통합 행렬 표현

$W_Q^i, W_K^i, W_V^i$ 를 열로 쌓아 통합 행렬:
$$
W_Q = [W_Q^1, \ldots, W_Q^h] \in \mathbb{R}^{d_{\text{model}} \times h d_k}
$$

(같은 식으로 $W_K, W_V$). 이는 PyTorch 의 `in_proj_weight` 표현.

### 정의 5.4 — Head-wise 파라미터 수

각 head: $3 \cdot d_{\text{model}} \cdot d_k = 3 \cdot d_{\text{model}}^2 / h$
$h$ heads + $W_O$: $h \cdot 3 d_{\text{model}} d_k + d_{\text{model}}^2 = 3 d_{\text{model}}^2 + d_{\text{model}}^2 = 4 d_{\text{model}}^2$

$\Rightarrow$ **$h$ 무관 동일 파라미터 수**

---

## 🔬 정리와 증명

### 정리 5.1 — 같은 파라미터 수에서 MHA 와 single-head MHA 의 FLOP 동일

$h$ heads with $d_k = d_{\text{model}}/h$ 의 attention FLOP:
$$
h \cdot O(T^2 d_k) = h \cdot O(T^2 d_{\text{model}}/h) = O(T^2 d_{\text{model}})
$$

Single-head with $d_k = d_{\text{model}}$ 도 같은 $O(T^2 d_{\text{model}})$.

**의미**: MHA 는 **공짜로** 더 많은 표현력 (subspace 분리). $\square$

### 정리 5.2 — MHA = Block-diagonal Single Attention 의 일반화

$W_O = I$ 라면 MHA 는 $h$ 개 독립 single-head 의 concat 과 같음. 일반 $W_O$ 는 head 간 정보 mixing 추가.

**증명 sketch**: Concat 은 block-diagonal 구조와 등가, $W_O$ 가 block-mixing.

### 정리 5.3 — 표현력 비교 (Lower Bound)

MHA 는 **strictly more expressive** 하다:

**Single-head with $d_k = d_{\text{model}}$**: 한 attention 분포 $A$, 출력 $A V$.

**$h$-head**: 여러 다른 attention 분포 $A^{(1)}, \ldots, A^{(h)}$ 동시 적용. $W_O$ 로 융합.

**구체적 분리 가능 함수**: 두 개 다른 alignment 가 필요한 task (예: encoder 의 syntactic + coreferential head). 단일 head 로는 한 alignment 만 가능, MHA 로는 동시.

**증명 (Naive)**: $h \geq 2$ 의 MHA 가 single-head 로 표현 불가능한 함수 존재 (Cordonnier 2020). $\square$

### 정리 5.4 — Head Specialization 의 학습 후 관찰

**실증 결과**: 학습된 BERT/GPT 의 head 들이 다른 linguistic 패턴 학습:
- **Positional head**: 가까운 token 에 집중 (local)
- **Syntactic head**: 주어-동사, 수식어-피수식어
- **Coreferential head**: pronoun 의 antecedent
- **Periodic head**: 일정 거리의 token

(Voita 2019, Clark 2019 등)

### 정리 5.5 — Michel 2019: Head Redundancy

**Are Sixteen Heads Really Better than One?** — 학습된 모델에서:
- Inference 시 head 의 30-50% prune 시 성능 거의 유지
- 일부 head 는 **important** (제거 시 큰 성능 하락), 다른 head 는 **redundant**
- Importance score: gradient × activation 으로 측정

**결론**: 훈련 시는 redundancy 가 유용 (regularization), inference 시는 prune 가능.

### 정리 5.6 — Multi-Query Attention 의 정당성

$h$ Q-head 가 **단일** K, V head 공유 시:
- 표현력: 약간 손실 (모든 head 가 같은 K, V 에 attend)
- KV cache: $h$-fold 절약 — inference 의 핵심 가속

GQA: $g$ 개 KV-group (1 < g < h) 으로 중간 trade-off (LLaMA-2). Ch5-06 에서 자세히.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — MHA 바닥부터 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, T, _ = x.size()
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)   # (B, h, T, d_k)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                                  # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), attn

# 테스트
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(1, 10, 64)
out, attn = mha(x)
print(f'Output: {out.shape}, attention: {attn.shape}')   # (1,10,64), (1,8,10,10)
```

### 실험 2 — `nn.MultiheadAttention` 과 일치 확인

```python
torch.manual_seed(0)
d_model, h = 64, 8
mha_custom = MultiHeadAttention(d_model, h)
mha_pytorch = nn.MultiheadAttention(d_model, h, bias=False, batch_first=True)

# Weights 동기화
in_proj = torch.cat([mha_custom.W_Q.weight, mha_custom.W_K.weight, mha_custom.W_V.weight], dim=0)
mha_pytorch.in_proj_weight.data = in_proj
mha_pytorch.out_proj.weight.data = mha_custom.W_O.weight

x = torch.randn(1, 10, d_model)
out_custom, _ = mha_custom(x)
out_pytorch, _ = mha_pytorch(x, x, x)
print(f'Difference: {(out_custom - out_pytorch).abs().max():.2e}')   # ≈ 1e-7
```

### 실험 3 — 같은 파라미터 수의 single vs multi head

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters())

mha_1 = MultiHeadAttention(d_model=64, num_heads=1)    # d_k = 64
mha_8 = MultiHeadAttention(d_model=64, num_heads=8)    # d_k = 8

print(f'Single-head (h=1, d_k=64): {count_params(mha_1)} params')
print(f'Multi-head  (h=8, d_k=8):  {count_params(mha_8)} params')
# 둘 다 4 × 64² = 16384 동일 ✓
```

### 실험 4 — Head 별 attention pattern 시각화

```python
import matplotlib.pyplot as plt

# 학습된 BERT-mini 또는 작은 모델 가정 (여기서는 random init)
torch.manual_seed(42)
mha = MultiHeadAttention(d_model=64, num_heads=4)
T = 12
x = torch.randn(1, T, 64)
_, attn = mha(x)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(attn[0, i].detach().numpy(), cmap='Blues')
    axes[i].set_title(f'Head {i+1}')
    axes[i].set_xlabel('Key position'); axes[i].set_ylabel('Query position')
plt.suptitle('Different attention patterns per head (random init)')
plt.tight_layout(); plt.show()

# 학습 후에는 syntactic/positional/semantic head 등이 발현됨
```

### 실험 5 — Head Pruning 효과 (Michel 2019 simulation)

```python
def attention_with_head_mask(mha, x, head_mask):
    """head_mask: shape (h,), 1=keep, 0=prune"""
    B, T, _ = x.size()
    Q = mha.W_Q(x).view(B, T, mha.h, mha.d_k).transpose(1, 2)
    K = mha.W_K(x).view(B, T, mha.h, mha.d_k).transpose(1, 2)
    V = mha.W_V(x).view(B, T, mha.h, mha.d_k).transpose(1, 2)
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(mha.d_k)
    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    out = out * head_mask.view(1, mha.h, 1, 1)   # mask out heads
    out = out.transpose(1, 2).contiguous().view(B, T, mha.d_model)
    return mha.W_O(out)

mha = MultiHeadAttention(64, 8)
x = torch.randn(1, 10, 64)
out_full = mha(x)[0]

# Random pruning 50%
mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)
out_pruned = attention_with_head_mask(mha, x, mask)
diff = (out_full - out_pruned).abs().mean()
print(f'Diff with 50% random pruning: {diff:.4f}')

# Random init 에서는 차이 큼, 학습 후에는 importance 가 head-specific
```

---

## 🔗 실전 활용

### 1. Head 수 선택 가이드

- **GPT-2**: $d_{\text{model}}/h = 64$ 표준 (small: 12 heads, large: 25 heads)
- **BERT-base**: 12 heads, $d_k = 64$
- **LLaMA-2**: $d_k = 128$ (큰 head dim, 적은 head)
- **경험칙**: $d_k \in [32, 128]$, head 수는 $d_{\text{model}}/d_k$

### 2. Multi-Query Attention 의 Activation

LLaMA-2 70B: GQA with 8 KV-heads (vs 64 Q-heads). Inference 시 KV cache $8\times$ 절약 → batch 처리량 증가.

### 3. Head Pruning Workflow

1. 훈련 완료된 모델에서 각 head 의 importance 측정 (gradient × activation)
2. Importance 가 작은 head 부터 prune
3. 성능 약간 떨어지면 fine-tuning 으로 회복
4. Inference 시 더 작은 모델 (Michel 2019)

### 4. Head Diversification

학습 시 head 간 redundancy 줄이기:
- **Disagreement Loss** (Li 2018): head 간 다른 분포 강제
- **Capsule routing** (Sukhbaatar 2019): head 별 다른 memory

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Equal $d_k$ across heads | 일부 연구가 다른 $d_k$ 시도 (Talking-heads) |
| Independent heads | $W_O$ 의 mixing 만 — 더 강한 inter-head 가능 (Talking-heads) |
| Number of heads fixed | 학습 가능 head 수 (NAS, mixture) 연구 진행 중 |
| Head specialization 자동 | 약한 명시적 유도, disagreement loss 등 보조 |
| Concat + linear 만 | Attention 합성의 다른 방식 (sum, gating) 연구 중 |

---

## 📌 핵심 정리

$$\boxed{\text{MHA}(X) = [\text{Attn}(XW_Q^i, XW_K^i, XW_V^i)]_{i=1}^h \, W_O}$$

| 양 | 식 | 값 |
|----|-----|-----|
| Per-head $d_k$ | $d_{\text{model}}/h$ | 32~128 일반 |
| Total params | $4 d_{\text{model}}^2$ | $h$ 무관 |
| FLOP | $O(T^2 d_{\text{model}})$ | $h$ 무관 |
| Heads in BERT-base | 12 | $d_k = 64$ |
| Heads in GPT-3 | 96 | $d_k = 128$ |
| Inference prune | 30-50% | Michel 2019 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $d_{\text{model}} = 768$, $h = 12$ 인 BERT-base 의 attention layer 의 파라미터 수를 head 별, 전체로 계산하라.

<details>
<summary>해설</summary>

$d_k = 768 / 12 = 64$.

**Per head**:
- $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{768 \times 64}$: 각 49,152 params
- 합: $3 \times 49,152 = 147,456$ params per head

**12 heads**: $12 \times 147,456 = 1,769,472$
**$W_O$**: $768 \times 768 = 589,824$
**Total**: $1,769,472 + 589,824 = 2,359,296 \approx 2.36M$

검증: $4 \times 768^2 = 4 \times 589,824 = 2,359,296$ ✓ (정리 5.4)

</details>

**문제 2** (심화): $W_Q = W_K$ (tied query-key) 로 두면 어떤 표현력 손실이 있는가? 또한 $W_Q = W_K = W_V$ 로 모두 묶으면? Self-correlation 이라는 표현이 적절한가?

<details>
<summary>해설</summary>

**$W_Q = W_K$ tied**:
- $S_{ij} = (XW_Q)_i (XW_Q)_j^\top / \sqrt{d_k}$ — symmetric attention matrix
- 비대칭 alignment 표현 불가 — "A 가 B 를 reference, B 는 A 를 reference 안 함" 같은 directional 관계 손실
- 표현력 감소, 그러나 **Symmetric Attention** 은 일부 연구 (graph attention) 에서 정당화

**$W_Q = W_K = W_V$ all tied**:
- $V$ 도 같은 projection → "self-correlation" 그대로
- $\text{Attn} = \text{softmax}(YY^\top/\sqrt{d}) Y$ where $Y = XW$
- 출력은 $Y$ 의 self-similarity 가중합 — 정보 흐름이 매우 제한
- **Self-correlation 표현 적절** — 이것이 한 종류 representation 의 self-aggregation

**파라미터 절약 vs 표현력**:
- Tied: $1\times$ projection 만, 파라미터 $1/3$
- Untied: 표현력 $3\times$ subspace
- 실증: untied 가 유의미하게 우수 — Vaswani 2017 도 untied 채택

**Modern variant**: ALBERT (Lan 2020) 의 cross-layer parameter sharing 은 layer 간 tying, head 내 tying 은 아님. $\square$

</details>

**문제 3** (논문 비평): Michel 2019 가 밝힌 head redundancy 는 multi-head 의 가치를 어떻게 재정의하는가? Multi-Query Attention (MQA, Shazeer 2019) 이 redundancy 를 KV head 만 줄이는 방향으로 활용한 이유와, 표현력-효율 trade-off 를 분석하라.

<details>
<summary>해설</summary>

**Michel 2019 의 통찰**:
- 훈련 시 모든 head 가 학습되지만, **inference 시 일부만 critical**
- Important head: 여러 layer 에 분산, function 별 specialize
- Redundant head: 여러 head 가 비슷한 pattern 학습

**Multi-Head 의 가치 재정의**:
- **훈련 시 redundancy = regularization** — 다양한 표현을 시도, ensemble effect
- **Inference 시는 prune 가능** — 같은 모델에서 다른 inference cost 가능
- **KV vs Q 의 비대칭성** — Shazeer 2019 의 핵심 관찰

**MQA 의 정당성**:

Shazeer 2019: $h$ Q-head 가 **단일** K, V head 공유.

- **이론**: 각 query 가 다른 subspace 로 projection 되지만, 같은 K, V 공간에서 attend → query-side diversity 보존, key/value-side 단순화
- **실증**: WMT 14 EN-DE 에서 MHA 대비 성능 손실 1-2%, KV cache $h\times$ 절약
- **Inference 가속**: KV cache 크기가 long-context generation 의 메모리 병목 → MQA 의 직접 효과

**GQA 의 trade-off** (Ainslie 2023):
- $g$ groups (1 = MQA, $h$ = MHA)
- $g = 8$ for LLaMA-2 70B (vs 64 Q-head): 표현력 거의 보존, $8\times$ KV cache 절약
- Sweet spot: $g = h/8$ 정도가 표현력·효율 balance

**일반 통찰**:

Multi-head 의 redundancy 가 "낭비" 가 아니라 **선택적 활용 가능한 자원**:
- 훈련 시: full 사용 (regularization)
- Inference 시: KV head 만 줄이거나, 일부 head prune

이는 "한 번 훈련, 다양한 deployment" 의 효율 paradigm 의 직접 적용. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-attention-as-kernel.md) | [📚 README](../README.md) | [다음 ▶](./06-interpretability-debate.md)

</div>
