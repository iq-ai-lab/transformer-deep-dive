# 01. Positional Encoding 의 필요성 — Permutation Equivariance

## 🎯 핵심 질문

- Self-attention 은 왜 permutation-equivariant 인가? — 수학적으로 정확히 무엇을 의미하는가?
- 순서 정보 없는 NN 이 sequence task 에 부적합한 이유는?
- PE 를 어떻게 주입하는가 — sum vs concatenate 의 trade-off?
- 학습 가능 (Learned) vs 고정 (Sinusoidal) PE 의 inductive bias 차이는?
- PE 가 모든 layer 에서 활용되는가, 아니면 첫 layer 만?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Positional Encoding 은 **Transformer 의 most underrated component** 입니다. 없으면:

1. **Sequence task 작동 불가** — "John loves Mary" 와 "Mary loves John" 구분 못함
2. **Attention 의 permutation-equivariance 가 한계** — 순서가 의미를 결정하는 task 에서 무력
3. **Inductive bias 부재** — RNN/CNN 의 자연스러운 spatial bias 가 없음

PE 의 선택이 모델의 long-context 처리, extrapolation 능력, computational efficiency 를 모두 결정합니다 (RoPE, ALiBi, Ch3-05).

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md) — Permutation equivariance 정리
- 군론 (선택): Permutation group $S_n$, equivariance vs invariance

---

## 📖 직관적 이해

### Permutation Invariance vs Equivariance

```
Invariant (집합):     f(perm(x)) = f(x)         예: sum(x), max(x)
Equivariant (sequence): f(perm(x)) = perm(f(x))  예: self-attention
```

**Self-attention 은 equivariant**: token 순서를 바꾸면 output 순서도 같이 바뀜. 그러나 **token 자체의 representation 은 같음** — 즉 **위치 정보가 없음**.

### "Bag of words" 문제

Self-attention 의 결과는 "bag of words" representation:
- "John loves Mary"
- "Mary loves John"

같은 token set → 같은 representation set (순서만 다름) → semantic 차이 못 잡음.

### Solution: Add positional information

```
x_t  = token embedding
pe_t = positional embedding (위치 t 에 따라)
input = x_t + pe_t   (또는 concat)
```

각 token 의 representation 이 위치-dependent → attention 이 위치 정보 활용 가능.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Permutation Matrix

$P \in \{0, 1\}^{T \times T}$ 가 **permutation matrix** 이려면 각 행과 열에 정확히 하나의 1, 나머지 0.

$P^\top P = I$ (orthogonal).

### 정의 1.2 — Permutation Equivariance

함수 $f: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ 가 **permutation-equivariant** 이려면:
$$
f(PX) = P f(X) \quad \forall P
$$

### 정의 1.3 — Positional Encoding

각 위치 $t \in \{1, \ldots, T\}$ 에 vector $\text{PE}_t \in \mathbb{R}^d$ 할당.

**Additive PE**:
$$
\tilde{x}_t = x_t + \text{PE}_t
$$

**Concatenated PE**:
$$
\tilde{x}_t = [x_t; \text{PE}_t] \in \mathbb{R}^{d + d_{\text{pe}}}
$$

### 정의 1.4 — Absolute vs Relative PE

- **Absolute**: $\text{PE}_t$ 가 위치 $t$ 의 함수 (Sinusoidal, Learned)
- **Relative**: 두 위치의 차이 $i - j$ 의 함수 (Shaw 2018, RoPE, ALiBi)

### 정의 1.5 — PE 의 차원

$\text{PE} \in \mathbb{R}^{T_{\max} \times d}$ — 모든 위치의 PE 를 행렬로.

---

## 🔬 정리와 증명

### 정리 1.1 — Self-Attention 의 Permutation Equivariance

$$
\text{SelfAttn}(PX) = P \, \text{SelfAttn}(X)
$$

**증명**: $Q' = (PX) W_Q = P(XW_Q) = PQ$, 마찬가지로 $K' = PK, V' = PV$.

$$
S' = Q'(K')^\top / \sqrt{d_k} = PQ K^\top P^\top / \sqrt{d_k} = P S P^\top
$$

Row-wise softmax 는 permutation 과 commute (row 안의 entry 순서가 바뀌면 softmax 도 같은 entry 의 함수):

엄밀히, $\text{softmax}(P A P^\top)$ 의 $(i, j)$ 성분은 $\text{softmax}(A)_{P^{-1}i, P^{-1}j}$. 즉 $\text{softmax}(P A P^\top) = P \text{softmax}(A) P^\top$.

$$
A' = \text{softmax}(S') = P \text{softmax}(S) P^\top = P A P^\top
$$

$$
\text{Attn}' = A' V' = P A P^\top P V = P A V = P \text{Attn} \quad \square
$$

### 정리 1.2 — PE 추가 후 Permutation Sensitivity

$\tilde{X} = X + \text{PE}$ 에 대해:
$$
\text{Attn}(P \tilde{X}) = \text{Attn}(P X + P \text{PE}) \neq P \text{Attn}(\tilde{X})
$$

(단, $P \text{PE} \neq \text{PE}$ 일 때 — 즉 PE 가 위치별로 다를 때)

**의미**: PE 가 permutation 을 깨뜨림 → 순서 정보 활용 가능.

### 정리 1.3 — Sum vs Concat 의 표현력

**Sum**: 더 간결, 같은 차원 유지. 단점: PE 와 token embedding 이 한 vector 로 섞임 — 모델이 분리 어려움.

**Concat**: 명확한 분리. 단점: 차원 증가, 첫 attention 에서 $W_Q, W_K$ 가 두 부분을 다르게 다룸 (학습 어려움).

**경험**: Sum 이 표준 (GPT, BERT). 충분한 차원에서 모델이 자연스럽게 분리 학습.

**Tsai 2019 분석**: $W_Q, W_K$ 의 행을 두 부분 ($d_{\text{tok}}, d_{\text{pe}}$) 으로 분리하면 sum 과 concat 이 등가.

### 정리 1.4 — PE 가 모든 layer 에서 영향

PE 는 입력 layer 에서 추가, 그러나 **residual connection** 으로 모든 layer 에서 propagate:
$$
h^{(l)} = h^{(0)} + \sum_{k} \Delta_k = X + \text{PE} + \sum_k \Delta_k
$$

따라서 깊은 layer 에서도 PE 정보 유지. 단, 학습 시 PE 정보가 부분적으로 "삼켜질" 수 있음 — 이것이 RoPE 같은 layer-wise PE injection 의 동기.

### 정리 1.5 — Learned vs Fixed 의 Inductive Bias

- **Learned PE**: data-driven, 학습 데이터 분포에 최적화. 단점: extrapolation 불가 (max length 고정), 새 위치 학습 필요
- **Fixed (Sinusoidal)**: 임의 위치 generalization (수학적 정의), 약간의 inductive bias (smooth periodic). 단점: data-specific 최적화 ↓

각각의 strength 가 다른 task 에 적합.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Permutation Equivariance 검증

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)

class SelfAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]

T, d = 5, 16
attn = SelfAttn(d, 4)
x = torch.randn(1, T, d)

# Permutation
perm_idx = torch.randperm(T)
P = torch.zeros(T, T)
for i, p in enumerate(perm_idx):
    P[i, p] = 1
x_perm = (P @ x.squeeze()).unsqueeze(0)

# Self-attn
y_orig = attn(x)
y_perm = attn(x_perm)

# Permutation equivariance check
y_orig_perm = (P @ y_orig.squeeze()).unsqueeze(0)
diff = (y_perm - y_orig_perm).abs().max()
print(f'Permutation equivariance: max diff = {diff:.6f}  (should be ~0)')
```

### 실험 2 — PE 추가 시 Permutation 깨짐

```python
# Random PE
pe = torch.randn(T, d) * 0.1   # Position-dependent

x_with_pe = x + pe.unsqueeze(0)
x_perm_with_pe = x_perm + pe.unsqueeze(0)   # PE 는 같은 순서 (위치별 fix)

y_orig_pe = attn(x_with_pe)
y_perm_pe = attn(x_perm_with_pe)

y_orig_perm_pe = (P @ y_orig_pe.squeeze()).unsqueeze(0)
diff_pe = (y_perm_pe - y_orig_perm_pe).abs().max()
print(f'With PE (different): max diff = {diff_pe:.4f}  (large → equivariance broken)')
```

### 실험 3 — Sum vs Concat 비교

```python
# Sum-based
class SumPE(nn.Module):
    def __init__(self, T_max, d):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(T_max, d) * 0.02)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

# Concat-based
class ConcatPE(nn.Module):
    def __init__(self, T_max, d_model, d_pe):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(T_max, d_pe) * 0.02)
        self.proj = nn.Linear(d_model + d_pe, d_model)
    def forward(self, x):
        T = x.size(1)
        pe = self.pe[:T].unsqueeze(0).expand(x.size(0), -1, -1)
        return self.proj(torch.cat([x, pe], dim=-1))

sum_pe = SumPE(100, 16)
concat_pe = ConcatPE(100, 16, 8)

x = torch.randn(1, 10, 16)
print(f'Sum output shape:    {sum_pe(x).shape}')
print(f'Concat output shape: {concat_pe(x).shape}')
print(f'Sum params:    {sum(p.numel() for p in sum_pe.parameters())}')
print(f'Concat params: {sum(p.numel() for p in concat_pe.parameters())}')
```

### 실험 4 — Bag-of-Words 문제 시연

```python
# 같은 token set, 다른 순서
torch.manual_seed(42)
tokens_a = torch.randn(1, 5, 16)
tokens_b = tokens_a[:, [4, 3, 2, 1, 0]]   # reverse order

attn = SelfAttn(16, 4)

# Without PE
y_a = attn(tokens_a)
y_b = attn(tokens_b)
y_a_set = y_a.sort(dim=1).values
y_b_set = y_b.sort(dim=1).values
print(f'Without PE - sorted output difference: {(y_a_set - y_b_set).abs().max():.6f}')
# 정렬 후 같음 → 같은 set, 다른 순서로 같은 representation set

# With PE
pe = torch.randn(5, 16) * 0.5
y_a_pe = attn(tokens_a + pe.unsqueeze(0))
y_b_pe = attn(tokens_b + pe.unsqueeze(0))
y_a_pe_set = y_a_pe.sort(dim=1).values
y_b_pe_set = y_b_pe.sort(dim=1).values
print(f'With PE - sorted output difference: {(y_a_pe_set - y_b_pe_set).abs().max():.4f}')
# 정렬 후에도 다름 → 순서가 의미를 갖게 됨
```

### 실험 5 — Layer 깊이 별 PE 정보 유지

```python
# 깊은 layer 에서 PE 정보가 얼마나 보존되는지

class Transformer(nn.Module):
    def __init__(self, num_layers, d, h):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True)
            for _ in range(num_layers)
        ])
    def forward(self, x, return_intermediate=False):
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            if return_intermediate:
                intermediates.append(x.clone())
        return (x, intermediates) if return_intermediate else x

torch.manual_seed(0)
model = Transformer(6, 32, 4)
x = torch.randn(1, 8, 32)
pe = torch.randn(8, 32) * 0.5

# PE 의 첫 layer 영향 → 깊은 layer 까지 propagate
_, intermediates = model(x + pe.unsqueeze(0), return_intermediate=True)

# 각 layer 에서 PE 와의 cosine similarity (PE 정보가 보존되는 척도)
for l, h in enumerate(intermediates):
    # 위치별 hidden 의 평균 norm 기준으로 PE 와 similarity
    h_mean = h.mean(0).detach()
    sim = F.cosine_similarity(h_mean.flatten(0), pe.flatten(0), dim=0)
    print(f'Layer {l+1}: similarity with PE = {sim:.4f}')
```

---

## 🔗 실전 활용

### 1. PE 의 layer 별 transparent 적용

원래 Vaswani: 첫 layer 만 PE 추가, residual 로 propagate.

**Modern variants**:
- RoPE: 매 attention layer 에서 PE 적용 — Q, K 에 회전 (Ch3-05)
- ALiBi: attention bias 로 매 layer 에서 distance 추가
- 효과: layer 깊이에서도 PE 정보 강화

### 2. PE 의 학습 가능성 trade-off

- BERT: learned PE — 512 token 한계
- GPT-2: learned PE — 1024 token 한계
- T5: relative PE (bucketed) — 더 긴 sequence
- LLaMA: RoPE — 좋은 extrapolation

### 3. Long Context 에서의 PE 선택

- 8K context: learned 도 가능 (충분한 데이터)
- 32K-100K context: RoPE / ALiBi 필수
- Million-token: ALiBi / NTK-aware RoPE / YaRN

### 4. Padding 과 PE

가변 길이 batch 에서:
- Padding token 의 PE 도 추가 (그러나 attention mask 로 제거)
- Position 0 부터 시작할지, padding 제외 후 시작할지 — 모델별 convention

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Discrete position $t \in \mathbb{Z}_+$ | Continuous PE (audio, time-series) 도 가능 |
| 1D sequence | 2D (image), 3D (video) 는 별도 PE |
| Position-only | Speaker, modality 등 추가 정보 가능 |
| Additive | Multiplicative, gating 도 시도 |

---

## 📌 핵심 정리

$$\boxed{\text{Self-Attn 은 permutation-equivariant} \Rightarrow \text{PE 필수}}$$

| 변형 | 정의 | 채택 | Pros | Cons |
|------|------|------|------|------|
| **Learned** | $\text{PE} \in \mathbb{R}^{T_{\max} \times d}$ | BERT, GPT-2 | Data-driven | Max length 고정 |
| **Sinusoidal** | $\sin/\cos$ 함수 | Vaswani 2017 | Extrapolation | Less optimized |
| **Relative (Shaw)** | $i - j$ 함수 | T5 | Local pattern | More compute |
| **RoPE** | 회전 행렬 | LLaMA | Auto relative + extrap | 이해 어려움 |
| **ALiBi** | Linear bias $-m\|i-j\|$ | BLOOM | Best extrapolation | Less expressive |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Self-attention 의 permutation equivariance 를 행렬 표기로 1줄 증명하라. (정리 1.1 의 압축 버전)

<details>
<summary>해설</summary>

$Q' = PQ$, $K' = PK$, $V' = PV$ (linear projection 적용 가능).

$\text{Attn}' = \text{softmax}(Q'(K')^\top/\sqrt{d}) V' = \text{softmax}(PQK^\top P^\top/\sqrt{d}) PV = P \text{softmax}(QK^\top/\sqrt{d}) P^\top P V = P \text{Attn}$.

(softmax row-wise + $P^\top P = I$) $\square$

</details>

**문제 2** (심화): PE 가 sum (additive) 으로 추가되는 것은 학습 시 token embedding 과 PE 가 한 vector 로 섞이는 것이다. 모델이 두 정보를 어떻게 분리 학습하는지 설명하라. $W_Q, W_K, W_V$ 의 row 가 분리에 어떤 역할을 하는가?

<details>
<summary>해설</summary>

**Sum 의 분리 메커니즘**:

$\tilde{x}_t = x_t + \text{PE}_t \in \mathbb{R}^d$. $W_Q \in \mathbb{R}^{d \times d_k}$ 의 변환:
$$
q_t = \tilde{x}_t W_Q = x_t W_Q + \text{PE}_t W_Q
$$

여기서 $x_t W_Q$ 와 $\text{PE}_t W_Q$ 는 **같은 vector 로 합쳐짐** — 자연스럽게 분리 안 됨.

**그러나 학습이 만드는 분리**:

$W_Q$ 의 column 들이 학습되면서 두 종류로 specialize:
- 일부 column 은 $x_t$ 의 특징 추출 (semantic)
- 다른 column 은 $\text{PE}_t$ 의 위치 정보 추출 (positional)

**Tsai 2019 의 명시적 분석**:

$W_Q$ 의 row 를 두 block 으로 분리:
$$
W_Q = \begin{pmatrix} W_Q^{\text{tok}} \\ W_Q^{\text{pos}} \end{pmatrix}
$$

$x_t W_Q^{\text{tok}}$ 와 $\text{PE}_t W_Q^{\text{pos}}$ 의 합을 $q_t$ 로 — 명시적 분리.

**Sum 이 implicit concat**: 충분한 차원에서 sum + linear projection = concat + linear projection. 학습이 이 분리 자동 구현.

**경험**: BERT/GPT 의 학습된 $W_Q, W_K$ 를 분석하면 명시적 token vs position specialization 발견 가능 — interpretability 연구.

**Modern alternative**: RoPE 는 이 implicit 분리를 회피 — $x_t$ 와 위치 정보를 곱셈으로 결합 (회전), additive sum 의 모호함 없음. $\square$

</details>

**문제 3** (논문 비평): "Why Position Embedding Matters" — RoPE, ALiBi 같은 modern PE 가 등장한 동기는 sinusoidal/learned PE 의 어떤 한계인가? Long context 의 본질적 어려움이 PE 만의 문제인가?

<details>
<summary>해설</summary>

**Sinusoidal/Learned PE 의 한계**:

1. **Extrapolation 실패** (가장 큰 문제):
   - Learned: max length 이상 학습 안 됨
   - Sinusoidal: 수학적으론 가능하지만 실증적으로 train length 의 2× 이상에서 성능 급감
   - Long context (32K+) 에서 명백한 문제

2. **Layer 깊이에서 PE 정보 약화**:
   - 첫 layer 에서만 PE 주입 → residual 로 propagate but 약화
   - 깊은 layer 의 attention 은 위치 정보 거의 망각

3. **Absolute vs Relative**:
   - Absolute PE 는 두 token 의 **distance** 가 아닌 **위치** 만 인코딩
   - "Token i 가 token j 에 attend" 시 i-j 의 거리가 더 자연스러움
   - 인간의 문법 직관 (수식어-피수식어 거리) 와 일치

**Modern PE 의 해결**:

1. **RoPE** (Su 2021):
   - $Q, K$ 에 회전 적용 → $\langle R(i)q, R(j)k \rangle = f(i-j, q, k)$
   - **자동 relative**: 수식이 i-j 만 의존
   - 매 layer 에서 적용 → 깊이 별 PE 정보 강화
   - Extrapolation 좋음 (회전 함수의 smooth periodic)

2. **ALiBi** (Press 2021):
   - 단순 linear bias $-m_h |i-j|$
   - 학습 가능 parameter 거의 없음 → 가장 단순
   - **Extrapolation 최강** — train 의 4×까지 robust

**Long Context 의 PE 외 어려움**:

PE 만의 문제가 아님:

1. **Attention 분포 문제**:
   - Long sequence 에서 attention 이 너무 spread (정보 dilution)
   - Sliding window (Longformer) 등 sparse attention 필요

2. **메모리·계산**:
   - $O(T^2)$ memory — Flash Attention (Ch5-05) 으로 mitigate
   - KV cache 크기 — MQA/GQA (Ch5-06)

3. **Long-range dependency 학습**:
   - 긴 거리 의존성은 데이터 부족 시 학습 어려움
   - PE 가 enable 해도 실제 사용은 별도

4. **Position 외 inductive bias**:
   - 단순 거리 외에 syntactic distance, dependency tree 등도 중요
   - Graph-based PE (Graphormer) 의 아이디어

**결론**:

PE 는 long context 의 **필요 조건** 이지만 **충분 조건이 아님**. RoPE/ALiBi 가 PE 측면을 해결, Flash Attention + MQA 가 efficiency 측면, scaling laws 가 학습 측면. **모든 측면의 동시 개선** 이 long-context LLM 가능하게. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch2-transformer-architecture/05-cross-attention.md) | [📚 README](../README.md) | [다음 ▶](./02-sinusoidal-pe.md)

</div>
