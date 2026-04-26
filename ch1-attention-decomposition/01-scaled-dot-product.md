# 01. Scaled Dot-Product Attention 완전 유도

## 🎯 핵심 질문

- Attention $\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k}) V$ 의 각 연산은 어떤 수학적 필연성에서 나오는가?
- Query $Q$, Key $K$, Value $V$ 의 분해는 왜 단일 입력 $X$ 에서 출발해 세 개의 다른 projection 을 거치는가?
- $QK^\top$ 의 의미는 무엇인가 — similarity score 가 왜 dot product 인가?
- Softmax 가 왜 row-wise 로 적용되며, 결과가 attention weight 로 해석되는 이유는?
- 계산 복잡도 $O(T^2 d)$ 와 메모리 $O(T^2)$ 가 왜 long-context 의 본질적 병목인가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Transformer 의 모든 표현력은 **scaled dot-product attention** 한 식에서 출발합니다. 그러나 많은 실무자가 `nn.MultiheadAttention(d, h)` 를 부르고 끝낼 뿐, 다음을 정확히 알지 못한 채 사용합니다:

1. **$Q, K, V$ projection 의 비대칭성** — 같은 $X$ 에서 세 개의 다른 행렬을 만드는 이유, 단일 projection 으로 환원 불가능한 이유
2. **Dot product 가 similarity 인 이유** — cosine 이 아니라 raw dot product 를 쓰는 동기, $\sqrt{d_k}$ scaling 의 필요성 (Ch1-02 에서 증명)
3. **Softmax 의 row-wise 적용** — column-wise 였다면 무엇이 깨지는가
4. **$O(T^2)$ 의 본질적 병목** — Linear / Sparse / Flash attention (Ch5) 모두 이 한계를 우회하려는 시도

이 문서에서는 attention 식의 **각 구성 요소를 분해해 정의**하고, 다음 문서들에서 다룰 수학적 분석의 토대를 다집니다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix multiplication, outer product, projection
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Linear layer, parameter sharing
- 확률론: i.i.d. 가정, expectation, variance
- [RNN & LSTM Deep Dive](https://github.com/iq-ai-lab/rnn-lstm-deep-dive): Bahdanau Attention (선행 권장) — soft alignment 의 기원

---

## 📖 직관적 이해

### Attention 의 두 가지 시각

Attention 은 두 관점으로 볼 수 있습니다:

1. **Soft alignment**: 출력의 각 위치가 입력의 어느 위치에 "주목" 하는지 가중치로 표현 (Bahdanau 2014 NMT 의 origin)
2. **Differentiable dictionary lookup**: $K$ 를 키, $V$ 를 값, $Q$ 를 질의로 보고 hash table 의 soft 버전으로 해석

Self-attention 은 두 관점이 같은 그래프 (token sequence) 위에서 작동합니다 — 각 token 이 다른 모든 token 을 query 하고, 그 결과를 가중합으로 모읍니다.

### Q, K, V 의 비대칭

같은 입력 $X \in \mathbb{R}^{T \times d}$ 에서:

```
Q = X W_Q  ─→ "내가 무엇을 찾는가" (질의)
K = X W_K  ─→ "내가 무엇을 가지고 있는가" (광고)
V = X W_V  ─→ "내가 실제로 전달하는 정보" (내용)
```

만약 $W_Q = W_K = W_V$ 라면 attention 이 self-correlation 으로 환원됩니다. 세 개의 독립 projection 이 **표현력의 핵심**입니다.

### Dot Product 의 직관

$q_i^\top k_j$ 가 클수록 $i$ 가 $j$ 에 "관심" 이 큽니다. Cosine similarity ($q^\top k / (\|q\| \|k\|)$) 가 아니라 raw dot product 를 쓰는 이유:

- **Magnitude 가 정보** — $\|q\|$ 가 클 때 더 강하게 attend, 학습이 자유도를 가짐
- **계산 단순** — normalization 없이 곱셈만으로 구현
- **단점**: $d_k$ 가 클 때 분산이 커져 softmax 포화 → $\sqrt{d_k}$ scaling 으로 해결 (Ch1-02)

### Softmax 의 row-wise 적용

```
scores ∈ ℝ^{T × T}:        softmax(row-wise):
  [ 0.3  -0.1   0.5 ]        [ 0.32  0.21  0.47 ]   ← row 합 = 1
  [-0.2   0.8   0.0 ]   →    [ 0.20  0.55  0.25 ]
  [ 0.1   0.0  -0.5 ]        [ 0.40  0.36  0.24 ]
```

각 query (row) 가 모든 key (column) 에 대한 확률 분포를 형성합니다. Column-wise 였다면 각 key 가 모든 query 에 대한 분포가 되는데, 이는 attention 의 의미와 다릅니다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Attention 의 입력

Sequence length $T$, embedding dimension $d$ 인 입력:
$$
X \in \mathbb{R}^{T \times d}
$$

각 행 $x_i \in \mathbb{R}^d$ 가 $i$-th token 의 embedding.

### 정의 1.2 — Q, K, V Projection

학습 가능한 projection matrix:
$$
W_Q \in \mathbb{R}^{d \times d_k}, \quad W_K \in \mathbb{R}^{d \times d_k}, \quad W_V \in \mathbb{R}^{d \times d_v}
$$

(통상 $d_k = d_v$ 또는 $d_k = d_v = d/h$ for multi-head)

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

차원: $Q, K \in \mathbb{R}^{T \times d_k}$, $V \in \mathbb{R}^{T \times d_v}$.

### 정의 1.3 — Attention Score Matrix

$$
S := \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
$$

$S_{ij} = q_i^\top k_j / \sqrt{d_k}$ — $i$-th query 와 $j$-th key 의 scaled similarity.

### 정의 1.4 — Attention Weight

Row-wise softmax:
$$
A := \text{softmax}(S), \qquad A_{ij} = \frac{\exp(S_{ij})}{\sum_{l=1}^{T} \exp(S_{il})}
$$

각 행 $A_{i,:}$ 는 확률 분포 ($\sum_j A_{ij} = 1$).

### 정의 1.5 — Scaled Dot-Product Attention

$$
\boxed{\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V \in \mathbb{R}^{T \times d_v}}
$$

각 출력 row:
$$
\text{Attn}(Q, K, V)_i = \sum_{j=1}^{T} A_{ij} \, v_j
$$

### 정의 1.6 — Masking

Optional mask $M \in \{0, -\infty\}^{T \times T}$:
$$
S_{ij}^{\text{masked}} = S_{ij} + M_{ij}
$$

- **Causal mask** (decoder): $M_{ij} = -\infty$ if $i < j$ — 미래 차단
- **Padding mask**: padding token 위치 차단

---

## 🔬 정리와 증명

### 정리 1.1 — Permutation Equivariance

Self-attention 은 **permutation-equivariant**: 임의 순열 행렬 $P \in \{0, 1\}^{T \times T}$ 에 대해:
$$
\text{Attn}(PQ, PK, PV) = P \cdot \text{Attn}(Q, K, V)
$$

**증명**:

$$
(PQ)(PK)^\top = P Q K^\top P^\top
$$

$\text{softmax}$ 는 row-wise 연산이므로 row 순서 변경에 commutative:
$$
\text{softmax}(P S P^\top) = P \, \text{softmax}(S) \, P^\top
$$

(엄밀히는 row-wise softmax 이므로 column 순서 변경 시 각 row 안에서 entry 순서만 바뀜, 그러나 sum 은 동일)

$$
\text{softmax}(P S P^\top) (P V) = P \, \text{softmax}(S) \, P^\top P V = P \, \text{softmax}(S) V
$$

($P^\top P = I$). 따라서 **순서 정보가 attention 자체에 없음** $\square$ — 이것이 PE (Ch3) 의 필요성의 직접적 동기.

### 정리 1.2 — 계산 복잡도

$$
\text{Time: } O(T^2 d_k + T^2 d_v), \qquad \text{Memory: } O(T^2 + T d_v)
$$

**증명**:

- $QK^\top$ 계산: $T \times T$ 행렬, 각 entry 가 $d_k$ 곱 → $O(T^2 d_k)$
- Softmax: $O(T^2)$
- $A V$ 계산: $T \times d_v$ 행렬, 각 entry 가 $T$ 곱 → $O(T^2 d_v)$
- Memory: attention matrix $A \in \mathbb{R}^{T \times T}$ 가 dominant → $O(T^2)$

$\square$

이것이 Ch5 의 효율화 동기의 핵심입니다 — $T = 32{,}768$ (32K context) 에서 $T^2 \approx 10^9$, $A$ 자체가 4GB (FP32).

### 정리 1.3 — Softmax 의 미분

$$
\frac{\partial A_{ij}}{\partial S_{ik}} = A_{ij} (\delta_{jk} - A_{ik})
$$

**증명**: $A_{ij} = e^{S_{ij}} / Z_i$ where $Z_i = \sum_l e^{S_{il}}$.

$$
\frac{\partial A_{ij}}{\partial S_{ik}} = \frac{\delta_{jk} e^{S_{ij}} Z_i - e^{S_{ij}} e^{S_{ik}}}{Z_i^2} = A_{ij} \delta_{jk} - A_{ij} A_{ik} = A_{ij}(\delta_{jk} - A_{ik}) \quad \square
$$

이 식은 Ch1-03 (softmax saturation) 에서 gradient vanishing 분석의 핵심.

### 정리 1.4 — Attention 의 출력은 V 의 convex combination

각 출력 $\text{Attn}_i = \sum_j A_{ij} v_j$ 는 $\{v_1, \ldots, v_T\}$ 의 **convex combination** ($A_{ij} \geq 0$, $\sum_j A_{ij} = 1$).

**따름**: $\|\text{Attn}_i\|_\infty \leq \max_j \|v_j\|_\infty$ — 출력은 입력 value 의 convex hull 안에. 이는 attention 이 representation 을 "압축" 만 하고 "확장" 하지 못함을 의미하며, FFN 의 비선형 확장이 표현력에 필수인 이유 (Ch2-02).

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Scaled Dot-Product Attention 바닥부터

```python
import torch
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (T, d_k)
    V:    (T, d_v)
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)   # (T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)                     # row-wise
    return attn @ V, attn

# 작은 예제
torch.manual_seed(0)
T, d_k, d_v = 5, 8, 8
Q = torch.randn(T, d_k); K = torch.randn(T, d_k); V = torch.randn(T, d_v)
out, attn = scaled_dot_product_attention(Q, K, V)
print(f'Output shape: {out.shape}')              # (5, 8)
print(f'Attn shape:   {attn.shape}')             # (5, 5)
print(f'Attn row sum: {attn.sum(-1)}')           # 모두 1
```

### 실험 2 — `nn.MultiheadAttention` 과의 일치 확인

```python
import torch.nn as nn

# 단일 head 모드로 비교 (multi-head 는 Ch1-05)
torch.manual_seed(0)
d_model, num_heads = 8, 1
mha = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)

# in_proj_weight 에서 W_Q, W_K, W_V 분리 (PyTorch 내부 표현)
W_Q = mha.in_proj_weight[:d_model].T              # (d, d)
W_K = mha.in_proj_weight[d_model:2*d_model].T
W_V = mha.in_proj_weight[2*d_model:].T
W_O = mha.out_proj.weight.T

# 입력
x = torch.randn(1, T, d_model)
Q = x @ W_Q; K = x @ W_K; V = x @ W_V

# 직접 구현
out_manual, _ = scaled_dot_product_attention(Q.squeeze(0), K.squeeze(0), V.squeeze(0))
out_manual = out_manual @ W_O

# nn.MultiheadAttention
out_nn, _ = mha(x, x, x)
print(f'Max difference: {(out_manual - out_nn.squeeze(0)).abs().max():.2e}')   # ≈ 1e-7
```

### 실험 3 — Permutation Equivariance 검증

```python
P_idx = torch.randperm(T)
Q_p, K_p, V_p = Q[P_idx], K[P_idx], V[P_idx]

out_orig, _ = scaled_dot_product_attention(Q, K, V)
out_perm, _ = scaled_dot_product_attention(Q_p, K_p, V_p)

# permutation 한 출력과 원래 출력에 같은 permutation 적용한 결과 비교
print(f'Equivariance error: {(out_perm - out_orig[P_idx]).abs().max():.2e}')   # ≈ 0
```

### 실험 4 — Causal Mask 효과

```python
T = 6
Q = torch.randn(T, d_k); K = torch.randn(T, d_k); V = torch.randn(T, d_v)

# Lower-triangular mask: i ≥ j 만 허용
causal = torch.tril(torch.ones(T, T))
out, attn = scaled_dot_product_attention(Q, K, V, mask=causal)
print('Causal attention matrix (upper-triangle = 0):')
print(attn.round(decimals=3))
# 대각선 위쪽 entry 가 모두 0 → 미래 token 참조 불가 ✓
```

### 실험 5 — 계산 복잡도 측정

```python
import time

for T in [128, 512, 2048, 8192]:
    Q = torch.randn(T, 64); K = torch.randn(T, 64); V = torch.randn(T, 64)
    t0 = time.time()
    for _ in range(5):
        scaled_dot_product_attention(Q, K, V)
    elapsed = (time.time() - t0) / 5
    mem_attn = T * T * 4 / 1024 / 1024  # FP32 attention matrix in MB
    print(f'T={T:5d}: time={elapsed*1000:7.2f}ms, attention matrix={mem_attn:.2f}MB')
# T=8192 시 attention matrix ≈ 256MB → O(T²) 의 직접적 비용
```

---

## 🔗 실전 활용

### 1. Self-Attention vs Cross-Attention

- **Self-attention**: $Q = K = V = X$ (같은 sequence 에서) — encoder, decoder self
- **Cross-attention**: $Q$ 는 decoder, $K, V$ 는 encoder — translation 의 alignment (Ch2-05)

### 2. Causal Decoding 의 KV 재사용

Autoregressive generation 시 $K, V$ 는 이전 step 의 것을 cache 가능 (KV cache). 새로 계산하는 것은 새 $Q$ 만 → 메모리 trade-off, MQA/GQA (Ch5-06) 가 이를 절약.

### 3. PyTorch 2.0 의 `F.scaled_dot_product_attention`

PyTorch 2.0 부터 native API 제공 — 자동으로 Flash Attention 백엔드 선택:
```python
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)
```

### 4. Attention 의 변형

- **Additive attention** (Bahdanau): $a^\top \tanh(W_q q + W_k k)$ — 작은 차원에서만 유리
- **Dot-product attention** (Vaswani): $q^\top k / \sqrt{d_k}$ — 표준
- **Bilinear**: $q^\top W k$ — extra parameter

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Sequence length $T$ 고정 (forward 별) | KV cache 로 incremental 가능 (Ch5-06) |
| $O(T^2)$ memory | Long-context 의 본질 병목 → Linear / Sparse / Flash (Ch5) |
| Permutation equivariance | 순서 정보 없음 → PE 필수 (Ch3) |
| Dot product as similarity | $\sqrt{d_k}$ scaling 없으면 saturation (Ch1-02, Ch1-03) |
| Single similarity space | 다양한 관계 표현 한계 → Multi-Head (Ch1-05) |
| Convex combination of V | 표현력 제한 → FFN 필수 (Ch2-02) |
| Real-valued | Complex / quantum 확장은 별도 |

---

## 📌 핵심 정리

$$\boxed{\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V}$$

| 구성 | 정의 | 차원 |
|------|------|------|
| **Input** $X$ | Token embedding | $T \times d$ |
| **Q, K** | Query, Key projection | $T \times d_k$ |
| **V** | Value projection | $T \times d_v$ |
| **Score** $S$ | $QK^\top / \sqrt{d_k}$ | $T \times T$ |
| **Attn weight** $A$ | $\text{softmax}(S)$ row-wise | $T \times T$ |
| **Output** | $A V$ | $T \times d_v$ |
| **Time** | $O(T^2 d)$ | — |
| **Memory** | $O(T^2)$ | dominant |
| **Equivariance** | Permutation | → PE 필수 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $Q = K = V = X$ (self-attention 의 최단순 형태) 이고 $W_Q = W_K = W_V = I$ 일 때, attention 출력이 어떻게 되는지 분석하라. 표현력의 한계는?

<details>
<summary>해설</summary>

$Q = K = V = X$ 시:
$$
\text{Attn} = \text{softmax}(X X^\top / \sqrt{d}) X
$$

$X X^\top$ 는 token 간 내적 (Gram matrix). Self-similarity 가 큰 token 끼리 attend.

**한계**:
- 모든 token 이 자기 자신과 가장 비슷 → 대각선이 dominant
- $W_Q, W_K$ 분리 없으면 query-key 비대칭이 사라짐 — 같은 의미를 다르게 attend 할 수 없음
- $W_V$ 없으면 출력이 입력의 단순 reweighting → 표현력 ↓

따라서 세 개의 독립 projection 이 표현력의 핵심. $\square$

</details>

**문제 2** (심화): Softmax 대신 단순 normalization $A_{ij} = S_{ij} / \sum_j S_{ij}$ (필요 시 $S \geq 0$) 을 쓰면 어떤 문제가 발생하는가? Linear Attention (Ch5-02) 이 왜 이 방향을 시도하는가?

<details>
<summary>해설</summary>

**단순 normalization 의 문제**:
1. **음수 score** — $S_{ij} < 0$ 일 때 의미 모호. $|S|$ 등 hack 필요.
2. **Sharpness 부족** — softmax 의 exponential 효과가 없어 "선명한" attention 어려움
3. **Gradient 가 less informative** — softmax 의 saturation 영역이 자연스러운 sparsity 유도, normalization 만으로는 모든 token 균등 attend 경향

**Linear Attention 의 trick (Ch5-02)**:
- Feature map $\phi(x) \geq 0$ 으로 양수성 보장 ($\phi = \text{ELU} + 1$)
- Normalization 도 결합 순서 변경: $\phi(Q)(\phi(K)^\top V) / (\phi(Q) \phi(K)^\top \mathbf{1})$
- 결과: $O(T^2 d) \to O(T d^2)$, 단 표현력 일부 손실 (sharpness)

따라서 softmax 의 비선형성과 양수성이 attention 의 표현력에 본질적, Linear Attention 은 효율 vs 표현력 trade-off. $\square$

</details>

**문제 3** (논문 비평): Vaswani et al. 2017 의 식 $\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k}) V$ 는 RNN 기반 Bahdanau Attention 의 어떤 한계를 해결했는가? 또한 이 식이 RNN seq2seq 의 어떤 inductive bias 를 잃게 만들었고, 이 손실이 PE / 큰 데이터 / 큰 모델 로 보상 가능한 이유는?

<details>
<summary>해설</summary>

**Bahdanau Attention 의 한계**:
- **Sequential bottleneck**: encoder hidden state 를 RNN 으로 계산 → $O(T)$ 직렬, GPU 병렬화 불가
- **Gradient vanishing**: 긴 시퀀스에서 LSTM 도 멀리 떨어진 의존성 어려움
- **Context vector 1개**: decoder 가 각 step 마다 단일 context vector 만 → coarse-grained alignment

**Vaswani 의 해결**:
- **Pure attention**: RNN 제거, $T$ token 을 동시에 처리 → GPU 병렬
- **All-pairs interaction**: $T \times T$ matrix 로 모든 token 쌍 직접 연결, distance 무관
- **Multi-head**: 여러 alignment 를 병렬로

**잃은 inductive bias**:
- **순서 정보** — RNN 은 자연스럽게 순서, Transformer 는 permutation-equivariant → PE 추가 필수
- **Local structure** — RNN 의 sequential bias 가 작은 데이터에서 강점, Transformer 는 데이터로 학습 필요
- **Inductive locality** — CNN 처럼 가까운 token 우선 처리하는 prior 없음

**보상**:
- PE 가 순서 inductive bias 회복 (Ch3)
- 큰 데이터 (GPT-3 의 300B tokens) 가 inductive bias 부재 보상 — Sutton 의 "bitter lesson"
- 큰 모델 + scaling laws (Ch7-01) 가 데이터 효율을 trading

따라서 Transformer 는 inductive bias 를 줄이고 universality 를 늘린 trade-off, 데이터·계산 시대의 적합한 선택. $\square$

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-sqrt-dk-scaling.md)

</div>
