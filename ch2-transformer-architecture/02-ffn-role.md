# 02. Feed-Forward Network 의 역할

## 🎯 핵심 질문

- FFN 이 왜 Transformer 의 파라미터의 약 2/3 를 차지하는가? Width 4× 의 동기는?
- FFN 이 단순 nonlinearity 추가 이상의 역할을 하는가 — Geva 2021 의 "key-value memory" 해석은?
- ReLU 와 GELU, SwiGLU, GeGLU 같은 modern activation 의 차이가 무엇이고 왜 Transformer 에 중요한가?
- FFN 없이 attention 만 있는 Transformer 의 표현력 한계는?
- LLaMA, PaLM 의 SwiGLU 채택 이유는? Activation 선택이 모델 성능에 얼마나 영향?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

FFN 은 **Transformer 의 가장 underrated 구성요소** 입니다. Attention 이 정보를 모은다면 FFN 이 그 정보를 **변환** 합니다:

1. **표현력의 대부분** — 한 block 의 파라미터 2/3, capacity 의 핵심
2. **Memory 로서의 FFN** — Geva 2021: $W_1$ 행이 key, $W_2$ 열이 value 인 학습된 dictionary
3. **Per-token nonlinear transformation** — Attention 이 못하는 차원 간 mixing
4. **Modern variants** — SwiGLU (LLaMA), GeGLU (PaLM) 가 ReLU 대비 성능 ↑

이 문서는 FFN 의 **역할과 변형들** 을 분석하고, Geva 2021 의 key-value memory 해석을 직접 재현합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-transformer-block.md](./01-transformer-block.md)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): MLP, UAT, ReLU/GELU
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Outer product, rank, SVD

---

## 📖 직관적 이해

### FFN = Per-token MLP

```
For each token x_i:
  h_i = x_i ⋅ W_1   ∈ ℝ^{4d}      (expansion)
  h_i = activation(h_i)
  y_i = h_i ⋅ W_2   ∈ ℝ^{d}        (compression)
```

토큰 별로 독립 적용 — token mixing 없음. **Position-wise FFN**.

### Width 4× 의 동기

Vaswani 2017: $d_{\text{ff}} = 4 \times d_{\text{model}}$ 이 표준.
- $d = 512$ → $d_{\text{ff}} = 2048$
- $d = 4096$ (LLaMA-7B) → $d_{\text{ff}} = 11008$ (또는 16384, 8/3 × 4)

**왜 4×?** Empirical sweet spot:
- 작으면 표현력 부족
- 크면 redundancy, compute 낭비
- 4× 가 sparse representation + capacity 의 균형

### Key-Value Memory 해석 (Geva 2021)

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x) = \sum_{i=1}^{d_{\text{ff}}} (W_2)_{:,i} \cdot \max(0, (W_1)_i x)
$$

- $(W_1)_i \in \mathbb{R}^d$: $i$-th **key** (학습된 패턴)
- $(W_2)_{:,i} \in \mathbb{R}^d$: $i$-th **value** (해당 패턴 매칭 시 추가될 vector)
- $\max(0, (W_1)_i x)$: $x$ 와 $i$-th key 의 매칭 정도 (양수성)

→ FFN = $d_{\text{ff}}$ 개의 (key, value) pair 를 가진 **soft hash table**.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Standard FFN (Vaswani 2017)

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

with $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, $d_{\text{ff}} = 4d$ 통상.

### 정의 2.2 — GELU FFN (BERT, GPT-2)

ReLU 대신 GELU:
$$
\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2} (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))
$$

(Hendrycks & Gimpel 2016)

### 정의 2.3 — Gated Linear Units (GLU)

Dauphin 2017:
$$
\text{GLU}(x) = (x W_1) \odot \sigma(x V_1)
$$

(element-wise product of two projections, one gated by sigmoid)

### 정의 2.4 — SwiGLU (LLaMA)

Shazeer 2020 + LLaMA:
$$
\text{SwiGLU}(x) = (x W_1) \odot \text{Swish}(x V_1) \cdot W_2
$$

with Swish $= x \cdot \sigma(\beta x)$.

**파라미터 보정**: 원래 4d width 대신 $\frac{2}{3} \times 4d = \frac{8}{3} d$ 로 두 projection 사용 (총 파라미터 동일).

### 정의 2.5 — GeGLU (PaLM)

$$
\text{GeGLU}(x) = (x W_1) \odot \text{GELU}(x V_1) \cdot W_2
$$

### 정의 2.6 — FFN 의 표현으로서

$\text{FFN}(x) = \sum_i \alpha_i(x) \cdot v_i$ where:
- $v_i = (W_2)_{:,i}$: $i$-th value
- $\alpha_i(x) = \text{activation}((W_1)_i x)$: matching coefficient

---

## 🔬 정리와 증명

### 정리 2.1 — FFN 의 Parameter 수

각 FFN: $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$, total $8 d^2$ (bias 무시).

Attention: $4 d^2$ (Ch1-05 정리 5.4).

**비율**: FFN / Attention = $8 / 4 = 2$ → FFN 이 block 의 2/3 차지.

(LLaMA SwiGLU 의 $\frac{8}{3}$ width 시 $W_1, V_1, W_2$ 각 $\frac{8}{3} d^2$, total $8 d^2$ 동일)

### 정리 2.2 — Per-token Independence

FFN 은 token 별 독립 적용:
$$
\text{FFN}(X)_i = \text{FFN}(X_{i,:}) \quad \forall i
$$

**의미**: Token-mixing 없음 → 정보 교환은 attention 이 전담. **표현력 분담**: attention = mixing, FFN = transformation.

### 정리 2.3 — UAT 적용

FFN 은 standard 2-layer MLP — Hornik 1989 의 UAT 에 의해 임의의 continuous function $\mathbb{R}^d \to \mathbb{R}^d$ 를 충분한 width 로 근사 가능.

**Transformer 에의 의미**: 한 token 의 임의 nonlinear 변환 가능, attention 이 못하는 부분 보완.

### 정리 2.4 — Geva 2021 의 Key-Value Memory

$\text{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}} \alpha_i(x) v_i$

$\alpha_i(x) = \text{ReLU}((W_1)_i x)$ 가 **selective activation**:
- $(W_1)_i x > 0$: $i$-th value $v_i$ contribute
- $(W_1)_i x \leq 0$: contribute 안 함

**실증 발견** (Geva 2021):
- 각 key $(W_1)_i$ 가 specific input pattern (e.g., "the cat") 에 매칭
- 해당 value $(W_2)_{:,i}$ 는 specific output bias (next token 후보)
- FFN = soft retrieval system

### 정리 2.5 — FFN 없는 Transformer 의 한계

Attention only → token 의 convex combination 출력 (Ch1 정리 1.4):
$$
\text{Attn}_i = \sum_j A_{ij} v_j \in \text{Conv}(\{v_1, \ldots, v_T\})
$$

**한계**: 출력이 입력 V 의 convex hull 안 → 새로운 representation 못 생성. 표현력은 **input-spanned subspace** 로 제한.

**FFN 추가 시**: nonlinear transformation 으로 hull 밖 representation 생성 가능 → universal approximation (Yun 2020).

### 정리 2.6 — SwiGLU 의 우수성 (Shazeer 2020)

GLU variants 의 ablation:
- ReLU FFN: baseline
- GLU: -0.1% perplexity (약간 우수)
- GeGLU: -0.2% (더 우수)
- SwiGLU: -0.2% (가장 우수)

**Empirical**: 같은 파라미터 수에서 1-2% 성능 향상. **이론적 동기**: gating 이 selective activation 을 더 정확히 — key-value memory 의 "selectivity" 강화.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Standard FFN vs SwiGLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class GELUFFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        # 파라미터 동일 위해 d_ff = 2/3 * 4d = 8d/3 사용 (LLaMA)
        d_ff_eff = int(2 * d_ff / 3)
        self.w1 = nn.Linear(d, d_ff_eff, bias=False)
        self.v  = nn.Linear(d, d_ff_eff, bias=False)
        self.w2 = nn.Linear(d_ff_eff, d, bias=False)
    def forward(self, x):
        return self.w2(self.w1(x) * F.silu(self.v(x)))

# 비교
torch.manual_seed(0)
d, d_ff = 64, 256
ffn_std  = StandardFFN(d, d_ff)
ffn_gelu = GELUFFN(d, d_ff)
ffn_swi  = SwiGLU(d, d_ff)

print(f'Standard params: {sum(p.numel() for p in ffn_std.parameters())}')
print(f'GELU     params: {sum(p.numel() for p in ffn_gelu.parameters())}')
print(f'SwiGLU   params: {sum(p.numel() for p in ffn_swi.parameters())}')

x = torch.randn(2, 10, d)
print(f'Standard: {ffn_std(x).shape}')
print(f'GELU:     {ffn_gelu(x).shape}')
print(f'SwiGLU:   {ffn_swi(x).shape}')
```

### 실험 2 — Activation 함수 비교 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-4, 4, 200)
relu = F.relu(x)
gelu = F.gelu(x)
swish = F.silu(x)
glu_gate = torch.sigmoid(x)

plt.figure(figsize=(10, 4))
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, gelu, label='GELU', linewidth=2)
plt.plot(x, swish, label='Swish (SiLU)', linewidth=2)
plt.plot(x, glu_gate, label='σ (GLU gate)', linewidth=2, linestyle='--')
plt.legend(); plt.grid(alpha=0.3)
plt.xlabel('input'); plt.ylabel('output')
plt.title('Activation functions used in FFN variants')
plt.show()
```

### 실험 3 — Key-Value Memory 해석 시각화

```python
torch.manual_seed(0)
d, d_ff = 8, 16
ffn = StandardFFN(d, d_ff)

# 학습된 W_1 의 행이 key
keys = ffn.fc1.weight    # (d_ff, d) — 행이 key
values = ffn.fc2.weight  # (d, d_ff) — 열이 value

# 임의 입력 x 에 대한 activation
x = torch.randn(d)
activations = F.relu(keys @ x + ffn.fc1.bias)   # (d_ff,)
print(f'Activations (matching score): {activations.detach().round(decimals=3)}')

# 활성화된 key 와 그 value 의 contribution
top_k = activations.topk(3).indices
print(f'Top-3 active keys: {top_k.tolist()}')
print(f'Their values shape: {values[:, top_k].shape}')

# FFN 출력 = sum of (activation × value)
out = sum(activations[i] * values[:, i] for i in range(d_ff))
out_check = ffn(x.unsqueeze(0)).squeeze() - ffn.fc2.bias
print(f'Reconstruction error: {(out - out_check).abs().max():.6f}')
```

### 실험 4 — FFN vs No-FFN 표현력 비교

```python
class AttnOnly(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        return self.ln(x + self.attn(x, x, x)[0])

class AttnFFN(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = StandardFFN(d, d_ff)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.ffn(x))
        return x

# 표현력 측정: 작은 dataset 에서 학습
torch.manual_seed(0)
T, d = 10, 16
X = torch.randn(100, T, d)
Y = torch.randn(100, T, d) * 0.5 + (X ** 2)   # nonlinear target

for model_class, name in [(lambda: AttnOnly(d, 4), 'Attn-only'),
                          (lambda: AttnFFN(d, 4, d*4), 'Attn+FFN')]:
    model = model_class()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(200):
        opt.zero_grad()
        loss = F.mse_loss(model(X), Y)
        loss.backward(); opt.step()
    print(f'{name:10s}: final loss = {loss.item():.4f}')
# Attn+FFN 이 nonlinear target 더 잘 학습
```

### 실험 5 — Width 별 ablation

```python
torch.manual_seed(0)
class TestModel(nn.Module):
    def __init__(self, d_ff_mult):
        super().__init__()
        self.ffn = StandardFFN(64, 64 * d_ff_mult)
    def forward(self, x):
        return self.ffn(x)

for mult in [1, 2, 4, 8]:
    model = TestModel(mult)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(100, 64)
    y = torch.randn(100, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(500):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward(); opt.step()
    print(f'd_ff = {mult}d ({n_params:6d} params): final loss = {loss.item():.4f}')
# 4× 가 sweet spot — 더 크면 overfitting, 작으면 표현력 부족
```

---

## 🔗 실전 활용

### 1. LLaMA 의 SwiGLU + RMSNorm

```python
class LLaMABlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = SwiGLU(d, 4 * d)   # 8d/3 effective width
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

LLaMA-2 70B: 80 layer, $d = 8192$, FFN width = $8/3 \times 8192 \approx 21845$.

### 2. PaLM 의 GeGLU

PaLM: GeGLU + parallel attention/FFN (residual 한 번):
$$
y = x + \text{Attn}(\text{LN}(x)) + \text{FFN}(\text{LN}(x))
$$

(병렬 → 같은 LN 입력, computation 일부 중복 but 더 빠름)

### 3. MoE FFN

Switch Transformer (Ch6-05) 등에서 FFN 을 expert pool 로:
$$
\text{MoE-FFN}(x) = \sum_e \text{router}(x)_e \cdot \text{FFN}_e(x)
$$

각 token 이 top-k expert 만 활성화 → 파라미터 ↑ 계산 →

### 4. FFN Pruning

Geva 2021 의 key-value memory 해석을 활용:
- Important key (high activation) 는 보존
- Rare key 는 prune
- Magnitude pruning, structured pruning 의 직접 동기

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| $d_{\text{ff}} = 4d$ | LLaMA: $8d/3$, 다양한 비율 가능 |
| Per-token independence | Token-mixing 은 attention 만 — 한계? Mixer 등 변형 |
| Single nonlinearity | GLU 가 두 projection 사용 |
| Dense FFN | MoE 가 sparse |
| Fixed across layer | Layer-wise 다른 $d_{\text{ff}}$ 가능 (PaLM-X 등) |

---

## 📌 핵심 정리

$$\boxed{\text{FFN}(x) = \sum_{i=1}^{d_{\text{ff}}} \text{act}((W_1)_i x) \cdot (W_2)_{:,i} \quad \text{— soft key-value memory}}$$

| 변형 | 식 | 채택 |
|------|-----|------|
| **ReLU** | $\max(0, xW_1) W_2$ | Vaswani 2017 |
| **GELU** | $\text{GELU}(xW_1) W_2$ | BERT, GPT-2 |
| **SwiGLU** | $(xW_1) \odot \text{Swish}(xV_1) \cdot W_2$ | LLaMA, Mistral |
| **GeGLU** | $(xW_1) \odot \text{GELU}(xV_1) \cdot W_2$ | PaLM |

| 양 | 값 |
|----|-----|
| $d_{\text{ff}}/d$ | 4× 표준 (또는 $8/3$ for GLU variants) |
| FFN 파라미터 비중 | block 의 ~2/3 |
| Per-token | Yes — token mixing 없음 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LLaMA-2 7B (d=4096, 32 layers) 의 FFN 의 총 파라미터 수를 계산하라. SwiGLU width 보정은 $\frac{8}{3} d$ 사용.

<details>
<summary>해설</summary>

LLaMA-2 7B: $d = 4096$, $d_{\text{ff}} = 11008$ (≈ $8/3 \times 4096 = 10923$, 약간 보정)

SwiGLU 한 layer:
- $W_1, V$ each: $4096 \times 11008 = 45M$
- $W_2$: $11008 \times 4096 = 45M$
- Total: $135M$

32 layers: $32 \times 135M = 4.32B$

전체 7B 중 약 60% — 정리 2.1 의 비율 ($2/3$) 과 거의 일치 ✓ $\square$

</details>

**문제 2** (심화): FFN 의 width $d_{\text{ff}} = 4d$ 가 sweet spot 인 직관적 / 이론적 이유는? Width 를 늘리면 항상 성능이 향상되는가?

<details>
<summary>해설</summary>

**Sweet spot 의 이유**:

1. **Sparsity 가정**: 학습된 representation 이 sparse — 한 input 에 대해 활성화되는 key 비율이 ~10-25%. $d_{\text{ff}} = 4d$ 시 활성 key 가 $\sim d$ 개 → input dim 과 매치.

2. **UAT 의 width 충분조건**: 임의 continuous function 근사에 충분한 width. 그러나 너무 크면 overfitting / compute 낭비.

3. **Empirical**: Vaswani 2017 ablation, GPT-2/3 ablation 모두 4× 가 best trade-off.

**Width 증가의 한계**:

- **Overfitting**: small data 에서 큰 FFN 은 overfit
- **Compute**: FLOP 가 width 에 linear, latency 직접 영향
- **Diminishing returns**: $d_{\text{ff}} = 8d$ 에서 4× 보다 약간 우수 but $2\times$ compute
- **Mixture of Experts**: MoE 가 effective width 를 늘리는 efficient 방법 — top-k routing 으로 파라미터 ↑ 계산 →

**LLaMA 의 $8/3 d$ 선택**: SwiGLU 가 두 projection 사용해서 같은 파라미터에 효율적, 따라서 width 줄임. 같은 파라미터 / 더 좋은 활성화. $\square$

</details>

**문제 3** (논문 비평): Geva 2021 의 "FFN as key-value memory" 해석은 mechanistic interpretability (Ch1-06) 에 어떤 시사를 주는가? 또한 이 해석이 sparse activation 과 MoE 의 정당화에 어떻게 사용되는가?

<details>
<summary>해설</summary>

**Mechanistic Interpretability 와의 연결**:

Geva 2021 의 핵심 발견:
- Layer-wise: 얕은 layer 의 key 는 syntactic pattern, 깊은 layer 는 semantic
- Specific key: 학습 후 특정 input pattern 에 정확히 매칭 (예: "Apple Inc.")
- Value: 해당 pattern 매칭 시 다음 token 분포에 specific bias

**시사점**:

1. **FFN 도 explanation 가능**: Attention 만 분석 대상이 아님 — FFN 의 key 들이 transparent computation
2. **Feature dictionary**: 모델이 학습한 feature 들이 $W_1$ 의 행으로 명시화
3. **Sparse autoencoder 와 직접 연결**: Anthropic 2024 의 SAE 가 FFN activation 을 monosemantic feature 로 분해

**Sparse Activation 의 정당화**:

학습된 FFN 에서 (Geva 2021):
- 한 input 당 활성 key 비율 5-15%
- 대부분의 key 는 0 — sparse activation
- 이는 **MoE 의 토대**: 어차피 active 안 되는 key 들을 별도 expert 로 분리

**MoE 의 직접 동기** (Switch, Mixtral):
- $d_{\text{ff}}$ 를 $E$ expert 로 분리, 각 expert 가 $d_{\text{ff}} / E$ width
- Router 가 input-dependent 로 top-k expert 선택
- "어차피 sparse" 를 explicit conditional computation 으로 변환
- 결과: 파라미터 $E \times$, 계산 $1\times$

**Sparse Autoencoder (SAE, Anthropic 2024)**:
- 학습된 hidden $h$ 를 $h \approx \sum_i c_i f_i$ 로 분해 ($f_i$ = monosemantic feature)
- $c_i$ 는 sparse — 대부분 0
- 이것이 Geva 의 key-value memory 의 modern formalization
- AI safety 에서 deception, manipulation feature 식별 시도

따라서 FFN 의 mechanistic 분석 → sparse activation 발견 → MoE / SAE → modern interpretability 의 lineage. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-transformer-block.md) | [📚 README](../README.md) | [다음 ▶](./03-pre-ln-vs-post-ln.md)

</div>
