# 01. Transformer Block 의 완전 도식

## 🎯 핵심 질문

- Transformer block 은 Attention + FFN + LayerNorm + Residual 의 어떤 조합인가?
- Pre-LN 과 Post-LN 의 차이는 단순히 LN 의 위치 차이 이상인가? Gradient flow 가 어떻게 다른가?
- Residual connection 이 왜 deep Transformer 의 필수 요소인가? CNN 의 ResNet 과 같은 동기인가?
- 한 block 이 표현력 측면에서 universal approximator 의 어떤 부분을 담당하는가?
- Block 을 stacking 하면 표현력은 어떻게 증가하는가 — Yun 2020 의 universal approximation 결과의 의미?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Transformer block 은 **하나의 layer 안에 여러 mechanism 을 packed** 한 설계입니다:

1. **Attention**: token 간 정보 교환 (mixing)
2. **FFN**: token 별 비선형 변환 (transformation)
3. **LayerNorm**: 분포 안정화 (Ch1-02 의 분산 가정 유지)
4. **Residual**: gradient highway, identity preservation

이 네 요소의 **순서와 조합** 이 Pre-LN vs Post-LN 의 구분을 만들고, 훈련 안정성·warmup 필요 여부·deep stacking 가능성을 결정합니다. 이 문서는 block 의 전체 식을 분해하고, 다음 문서 (Ch2-02 FFN, Ch2-03 LN 위치) 에서 각 요소를 깊이 분석할 토대를 만듭니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md), [05-multi-head.md](../ch1-attention-decomposition/05-multi-head.md)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Residual, MLP, backpropagation
- [Regularization Theory Deep Dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive): LayerNorm

---

## 📖 직관적 이해

### Block = Attention + FFN

```
┌──────────────────────────────────────┐
│         Transformer Block             │
│                                        │
│   x ──→ [Attention] ─┐                │
│   │                   │                │
│   └───── (residual) ──+──→ [FFN] ─┐   │
│                       │            │   │
│                       └─ (residual)+ ──→ output
│                                        │
└──────────────────────────────────────┘
```

두 개의 sub-layer (Attention, FFN), 각각 residual connection 으로 감싸짐.

### LayerNorm 의 위치 두 가지

```
Pre-LN:                         Post-LN:
x' = x + Attn(LN(x))           x' = LN(x + Attn(x))
y  = x' + FFN(LN(x'))          y  = LN(x' + FFN(x'))

  ↑ LN 이 sub-layer 입력 측      ↑ LN 이 residual 후
```

Pre-LN 은 residual path 에 LN 없음 → gradient 가 직접 흐름. Post-LN 은 residual 에도 LN — depth 별 누적 문제.

### Information flow 의 두 axis

- **Token-mixing axis**: Attention (token 간 교환)
- **Feature-mixing axis**: FFN (한 token 안의 차원 간 교환)

이 분리는 MLP-Mixer (Tolstikhin 2021) 같은 후속 연구에서 명시화됨.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Sub-layer

- **MHA sublayer**: $\text{Attn}(x) := \text{MultiHeadAttention}(x, x, x)$ (self-attention, Ch1-05)
- **FFN sublayer**: $\text{FFN}(x) := \max(0, xW_1 + b_1) W_2 + b_2$ (Ch2-02)

### 정의 1.2 — LayerNorm

$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

with $\mu = \frac{1}{d} \sum_i x_i$, $\sigma^2 = \frac{1}{d} \sum_i (x_i - \mu)^2$. Per-token, per-feature normalization.

### 정의 1.3 — Pre-LN Block

$$
\begin{aligned}
x' &= x + \text{Attn}(\text{LN}_1(x)) \\
y &= x' + \text{FFN}(\text{LN}_2(x'))
\end{aligned}
$$

### 정의 1.4 — Post-LN Block (Vaswani 2017 원전)

$$
\begin{aligned}
x' &= \text{LN}_1(x + \text{Attn}(x)) \\
y &= \text{LN}_2(x' + \text{FFN}(x'))
\end{aligned}
$$

### 정의 1.5 — Residual Connection

$y = x + f(x)$ where $f$ 는 sub-layer. Identity shortcut 이 gradient highway 제공.

### 정의 1.6 — N-layer Transformer

$$
h^{(0)} = X + \text{PE}, \quad h^{(l+1)} = \text{Block}(h^{(l)}), \quad l = 0, \ldots, N-1
$$

Output $h^{(N)}$ 가 final representation.

---

## 🔬 정리와 증명

### 정리 1.1 — Pre-LN 의 Gradient Highway

Pre-LN block 의 input-output relation:
$$
y = x + \text{Attn}(\text{LN}_1(x)) + \text{FFN}(\text{LN}_2(x'))
$$

Gradient:
$$
\frac{\partial y}{\partial x} = I + \text{Attn-Jacobian} + \text{FFN-Jacobian}
$$

**$I$ 항이 직접 gradient highway** — sub-layer Jacobian 이 작아도 gradient 가 흐름.

### 정리 1.2 — Post-LN 의 Layer-wise Gradient 누적

Post-LN block $y = \text{LN}(x + f(x))$:

$$
\frac{\partial y}{\partial x} = \text{LN-Jacobian} \cdot (I + f-\text{Jacobian})
$$

LN-Jacobian 이 작으면 (saturation 등) residual highway 가 깨짐. **$L$-layer 후 gradient norm 이 $O(L)$ 로 누적/감소** (Xiong 2020, Ch2-03).

### 정리 1.3 — Residual 의 표현력 보존

$y = x + f(x)$ 형태는 $f(x) = 0$ 시 identity 함수. 따라서 sub-layer 는 **항등 함수 학습 자유** — 표현력은 단조 증가, 깊이 추가가 절대 손해 안 됨.

**증명**: $\text{Identity} \in \{x \mapsto x + f(x) : f \in \mathcal{F}\}$ ($f \equiv 0$). Block 추가는 expressive class 를 strict 하게 확장 (또는 같음) $\square$.

### 정리 1.4 — Universal Approximation (Yun 2020)

**Are Transformers universal approximators of sequence-to-sequence functions?**

Theorem (informal): Sufficient depth + width Transformer 는 임의의 continuous permutation-equivariant sequence-to-sequence function 을 임의 정밀도로 근사.

**Sketch**:
1. Attention 으로 token 간 임의 정보 mixing 가능
2. FFN 으로 token 별 임의 nonlinear 변환 (UAT)
3. 충분한 layer 로 합성

PE 추가 시 permutation-equivariance 제약 풀려 **임의 sequence-to-sequence function** 근사 가능.

### 정리 1.5 — Block 의 표현력 분해 (Token-mixing + Feature-mixing)

각 block 은:
- Attention: token-mixing matrix $A \in \mathbb{R}^{T \times T}$ 로 정보 교환
- FFN: per-token nonlinear $\phi: \mathbb{R}^d \to \mathbb{R}^d$

**MLP-Mixer (Tolstikhin 2021)** 는 attention 을 fixed token-mixing MLP 로 대체 — Transformer 의 부분 일반화. 두 axis 의 분리가 본질.

### 정리 1.6 — Residual 없는 Transformer 의 실패

Residual 제거 시:
- Gradient vanishing: deep 에서 학습 불가
- Identity 학습 불가능: 모든 layer 가 nonlinear transformation 강제
- 실증: 6+ layer 부터 학습 거의 불가

ResNet (He 2015) 의 동기와 정확히 같음 — depth 의 본질적 enabler.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Pre-LN Block 바닥부터 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class PreLNBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn  = FFN(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x_ln = self.ln1(x)
        x = x + self.attn(x_ln, x_ln, x_ln)[0]
        x = x + self.ffn(self.ln2(x))
        return x

class PostLNBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn  = FFN(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.ffn(x))
        return x

# 테스트
torch.manual_seed(0)
d, h, df = 64, 8, 256
block = PreLNBlock(d, h, df)
x = torch.randn(2, 10, d)
y = block(x)
print(f'Input: {x.shape}, Output: {y.shape}')   # (2, 10, 64)
```

### 실험 2 — Pre-LN vs Post-LN gradient norm 비교

```python
def measure_input_grad_norm(block_class, num_layers, d=64, h=8, df=256):
    blocks = nn.Sequential(*[block_class(d, h, df) for _ in range(num_layers)])
    x = torch.randn(1, 10, d, requires_grad=True)
    y = blocks(x)
    y.sum().backward()
    return x.grad.norm().item()

print(f'{"layers":>7} | {"Pre-LN |grad|":>14} | {"Post-LN |grad|":>15}')
print('-' * 42)
for L in [1, 4, 8, 12, 24]:
    g_pre  = measure_input_grad_norm(PreLNBlock, L)
    g_post = measure_input_grad_norm(PostLNBlock, L)
    print(f'{L:7d} | {g_pre:14.4f} | {g_post:15.4f}')
# Post-LN 의 gradient norm 이 layer 증가에 따라 더 빠르게 변화 (vanish or explode)
```

### 실험 3 — Residual 있고 없을 때 deep training 시뮬레이션

```python
class BlockNoResidual(nn.Module):
    def __init__(self, d, h, df):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn  = FFN(d, df)
        self.ln1  = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        x = self.ln1(self.attn(x, x, x)[0])   # NO residual
        x = self.ln2(self.ffn(x))
        return x

# Deep stacking
torch.manual_seed(0)
blocks_with    = nn.Sequential(*[PreLNBlock(64, 8, 256) for _ in range(12)])
blocks_without = nn.Sequential(*[BlockNoResidual(64, 8, 256) for _ in range(12)])

x = torch.randn(1, 10, 64)
y_with    = blocks_with(x)
y_without = blocks_without(x)
print(f'With residual    activation norm: {y_with.norm():.4f}')
print(f'Without residual activation norm: {y_without.norm():.4f}')
# Without residual 시 norm 이 layer 마다 unstable
```

### 실험 4 — Identity 학습 가능성 확인

```python
# Block 의 weight 가 0 이면 identity 인지
torch.manual_seed(0)
block = PreLNBlock(64, 8, 256)

# 모든 sub-layer weight 를 0 으로 (실제로는 작은 값)
for p in block.parameters():
    p.data.zero_()

x = torch.randn(1, 10, 64)
y = block(x)
diff = (y - x).abs().mean()
print(f'Pre-LN with zero weights: |y - x| = {diff:.6f}')   # ≈ 0 (LN 의 γ, β 효과만)

# Post-LN 도 비교
block_post = PostLNBlock(64, 8, 256)
for p in block_post.parameters():
    p.data.zero_()
y_post = block_post(x)
diff_post = (y_post - x).abs().mean()
print(f'Post-LN with zero weights: |y - x| = {diff_post:.6f}')
# Post-LN 은 LN 이 outer 라 identity 보존 어려움
```

### 실험 5 — Block 시각화 (information flow)

```python
import matplotlib.pyplot as plt

# Block forward 의 각 단계 activation 추적
class TracedBlock(PreLNBlock):
    def forward(self, x):
        h0 = x
        h1 = self.ln1(x)
        h2 = self.attn(h1, h1, h1)[0]
        h3 = x + h2                    # residual 1
        h4 = self.ln2(h3)
        h5 = self.ffn(h4)
        h6 = h3 + h5                   # residual 2
        return [h0, h1, h2, h3, h4, h5, h6]

torch.manual_seed(0)
block = TracedBlock(64, 8, 256)
x = torch.randn(1, 10, 64)
hs = block(x)

names = ['x', 'LN(x)', 'Attn', 'x+Attn', 'LN(x+Attn)', 'FFN', 'output']
norms = [h.norm().item() for h in hs]

plt.figure(figsize=(9, 4))
plt.bar(names, norms)
plt.ylabel('||h||'); plt.title('Pre-LN block: activation norms')
plt.xticks(rotation=30); plt.tight_layout(); plt.show()
```

---

## 🔗 실전 활용

### 1. Pre-LN 의 모던 표준

GPT-2, GPT-3, LLaMA, PaLM 모두 Pre-LN. 이유:
- Warmup 없이 안정적 (Xiong 2020, Ch2-03)
- 큰 LR 가능 (학습 빠름)
- Deep stacking (96+ layer) 가능

Post-LN 은 BERT 같은 일부 초기 모델, 그러나 modern recipe 는 Pre-LN.

### 2. RMSNorm (LLaMA)

LayerNorm 의 mean centering 제거:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma
$$

- 성능 거의 동일, 7-10% 빠름
- LLaMA, Mistral 등 채택

### 3. Sandwich-LN (CogView)

$y = \text{LN}(\text{LN}(x) + f(\text{LN}(x)))$ — Pre + Post 결합. 매우 deep 모델에서 안정.

### 4. DeepNorm (Wang 2022)

$y = \text{LN}(\alpha x + f(x))$ with $\alpha > 1$ — 1000+ layer 학습 가능.

### 5. Block 의 GPU efficient implementation

- **Fused kernels**: Attention + FFN + LN 을 하나의 CUDA kernel 로 (Megatron, Flash Attention)
- **Mixed precision**: FP16/BF16 forward, FP32 master weight (Ch4-05)
- **Activation checkpointing**: gradient 저장 메모리 절약

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Self-attention 만 (encoder/decoder self) | Decoder 는 cross-attention 추가 (Ch2-05) |
| Pre-LN 또는 Post-LN | DeepNorm, Sandwich 등 변형 |
| ReLU FFN | GELU, SwiGLU 변형 더 우수 (Ch2-02) |
| 단일 token mixing (attention) | MoE 같은 conditional computation 변형 |
| LN per-token | BatchNorm, GroupNorm 등 alternative (Transformer 에는 LN 표준) |

---

## 📌 핵심 정리

$$\boxed{\text{Pre-LN: } \quad x' = x + \text{Attn}(\text{LN}(x)), \quad y = x' + \text{FFN}(\text{LN}(x'))}$$

$$\boxed{\text{Post-LN: } \quad x' = \text{LN}(x + \text{Attn}(x)), \quad y = \text{LN}(x' + \text{FFN}(x'))}$$

| 요소 | 역할 |
|------|------|
| **Attention** | Token 간 정보 교환 (mixing) |
| **FFN** | Token 별 비선형 변환 |
| **LayerNorm** | 분포 안정화, 분산 ≈ 1 |
| **Residual** | Gradient highway, identity 학습 가능 |
| **Pre-LN** | Modern 표준, warmup 불필요 |
| **Post-LN** | 원전, warmup 필수 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Pre-LN block 의 forward 식을 step-by-step 으로 6 step 이내에 적고, 각 step 의 차원을 표시하라.

<details>
<summary>해설</summary>

입력: $x \in \mathbb{R}^{T \times d}$.

1. $\tilde{x}_1 = \text{LN}_1(x) \in \mathbb{R}^{T \times d}$
2. $a = \text{MHA}(\tilde{x}_1) \in \mathbb{R}^{T \times d}$
3. $x' = x + a \in \mathbb{R}^{T \times d}$ (residual 1)
4. $\tilde{x}_2 = \text{LN}_2(x') \in \mathbb{R}^{T \times d}$
5. $f = \text{FFN}(\tilde{x}_2) \in \mathbb{R}^{T \times d}$
6. $y = x' + f \in \mathbb{R}^{T \times d}$ (residual 2)

차원은 모든 step 에서 동일 (Transformer 의 핵심 invariance — uniform width). $\square$

</details>

**문제 2** (심화): Pre-LN block 의 $\partial y / \partial x$ 를 분해하라. $L$-layer 쌓을 때 effective gradient 가 어떻게 작동하는지 분석하라.

<details>
<summary>해설</summary>

Pre-LN block:
$$
y = x + g_1(x) + g_2(x + g_1(x))
$$

where $g_1 = \text{Attn} \circ \text{LN}_1$, $g_2 = \text{FFN} \circ \text{LN}_2$.

$$
\frac{\partial y}{\partial x} = I + \frac{\partial g_1}{\partial x} + \frac{\partial g_2}{\partial x'} \cdot \frac{\partial x'}{\partial x}
$$

with $x' = x + g_1(x)$ → $\partial x'/\partial x = I + \partial g_1 / \partial x$.

**핵심**: $I$ 항이 항상 존재 → identity gradient highway.

**$L$-layer**:
$$
\frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{l=1}^L \left( I + (\text{block Jacobian}) \right)
$$

각 항이 $I + \delta$ 형태 → product 가 $I + O(L \delta)$, 만약 $\delta$ 가 small 이면 stable.

**Post-LN 비교**: $\partial y/\partial x = \text{LN}_J \cdot (I + f_J)$, LN-Jacobian 이 saturated 시 → highway 깨짐, $L$-layer 후 gradient 가 $O(\prod \text{LN}_J^{(l)})$ 로 누적 감쇠.

따라서 Pre-LN 이 deep stacking 의 직접적 enabler. Ch2-03 에서 자세히. $\square$

</details>

**문제 3** (논문 비평): MLP-Mixer (Tolstikhin 2021) 는 Attention 을 token-mixing MLP 로 대체한다. 이 변형이 Transformer block 의 어떤 본질을 보존하고 어떤 것을 잃는가? 또한 이것이 Transformer 의 inductive bias 분석에 어떤 통찰을 주는가?

<details>
<summary>해설</summary>

**MLP-Mixer 의 구조**:
- Token-mixing MLP: $T \times T$ matrix (vs attention 의 학습된 $A$)
- Channel-mixing MLP: 같은 FFN

**보존되는 것**:
1. **Two-axis 구조**: token-mixing + feature-mixing 분리 — Transformer 의 본질
2. **Residual + LN**: gradient highway
3. **표현력**: 충분한 width 시 universal approximator 가능

**잃는 것**:
1. **Data-dependent mixing**: Attention 은 input-dependent $A$, MLP-Mixer 는 fixed (token position 만 의존)
2. **Variable length**: MLP-Mixer 는 fixed token 수 가정 (FC layer)
3. **Permutation equivariance**: PE 효과가 학습된 attention 으로 자연스럽게 결합 안 됨

**Inductive Bias 통찰**:

- **Attention 이 essential 이 아님**: Token-mixing 만 있으면 Transformer-like 성능 가능 (vision)
- **Attention 의 가치 = data-dependent mixing**: input 에 따라 다른 $A$ — 이것이 NLP 의 변동성·복잡성에 적합
- **Vision: simpler mixing 충분** — image patch 의 spatial structure 가 약한 inductive bias
- **NLP: attention 의 flexibility 필수** — 문장의 syntactic / semantic 다양성

**모던 통찰**:
- **MLP-Mixer / ConvMixer / FNet (Lee-Thorp 2022)**: vision 에서 attention 대체 가능
- **NLP**: attention 의 data-dependent 특성이 여전히 우월
- **Transformer ⊃ MLP-Mixer**: attention 이 더 일반적인 mixing

따라서 "Attention is all you need" 가 아닐 수 있다 — **two-axis structure + data-dependent mixing for hard cases** 가 핵심 inductive bias. Ch5 의 efficient attention 들도 이 두 axis 를 보존하면서 mixing 의 sparsity 만 조정. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch1-attention-decomposition/06-interpretability-debate.md) | [📚 README](../README.md) | [다음 ▶](./02-ffn-role.md)

</div>
