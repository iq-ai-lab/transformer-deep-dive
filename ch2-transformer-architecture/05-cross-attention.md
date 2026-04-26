# 05. Cross-Attention 메커니즘

## 🎯 핵심 질문

- Cross-attention 의 정의 — Q 가 decoder, K/V 가 encoder 인 비대칭 attention 의 의미는?
- Self-attention 과 cross-attention 은 같은 식이지만 어떤 정보 흐름의 차이가 있는가?
- Translation 에서 encoder-decoder cross-attention 이 alignment 와 어떻게 등가인가?
- Multi-modal (CLIP, BLIP 등) 에서 cross-attention 이 어떻게 modality 간 bridge 역할을 하는가?
- Decoder-only 모델에서 cross-attention 부재가 conditional generation 에 어떤 영향을 주는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Cross-attention 은 **두 sequence 간의 information bridge** 입니다:

1. **Translation 의 핵심** — encoder 의 source, decoder 의 target 을 연결
2. **Soft alignment** — Bahdanau Attention 의 일반화, 명시적 alignment 학습 불필요
3. **Multi-modal fusion** — image-text, audio-text 같은 modality 간 정보 통합
4. **Encoder-decoder 의 정체성** — cross-attention 이 없으면 encoder-decoder ≈ decoder-only

이 문서는 cross-attention 의 **수학적 정의, alignment 해석, multi-modal 응용** 을 정리합니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md) — Q, K, V 분해
- 이전 문서: [04-encoder-vs-decoder.md](./04-encoder-vs-decoder.md) — encoder-decoder 구조
- (선택) [RNN & LSTM Deep Dive](https://github.com/iq-ai-lab/rnn-lstm-deep-dive): Bahdanau Attention 의 origin

---

## 📖 직관적 이해

### Self vs Cross

```
Self-attention:           Cross-attention:
  X ─→ Q                    X_dec ─→ Q
  X ─→ K  같은 X에서!        X_enc ─→ K  다른 X에서!
  X ─→ V                    X_enc ─→ V
```

- **Self**: 한 sequence 가 자기 자신에 attend (정보 mixing 내부)
- **Cross**: 한 sequence (Q) 가 다른 sequence (K, V) 에 attend (정보 가져옴)

### Translation 에서의 동작

```
Encoder:    "The cat sat"  ──→  z = [z_The, z_cat, z_sat]
Decoder:    "Le chat" → "_"
            (each decoder token queries z)

  Q (decoder) = "Le chat" 의 hidden
  K, V (encoder) = z

  Cross-attention 이 alignment 학습:
    "Le"   ── attends ──→ "The"
    "chat" ── attends ──→ "cat"
```

### Multi-modal Bridge

```
Vision encoder:   image_patches ──→ [v_1, ..., v_N]
Text decoder:     "Caption: a"

  Q = decoder text
  K, V = image patches

  → Decoder 가 image 를 query, caption 생성
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Cross-Attention

Source sequence $Z \in \mathbb{R}^{T_z \times d}$, target sequence $X \in \mathbb{R}^{T_x \times d}$:
$$
\text{CrossAttn}(X, Z) = \text{softmax}\!\left(\frac{(X W_Q)(Z W_K)^\top}{\sqrt{d_k}}\right) (Z W_V)
$$

- $Q = X W_Q \in \mathbb{R}^{T_x \times d_k}$ (target 에서)
- $K = Z W_K \in \mathbb{R}^{T_z \times d_k}$ (source 에서)
- $V = Z W_V \in \mathbb{R}^{T_z \times d_v}$ (source 에서)

출력: $\mathbb{R}^{T_x \times d_v}$ — target 길이 유지.

### 정의 5.2 — Encoder-Decoder Block (T5, BART)

Decoder block 이 cross-attention 추가:
$$
\begin{aligned}
x' &= x + \text{SelfAttn}(\text{LN}(x); M^{\text{causal}}) \\
x'' &= x' + \text{CrossAttn}(\text{LN}(x'), \text{LN}(z)) \\
y &= x'' + \text{FFN}(\text{LN}(x''))
\end{aligned}
$$

with $z$ 가 encoder 출력 (frozen during decoder forward).

### 정의 5.3 — Alignment Matrix

Cross-attention weight $A \in \mathbb{R}^{T_x \times T_z}$:
$$
A_{ij} = \text{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)_j
$$

$A_{ij}$ = target token $i$ 가 source token $j$ 에 부여하는 alignment weight.

### 정의 5.4 — Coverage Constraint

Translation 에서 **coverage**: source 의 각 token 이 충분히 attended 됐는가:
$$
c_j = \sum_{i=1}^{T_x} A_{ij}
$$

Ideal: $c_j \approx 1$ for all $j$. 실패 시: under-translation 또는 over-translation.

---

## 🔬 정리와 증명

### 정리 5.1 — Cross-Attention = Self-Attention 의 일반화

$Z = X$ 일 때 cross-attention 은 self-attention:
$$
\text{CrossAttn}(X, X) = \text{SelfAttn}(X)
$$

**증명**: Q, K, V 모두 같은 $X$ 에서 → 정의상 self-attention $\square$.

따라서 self 는 cross 의 special case. 그러나 inductive bias 가 다름 — self 는 within-sequence mixing, cross 는 between-sequence bridge.

### 정리 5.2 — Encoder Output 의 Frozen 성질

Decoder forward 에서 encoder 출력 $z$ 는 fix:
$$
z = \text{Encoder}(\text{source}) \quad \text{(computed once)}
$$

각 decoder layer 의 cross-attention 은 같은 $z$ 에 attend.

**효율 의미**: KV cache 의 자연스러운 적용 — encoder 의 K, V 를 한 번 계산하고 모든 decoder layer · 모든 generation step 에서 재사용.

### 정리 5.3 — Bahdanau Attention 과의 등가성

Bahdanau 2014 의 attention:
$$
e_{ij} = a^\top \tanh(W_q s_i + W_k h_j), \quad \alpha_{ij} = \text{softmax}(e_{ij})
$$

(additive attention, RNN 기반)

Vaswani 의 dot-product cross-attention:
$$
e_{ij} = q_i^\top k_j / \sqrt{d_k}, \quad \alpha_{ij} = \text{softmax}(e_{ij})
$$

**같은 framework**: 모두 (decoder query, encoder key) → score → weighted (encoder value). 차이는:
- Bahdanau: additive scoring + RNN encoder
- Vaswani: dot-product scoring + Transformer encoder

따라서 Vaswani 의 cross-attention = **scaled dot-product 변형의 Bahdanau** $\square$.

### 정리 5.4 — Cross-Attention 의 Asymmetry

$\text{CrossAttn}(X, Z) \neq \text{CrossAttn}(Z, X)$

**증명**: 출력 차원이 다름 ($T_x \times d_v$ vs $T_z \times d_v$). 또한 $Q, K, V$ 의 출처가 다르므로 학습된 mapping 도 달라짐 $\square$.

이 비대칭성이 Q (target) → K, V (source) 의 정보 흐름 방향을 결정.

### 정리 5.5 — Cross-Attention as Soft Lookup

$$
\text{CrossAttn}_i = \sum_{j=1}^{T_z} A_{ij} v_j
$$

Target token $i$ 의 출력 = source token 들의 weighted average. **Soft dictionary lookup**: $q_i$ 가 query, $k_j$ 가 keys, $v_j$ 가 values.

→ Differentiable retrieval system, gradient 가 alignment 로 흐름.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Cross-Attention 바닥부터

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, z):
        """x: target (B, T_x, d), z: source (B, T_z, d)"""
        B, T_x, d = x.size(); T_z = z.size(1)
        Q = self.W_Q(x).view(B, T_x, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(z).view(B, T_z, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(z).view(B, T_z, self.h, self.d_k).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T_x, d)
        return self.W_O(out), attn

# 테스트
torch.manual_seed(0)
ca = CrossAttention(64, 8)
x = torch.randn(1, 5, 64)   # target: 5 tokens
z = torch.randn(1, 8, 64)   # source: 8 tokens
out, attn = ca(x, z)
print(f'Target: {x.shape}, Source: {z.shape}')
print(f'Output: {out.shape}, Attention: {attn.shape}')   # (1,5,64), (1,8,5,8)
```

### 실험 2 — `nn.MultiheadAttention` 의 cross-attention 모드

```python
mha = nn.MultiheadAttention(64, 8, bias=False, batch_first=True)
out, attn = mha(x, z, z)   # Q=x, K=V=z
print(f'PyTorch cross-attn output: {out.shape}, attn: {attn.shape}')
# Same as above (단, attn 은 head-averaged 가 default)
```

### 실험 3 — Translation Alignment 시각화

```python
# 가상의 영-한 alignment
en_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
ko_tokens = ['고양이가', '매트', '위에', '앉았다']

T_en, T_ko = len(en_tokens), len(ko_tokens)

# 학습 후 기대되는 alignment (ideal)
ideal_alignment = np.array([
    [0.05, 0.7, 0.05, 0.05, 0.05, 0.10],   # 고양이가 → cat
    [0.05, 0.05, 0.05, 0.05, 0.20, 0.60],  # 매트 → mat
    [0.05, 0.05, 0.05, 0.75, 0.05, 0.05],  # 위에 → on
    [0.05, 0.05, 0.85, 0.00, 0.00, 0.05],  # 앉았다 → sat
])

plt.figure(figsize=(8, 4))
plt.imshow(ideal_alignment, cmap='Blues', aspect='auto')
plt.xticks(range(T_en), en_tokens); plt.yticks(range(T_ko), ko_tokens)
plt.xlabel('Source (English)'); plt.ylabel('Target (Korean)')
plt.title('Cross-attention as soft alignment (translation)')
plt.colorbar()
for i in range(T_ko):
    for j in range(T_en):
        plt.text(j, i, f'{ideal_alignment[i,j]:.2f}', ha='center', va='center',
                 color='white' if ideal_alignment[i,j] > 0.4 else 'black', fontsize=8)
plt.tight_layout(); plt.show()
```

### 실험 4 — Cross-Attention 의 Gradient Flow

```python
# Encoder output 의 gradient 가 어떻게 흐르는지
torch.manual_seed(0)
encoder_out = torch.randn(1, 8, 64, requires_grad=True)
decoder_in = torch.randn(1, 5, 64, requires_grad=True)
ca = CrossAttention(64, 8)
out, attn = ca(decoder_in, encoder_out)
out.sum().backward()

print(f'Encoder gradient norm: {encoder_out.grad.norm():.4f}')
print(f'Decoder gradient norm: {decoder_in.grad.norm():.4f}')
# 둘 다 nonzero → cross-attention 으로 gradient 가 양쪽 흐름
```

### 실험 5 — Encoder-Decoder Block 통합

```python
class EncDecBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln3 = nn.LayerNorm(d)
    
    def forward(self, x, z):
        # 1. Causal self-attention
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
        x_ln = self.ln1(x)
        x = x + self.self_attn(x_ln, x_ln, x_ln, attn_mask=causal)[0]
        # 2. Cross-attention to encoder
        x = x + self.cross_attn(self.ln2(x), z, z)[0]
        # 3. FFN
        x = x + self.ffn(self.ln3(x))
        return x

torch.manual_seed(0)
block = EncDecBlock(64, 8, 256)
x_dec = torch.randn(1, 5, 64)
z_enc = torch.randn(1, 8, 64)
y = block(x_dec, z_enc)
print(f'Encoder-Decoder output: {y.shape}')   # (1, 5, 64)

# Parameter 수 비교
print(f'Encoder-Decoder block params: {sum(p.numel() for p in block.parameters())}')
# vs decoder-only block 보다 cross-attention 만큼 더 많음
```

---

## 🔗 실전 활용

### 1. T5 / BART 의 Cross-Attention

T5 (Raffel 2020): 12 encoder + 12 decoder layer (base). 각 decoder layer 에 cross-attention.
- Encoder K, V 는 한 번 계산 → 모든 decoder layer + 모든 generation step 에서 재사용
- Inference 시 KV cache 가 매우 효과적

### 2. 음성 인식 (Whisper)

OpenAI Whisper: encoder = audio mel-spectrogram, decoder = text.
- Cross-attention 이 audio frame 과 text token alignment 학습
- 학습된 alignment 가 timestamp 추출에 활용

### 3. Multi-modal (CLIP, BLIP)

- **CLIP**: 별도 encoder 두 개 (image, text) → contrastive loss, cross-attention 명시적 X
- **BLIP-2**: Q-Former — learnable query 가 image features 에 cross-attend, text decoder 로 전달
- **Flamingo (DeepMind)**: Frozen LLM + interleaved cross-attention to vision encoder

### 4. Retrieval-Augmented Generation (RAG)

검색된 document 를 cross-attention 의 K, V 로:
- $X$ = generation context
- $Z$ = retrieved passages
- Decoder 가 retrieved 정보를 cross-attend → factual grounding

### 5. Decoder-only LLM 의 conditional generation 우회

GPT 류는 cross-attention 없음. Conditional generation 은:
- Concatenation: "[Source]: ... [Target]: ..."
- ICL: few-shot examples 로 implicit conditioning
- Prompting: instruction 형태로

이 방법들이 cross-attention 만큼 효과적인 이유는 **충분한 scale + 학습** 으로 implicit cross-attention 학습 가능 (Wang 2022).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| $T_x, T_z$ 독립 | Both 가 길면 cross-attention 비용 $O(T_x T_z)$ |
| Encoder 가 frozen during decoder | 일부 변형이 joint training |
| Single source | Multi-source attention (multi-document) 가능 |
| Same dimension | 서로 다른 dim 시 추가 projection 필요 |
| Soft alignment | Hard alignment (HMM 등) 도 가능 — Transformer 는 softmax |

---

## 📌 핵심 정리

$$\boxed{\text{CrossAttn}(X, Z) = \text{softmax}\!\left(\frac{(XW_Q)(ZW_K)^\top}{\sqrt{d_k}}\right) (ZW_V)}$$

| 양 | 정의 | 출처 |
|----|------|------|
| Q | $X W_Q$ | Target (decoder) |
| K, V | $Z W_K, Z W_V$ | Source (encoder) |
| Output dim | $T_x \times d_v$ | Target 길이 유지 |
| Attention shape | $T_x \times T_z$ | Alignment matrix |
| Complexity | $O(T_x T_z d_k)$ | 두 길이 곱 |

| 응용 | 두 sequence |
|------|------------|
| Translation | Source ↔ target |
| Speech recognition | Audio ↔ text |
| Image captioning | Image patches ↔ text |
| RAG | Context ↔ retrieved docs |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T_x = 4, T_z = 6, d = 8, d_k = 8$ 인 cross-attention 의 Q, K, V, attention matrix 의 shape 를 구체적으로 적어라. 각 step 의 차원을 추적하라.

<details>
<summary>해설</summary>

- $X \in \mathbb{R}^{4 \times 8}$ (target), $Z \in \mathbb{R}^{6 \times 8}$ (source)
- $W_Q, W_K, W_V \in \mathbb{R}^{8 \times 8}$
- $Q = X W_Q \in \mathbb{R}^{4 \times 8}$
- $K = Z W_K \in \mathbb{R}^{6 \times 8}$
- $V = Z W_V \in \mathbb{R}^{6 \times 8}$
- $S = QK^\top / \sqrt{8} \in \mathbb{R}^{4 \times 6}$
- $A = \text{softmax}(S) \in \mathbb{R}^{4 \times 6}$ (each row sums to 1)
- $\text{Output} = A V \in \mathbb{R}^{4 \times 8}$

Target 길이 $4$ 보존 ✓ — 이것이 cross-attention 의 핵심: target 에 source 정보 주입, target 길이는 그대로. $\square$

</details>

**문제 2** (심화): Decoder-only 모델 (GPT) 가 cross-attention 없이 conditional generation 을 어떻게 implicit 하게 학습하는가? Concatenation 방식이 cross-attention 과 정확히 같은 표현력을 갖는지, 아니면 한계가 있는지 분석하라.

<details>
<summary>해설</summary>

**Concatenation 방식**:

Input 을 "[source] [SEP] [target]" 으로 concat, causal LM 으로 학습.

**자연스러운 conditional generation**:
- Causal mask 로 target 부분이 source 를 attend 가능 (이전 token 들을 봄)
- Generation 시 source + target prefix 를 prompt 로 제공

**정확히 cross-attention 과 같은가?**

**같은 점**:
- Decoder 가 source 정보 활용 가능
- 충분한 scale 에서 same expressivity (Wang 2022)
- Single sequence 처리, 단순한 architecture

**다른 점**:

1. **Self-attention 만 사용** — source 가 target 의 self-attention 안에서 처리. 같은 attention head 에서 source-source, target-target, target-source 모두.
2. **Computation**: source 부분이 매 generation step 마다 K, V 계산 (KV cache 로 mitigate). Cross-attention 은 source K, V 를 한 번만.
3. **Inductive bias**: Cross-attention 은 명시적으로 "source → target" 정보 흐름. Concatenation 은 implicit — 학습이 더 어려울 수 있지만 scale 로 극복.

**Wang 2022 의 ablation**:
- 작은 scale (~1B): encoder-decoder 가 약간 우수
- 큰 scale (~10B+): decoder-only + concatenation 이 동등하거나 우수
- ICL 능력은 decoder-only 가 더 자연스러움

**Modern 결론**:

Decoder-only 의 단순함 + scale 이 cross-attention 의 inductive bias 를 학습으로 대체. 그러나 specific task (translation, structured generation) 는 encoder-decoder 가 여전히 효율적 — task 의 inductive bias 가 명확할 때.

**한 줄 요약**: Concatenation = scale 의 힘으로 cross-attention 의 효과 emergent. Architecture 는 simpler, training 은 harder, scale 에서 그 trade-off 가 worth. $\square$

</details>

**문제 3** (논문 비평): Multi-modal 모델 (Flamingo, BLIP-2) 이 cross-attention 으로 vision-language 를 연결하는 방식과 native multimodal (GPT-4V, Gemini) 이 single sequence 로 처리하는 방식을 비교 분석하라. 각각의 inductive bias 와 scaling 의 trade-off 는?

<details>
<summary>해설</summary>

**Cross-Attention 방식 (Flamingo, BLIP-2)**:

```
Vision Encoder (frozen) ─→ image features
                              │
                              │ (cross-attention)
                              ↓
LLM Decoder (frozen / partial) ── text generation
```

- **장점**:
  - Frozen LLM 활용 — 새 모달리티 추가에 최소 비용
  - Specialized vision encoder (CLIP, ViT) 활용
  - Modality 간 명확한 분리

- **단점**:
  - Cross-attention 만큼만 정보 흐름 — bottleneck
  - 각 layer 에 cross-attention 추가 → 추가 파라미터
  - 두 modality 가 다른 representation space (불일치)

**Single Sequence 방식 (GPT-4V, Gemini)**:

```
Image patches → tokens → ─┐
                          │
Text tokens →            ─┴──→ Single sequence → Decoder LM
```

- Image patch 를 text 와 같은 embedding space 로
- Self-attention 하나에서 image-text 자유 mixing
- Decoder-only 로 unified

- **장점**:
  - Cross-modal interaction 이 자유로움 (모든 layer 에서)
  - Architecture 단순 — scaling 쉬움
  - ICL 자연스럽게 (image 와 text 같은 prompt)
  - Modality 간 distinction 학습으로 emergent

- **단점**:
  - 처음부터 multimodal 학습 필요 (frozen LLM 활용 제한)
  - Image tokenizer 학습 / 선택 critical
  - Sequence 길이 빠르게 증가 (image 가 많은 token 차지)

**Inductive Bias 비교**:

| 방식 | Inductive Bias | 학습 난이도 |
|------|---------------|------------|
| Cross-attention | Modality 분리 + bridge | 작은 데이터로 가능 (frozen 활용) |
| Single sequence | No prior, free mixing | 큰 데이터 필요 (scale 의존) |

**Scaling Trade-off**:

- **Small scale**: Cross-attention 우수 (inductive bias 가 sample efficiency 증가)
- **Large scale**: Single sequence 우수 (free mixing 의 표현력)
- **Engineering**: Single sequence 가 simpler — scaling laws (Ch7-01) 적용 더 명확

**Modern 추세**:

- Frontier models (GPT-4V, Gemini, Claude 3) 가 single sequence 채택
- Open source (LLaVA, MiniGPT-4) 가 cross-attention 유사 (frozen vision + adapter)
- Trade-off: 비용 (frontier) vs accessibility (open)

**Ch5-04 sparse attention 과의 연결**: Long multimodal sequence 에서 attention sparsity 가 critical. Cross-attention 은 자연스럽게 "modality A only attend modality B" 의 sparsity 를 강제, single sequence 는 학습으로. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-encoder-vs-decoder.md) | [📚 README](../README.md) | [다음 ▶](../ch3-positional-encoding/01-pe-necessity.md)

</div>
