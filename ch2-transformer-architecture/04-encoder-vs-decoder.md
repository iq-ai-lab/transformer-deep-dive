# 04. Encoder vs Decoder 의 차이

## 🎯 핵심 질문

- Encoder 와 Decoder 는 같은 Transformer block 인데 무엇이 다른가? — 답은 **masking matrix** 하나뿐
- Causal mask 의 수학적 정의는? Lower-triangular 가 왜 미래 차단을 의미하는가?
- BERT (encoder-only) 와 GPT (decoder-only) 가 architectural 으로 거의 동일한데 왜 task 가 다른가?
- Decoder 의 cross-attention 추가는 어떤 정보 흐름을 만드는가? (Ch2-05 에서 자세히)
- Bidirectional (BERT) vs Causal (GPT) attention 의 표현력·태스크 적합성 차이는?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Encoder/Decoder 의 구분은 **architecture 가 아니라 attention mask** 의 차이입니다:

1. **같은 block, 다른 mask** — 한 코드베이스로 BERT, GPT 모두 구현 가능
2. **Bidirectional → NLU**: 양쪽 context 활용 → 이해 task (분류, NER)
3. **Causal → Generation**: 미래 차단 → autoregressive generation
4. **Cross-attention 의 추가**: encoder-decoder (T5) 의 핵심 — Q 가 decoder, K/V 가 encoder

이 문서는 **mask matrix 의 명시적 분석** 으로 두 변형의 차이를 분해하고, 각각의 task 적합성을 정리합니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md) — masking 정의
- 이전 문서: [01-transformer-block.md](./01-transformer-block.md) — block 구조
- 정보이론: Conditional probability $p(x_t | x_{<t})$, autoregressive factorization

---

## 📖 직관적 이해

### Bidirectional Attention (Encoder)

```
   x_1   x_2   x_3   x_4   x_5
    │     │     │     │     │
    └──┬──┴──┬──┴──┬──┴──┬──┘
       │     │     │     │
   각 token 이 모든 다른 token 에 attend
   Mask = ones(T, T)  (어디든 attend 가능)
```

$T$ tokens 가 동시에 모든 다른 token 정보를 사용 — "이해" 에 적합.

### Causal Attention (Decoder)

```
   x_1   x_2   x_3   x_4   x_5
    │     │     │     │     │
    │     ←──   ←─────  ←───── (각 token 이 자신과 이전만 봄)
    └─→   ✗    ✗    ✗    ✗
          └─→  ✗    ✗    ✗
                └─→  ✗    ✗
                      └─→  ✗
   Mask = lower-triangular   (미래 차단)
```

$x_t$ 는 $x_{t}, x_{t-1}, \ldots, x_1$ 만 attend — autoregressive.

### Mask Matrix 시각화

```
Bidirectional:        Causal:
[1 1 1 1 1]         [1 0 0 0 0]
[1 1 1 1 1]         [1 1 0 0 0]
[1 1 1 1 1]         [1 1 1 0 0]
[1 1 1 1 1]         [1 1 1 1 0]
[1 1 1 1 1]         [1 1 1 1 1]
```

Mask 가 0 인 위치 → score 에 $-\infty$ 추가 → softmax 후 0.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Attention Mask

$M \in \{0, -\infty\}^{T \times T}$ (또는 $\{0, 1\}$ with 다른 convention).

$$
S^{\text{masked}} = S + M
$$

$M_{ij} = -\infty$ 면 $A_{ij} = 0$.

### 정의 4.2 — Bidirectional Mask (Encoder)

$$
M^{\text{bi}}_{ij} = 0 \quad \forall i, j \in [T]
$$

(또는 padding mask 만 있고 나머지는 0)

### 정의 4.3 — Causal Mask (Decoder)

$$
M^{\text{causal}}_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}
$$

행렬: lower-triangular ones, upper-triangular $-\infty$.

### 정의 4.4 — Encoder Block

Pre-LN encoder block:
$$
\begin{aligned}
x' &= x + \text{MHA}(\text{LN}(x), \text{LN}(x), \text{LN}(x); M^{\text{bi}}) \\
y &= x' + \text{FFN}(\text{LN}(x'))
\end{aligned}
$$

### 정의 4.5 — Decoder Block (Encoder-Decoder)

Decoder 는 두 attention sub-layer 를 가짐:
1. **Self-attention with causal mask**: decoder 내 미래 차단
2. **Cross-attention**: encoder 출력에 attend (Ch2-05)

$$
\begin{aligned}
x' &= x + \text{MHA}^{\text{self}}(\text{LN}(x); M^{\text{causal}}) \\
x'' &= x' + \text{MHA}^{\text{cross}}(\text{LN}(x'), \text{LN}(z), \text{LN}(z)) \\
y &= x'' + \text{FFN}(\text{LN}(x''))
\end{aligned}
$$

with $z$ 가 encoder 출력.

### 정의 4.6 — Decoder-only Block (GPT)

Cross-attention 제거, self-attention 에 causal mask 만:
$$
\begin{aligned}
x' &= x + \text{MHA}(\text{LN}(x); M^{\text{causal}}) \\
y &= x' + \text{FFN}(\text{LN}(x'))
\end{aligned}
$$

---

## 🔬 정리와 증명

### 정리 4.1 — Causal Mask 의 Autoregressive Factorization

Decoder 의 출력 $h^{(L)}_t$ 는 오직 $x_1, \ldots, x_t$ 에만 의존:
$$
h^{(L)}_t = f(x_{1:t})
$$

**증명** (induction on layer):

**Base** ($L = 0$): $h^{(0)}_t = x_t$ — trivially $f(x_t)$.

**Inductive step**: $h^{(l)}_t$ 가 $x_{1:t}$ 에만 의존 가정. Causal attention:
$$
h^{(l+1)}_t = \sum_{j=1}^t A_{tj}^{(l)} v_j^{(l)}
$$

($A_{tj} = 0$ for $j > t$). $v_j^{(l)} = h^{(l)}_j W_V$ 가 $x_{1:j} \subseteq x_{1:t}$ 에만 의존. 따라서 $h^{(l+1)}_t$ 도 $x_{1:t}$ 에만 의존. $\square$

**의미**: Autoregressive language modeling $p(x_t | x_{<t})$ 를 정확히 구현.

### 정리 4.2 — Causal Attention 의 Output Independence

$x_{t+1}, \ldots, x_T$ 를 변경해도 $h^{(L)}_t$ 가 변하지 않음.

**증명**: 정리 4.1 의 따름 — $h^{(L)}_t = f(x_{1:t})$ 에서 $x_{>t}$ 는 안 들어감 $\square$.

이것이 **batch training 의 효율** 의 핵심 — causal mask 만 쓰면 한 sequence 의 모든 position 의 prediction 을 한 forward 에 계산.

### 정리 4.3 — Bidirectional 의 Symmetric Information Flow

Encoder 의 $h^{(L)}_t$ 는 $x_{1:T}$ 모두에 의존:
$$
h^{(L)}_t = f(x_{1:T}) \quad \forall t
$$

**의미**: 같은 입력에 대해 모든 token 이 같은 정보 access — bidirectional context.

**한계**: Generation 시 미래 token 정보 access 불가 (정의상) → autoregressive generation 어려움. BERT 가 generation 에 부적합한 이유.

### 정리 4.4 — Pre-fix Causal Mask

Hybrid: 일부 prefix 는 bidirectional, 그 후는 causal. T5 / UL2 등에서 사용.

$$
M^{\text{prefix}}_{ij} = \begin{cases} 0 & j \leq T_{\text{prefix}} \text{ or } i \geq j \\ -\infty & \text{otherwise} \end{cases}
$$

(prefix 영역은 bidirectional, 그 후는 causal)

### 정리 4.5 — Encoder vs Decoder 의 Parameter 차이

**Encoder block** (Pre-LN): MHA + FFN + 2 LN
**Decoder block** (Encoder-Decoder): self-MHA + cross-MHA + FFN + 3 LN

Cross-attention 의 추가로 decoder block 이 encoder block 보다 파라미터 ~50% 많음 (대략).

**Decoder-only (GPT)**: encoder block 과 같은 구조 — cross-attention 없음.

### 정리 4.6 — Computational Equivalence (Same Block, Different Mask)

같은 PyTorch implementation 으로 encoder/decoder 모두 표현 가능:
- `mask=None` → encoder
- `mask=causal_mask` → decoder

→ "Encoder/Decoder 의 차이 = mask" 가 정확히 코드 수준 truth.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Causal Mask 의 효과

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

T = 8
mask_bi    = torch.zeros(T, T)
mask_causal = torch.zeros(T, T)
mask_causal.masked_fill_(torch.triu(torch.ones(T, T), diagonal=1).bool(), float('-inf'))

print('Bidirectional mask:')
print(mask_bi)
print('\nCausal mask:')
print(mask_causal)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow((mask_bi == 0).float(), cmap='Blues')
axes[0].set_title('Bidirectional (encoder)')
axes[1].imshow((mask_causal == 0).float(), cmap='Blues')
axes[1].set_title('Causal (decoder)')
for ax in axes: ax.set_xlabel('Key'); ax.set_ylabel('Query')
plt.tight_layout(); plt.show()
```

### 실험 2 — Causal 의 Autoregressive 성질 검증

```python
torch.manual_seed(0)
d, h, T = 32, 4, 6

class CausalBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
    def forward(self, x):
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return self.attn(x, x, x, attn_mask=causal)[0]

block = CausalBlock(d, h)

# 원래 입력
x_orig = torch.randn(1, T, d)
out_orig = block(x_orig)

# 미래 token 만 변경 (position 4, 5)
x_modified = x_orig.clone()
x_modified[0, 4:] = torch.randn(2, d)
out_modified = block(x_modified)

# Position 0~3 은 변하지 않아야 함
diff_past = (out_orig[0, :4] - out_modified[0, :4]).abs().max()
diff_future = (out_orig[0, 4:] - out_modified[0, 4:]).abs().max()
print(f'Diff in past tokens (0-3):   {diff_past:.6f}  (should be 0)')
print(f'Diff in future tokens (4-5): {diff_future:.4f}  (changed)')
```

### 실험 3 — Bidirectional vs Causal Hidden Visualization

```python
torch.manual_seed(0)

class EncoderBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]

bi  = EncoderBlock(d, h)
ca  = CausalBlock(d, h)

# 가중치 동기화
ca.load_state_dict(bi.state_dict())

x = torch.randn(1, T, d)
y_bi    = bi(x)
y_causal = ca(x)

# 같은 가중치, 다른 mask → 다른 출력
print(f'Different by {((y_bi - y_causal) ** 2).mean().sqrt():.4f}')
# 같은 weight 라도 attention pattern 이 다르면 출력 다름
```

### 실험 4 — KV Cache (Causal 의 incremental computation)

```python
# Decoder 는 generation 시 한 token 씩 추가, 이전 K, V 는 reuse
torch.manual_seed(0)
d_model, num_heads = 32, 4
attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

# Full forward (한 번에 모두)
T_full = 5
x_full = torch.randn(1, T_full, d_model)
causal_mask = torch.triu(torch.ones(T_full, T_full), diagonal=1).bool()
out_full, _ = attn(x_full, x_full, x_full, attn_mask=causal_mask)

# Incremental: 한 token 씩
out_incr = []
x_seen = []
for t in range(T_full):
    x_t = x_full[:, t:t+1]
    x_seen.append(x_t)
    x_curr = torch.cat(x_seen, dim=1)
    T_curr = x_curr.size(1)
    cm = torch.triu(torch.ones(T_curr, T_curr), diagonal=1).bool()
    o, _ = attn(x_curr, x_curr, x_curr, attn_mask=cm)
    out_incr.append(o[:, -1:])
out_incr = torch.cat(out_incr, dim=1)

print(f'Full vs incremental: {(out_full - out_incr).abs().max():.6f}')
# 같음 — KV cache 정당화 (Ch5-06)
```

### 실험 5 — BERT vs GPT 차이를 mask 로 시뮬레이션

```python
# 같은 모델, mask 만 변경 시 학습 task 도 다름
torch.manual_seed(0)

# 가짜 sequence
T, d = 8, 16
x = torch.randn(1, T, d)
target_ar = torch.randn(1, T, d)   # autoregressive target (next token)
target_mlm = x.clone()
target_mlm[0, [2, 5]] = 0          # mask 두 position (MLM target)

# 같은 block, 다른 mask
class FlexibleBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x

block = FlexibleBlock(d, 4)

# Encoder mode
y_encoder = block(x, attn_mask=None)
print(f'Encoder output norm: {y_encoder.norm():.4f}')

# Decoder (causal) mode
causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
y_decoder = block(x, attn_mask=causal)
print(f'Decoder output norm: {y_decoder.norm():.4f}')
```

---

## 🔗 실전 활용

### 1. BERT 의 Bidirectional 학습

- **MLM** (Masked Language Modeling): 15% token 마스킹, 양쪽 context 로 예측
- **Bidirectional 의 직접 활용** — generation 못 하지만 understanding 우수
- Cls 분류, NER, QA 등 standard

### 2. GPT 의 Autoregressive 학습

- **Causal LM**: $p(x_t | x_{<t})$, 한 forward 에 모든 position 의 loss 계산 가능 (causal mask 덕분)
- **Generation**: 학습 그대로 사용 — argmax / sampling 으로 다음 token
- KV cache 로 incremental generation 가속

### 3. T5 의 Encoder-Decoder

- Encoder 는 input 처리 (bidirectional)
- Decoder 는 output 생성 (causal + cross-attention)
- Translation, summarization 등 seq2seq 표준
- **Cross-attention** 이 encoder 출력을 decoder 가 활용 (Ch2-05)

### 4. Prefix LM (UL2, GLM)

Prefix 부분은 bidirectional (BERT-like), 그 후는 causal (GPT-like). 한 모델에서 두 paradigm 결합. Span corruption objective 에 적합.

### 5. Encoder-Decoder vs Decoder-only

GPT-3 이후 추세는 **decoder-only** 이 simpler:
- Encoder-Decoder: 더 좋은 conditional generation, but 2× 파라미터·메모리
- Decoder-only: simpler, larger scale 가능, ICL 자연스러움

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Token 단위 | Pixel/byte 단위도 가능 (ViT, ByteNet) |
| Causal mask 만 | Sliding window (Longformer) 등 hybrid 가능 |
| Single direction | Non-causal decoder (XLNet) 가 양방향 LM |
| Discrete sequence | Continuous (audio, image) 도 동일 mask 적용 |

---

## 📌 핵심 정리

$$\boxed{\text{Encoder = Bidirectional Mask, Decoder = Causal Mask + (optionally) Cross-Attn}}$$

| 요소 | Encoder (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|---------------|-------------------|---------------------|
| Self-attn mask | Bidirectional | Causal | Encoder: Bi, Decoder: Causal |
| Cross-attn | None | None | Decoder: Q from dec, K/V from enc |
| Sub-layers per block | 2 (Attn + FFN) | 2 (Attn + FFN) | Encoder 2, Decoder 3 |
| Task | NLU (분류, NER) | Generation, ICL | seq2seq (번역, 요약) |
| Examples | BERT, RoBERTa | GPT-3/4, LLaMA | T5, BART |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Causal mask matrix 를 5×5 로 그리고, $A = \text{softmax}(QK^\top + M)$ 의 row 1 (첫 token) 과 row 5 (마지막 token) 의 entry 가 어떻게 분포하는지 분석하라.

<details>
<summary>해설</summary>

Causal mask:
$$
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty & -\infty \\ 0 & 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}
$$

**Row 1**: 첫 token (position 0) 은 자기 자신만 attend 가능. $A_{11} = 1$, 나머지 0. → Self-attention 만, 정보 없음.

**Row 5**: 마지막 token 은 5 entries 모두 attend 가능. $A_{5,j} = \text{softmax}(S_{5,1:5})_j$ 정상 분포.

**의미**:
- 초기 token 은 정보 부족 (자기만 봄)
- 끝 token 은 풍부한 context — 이것이 right-to-left dependency 의 정보 비대칭

**Token 0 의 첫 token 표현이 약함을 보완하는 mechanism**:
- Layer 누적: $L$ layer 후 token 0 이 (자기 자신만 보지만) 학습된 W_Q, W_K, W_V 로 풍부한 representation 형성
- Position embedding: 위치 정보 + token 정보 결합
- 그러나 본질적으로 첫 token 은 "정보 sink" — Streaming LLM (Xiao 2023) 의 직접 동기 $\square$

</details>

**문제 2** (심화): BERT 의 MLM objective 로 학습한 모델을 GPT-style 로 generation 시키려면? 가능한 방법과 한계를 분석하라.

<details>
<summary>해설</summary>

**Direct generation 의 한계**:

BERT 는 bidirectional → 학습 시 모든 position 의 mask 된 token 을 양쪽 context 로 예측. Causal generation 시:
- $t = 1$: $x_1$ 만 보고 $x_2$ 예측 — 학습한 적 없는 setup
- BERT 의 representation 이 right-context 를 가정 — left-only 입력에 부정확

**가능한 방법들**:

1. **Iterative refinement** (Mansimov 2019, BERT-NMT):
   - Random 초기화로 시작
   - 매 iteration 마다 일부 token 을 BERT 로 update
   - 수렴 시까지 반복
   - 단점: slow, sequential generation 보다 비효율

2. **Mask-Predict** (Ghazvininejad 2019):
   - 모든 position 을 mask
   - Parallel 하게 모든 position 예측
   - Confidence 낮은 position 다시 mask, 반복
   - 단점: parallel decoding 의 quality 한계

3. **BART / T5 적용**: BERT 같은 bidirectional encoder + autoregressive decoder
   - Encoder 가 BERT-like, decoder 가 GPT-like
   - "BERT 만으로 generation" 보다 자연스러움
   - 단점: 모델 더 큼 (encoder + decoder)

**근본 한계**:

BERT 의 표현은 **양방향 context 가정** 으로 학습 — left-only 입력에 fundamentally biased. Modern 추세는:
- Generation 이 목표 → GPT-style decoder-only
- Understanding 이 목표 → BERT-style encoder
- Both → encoder-decoder (T5)

각 paradigm 의 inductive bias 가 task 와 일치해야 효율적. $\square$

</details>

**문제 3** (논문 비평): GPT-3 이후 LLM 은 거의 모두 **decoder-only** 다. T5/BART 같은 encoder-decoder 의 강점을 LLM 이 어떻게 흡수했는가? 그리고 In-Context Learning 과의 관계는?

<details>
<summary>해설</summary>

**Encoder-Decoder 의 강점**:
1. **Conditional generation**: input → output 명확 분리
2. **Different lengths**: input 과 output 길이 자유로움
3. **Cross-attention**: 매 decoder layer 가 encoder 정보 access

**Decoder-only 가 흡수한 방법**:

1. **Concatenation pattern**: input 과 output 을 single sequence 로
   - "[Input]: ... [Output]: ..."
   - Causal LM 으로 학습, output 부분만 loss 계산
   - GPT-3 의 prompt 형식과 등가

2. **In-Context Learning** (Brown 2020):
   - "Example1 → Output1, Example2 → Output2, Test →"
   - Few-shot 으로 task 학습 (no weight update)
   - Encoder-decoder 의 conditional generation 이 자연스럽게 emergent

3. **Instruction Following** (InstructGPT):
   - "Translate the following: ..."
   - Conditional behavior 가 instruction-tuning 으로

**왜 encoder-decoder 보다 decoder-only?**

1. **Simpler architecture**: 한 종류 block 만 — scaling 쉬움
2. **Better scaling laws**: 같은 compute 에서 decoder-only 가 더 좋은 perplexity (Wang 2022 ablation)
3. **Single sequence training**: Encoder-decoder 의 left/right embedding 분리 불필요
4. **ICL 의 자연스러움**: prompt 가 그냥 sequence 의 prefix

**ICL 의 본질**:

ICL = encoder-decoder 의 "conditional generation" 이 decoder-only 에서 emergent 한 형태. Prompt 의 example 들이 implicit "encoder" 역할, generation 이 "decoder" 역할.

**Theoretical view** (Akyürek 2023, Ch7-02):
- 한 layer attention 이 한 step gradient descent 와 등가
- 충분한 layer + scale 에서 ICL 이 implicit fine-tuning 처럼 작동
- 따라서 decoder-only + scale = encoder-decoder + explicit fine-tuning

**Modern 결론**: Encoder-Decoder 는 specific task (translation, summarization) 에 여전히 좋지만, **general-purpose LLM 은 decoder-only + ICL** 이 dominant. T5 의 정신은 LLM 의 instruction-tuning 으로 진화. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-pre-ln-vs-post-ln.md) | [📚 README](../README.md) | [다음 ▶](./05-cross-attention.md)

</div>
