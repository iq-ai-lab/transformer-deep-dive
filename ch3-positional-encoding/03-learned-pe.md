# 03. Learned Positional Embedding

## 🎯 핵심 질문

- Learned PE 의 정확한 정의 — 각 위치에 학습 가능한 vector 할당의 의미는?
- Sinusoidal vs Learned 의 trade-off: 표현력 vs extrapolation?
- 왜 BERT (learned) 의 max length 가 512 로 제한되는가? GPT-2 의 1024 는?
- Learned PE 가 학습 후 어떤 패턴을 보이는가 — 실증적으로 sinusoidal-like 인지?
- Learned PE 의 한계 — extrapolation 불가 — 가 long context 시대에 fatal 한 이유는?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Learned PE 는 **simplest, most data-driven** 한 PE:

1. **BERT, GPT-1/2 의 표준** — 충분한 데이터로 좋은 위치 representation 학습
2. **Inductive bias 최소** — 모델이 데이터에서 직접 위치 정보 학습
3. **Fundamental 한계** — max length 고정, extrapolation 불가
4. **Modern 추세** — long context 가 RoPE/ALiBi 채택 강제

이 문서는 learned PE 의 **trade-off 와 학습 후 패턴** 을 분석합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-sinusoidal-pe.md](./02-sinusoidal-pe.md)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Embedding layer

---

## 📖 직관적 이해

### Embedding Table 으로서의 PE

```
position    learned PE
   0    →   [0.12, -0.34, 0.56, ...]   (학습된 vector)
   1    →   [0.45, 0.78, -0.12, ...]
   2    →   [-0.23, 0.91, 0.34, ...]
   ...
   T_max →  [0.05, -0.67, 0.89, ...]
```

각 position 에 free 학습 vector — 어떤 inductive bias 도 강제 안 함.

### Token Embedding 과의 유사성

Token embedding:
```
token "cat"  → vector
token "dog"  → vector
```

Learned PE:
```
position 0 → vector
position 1 → vector
```

같은 lookup table 구조, **discrete index → vector**.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Learned Positional Embedding

학습 가능 행렬:
$$
\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}
$$

각 위치 $t \in \{0, \ldots, T_{\max}-1\}$ 의 PE 는 $\mathbf{P}$ 의 $t$-th 행:
$$
\text{PE}_t = \mathbf{P}_{t, :}
$$

### 정의 3.2 — 적용

$$
\tilde{x}_t = x_t + \mathbf{P}_t
$$

(또는 concat — 표준은 add)

### 정의 3.3 — Initialization

통상 작은 random Gaussian:
$$
\mathbf{P}_{ti} \sim \mathcal{N}(0, 0.02^2)
$$

(또는 sinusoidal 으로 초기화 후 학습)

### 정의 3.4 — Max Length

$T_{\max}$ 가 모델 hyperparameter — 학습/inference 모두 이 값 이하의 sequence 만 처리 가능.

- BERT-base: $T_{\max} = 512$
- GPT-2: $T_{\max} = 1024$
- GPT-3: $T_{\max} = 2048$ (그러나 GPT-3 는 learned 가 아닐 수도)

---

## 🔬 정리와 증명

### 정리 3.1 — Learned PE 의 표현력

학습 가능 vector 이므로 **임의 위치 representation 학습 가능** — sinusoidal 의 specific 함수 form 보다 자유롭다.

**의미**: data 의 위치 분포에 perfectly fit 가능.

**한계**: $T_{\max}$ 고정 → 그 이상 위치는 학습 안 됨.

### 정리 3.2 — Extrapolation 불가능

$pos > T_{\max}$ 인 위치의 PE 는 정의되지 않음. Inference 시 truncation 또는 error.

**대응**:
- Truncation: 처음 $T_{\max}$ token 만 사용 (정보 손실)
- Sliding window: window 안에서만 attention
- Reuse: $\text{PE}_t = \text{PE}_{t \mod T_{\max}}$ (학습 안 했지만 ad-hoc)
- Fine-tune: 새 데이터로 longer length 학습 (cost ↑)

→ **Learned PE 의 fatal 한 한계**.

### 정리 3.3 — Sinusoidal vs Learned 의 등가성 (충분 데이터 가정)

**Hypothesis**: 충분한 데이터로 learned PE 를 학습하면 sinusoidal-like 패턴 emergent.

**실증** (BERT analysis):
- 학습된 PE 의 PCA 시 sinusoidal-similar 패턴 발견
- Frequency-like 구조 자연스럽게 학습
- 그러나 정확히 sinusoidal 은 아님 — data-specific 변형

**의미**: Sinusoidal 의 inductive bias 가 충분 데이터 시 redundant (불필요), 부족한 데이터 시 useful.

### 정리 3.4 — Initialization 의 영향

Random init vs sinusoidal init:
- Random: 학습 시간 더 길지만 final 성능 거의 동일
- Sinusoidal init: 빠른 수렴, 같은 final
- → Initialization 이 critical 하지 않음 (충분 학습 시)

### 정리 3.5 — Param 비용

$\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$ 의 param 수: $T_{\max} \times d$.

- BERT-base ($T_{\max}=512, d=768$): $0.4M$ — 작음
- GPT-3 ($T_{\max}=2048, d=12288$): $25M$ — 무시 못 함
- 100K context with $d=8192$: $0.8B$ — significant

→ 큰 context 모델은 learned PE 의 param cost 도 부담.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Learned PE 구현

```python
import torch
import torch.nn as nn

class LearnedPE(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        # Init small
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def forward(self, x):
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f'Sequence length {T} > max_len {self.max_len}')
        pos = torch.arange(T, device=x.device)
        return x + self.pe(pos).unsqueeze(0)

# 테스트
pe = LearnedPE(max_len=128, d_model=64)
x = torch.randn(2, 50, 64)
y = pe(x)
print(f'Output: {y.shape}')   # (2, 50, 64)

# Param 수
print(f'Learned PE params: {sum(p.numel() for p in pe.parameters())}')   # 128 × 64 = 8192
```

### 실험 2 — Sinusoidal Init 으로 Learned

```python
import numpy as np

def sinusoidal_init(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class LearnedPESinusoidalInit(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(sinusoidal_init(max_len, d_model))
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

# 두 PE 비교
pe_random = LearnedPE(128, 64)
pe_sinit  = LearnedPESinusoidalInit(128, 64)
print(f'Random init PE[0,:8]:    {pe_random.pe.weight[0, :8].detach()}')
print(f'Sinusoidal init PE[0,:8]: {pe_sinit.pe[0, :8].detach()}')
```

### 실험 3 — 학습 후 Sinusoidal-like 패턴 확인 (toy experiment)

```python
import matplotlib.pyplot as plt

# Toy task: 위치 분류
torch.manual_seed(0)
T_max, d = 32, 16

class PositionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Embedding(T_max, d)
        self.classifier = nn.Linear(d, T_max)
    def forward(self, pos):
        return self.classifier(self.pe(pos))

model = PositionClassifier()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# 학습: 각 position 을 분류
for ep in range(500):
    pos = torch.arange(T_max)
    target = pos
    logits = model(pos)
    loss = nn.CrossEntropyLoss()(logits, target)
    opt.zero_grad(); loss.backward(); opt.step()

# 학습된 PE 시각화
learned_pe = model.pe.weight.detach()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(learned_pe.numpy(), cmap='RdBu', aspect='auto')
axes[0].set_title(f'Learned PE (after training)')
axes[0].set_xlabel('Dim'); axes[0].set_ylabel('Position')

# Sinusoidal 비교
sin_pe = sinusoidal_init(T_max, d)
axes[1].imshow(sin_pe.numpy(), cmap='RdBu', aspect='auto')
axes[1].set_title('Sinusoidal PE (reference)')
axes[1].set_xlabel('Dim'); axes[1].set_ylabel('Position')
plt.tight_layout(); plt.show()
# 학습된 PE 가 약간의 periodic 구조 발현 가능
```

### 실험 4 — Extrapolation 실패 시연

```python
# Train 까지의 length 학습, 그 이상에서 random 을 inference
pe = LearnedPE(max_len=64, d_model=32)
x_train = torch.randn(1, 50, 32)   # Train length 50 (within max_len)
y_train = pe(x_train)
print(f'Train length 50: OK')

x_test_long = torch.randn(1, 100, 32)   # > max_len
try:
    y_test = pe(x_test_long)
except ValueError as e:
    print(f'Test length 100: ERROR — {e}')
# → Learned PE 는 strict max_len enforce
```

### 실험 5 — BERT 의 학습된 PE 패턴 분석 (시뮬레이션)

```python
# 가짜 학습된 BERT-style PE 의 PCA
torch.manual_seed(0)
fake_bert_pe = torch.randn(512, 768) * 0.02   # initial
# 실제 학습 후엔 sinusoidal-like 패턴 보일 수 있음

# PCA
U, S, V = torch.linalg.svd(fake_bert_pe, full_matrices=False)
plt.figure(figsize=(8, 4))
plt.plot(S[:30].numpy())
plt.xlabel('Singular value index'); plt.ylabel('Magnitude')
plt.title('PCA of (random init) PE — flat (no structure)')
plt.show()

# 실제 학습 후엔 처음 몇 singular value 가 dominant
# (frequency-like 패턴 학습 결과)
```

---

## 🔗 실전 활용

### 1. BERT 의 Learned PE

- Max length: 512
- 학습된 PE 의 PCA → 초기 차원이 frequency-like
- Fine-tune 시 PE 도 함께 학습 (또는 freeze)

### 2. GPT-2 의 Learned PE

- Max length: 1024
- $T_{\max} = 1024$ 가 응용의 hard limit
- Long document 처리 어려움 → context extension 기법 (RoPE 가 채택됨)

### 3. Learned 에서 Sinusoidal 으로의 회귀

Long context 시대:
- BERT (2018): learned ✓
- GPT-3 (2020): learned but length 2048
- LLaMA (2023): RoPE — learned 의 한계 인정

→ Modern LLM 거의 모두 RoPE/ALiBi 로 전환.

### 4. Hybrid — Learning Sinusoidal-form

일부 연구: sinusoidal frequency 를 학습 가능 parameter:
$$
\text{PE}_{(t, 2i)} = \sin(t \cdot \omega_i^{\text{learned}})
$$

→ Learned 의 expressivity + sinusoidal 의 extrapolation.

### 5. Position Interpolation (Chen 2023)

Learned PE 모델의 long context 확장:
- Inference 시 position $t$ 를 $t / k$ 로 scale (compress)
- 학습한 max_len 안에 맵핑
- Fine-tuning 약간 필요

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 정확한 max_len | Extrapolation 불가 — RoPE/ALiBi |
| Discrete position | Continuous (audio) 시 어려움 |
| Token embedding 과 sum | Concat/multiply 도 시도 |
| 학습 충분 가정 | Small data 에서 sinusoidal 보다 worse |
| Static (학습 후 fix) | Dynamic position 가능 (rare) |

---

## 📌 핵심 정리

$$\boxed{\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}, \quad \text{PE}_t = \mathbf{P}_t \quad \text{(learnable lookup)}}$$

| 양 | 값 | 비고 |
|----|-----|------|
| Param 수 | $T_{\max} \times d$ | 작은 모델은 negligible |
| Max length | Hard limit | Extrapolation 불가 |
| Init | $\mathcal{N}(0, 0.02^2)$ 또는 sinusoidal | Choice 무관 |
| Inductive bias | None | data-driven |

| 모델 | $T_{\max}$ | PE 종류 |
|------|------|--------|
| BERT-base | 512 | Learned |
| GPT-2 | 1024 | Learned |
| GPT-3 | 2048 | Learned (legacy) |
| LLaMA | 4K-32K | RoPE |
| Claude/Gemini | 100K+ | RoPE/ALiBi variants |

---

## 🤔 생각해볼 문제

**문제 1** (기초): BERT-base 의 learned PE 가 $\mathbb{R}^{512 \times 768}$ 이다. 만약 input length 가 400 이라면, 어떤 PE row 들이 사용되는가? 만약 1000 이라면?

<details>
<summary>해설</summary>

**Length 400**: PE row 0 부터 399 까지 사용. 나머지 row (400-511) 는 이번 forward 에 영향 없음 (gradient 도 0 — 학습 안 됨).

**Length 1000**: $1000 > 512 = T_{\max}$ → **error** (BERT 처리 불가).

대응:
- Truncation: 처음 512 token 만 사용 (정보 손실 488 token)
- Sliding window: 여러 chunk 로 나눠 처리, 결과 병합
- Long-document model 사용 (Longformer 등)

이것이 **learned PE 의 fundamental 한계** — input length 가 architecture 의 hard constraint. $\square$

</details>

**문제 2** (심화): Learned PE 가 학습 후 sinusoidal-like 패턴을 보이는 이유를 분석하라. 데이터의 어떤 통계적 성질이 이런 패턴을 emergent 하게 만드는가?

<details>
<summary>해설</summary>

**관찰** (실증):

학습된 BERT/GPT 의 PE 를 PCA 또는 spectral analysis 시:
- 초기 component 가 frequency-like (sinusoidal 의 lowest freq 와 비슷)
- 일부 dimension 이 quasi-periodic 패턴
- 모두는 아니지만 일부 명확

**Why sinusoidal-like?**

1. **위치 구분의 효율성**:
   - 모든 위치를 다르게 표현하려면 frequency 가 다양해야 함
   - 단일 frequency: 짧은 거리 잘 구분, 긴 거리 ambiguous
   - 다양한 frequency: 다양한 거리 모두 처리
   - → multi-scale frequency 가 자연스러운 optimal solution

2. **Smooth 패턴**:
   - 인접 위치의 PE 가 비슷하면 model 이 inter-position 일반화 쉬움
   - Smooth periodic = locality + 다양성

3. **Inner product 의 relative**:
   - $\langle \text{PE}_i, \text{PE}_j \rangle$ 가 $|i-j|$ 에 의존하는 것이 attention 에 유용
   - Sinusoidal 이 자연스럽게 이 성질
   - 학습이 이 성질을 emergent 하게 발견

**데이터의 역할**:

자연어 데이터의 통계적 성질:
- **Distance decay**: 가까운 token 이 더 관련 — local pattern 강조 (high freq 학습)
- **Long-range structure**: 문장/문단 단위 의존성 — low freq 학습
- 두 scale 모두 필요 → multi-scale frequency emergent

**Caveat**:

Sinusoidal-like 가 universal 아님:
- 일부 dimension 은 task-specific specialize
- Fine-tuning 으로 특정 task 패턴 학습 가능
- Pure sinusoidal 보다 약간 더 expressive

**결론**:

Learned PE 가 data-driven 으로 sinusoidal 의 essence 를 발견 — Vaswani 의 inductive bias 가 자연스러운 solution 임을 데이터가 confirm. 그러나 정확한 sinusoidal 보다 약간 적응된 form. $\square$

</details>

**문제 3** (논문 비평): "Position Interpolation" (Chen 2023) 은 learned PE 모델 (LLaMA-1) 의 context length 를 2K → 32K 로 확장했다. Inference 시 position $t$ 를 $t/k$ 로 scale 하는 trick 의 정당성은? 왜 이것이 RoPE 와 호환되는가?

<details>
<summary>해설</summary>

**Position Interpolation (PI) 의 정의**:

LLaMA-1 (학습 시 max_len = 2048) 을 32K 로 확장:
- Inference 시 position $t \in [0, 32000]$ 을 $t' = t / 16$ 로 scale → $t' \in [0, 2000]$ (학습 분포 내)
- 학습한 PE 분포에 맵핑 — extrapolation 대신 **interpolation**
- 약간의 fine-tuning (1000 step) 으로 회복

**왜 RoPE 와 호환?**

RoPE 의 회전 함수:
$$
R(t) = \text{rotation by angle } t \cdot \omega
$$

각 dimension $i$ 의 frequency $\omega_i$. $t$ 를 $t/k$ 로 scale 하면:
$$
R(t/k) = \text{rotation by } (t/k) \cdot \omega = \text{rotation by } t \cdot (\omega/k)
$$

→ Effective frequency 가 $\omega / k$ — 즉 **wavelength 가 $k$ 배 길어짐**.

**핵심 통찰**:

RoPE 는 frequency 의 모든 spectrum 을 사용하므로:
- Frequency scale 변경 = 다른 wavelength 의 covering
- 학습한 frequency 분포 안에서 long position 도 처리
- Extrapolation (못 본 frequency) 대신 interpolation (본 frequency 의 다른 사용)

**왜 fine-tuning 필요?**

원래 학습된 RoPE 는 specific $\omega$ 분포에 최적화. Scale 변경으로:
- Effective frequency 가 다른 분포 됨
- 모델이 적응 필요 (작지만 nontrivial)

→ 1000 step fine-tuning 으로 회복.

**Learned PE 와의 비교**:

Learned PE (BERT) 에 PI 적용 가능?
- 이론적으론 가능 — position $t$ 를 $\lfloor t/k \rfloor$ 또는 interpolation 으로 lookup
- 그러나 fundamentally 어려움 — discrete embedding 이 smooth 하지 않을 수 있음
- 실증적으로 RoPE PI 보다 worse

**RoPE 의 advantage**:

- Continuous mathematical structure (회전) → smooth interpolation 자연
- Frequency-based → scaling 의 mathematical 의미 명확
- 정확히 long context extension 의 enabler

**다른 RoPE extension 기법**:
- **NTK-aware** (Neural Tangent Kernel): high-freq 보존, low-freq scale
- **YaRN** (Peng 2023): NTK + temperature scaling
- **Dynamic NTK**: position 별 다른 scaling

→ RoPE 의 mathematical elegance 가 long context era 의 직접 enabler. Learned PE 는 이 era 의 primary choice 가 아님. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-sinusoidal-pe.md) | [📚 README](../README.md) | [다음 ▶](./04-relative-pe.md)

</div>
