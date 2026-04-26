# 02. Sinusoidal Positional Encoding 의 수학적 성질

## 🎯 핵심 질문

- Vaswani 2017 의 sinusoidal PE 식 $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$ 는 어떤 동기에서 도출되었는가?
- "$PE_{pos+k} = M_k \, PE_{pos}$" — 임의 offset $k$ 에 대해 선형 변환 (회전 행렬) 으로 표현 가능 성질을 어떻게 증명하는가?
- Frequency 가 $10000^{-2i/d}$ 로 geometric 하게 감소하는 의미는 — 다양한 wavelength 표현?
- Sinusoidal PE 의 extrapolation 능력은 실제로 어떤가?
- 왜 두 가지 함수 (sin, cos) 가 alternating 하게 사용되는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Sinusoidal PE 는 Vaswani 2017 의 **가장 영감 있는 설계** 중 하나입니다:

1. **Linear shift 성질** — relative position 을 자연스럽게 인코딩 (정리 2.1)
2. **무한 길이 지원** — 학습 가능 PE 와 달리 임의 위치 정의 가능
3. **Inductive bias** — smooth periodic 구조가 위치 패턴에 적합
4. **RoPE 의 토대** — Sinusoidal 의 회전 성질이 RoPE 로 직접 발전

이 문서는 sinusoidal PE 의 **수학적 성질을 엄밀히 증명** 하고, 다양한 frequency 의 의미를 분석합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-pe-necessity.md](./01-pe-necessity.md)
- 삼각함수: 합차 공식 $\sin(a+b) = \sin a \cos b + \cos a \sin b$
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 회전 행렬, block diagonal

---

## 📖 직관적 이해

### Sinusoidal PE 의 모양

```
position
    ↓
PE: ┌────────────────────────────┐
  0 │  sin(0)  cos(0)  sin(0)... │  ← all 0 (or 1 for cos)
  1 │  sin(1)  cos(1)  sin(1/100) │  ← high freq → low freq
  2 │  sin(2)  cos(2)  sin(2/100) │
    │   ...
 100 │  sin(100) cos(100) sin(1) │   ← longer wavelength
    └────────────────────────────┘
       low  →  high dimension
```

### Frequency 의 spectrum

```
dimension i:    frequency w_i = 1 / 10000^(2i/d)

i = 0:    w = 1         → 짧은 wavelength (인접 위치 구분)
i = d/2:  w = 1/100     → 중간 wavelength
i = d:    w = 1/10000   → 긴 wavelength (먼 위치 차별)

각 dimension 이 다른 "scale" 의 위치 정보 인코딩
```

### Linear Shift 직관

```
PE_{pos+k} = M_k · PE_{pos}

M_k = block diagonal of 2×2 회전:
  [ cos(w_i · k)  -sin(w_i · k) ]   per (2i, 2i+1) pair
  [ sin(w_i · k)   cos(w_i · k) ]
```

→ 임의 거리 $k$ 만큼 이동 = 회전. 회전각이 frequency × distance.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Sinusoidal PE (Vaswani 2017)

차원 $d$ (짝수 가정), position $pos \in \mathbb{Z}_+$:
$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d})
$$
$$
\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

각 $i \in \{0, 1, \ldots, d/2-1\}$.

### 정의 2.2 — Frequency

차원 $i$ 의 angular frequency:
$$
\omega_i := 1 / 10000^{2i/d} = 10000^{-2i/d}
$$

$\omega_0 = 1$, $\omega_{d/2} = 10000^{-1} = 10^{-4}$.

### 정의 2.3 — Wavelength

$$
\lambda_i = 2\pi / \omega_i = 2\pi \cdot 10000^{2i/d}
$$

- $\lambda_0 = 2\pi$ (가장 짧음)
- $\lambda_{d/2-1} \approx 2\pi \cdot 10000$ (가장 김)

### 정의 2.4 — Sinusoidal Pair

차원 $(2i, 2i+1)$ 의 pair:
$$
P_i(pos) := \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix} \in \mathbb{R}^2
$$

전체 PE 는 $d/2$ 개 pair 의 concatenation.

### 정의 2.5 — 2D Rotation Matrix

$$
R(\theta) := \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

---

## 🔬 정리와 증명

### 정리 2.1 — Linear Shift Property (핵심)

임의 offset $k \in \mathbb{Z}$ 에 대해:
$$
P_i(pos + k) = R(\omega_i k) \cdot P_i(pos)
$$

(2-dim pair 별 회전)

**증명**:

$$
P_i(pos+k) = \begin{pmatrix} \sin(\omega_i (pos+k)) \\ \cos(\omega_i (pos+k)) \end{pmatrix}
$$

삼각함수 합차 공식:
$$
\sin(\omega_i (pos+k)) = \sin(\omega_i pos) \cos(\omega_i k) + \cos(\omega_i pos) \sin(\omega_i k)
$$
$$
\cos(\omega_i (pos+k)) = \cos(\omega_i pos) \cos(\omega_i k) - \sin(\omega_i pos) \sin(\omega_i k)
$$

행렬 형태:
$$
\begin{pmatrix} \sin(\omega_i (pos+k)) \\ \cos(\omega_i (pos+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}
$$

이는 정확히 회전 $R(-\omega_i k)$ — 부호 convention 차이만, equivalent $\square$.

### 정리 2.2 — 전체 PE 의 Linear Shift

$\text{PE}_{pos} \in \mathbb{R}^d$ 전체에 대해:
$$
\text{PE}_{pos+k} = M_k \, \text{PE}_{pos}
$$

with $M_k$ 가 block-diagonal of $R(\omega_i k)$ for $i = 0, \ldots, d/2-1$.

**증명**: 각 pair 별로 정리 2.1 적용, block diagonal 로 합 $\square$.

### 정리 2.3 — Inner Product Invariance

$$
\langle \text{PE}_{pos_1+k}, \text{PE}_{pos_2+k} \rangle = \langle M_k \text{PE}_{pos_1}, M_k \text{PE}_{pos_2} \rangle = \langle \text{PE}_{pos_1}, \text{PE}_{pos_2} \rangle
$$

(rotation 은 inner product 보존)

**의미**: 두 PE 의 inner product 는 **위치 차이만 의존**:
$$
\langle \text{PE}_{pos}, \text{PE}_{pos+k} \rangle = \sum_i \cos(\omega_i k)
$$

(같은 pos 와 pos+k 의 sin/cos pair 의 dot product)

→ **자동 relative encoding** in inner product. RoPE 의 직접 동기.

### 정리 2.4 — Asymptotic Behavior

$d \to \infty$ 시, $\omega_i$ 가 $[10^{-4}, 1]$ 에 dense:
$$
\sum_{i=0}^{d/2-1} \cos(\omega_i k) \approx \frac{d}{2} \int_{10^{-4}}^{1} \cos(\omega k) \, d\omega \cdot \frac{1}{\omega}
$$

(geometric distribution of $\omega$)

**의미**: PE 가 다양한 frequency 의 weighted sum 으로 임의 거리 $k$ 의 정보 인코딩.

### 정리 2.5 — Extrapolation 가능성

Sinusoidal PE 는 임의 위치 $pos$ 에 대해 정의됨 (수학적):
$$
\text{PE}_{pos} = (\sin(\omega_i pos), \cos(\omega_i pos))_{i=0}^{d/2-1}
$$

**그러나 실증적으로**:
- $pos > 2 \times \text{train length}$ 시 성능 감소 (학습 안 한 위치 패턴)
- 특히 짧은 wavelength (high freq) 에서 unreliable

→ **이론상 extrapolation 가능, 실증적으론 한계** — RoPE/ALiBi 가 개선.

### 정리 2.6 — 100K Constant 의 의미

$10000$ 은 임의 선택 — 다양한 wavelength 를 보장하기 위함.

- 너무 작으면 (예: $100$): 모든 dimension 이 비슷한 wavelength
- 너무 크면 (예: $10^{10}$): 대부분 dimension 이 너무 긴 wavelength → 짧은 위치 차이 안 잡음

$10000$ 이 typical max sequence length (수천) 를 cover 하는 반면 짧은 position 도 분간하는 sweet spot.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Sinusoidal PE 생성과 시각화

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         -(np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

pe = sinusoidal_pe(100, 128)
print(f'PE shape: {pe.shape}')

plt.figure(figsize=(10, 4))
plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Dimension'); plt.ylabel('Position')
plt.title('Sinusoidal Positional Encoding (low → high dim)')
plt.colorbar(); plt.tight_layout(); plt.show()
```

### 실험 2 — Linear Shift Property 검증

```python
d = 128
pe = sinusoidal_pe(200, d)

def rotation_matrix_for_shift(k, d):
    """M_k: block diagonal of 2x2 rotations"""
    M = torch.zeros(d, d)
    for i in range(d // 2):
        omega_i = 1 / (10000 ** (2 * i / d))
        c, s = np.cos(omega_i * k), np.sin(omega_i * k)
        M[2*i,   2*i  ] =  c; M[2*i,   2*i+1] = s
        M[2*i+1, 2*i  ] = -s; M[2*i+1, 2*i+1] = c
    return M

# Test: PE_{pos+k} = M_k · PE_{pos}
k = 5
M_k = rotation_matrix_for_shift(k, d)

for pos in [10, 30, 80]:
    pe_pos = pe[pos]
    pe_pos_k = pe[pos + k]
    pe_predicted = M_k @ pe_pos
    diff = (pe_pos_k - pe_predicted).abs().max()
    print(f'pos={pos}, k={k}: |PE_{{pos+k}} - M_k · PE_{{pos}}| = {diff:.6e}')
# 모두 ≈ 0 → linear shift property 확인 ✓
```

### 실험 3 — Inner Product Invariance

```python
# Inner product 가 거리에만 의존하는지 검증
print('Inner product test (should depend on |i - j| only):')
for i, j in [(10, 15), (50, 55), (100, 105)]:
    sim = (pe[i] @ pe[j]).item()
    print(f'  pos {i} vs pos {j} (dist=5): {sim:.4f}')
# 같음 ≈ → invariance ✓

print('\nDifferent distances:')
pos = 50
for k in [1, 5, 10, 50, 100]:
    sim = (pe[pos] @ pe[pos+k]).item() / d
    print(f'  k={k:3d}: <PE_{{pos}}, PE_{{pos+k}}>/d = {sim:.4f}')
# k 가 커질수록 보통 작아지는 경향 (decorrelation)
```

### 실험 4 — Frequency Spectrum 시각화

```python
# 각 dimension 의 frequency
i_range = np.arange(0, d, 2)
omega = 1 / (10000 ** (i_range / d))
wavelength = 2 * np.pi / omega

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(i_range, omega)
axes[0].set_xlabel('Dimension index'); axes[0].set_ylabel('Frequency ω_i')
axes[0].set_yscale('log'); axes[0].set_title('Frequency spectrum')
axes[0].grid(alpha=0.3)

axes[1].plot(i_range, wavelength)
axes[1].set_xlabel('Dimension index'); axes[1].set_ylabel('Wavelength λ_i')
axes[1].set_yscale('log'); axes[1].set_title('Wavelength spectrum (log scale)')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

### 실험 5 — Extrapolation Test

```python
# Train length = 100, test 시 > 100 까지 확장
pe_train = sinusoidal_pe(100, d)
pe_test  = sinusoidal_pe(300, d)

# 잘 학습됐다면 train length 안에서는 잘 작동
# 그 너머는 attention 의 잘못된 작동 위험

# Inner product matrix
sim_matrix = pe_test @ pe_test.T

plt.figure(figsize=(7, 6))
plt.imshow(sim_matrix.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('pos j'); plt.ylabel('pos i')
plt.title('PE inner product matrix (extrapolated)')
plt.colorbar(); plt.tight_layout(); plt.show()
# 대각선 근처 (가까운 위치) 에 high similarity 의 stripe pattern
```

---

## 🔗 실전 활용

### 1. Vaswani 2017 의 표준 채택

원래 Transformer (2017): sinusoidal PE.
- 학습 데이터 의존 적음
- Translation task 에서 잘 작동
- Extrapolation 약간 가능 (train length 의 1.2× 정도)

### 2. BERT / GPT 가 learned 로 전환한 이유

- 데이터 풍부 — learned PE 가 약간 더 좋은 성능
- Max length 가 학습에서 정해짐 (BERT 512, GPT-2 1024)
- Extrapolation 무관 (대부분 응용이 짧은 sequence)

### 3. Sinusoidal 의 부활 (LongFormer 등)

Long context 모델에서 sinusoidal 다시 사용:
- Learned 의 max length 한계 극복
- Train length 보다 긴 sequence 가능
- 단, 큰 차이는 RoPE/ALiBi 가 더 좋은 결과

### 4. Learning Sinusoidal 으로 초기화

일부 모델: learned PE 를 sinusoidal 로 초기화 → 학습 진행. 두 paradigm 의 hybrid.

### 5. PyTorch 의 표준 구현

```python
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # buffer (not learned)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| $d$ 짝수 | 홀수 시 $d/2$ pair + 1 standalone — convention 처리 |
| Discrete position | Continuous PE (audio) 가능 — 같은 식 사용 |
| 1D | 2D, 3D 는 row/column 별 sinusoidal concat 또는 outer |
| 단일 $10000$ base | NTK-aware: base 조정으로 long context (YaRN) |
| Periodic 가정 | 실제 언어 위치는 quasi-periodic, 약간 mismatch |

---

## 📌 핵심 정리

$$\boxed{\text{PE}_{(pos, 2i)} = \sin(pos \cdot \omega_i), \quad \text{PE}_{(pos, 2i+1)} = \cos(pos \cdot \omega_i), \quad \omega_i = 10000^{-2i/d}}$$

$$\boxed{\text{PE}_{pos+k} = M_k \, \text{PE}_{pos} \quad \text{(linear shift, block-diagonal rotation)}}$$

| 양 | 식 | 값 |
|----|-----|-----|
| Frequency | $\omega_i = 10000^{-2i/d}$ | $[10^{-4}, 1]$ |
| Wavelength | $\lambda_i = 2\pi/\omega_i$ | $[2\pi, 2\pi \cdot 10^4]$ |
| Linear shift | $M_k = \bigoplus_i R(\omega_i k)$ | block-diagonal rotation |
| Inner product | depends only on $\|i - j\|$ | translation invariant |
| Extrapolation | Theoretical: ✓, Empirical: limited | RoPE/ALiBi 가 개선 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $d = 4$ 인 sinusoidal PE 의 처음 4 위치 ($pos = 0, 1, 2, 3$) 의 PE 값을 손으로 계산하라. $\omega_0 = 1$, $\omega_1 = 0.01$ 가정.

<details>
<summary>해설</summary>

$d = 4$ → 2 pair, $\omega_0 = 1$, $\omega_1 = 10000^{-2/4} = 10^{-1} = 0.1$ (실제), 문제 가정대로 $0.01$ 사용.

$$
\text{PE}_{pos} = (\sin(\omega_0 pos), \cos(\omega_0 pos), \sin(\omega_1 pos), \cos(\omega_1 pos))
$$

**$pos = 0$**: $(0, 1, 0, 1)$
**$pos = 1$**: $(\sin 1, \cos 1, \sin 0.01, \cos 0.01) \approx (0.841, 0.540, 0.010, 1.000)$
**$pos = 2$**: $(\sin 2, \cos 2, \sin 0.02, \cos 0.02) \approx (0.909, -0.416, 0.020, 1.000)$
**$pos = 3$**: $(\sin 3, \cos 3, \sin 0.03, \cos 0.03) \approx (0.141, -0.990, 0.030, 1.000)$

**관찰**:
- 첫 두 dim ($\omega_0$): 빠르게 변화 (high freq)
- 마지막 두 dim ($\omega_1$): 느리게 변화 (low freq)
- 다양한 scale 의 위치 정보 동시 인코딩 ✓ $\square$

</details>

**문제 2** (심화): 정리 2.1 의 linear shift 성질이 attention score $q_i^\top k_j$ 에 어떻게 작동하는가? PE 가 사용된 query $q_i = (x_i + \text{PE}_i) W_Q$ 와 key $k_j = (x_j + \text{PE}_j) W_K$ 의 내적을 분해해 어떤 항이 "relative" 인지 분석하라.

<details>
<summary>해설</summary>

$q_i = (x_i + \text{PE}_i) W_Q$, $k_j = (x_j + \text{PE}_j) W_K$.

$$
q_i^\top k_j = (x_i W_Q)(x_j W_K)^\top + (x_i W_Q)(\text{PE}_j W_K)^\top + (\text{PE}_i W_Q)(x_j W_K)^\top + (\text{PE}_i W_Q)(\text{PE}_j W_K)^\top
$$

네 항:
1. **Content-content**: $x_i^\top W_Q W_K^\top x_j$ — 순수 token 간 similarity
2. **Content-position**: $x_i^\top W_Q W_K^\top \text{PE}_j$ — 학습된 token 의 위치 의존성
3. **Position-content**: $\text{PE}_i^\top W_Q W_K^\top x_j$ — 위치별 token 의존성
4. **Position-position**: $\text{PE}_i^\top W_Q W_K^\top \text{PE}_j$ — 순수 position-position

**Sinusoidal 의 Position-position 항**:

If $W_Q W_K^\top \approx I$ (학습 후 가정), 그러면:
$$
\text{PE}_i^\top \text{PE}_j = \sum_k \cos(\omega_k (i - j))
$$

(정리 2.3) → **$|i - j|$ 만 의존** = relative.

**그러나 일반 $W_Q W_K^\top$ 시**:

$\text{PE}_i^\top W_Q W_K^\top \text{PE}_j$ 는 일반적으로 $i, j$ 모두에 의존 — relative 가 아닐 수 있음.

**Shaw 2018 의 직접 해결**:

Sinusoidal PE 를 사용하지 말고 명시적 relative position embedding $a_{ij}^K$ 추가:
$$
e_{ij} = (x_i W_Q)(x_j W_K + a_{ij}^K)^\top
$$

→ relative 항 $(x_i W_Q) a_{ij}^K$ 가 명시적으로.

**RoPE 의 더 elegant 해결**:

$Q, K$ 에 직접 회전 적용 → $\langle R(i) q, R(j) k \rangle = q^\top R(j-i) k$ — 자동 relative.

→ Sinusoidal 의 inner product invariance 를 attention 의 모든 부분으로 확장. Ch3-05 에서 자세히. $\square$

</details>

**문제 3** (논문 비평): Sinusoidal PE 가 RoPE 로 진화한 것은 단순한 변화가 아니라 fundamental 한 design 변화이다. 두 PE 가 inner product 를 다루는 방식의 차이와, 이것이 long context 에서 결정적 이유를 분석하라.

<details>
<summary>해설</summary>

**Sinusoidal 의 inner product**:

$\langle \text{PE}_i, \text{PE}_j \rangle = \sum_k \cos(\omega_k (i-j))$ — relative.

그러나 attention score 는 $(x_i + \text{PE}_i) W_Q W_K^\top (x_j + \text{PE}_j)$ — **PE 와 token 이 sum 으로 섞임**, $W_Q W_K^\top$ 가 학습 가능 → relative 보장 깨질 수 있음.

**RoPE 의 inner product**:

$$
\langle R(i) q, R(j) k \rangle = q^\top R(i)^\top R(j) k = q^\top R(j-i) k
$$

(rotation 의 합성: $R(i)^\top R(j) = R(j-i)$)

→ **항상 $j-i$ 만 의존**, $W_Q W_K^\top$ 와 무관.

**Long Context 에서의 차이**:

1. **Sinusoidal**:
   - Train length 안에서는 OK (학습이 token-PE coupling 을 train 분포에 맞춤)
   - Train 의 2× 이상에서 성능 급감 — $W_Q W_K^\top$ 가 못 본 PE 패턴
   - YaRN 같은 NTK-aware scaling 으로 해결 시도

2. **RoPE**:
   - Inner product 의 relative 성질이 architecture 에 baked-in
   - Token 과 position 이 **곱셈 (회전)** 으로 결합 — sum 의 모호함 없음
   - Extrapolation 자연스러움 (회전이 임의 각도)

**더 깊은 차이 — Attention 의 Information Flow**:

Sinusoidal 은 PE 를 token embedding 에 add — model 이 "이게 token 정보, 이게 PE 정보" 를 분리해야. 큰 모델 + 충분 데이터 시 가능, 작은 모델 / OOD 에서 fail.

RoPE 는 PE 를 attention 의 inner product 에 곱함으로 직접 적용 — "Q 와 K 가 어떻게 align 되는가" 의 metric 자체에 PE injection. 모델이 분리할 필요 없음.

**LLaMA 의 RoPE 채택**:

LLaMA-1: 2K context, RoPE
LLaMA-2: 4K → context extension 으로 32K (NTK-aware RoPE)
LLaMA-3: 128K (RoPE + position interpolation)

→ RoPE 의 mathematical robustness 가 long context scaling 의 직접 enabler.

**Sinusoidal 의 legacy**:

원래 sinusoidal 의 통찰 (relative encoding 가능) 이 RoPE 에서 완성. Sinusoidal 은 stepping stone, RoPE 가 logical conclusion.

**Modern LLM 의 통일 추세**: RoPE + ALiBi 가 standard. Sinusoidal/learned 은 legacy. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-pe-necessity.md) | [📚 README](../README.md) | [다음 ▶](./03-learned-pe.md)

</div>
