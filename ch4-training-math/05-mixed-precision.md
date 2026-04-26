# 05. Mixed Precision Training

## 🎯 핵심 질문

- FP16, BF16, FP32 의 차이 — exponent / mantissa 분포가 학습에 미치는 영향?
- FP16 의 underflow 문제 — gradient 의 작은 값이 0 으로 수렴 → loss scaling 의 필요성?
- BF16 이 FP16 보다 LLM 훈련에 우수한 이유 — 같은 16-bit 인데 dynamic range 차이?
- Master weight 가 FP32 인 이유 — 누적 오차 방지?
- Modern LLM 의 표준 mixed precision recipe — H100/A100 에서?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Mixed Precision Training 은 **modern GPU 학습의 사실상 표준**:

1. **2× 속도** — FP16/BF16 이 FP32 의 2배 throughput (Tensor Core)
2. **메모리 50% 절약** — activation, gradient 가 16-bit
3. **모든 LLM 채택** — GPT-3, LLaMA, PaLM 등
4. **BF16 의 부상** — A100/H100 의 native BF16, FP16 의 underflow 문제 회피

이 문서는 mixed precision 의 **이론적 토대, FP16 vs BF16, loss scaling** 을 다룹니다.

---

## 📐 수학적 선행 조건

- 컴퓨터 산술: IEEE 754, floating-point representation
- 이전 문서: [02-adamw.md](./02-adamw.md), [04-gradient-accumulation.md](./04-gradient-accumulation.md)

---

## 📖 직관적 이해

### Float Format 비교

```
FP32: [sign 1] [exponent 8 ] [mantissa 23]   total 32 bits
FP16: [sign 1] [exponent 5 ] [mantissa 10]   total 16 bits   ← Tensor Core
BF16: [sign 1] [exponent 8 ] [mantissa 7 ]   total 16 bits   ← TPU/A100/H100 native
```

- **Exponent**: dynamic range (큰/작은 수 표현)
- **Mantissa**: precision (정밀도)

FP16 vs BF16:
- FP16: 큰 mantissa (10) → 정밀, 그러나 작은 exponent (5) → 좁은 range
- BF16: 작은 mantissa (7) → 덜 정밀, 큰 exponent (8) → 넓은 range (FP32 와 동일)

### FP16 의 Underflow 문제

```
FP16 representable range:  ~6e-5 to ~6.5e4
FP16 normal range:         ~5.96e-8 to ~6.5e4 (subnormal 포함)

Gradient typical:          1e-3 to 1e-7
                                       ↑
                                FP16 underflow 위험
```

작은 gradient → FP16 에서 0 → 학습 안 됨.

### Loss Scaling 의 idea

```
Loss × S (scale, e.g., 1024)
   ↓ backward
Gradient × S   (이제 FP16 representable)
   ↓ FP32 로 변환
Gradient × S / S = Gradient (원래 값 복원)
   ↓ optimizer.step
```

Forward 에는 영향 없음, gradient computation 만 boosting.

### BF16 의 advantage

```
BF16 의 exponent = FP32 와 동일 → underflow 없음
Mantissa 작음 → 약간의 precision 손실
                 그러나 LLM 학습엔 충분
```

Loss scaling **불필요** — modern LLM 의 BF16 채택 이유.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — IEEE 754 Float

$x = (-1)^s \cdot 1.m \cdot 2^{e - \text{bias}}$

- $s$: sign bit
- $m$: mantissa (fraction)
- $e$: exponent
- bias: $2^{e_{\text{bits}}-1} - 1$

### 정의 5.2 — FP16 (IEEE half)

- Sign: 1 bit
- Exponent: 5 bits (bias 15)
- Mantissa: 10 bits

Range: $\pm 6.10 \times 10^{-5}$ to $\pm 65504$.

### 정의 5.3 — BF16 (Brain float)

- Sign: 1 bit
- Exponent: 8 bits (bias 127, FP32 와 동일)
- Mantissa: 7 bits

Range: $\pm 1.18 \times 10^{-38}$ to $\pm 3.4 \times 10^{38}$ (FP32 와 거의 동일).

### 정의 5.4 — Mixed Precision Forward/Backward

```
Forward:  FP16/BF16 (fast, GPU Tensor Core)
Loss:     FP32 (precision)
Backward: FP16/BF16 (fast)
Gradient: FP16/BF16
Master weight: FP32
Optimizer state (m, v): FP32
```

### 정의 5.5 — Loss Scaling

$$
L_{\text{scaled}} = L \cdot S, \quad g = \nabla L_{\text{scaled}} = S \cdot \nabla L
$$

FP16 backward 에 $S \cdot g$ 사용 → 작은 gradient 도 representable.

Optimizer step 전:
$$
g_{\text{actual}} = g_{\text{scaled}} / S
$$

(FP32 master weight 에서 division)

### 정의 5.6 — Dynamic Loss Scaling

$S$ 를 학습 중 자동 조정:
- Gradient 에 inf/nan 발생 시 $S$ 줄임 (overflow 방지)
- 안정적이면 $S$ 늘림 (underflow 방지)

PyTorch 의 `torch.cuda.amp.GradScaler` 가 자동.

---

## 🔬 정리와 증명

### 정리 5.1 — FP16 의 Underflow 발생 분석

FP16 의 minimum positive normal = $2^{-14} \approx 6.1 \times 10^{-5}$.

LLM gradient 분포 (typical):
- Median ~ $10^{-3}$
- Tail ~ $10^{-7}$ to $10^{-9}$

Tail 의 작은 gradient 가 FP16 에서 **subnormal** 또는 **0** → 학습 정보 손실.

**Loss scaling factor $S$**:

$g \cdot S$ 가 FP16 normal range 안에:
- $S = 1024$: gradient $> 10^{-7}$ representable
- $S = 65536$: gradient $> 10^{-9}$ representable

### 정리 5.2 — BF16 의 No-Underflow 성질

BF16 의 minimum positive normal = $2^{-126} \approx 1.18 \times 10^{-38}$.

Gradient $> 10^{-38}$ 까지 representable → **practical 모든 LLM gradient cover**.

→ BF16 는 loss scaling **불필요** (실증).

### 정리 5.3 — FP16 vs BF16 의 Precision

FP16: mantissa 10 bits → ~3-4 decimal digits precision.
BF16: mantissa 7 bits → ~2-3 decimal digits.

**LLM context**:
- Weight magnitude: $\sim 10^{-1}$ to $10^{-3}$
- Update step: $\eta \cdot g \sim 10^{-7}$
- Update / weight ratio: $\sim 10^{-4}$

→ BF16 의 7-bit mantissa 도 충분 (relative precision $2^{-7} \approx 0.78\%$).

### 정리 5.4 — Master Weight 의 정당성

FP16/BF16 weight 만 사용 시 update 누적 오차:
$$
\theta_{t+1} = \theta_t + \Delta_t
$$

각 $\Delta_t \sim 10^{-7}$, $\theta_t \sim 10^{-1}$ → $\theta_t + \Delta_t \approx \theta_t$ (in FP16/BF16, mantissa 부족).

**FP32 master**:
- 정밀하게 누적
- Forward 시 FP16/BF16 으로 cast (single-step error 작음)

### 정리 5.5 — Mixed Precision 의 Speed

NVIDIA Tensor Core (V100, A100, H100):
- FP16 matmul: 2-4× faster than FP32
- BF16 matmul: same as FP16 (A100+)
- TF32 (Tensor Float 32): A100, FP32-like precision + FP16 speed

→ Mixed precision = 거의 free 2× speedup.

### 정리 5.6 — TF32 (NVIDIA)

NVIDIA A100 의 TF32:
- Exponent: 8 bits (FP32 같음)
- Mantissa: 10 bits (FP16 같음)
- Total: 19 bits, FP32 처럼 사용

A100 default: TF32 for matmul (no code change), 2× speed vs FP32, 거의 같은 precision.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — FP16 vs BF16 의 Range

```python
import torch

# FP16 의 finfo
print('FP16:', torch.finfo(torch.float16))
# min: 6.10e-5, max: 65504, eps: 9.77e-4

# BF16 의 finfo
print('BF16:', torch.finfo(torch.bfloat16))
# min: 1.18e-38, max: 3.39e+38, eps: 7.81e-3

# FP32
print('FP32:', torch.finfo(torch.float32))
# min: 1.18e-38, max: 3.40e+38, eps: 1.19e-7
```

### 실험 2 — FP16 Underflow 시연

```python
# 작은 gradient 가 FP16 에서 0 으로
small_val = torch.tensor(1e-6, dtype=torch.float32)
print(f'FP32: {small_val}')
print(f'FP16: {small_val.half()}')   # 0 또는 subnormal
print(f'BF16: {small_val.bfloat16()}')   # 정확히 representable

# 다양한 magnitude
for v in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    x = torch.tensor(v, dtype=torch.float32)
    print(f'{v:.0e}: FP16={x.half().item():.2e}, BF16={x.bfloat16().item():.2e}')
```

### 실험 3 — Mixed Precision Training (PyTorch AMP)

```python
import torch.nn as nn
import torch.nn.functional as F

# AMP automatic mixed precision
model = nn.Linear(10, 1).cuda() if torch.cuda.is_available() else nn.Linear(10, 1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()   # Loss scaling for FP16

device = next(model.parameters()).device

for step in range(10):
    x = torch.randn(32, 10, device=device)
    y = torch.randn(32, 1, device=device)
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        # Forward in FP16
        pred = model(x)
        loss = F.mse_loss(pred, y)
    
    # Backward with loss scaling
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
```

### 실험 4 — BF16 (no loss scaling needed)

```python
for step in range(10):
    x = torch.randn(32, 10, device=device)
    y = torch.randn(32, 1, device=device)
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model(x)
        loss = F.mse_loss(pred, y)
    
    loss.backward()   # No GradScaler needed
    opt.step()
    opt.zero_grad()
```

### 실험 5 — Loss Scaling 의 동작 시연

```python
# Manual loss scaling
def manual_loss_scaling(model, x, y, S=1024):
    pred = model(x).half()   # FP16
    loss = F.mse_loss(pred, y.half())
    
    # Scaled backward
    scaled_loss = loss * S
    scaled_loss.backward()
    
    # Gradient 가 FP16 에서 representable 한 영역
    for p in model.parameters():
        if p.grad is not None:
            # Unscale before optimizer step
            p.grad = p.grad.float() / S   # FP32 로 변환 후 unscale
    return loss.item()

# Without loss scaling - underflow 가능
def no_loss_scaling(model, x, y):
    pred = model(x).half()
    loss = F.mse_loss(pred, y.half())
    loss.backward()
    # Gradient 가 너무 작아 FP16 에서 0 가능
    return loss.item()
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 Recipe

| Model | Format | Loss Scaling |
|-------|--------|--------------|
| GPT-3 | FP16 | Dynamic |
| LLaMA | BF16 | Not needed |
| PaLM | BF16 | Not needed |
| Mistral | BF16 | Not needed |

**Trend**: BF16 dominant in 2023+. FP16 은 legacy / V100.

### 2. NVIDIA H100 의 FP8

H100 부터 FP8 (E4M3, E5M2) 지원:
- 2× speedup over BF16
- Inference 에서 표준 (양자화)
- Training 에서 Transformer Engine (NVIDIA library) 자동

### 3. PyTorch AMP (Automatic Mixed Precision)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for x, y in dataloader:
    with autocast(dtype=torch.bfloat16):
        loss = model(x, y)
    
    scaler.scale(loss).backward()   # FP16 시 필요, BF16 시 no-op
    scaler.unscale_(opt)            # gradient clipping 전
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
```

### 4. Memory Saving

| Component | FP32 | Mixed (BF16) |
|-----------|------|--------------|
| Weights (master) | 4 bytes/param | 4 bytes/param |
| Weights (compute) | - | 2 bytes/param |
| Activations | 4 bytes | 2 bytes |
| Gradients | 4 bytes/param | 2 bytes/param |
| Optimizer (Adam m, v) | 8 bytes/param | 8 bytes/param |
| **Total** | 16 bytes/param | 12 bytes/param (25% saving) |

**8-bit Adam**: optimizer state 도 8-bit → 6 bytes/param.

### 5. Specific Ops 의 FP32 강제

LayerNorm, Softmax 같은 numerical 민감 op 은 FP32 강제:
- AMP 가 자동 (autocast 의 op-level whitelist)
- LN 의 $\sigma$ 계산 등

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Tensor Core GPU | Older GPU (P100 등) 는 FP16 native support 없음 |
| FP16 numerical issues | BF16 이 더 robust |
| Master weight FP32 | 8-bit AdamW 같은 변형도 가능 |
| BF16 의 mantissa 부족 | Edge case 에서 일부 task 영향 (rare) |
| Symmetric quantization | Asymmetric 도 가능 (advanced) |

---

## 📌 핵심 정리

| Format | Bits | Exp | Mantissa | Range | Mantissa precision |
|--------|------|-----|----------|-------|------|
| **FP32** | 32 | 8 | 23 | $\pm 10^{38}$ | 7 digits |
| **FP16** | 16 | 5 | 10 | $\pm 65504$ | 3-4 digits |
| **BF16** | 16 | 8 | 7 | $\pm 10^{38}$ | 2-3 digits |
| **TF32** | 19 | 8 | 10 | $\pm 10^{38}$ | 3-4 digits |

| 결정 | Choice |
|------|--------|
| Modern LLM | BF16 (no loss scaling) |
| Legacy / V100 | FP16 + dynamic scaling |
| Master weight | FP32 |
| Optimizer state | FP32 (or 8-bit AdamW) |
| LayerNorm, Softmax | FP32 enforced |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $1.5 \times 10^{-7}$ 이 FP16, BF16, FP32 에서 정확히 representable 한지 확인하라.

<details>
<summary>해설</summary>

**FP16**:
- Min normal: $2^{-14} \approx 6.10 \times 10^{-5}$
- $1.5 \times 10^{-7} < 6.10 \times 10^{-5}$ → **subnormal**
- Subnormal range: down to $2^{-24} \approx 5.96 \times 10^{-8}$
- $1.5 \times 10^{-7} > 5.96 \times 10^{-8}$ → subnormal representable, but 정밀도 ↓
- Effective: representable but 일부 mantissa precision 손실

**BF16**:
- Min normal: $2^{-126} \approx 1.18 \times 10^{-38}$
- $1.5 \times 10^{-7} \gg 1.18 \times 10^{-38}$ → **normal representable**
- Mantissa 7-bit precision: $\approx 0.78\%$ 상대 오차

**FP32**:
- Min normal: $1.18 \times 10^{-38}$
- $1.5 \times 10^{-7}$ → **normal representable**
- Mantissa 23-bit precision: $\approx 1.2 \times 10^{-7}$ 상대 오차 (매우 정확)

**결론**:
- FP32 가 가장 정밀
- BF16 정확히 representable, mild precision 손실
- FP16 subnormal — precision 더 손실

→ Gradient $10^{-7}$ scale 이 LLM 에서 흔함, BF16 우위. $\square$

</details>

**문제 2** (심화): Dynamic loss scaling 의 algorithm 을 sketch 하라. inf/nan 발생 시 와 발생 안 할 때의 $S$ 조정 logic 은?

<details>
<summary>해설</summary>

**Dynamic Loss Scaling Algorithm** (PyTorch GradScaler):

**Init**: $S = 2^{16} = 65536$ (default initial scale).

**Step**:
1. Forward: $L = \text{model}(x)$.
2. Scaled backward: $\nabla(L \cdot S) = S \cdot \nabla L$.
3. Check gradient for inf/nan:
   - If found:
     - **Skip optimizer.step()** (이번 update 건너뜀)
     - $S \leftarrow S / 2$ (scale 줄임)
     - Reset growth counter
   - Otherwise:
     - Optimizer.step() (정상 update)
     - Growth counter ++
     - If counter > growth_interval (e.g., 2000):
       - $S \leftarrow S \times 2$
       - Reset counter

**Logic**:
- **Inf/Nan**: gradient overflow 발생 → $S$ 너무 큼 → 줄임
- **장기 안정**: 더 작은 gradient 도 representable 시도 → $S$ 늘림
- **Equilibrium**: optimal $S$ 찾음 (충분 정밀, no overflow)

**Code skeleton**:
```python
class GradScaler:
    def __init__(self, init_scale=65536, growth_factor=2, growth_interval=2000):
        self.scale = init_scale
        self.growth_counter = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def step(self, optimizer):
        # Check for inf/nan in gradients
        has_inf = any((p.grad == float('inf') | p.grad.isnan()).any() 
                      for p in optimizer.param_groups[0]['params'])
        
        if has_inf:
            # Skip step, reduce scale
            self.scale /= self.growth_factor
            self.growth_counter = 0
            optimizer.zero_grad()
        else:
            # Unscale and step
            for p in optimizer.param_groups[0]['params']:
                p.grad /= self.scale
            optimizer.step()
            self.growth_counter += 1
            
            if self.growth_counter > self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_counter = 0
```

**왜 이 algorithm?**:
- Initial scale 큼 → 작은 gradient 도 representable
- 학습 중 gradient distribution 변화 시 자동 적응
- Overflow 시 잠깐 step skip — 큰 비용 아님
- Long-term equilibrium → optimal scale

**BF16 의 의미**:
- $S = 1$ (no scaling) 으로 충분
- Underflow 위험 거의 없음 (BF16 의 wide range)
- → simpler training pipeline. $\square$

</details>

**문제 3** (논문 비평): NVIDIA H100 의 FP8 training 이 BF16 보다 2× 빠르다. FP8 의 mantissa 가 4 bit (E4M3) 인데도 LLM 학습이 가능한 이유는? Block-wise scaling 같은 기법이 어떻게 정밀도를 보완하는가?

<details>
<summary>해설</summary>

**FP8 Format (NVIDIA H100)**:

두 variants:
- **E4M3**: Exponent 4, Mantissa 3 (sign 1) — wider precision, less range
- **E5M2**: Exponent 5, Mantissa 2 — wider range, less precision

**Why FP8 works for LLM**:

1. **Block-wise scaling**:
   - 단순 FP8 cast 는 정밀도 부족
   - **Per-block scale factor** (e.g., 32 elements per block)
   - 각 block 의 max 로 scale → block 안에서 모든 value representable
   - $\to$ effective dynamic range 확장

2. **Mixed FP8**:
   - **E4M3 for forward** (precision 우선)
   - **E5M2 for backward** (range 우선 — gradient 분포가 wider)
   - 각 op 별 적절한 format

3. **FP32 master + BF16 critical ops**:
   - LayerNorm, Softmax: BF16 또는 FP32
   - Optimizer state: FP32
   - Compute (matmul): FP8

4. **NVIDIA Transformer Engine**:
   - Auto-conversion FP8 ↔ BF16
   - Per-tensor scaling 자동 관리
   - No code change for users

**Trade-offs**:

**Pro**:
- 2× speedup over BF16
- 50% memory of BF16 (training)
- H100 의 native support — Tensor Core throughput 최대

**Con**:
- Numerical sensitivity ↑ (mantissa 매우 작음)
- 일부 task 에서 약간 worse (reasoning, math)
- Pipeline complexity (conversion overhead)

**현재 (2024-2026) 상황**:

- NVIDIA H100 + Transformer Engine: production ready
- LLaMA-3, Mistral-8x22B 등: BF16 + 일부 FP8 layer
- Open frontier (GPT-4, Claude): BF16 + 일부 FP8 (estimate)

**미래**:

- FP4 (4-bit float) 도 inference 에서 사용 시작
- Block-wise scaling 의 더 정교한 variant
- Hardware (GPU, TPU) 의 native lower-precision support

**핵심 통찰**:

LLM 의 weight, activation, gradient distribution 이 long-tailed but bounded. **Block-wise statistics** 활용 시 매우 낮은 precision 으로도 학습 가능. Standard IEEE FP 의 single-element representation 한계를 block-level scaling 으로 우회.

이는 attention 의 $\sqrt{d_k}$ scaling 과 유사한 idea — 분산을 정규화하면 fewer bits 로도 충분. Mixed precision 의 logical conclusion. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-gradient-accumulation.md) | [📚 README](../README.md) | [다음 ▶](../ch5-attention-efficiency/01-quadratic-bottleneck.md)

</div>
