# 04. Gradient Accumulation 과 Linear Scaling Rule

## 🎯 핵심 질문

- Gradient accumulation 의 정확한 mechanism — micro-batch 와 effective batch size 의 관계?
- Linear Scaling Rule (Goyal 2017): batch $B \to kB$ 시 LR $\eta \to k\eta$ — 왜 linear 인가?
- Large batch 의 한계 — sharp minima, generalization 저하 — 의 이론적 근거?
- LAMB optimizer 가 어떻게 large batch (32K+) 를 가능하게 하는가?
- Modern LLM 의 effective batch size — GPT-3 의 3.2M tokens 같은 거대 batch 의 의미?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Gradient Accumulation + Linear Scaling 은 **large model training 의 핵심 enablers**:

1. **GPU 메모리 우회** — micro-batch 로 작은 메모리, accumulation 으로 큰 effective batch
2. **Distributed training** — multi-GPU 의 자연스러운 결합
3. **LR scaling rule** — 다양한 batch size 에서 일관된 학습
4. **LLM 의 거대 batch** — GPT-3 의 3.2M tokens batch

이 문서는 accumulation 과 scaling rule 의 **이론과 실전** 을 정리합니다.

---

## 📐 수학적 선행 조건

- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): SGD, batch gradient 의 분산
- 이전 문서: [02-adamw.md](./02-adamw.md)

---

## 📖 직관적 이해

### Gradient Accumulation 의 mechanism

```
일반 (batch size B, 한 번에):
  forward(B examples) → loss → backward → optimizer.step()

Accumulation (micro-batch B_micro, K steps):
  for k in range(K):
    forward(B_micro examples) → loss / K → backward
    # 이 단계에서 gradient accumulate (sum)
  optimizer.step()   # 한 번만
  optimizer.zero_grad()
```

Effective batch size = $B_{\text{micro}} \times K$.

**메모리**: forward 시 activation 만 저장 (B_micro 분), gradient 는 누적 (1× model size).

### Linear Scaling Rule

```
같은 모델, 같은 epoch 수:
  Batch B,  LR η     →  loss curve A
  Batch 2B, LR η     →  loss 학습 느림 (under-step)
  Batch 2B, LR 2η    →  loss curve A 와 거의 일치 (linear scaling)
```

→ Batch ↑ k 배 시 LR ↑ k 배 → 같은 effective progress.

### Large Batch 의 한계

```
Batch B (small):  많은 update step / epoch → fine convergence
Batch 10B:        적은 step / epoch → coarse convergence
Batch 100B:       극히 적은 step → sharp minima (잘못된 local min)
```

너무 큰 batch 는 generalization 저하 — **sharp minima hypothesis** (Keskar 2017).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Mini-Batch SGD

Loss 의 unbiased estimate:
$$
\hat{L}(\theta; B) = \frac{1}{B} \sum_{i=1}^B L_i(\theta)
$$

Gradient:
$$
\hat{g}(\theta; B) = \frac{1}{B} \sum_i \nabla L_i = \frac{1}{B} \sum_i g_i
$$

### 정의 4.2 — Gradient Accumulation

$K$ micro-batches each of size $B_{\text{micro}}$, total $B = K B_{\text{micro}}$:
$$
g^{(\text{accum})} = \frac{1}{K} \sum_{k=1}^K \hat{g}(\theta; B_{\text{micro}}^{(k)}) = \frac{1}{B} \sum_{i=1}^B g_i
$$

(각 micro-batch 의 gradient 를 평균화 — 큰 batch 의 gradient 와 동일)

### 정의 4.3 — Linear Scaling Rule (Goyal 2017)

Batch size $B$ → $kB$ 시:
$$
\eta_{\text{new}} = k \eta_{\text{old}}
$$

(linear)

### 정의 4.4 — Sharp / Flat Minima (Keskar 2017)

Eigenvalues of Hessian $H = \nabla^2 L$:
- **Sharp**: 큰 eigenvalue → loss 가 perturbation 에 sensitive
- **Flat**: 작은 eigenvalue → robust

Generalization gap $\propto$ sharpness.

### 정의 4.5 — Effective Batch Size

$$
B_{\text{eff}} = B_{\text{micro}} \times K_{\text{accum}} \times N_{\text{GPUs}}
$$

---

## 🔬 정리와 증명

### 정리 4.1 — Accumulation 의 Equivalence

Gradient accumulation with micro-batch $B_m$, $K$ steps = full batch $B = K B_m$:
$$
\frac{1}{K} \sum_{k=1}^K \hat{g}(\theta; B_m^{(k)}) = \hat{g}(\theta; B)
$$

(정의상 같은 average)

**증명**: 두 식이 같은 sample 의 gradient 평균 — average of averages = total average (모든 sample 동일 weight) $\square$.

### 정리 4.2 — Linear Scaling Rule 의 정당성 (Goyal 2017)

Mini-batch SGD 의 update:
$$
\theta_{t+1} = \theta_t - \eta \hat{g}(\theta_t; B)
$$

$k$ steps 후:
$$
\theta_{t+k} \approx \theta_t - \eta \sum_{i=0}^{k-1} \hat{g}(\theta_{t+i}; B)
$$

Small LR + smooth function 가정으로 $\theta_{t+i} \approx \theta_t$:
$$
\theta_{t+k} \approx \theta_t - k\eta \cdot \mathbb{E}[\hat{g}(\theta_t; B)]
$$

Single step with batch $kB$ + LR $k\eta$:
$$
\theta_{t+1} = \theta_t - k\eta \hat{g}(\theta_t; kB) \approx \theta_t - k\eta \cdot \mathbb{E}[\hat{g}(\theta_t; kB)]
$$

두 expectation 이 같음 (unbiased estimate) → equivalent $\square$.

### 정리 4.3 — Linear Rule 의 Limit

Large LR 시 $\theta_{t+i} \neq \theta_t$ → equivalence 깨짐.

Practical limit: batch size 가 너무 커지면 (예: 10K+) linear scaling 가 잘 안 작동, sub-linear scaling 또는 plateau.

### 정리 4.4 — Square-root Scaling (Adaptive Optimizers)

LAMB / AdamW + large batch 에서 종종 사용:
$$
\eta_{\text{new}} = \sqrt{k} \eta_{\text{old}}
$$

**동기**: Adam 의 adaptive LR 이 이미 일부 scaling 자동 처리 → linear 보다 conservative.

**경험**: small-medium batch 는 linear, very large batch 는 sqrt.

### 정리 4.5 — Sharp Minima 의 Generalization

Keskar 2017: large batch 가 학습 시 sharp minima 로 수렴 경향:
- Train loss 같지만 test loss 큼
- Generalization gap $\propto$ Hessian 의 max eigenvalue

**대응**:
- Warmup (Goyal 2017)
- LARS / LAMB (You 2017, 2019)
- Layer-wise LR scaling

### 정리 4.6 — LAMB Optimizer (You 2019)

Layer-wise Adaptive Moments for Batch training:
$$
\theta_{t+1} = \theta_t - \eta \cdot \phi(\|\theta\|) / \|\hat{m}/\sqrt{\hat{v}}\| \cdot \hat{m}/\sqrt{\hat{v}}
$$

- Layer 별 normalize: $\phi(\|\theta\|)$ scaling
- 32K batch BERT 학습 가능 (vs Adam 의 8K 한계)
- TPU pod 같은 large-scale training 의 토대

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Gradient Accumulation

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Method 1: 큰 batch 한 번에
B = 32
x = torch.randn(B, 10); y = torch.randn(B, 1)
opt.zero_grad()
loss = ((model(x) - y) ** 2).mean()
loss.backward()
big_batch_grad = model.weight.grad.clone()

# Method 2: 작은 batch + accumulation
torch.manual_seed(0)
model2 = nn.Linear(10, 1)
opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)

B_micro = 8; K_steps = 4   # B_eff = 32
opt2.zero_grad()
for k in range(K_steps):
    x_micro = x[k*B_micro:(k+1)*B_micro]
    y_micro = y[k*B_micro:(k+1)*B_micro]
    loss_micro = ((model2(x_micro) - y_micro) ** 2).mean() / K_steps
    loss_micro.backward()

accum_grad = model2.weight.grad.clone()

print(f'Big batch grad: {big_batch_grad}')
print(f'Accum    grad: {accum_grad}')
print(f'Difference:    {(big_batch_grad - accum_grad).abs().max():.2e}')
# 거의 0 — accumulation 이 큰 batch 와 등가 ✓
```

### 실험 2 — Linear Scaling Rule

```python
import torch.nn.functional as F

def train_with_batch(B, lr, n_steps=100):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 5))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x = torch.randn(B, 20); y = torch.randint(0, 5, (B,))
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

import matplotlib.pyplot as plt

# Reference: B=32, LR=0.01
ref = train_with_batch(B=32, lr=0.01)

# Larger batch with linear scaling
large_linear = train_with_batch(B=128, lr=0.04)   # 4x batch, 4x LR

# Larger batch without scaling (under-step)
large_noscale = train_with_batch(B=128, lr=0.01)

plt.plot(ref,         label='B=32, LR=0.01 (ref)')
plt.plot(large_linear,label='B=128, LR=0.04 (linear scale)')
plt.plot(large_noscale,label='B=128, LR=0.01 (no scale)')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
plt.title('Linear Scaling Rule'); plt.show()
# Linear scale 이 reference 와 비슷, no scale 은 학습 느림
```

### 실험 3 — Sharp vs Flat Minima Visualization

```python
# 학습된 모델의 loss landscape 1D 시각화
torch.manual_seed(0)
model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습
for step in range(500):
    x = torch.randn(32, 10); y = torch.randn(32, 1)
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 1D perturbation 따라 loss
direction = torch.randn_like(model.weight) * 0.01
alphas = torch.linspace(-2, 2, 50)
losses_along = []
original_w = model.weight.data.clone()
for a in alphas:
    model.weight.data = original_w + a * direction
    x = torch.randn(100, 10); y = torch.randn(100, 1)
    loss = ((model(x) - y) ** 2).mean()
    losses_along.append(loss.item())
model.weight.data = original_w

plt.plot(alphas.numpy(), losses_along)
plt.xlabel('perturbation α'); plt.ylabel('loss')
plt.title('Loss landscape (sharper near optimum suggests sharp minimum)')
plt.show()
```

### 실험 4 — LAMB-style Layer-wise scaling

```python
# Layer-wise LR scaling 시뮬레이션
class LayerWiseAdam:
    def __init__(self, params_list, lr=1e-3):
        self.params = list(params_list)
        self.optimizers = [torch.optim.AdamW([p], lr=lr * (1 + i*0.1)) 
                           for i, p in enumerate(self.params)]
    def step(self):
        for opt in self.optimizers:
            opt.step()
    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

# Layer 별 다른 LR 적용 — layer 깊이에 따라 조정
torch.manual_seed(0)
model = nn.Sequential(*[nn.Linear(20, 20) for _ in range(5)])
# LAMB-like: 각 layer 의 weight norm 에 따라 LR 조정 (실제 구현 simplified)
```

### 실험 5 — Gradient Accumulation in PyTorch (표준 패턴)

```python
def training_loop_with_accumulation(model, dataloader, optimizer, accum_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for step, (x, y) in enumerate(dataloader):
        loss = compute_loss(model, x, y) / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

def compute_loss(model, x, y):
    return F.mse_loss(model(x), y)

# 효과: B_micro = 32 시 effective batch = 128
# 메모리: B_micro 만큼만 (accumulation 은 gradient 만 누적)
```

---

## 🔗 실전 활용

### 1. LLM 의 Effective Batch Size

| Model | Micro batch | Accum | GPUs | Effective tokens |
|-------|-------------|-------|------|-----------------|
| GPT-3 | 32 | 384 | 1024 | 12M tokens |
| LLaMA-2 70B | 4 | 1024 | 2048 | 16M tokens |
| Chinchilla | 32 | 256 | 1024 | 8M tokens |

매우 큰 effective batch — gradient noise 줄여 stable training.

### 2. Distributed Data Parallel (DDP)

```python
# PyTorch DDP 의 자동 gradient averaging
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model)

# 각 GPU 가 own batch 처리 후 자동 gradient sync
# Effective batch = N_GPUs × B_per_GPU
```

### 3. Gradient Checkpointing 과의 결합

큰 모델 + 긴 sequence:
- Gradient checkpointing: forward activation 일부만 저장, backward 시 재계산
- Memory 절약 vs 시간 증가
- Gradient accumulation 과 함께 effective batch 의 시간/메모리 trade-off 조정

### 4. Adaptive Batch Size

학습 진행 따라 batch 변화:
- 초기: 작은 batch (빠른 progress)
- 후기: 큰 batch (stable convergence)

→ 일부 모델 채택, 표준은 fixed.

### 5. Warmup 과의 결합

Linear scaling rule + warmup:
- 큰 batch 에서 LR 도 큼 → 더 강한 warmup 필요
- LLaMA: $T_w$ 가 effective batch 와 함께 조정

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Linear scaling rule | Very large batch 에서 sub-linear |
| Same loss curve | Small differences in dynamics |
| BatchNorm 무시 | BN 은 batch 별 다른 statistics |
| Stationary problem | RL 등 non-stationary 별도 |
| Single-task | Multi-task 시 task 별 batch 비율 |

---

## 📌 핵심 정리

$$\boxed{B_{\text{eff}} = B_{\text{micro}} \times K_{\text{accum}} \times N_{\text{GPUs}}}$$

$$\boxed{\text{Linear Scaling: } B \to kB \implies \eta \to k\eta}$$

| 기법 | 장점 | 한계 |
|------|------|------|
| Gradient accumulation | 메모리 절약 | Wall-clock time 증가 |
| Linear scaling rule | 일관된 학습 | Very large batch 에서 break |
| LAMB | 32K+ batch 가능 | Hyperparameter tuning |
| Sharp minima 회피 | Warmup, LARS | 큰 batch 의 generalization 위험 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GPU memory 가 micro-batch 16 만 가능. Effective batch 256 을 만들기 위한 accumulation steps 와, 이 경우 4 GPU 사용 시 per-GPU accumulation steps 를 계산하라.

<details>
<summary>해설</summary>

**1 GPU**:
$B_{\text{eff}} = B_{\text{micro}} \times K = 16 \times K = 256$ → $K = 16$ steps.

**4 GPU (DDP)**:
각 GPU 가 own micro-batch 처리, gradient sync.
$B_{\text{eff}} = N_{\text{GPU}} \times B_{\text{micro}} \times K = 4 \times 16 \times K = 256$ → $K = 4$ steps per GPU.

→ Wall-clock time 4× 빠름 (4 GPUs parallel).

**Memory**:
- 1 GPU: micro-batch 16 의 activation memory
- 4 GPU: 같은 (각 GPU 가 micro-batch 16 처리)

DDP 의 효과: same effective batch with same memory per GPU, 4× wall-clock 가속. $\square$

</details>

**문제 2** (심화): Linear scaling rule 이 모든 optimizer 에 적용되는가? Adam 의 경우 batch 증가 시 어떻게 다르게 작동하는가? Square-root scaling 을 사용하는 이유는?

<details>
<summary>해설</summary>

**SGD 의 linear scaling**:

Goyal 2017 의 정당성: $\sum_{i=1}^k g_i \approx k \mathbb{E}[g]$ (small LR), 따라서 batch $\times k$ + LR $\times k$ = same effective.

**Adam 의 차이**:

Adam 의 update:
$$
\Delta\theta = -\eta \frac{\hat{m}}{\sqrt{\hat{v}}}
$$

Batch 증가 시:
- $\hat{m}$ (gradient EMA): scale 무관 (averaging)
- $\hat{v}$ (squared gradient EMA): batch 증가 시 variance 줄어 $\hat{v}$ 작아짐
- $\Delta\theta = \hat{m}/\sqrt{\hat{v}}$ — denominator 작아져 effective LR **자동 증가**

**Adam 이 linear scaling 일부 자동**:
- Pure linear LR scaling 시 over-scaling — 너무 큰 step
- $\sqrt{k}$ scaling 이 적절 — Adam 의 자동 scaling 과 함께 effective $k$ scaling

**Empirical**:

| Optimizer | Best scaling rule |
|-----------|-------------------|
| SGD (with momentum) | Linear ($\eta \to k\eta$) |
| Adam (small batch) | $\sqrt{k}$ to linear |
| AdamW (large batch) | $\sqrt{k}$ |
| LAMB | $\sqrt{k}$ + layer-wise |

**LAMB 의 layer-wise normalization**:

각 layer 의 update 를 norm 으로 normalize:
$$
\Delta \theta_l = -\eta \cdot \phi(\|\theta_l\|) / \|m_l/\sqrt{v_l}\| \cdot m_l / \sqrt{v_l}
$$

- Layer 별 다른 effective LR
- Large batch (32K+) 에서 SGD/Adam 이 fail 하는 영역에서 작동
- BERT 32K batch 학습의 enabler

**결론**:

- SGD: linear scaling 표준
- Adam: $\sqrt{k}$ 가 일반적, very small batch 에서는 linear 도 OK
- LAMB / LARS: 32K+ batch 의 layer-wise scaling

각 optimizer 의 internal dynamics 가 batch 증가 시 다른 effect — careful tuning 필요. $\square$

</details>

**문제 3** (논문 비평): GPT-3 의 effective batch size 가 12M tokens (~3.2M samples) 다. 이렇게 거대한 batch 가 generalization 에 해롭지 않은 이유는? Sharp minima hypothesis 와 reconcile 하라.

<details>
<summary>해설</summary>

**Sharp Minima Hypothesis (Keskar 2017)**:
- Large batch → sharp minima → 일반화 gap ↑
- 8K+ batch (image classification) 에서 관찰

**그러나 GPT-3 의 12M tokens batch**:
- 학습 안정, generalization 우수
- 어떻게?

**Reconciliation**:

1. **Pre-training 의 다른 nature**:
   - Image classification: discrete classes, finite samples → overfitting 위험
   - Language modeling: infinite-like token 분포, 매우 다양한 context → sharp minima 의 generalization 영향 적음
   - Pre-training loss 와 downstream performance 가 다른 metric

2. **Implicit regularization**:
   - 거대한 dataset (300B+ tokens for GPT-3) — inherent variability
   - Each batch 가 매우 다양한 sample → effectively SGD-like noise 보존
   - Sharp minima 의 위험 자연스럽게 mitigate

3. **Warmup 과 careful tuning**:
   - GPT-3 의 warmup ratio 0.1% (375M tokens)
   - Linear scaling 대신 careful schedule
   - LR 도 매우 작음 ($1.2 \times 10^{-4}$)

4. **Adam optimizer 의 자동 normalization**:
   - $\hat{v}$ 의 division 이 layer-wise scaling 효과
   - Large batch 의 LR effect 자동 mitigate
   - LAMB 의 effect 일부 자동

5. **Architecture 의 robustness**:
   - Pre-LN, residual, layer norm — sharp minima 에서도 reasonable
   - Image classification 의 BatchNorm 처럼 sensitive 하지 않음

6. **Token-level vs sequence-level**:
   - 12M "tokens" batch ≈ 6K sequences 정도 (각 2K tokens)
   - Sequence 단위로는 medium batch — sharp minima warning 영역 안
   - Token 단위 view 가 misleading

**Empirical**:

GPT-3 학습 시:
- Loss curve 가 매우 smooth
- Validation gap 작음
- Few-shot performance 우수

→ Sharp minima 의 우려가 LLM 에서는 minor.

**미래 (Chinchilla 등)**:

Chinchilla 의 finding: GPT-3 가 under-trained — 더 많은 데이터, 더 작은 모델 권장.
- Effective batch size 는 비슷
- 그러나 more diverse data 가 sharp minima 위험 더 줄임

**결론**:

Sharp minima hypothesis 는 specific context (image classification, finite data) 의 phenomenon. LLM 의 거대 dataset + token-level training + careful schedule 이 거대 batch 의 위험 mitigate. **"Bigger batch always worse"** 의 단순한 statement 가 LLM context 에 적용 안 됨. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-label-smoothing.md) | [📚 README](../README.md) | [다음 ▶](./05-mixed-precision.md)

</div>
