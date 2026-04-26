# 01. Warmup 의 필요성 (Xiong 2020)

## 🎯 핵심 질문

- 왜 Transformer 훈련이 거의 항상 warmup 을 필요로 하는가? — Post-LN 의 gradient 분석에서 직접 도출
- Warmup schedule 의 표준 형태 — linear / cosine / inverse sqrt 의 차이는?
- 큰 모델에서 warmup ratio 가 작아지는 이유는?
- Warmup 없이 학습 가능한 architecture variants — DeepNorm, T-Fixup, Admin 의 idea?
- Cosine decay 후 final LR 이 0 인 vs 작은 값인 영향?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Warmup 은 Transformer 훈련의 **most underrated technique**:

1. **Post-LN 의 발산 회피** — Ch2-03 의 gradient $O(L)$ 누적 분석의 직접 응용
2. **Pre-LN 도 권장** — 큰 모델 (GPT-3+) 에서 안정성 확보
3. **Cosine schedule 의 표준** — warmup → cosine decay 가 modern recipe
4. **LR sweet spot 발견** — warmup 후 stable 영역에서 학습

이 문서는 warmup 의 **이론적 정당성, schedule 변형, modern recipe** 를 정리합니다.

---

## 📐 수학적 선행 조건

- Chapter 2: [03-pre-ln-vs-post-ln.md](../ch2-transformer-architecture/03-pre-ln-vs-post-ln.md) — gradient norm 분석
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Learning rate, gradient flow

---

## 📖 직관적 이해

### Warmup 의 두 단계

```
Loss
 │
 │     ╲ ╲___          <- warmup: small LR, gentle progress
 │      ╲    ╲______   <- main training: stable progress
 │       ↑↑    ↓↓↓↓   <- cosine decay
 │      warmup
 └────────────────────→ steps
     0   T_w    T
```

**Phase 1 (warmup)**: $\eta$ 가 작음 → gradient 가 explode 해도 step 작음 → safe 학습.
**Phase 2 (main)**: 정상 LR, 학습 본격 진행.
**Phase 3 (decay)**: LR 감소 → fine convergence.

### Why does Post-LN explode without warmup?

Ch2-03 정리: Post-LN gradient 가 layer 깊이 $L$ 에 따라 $O(L)$ 누적. 큰 LR + 큰 gradient = 발산.

Warmup 의 small LR 이 초기 gradient 의 불안정 영역을 통과 — 그 후 LN parameters $\gamma, \beta$ 가 학습되면서 gradient 가 stabilize.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Linear Warmup

$$
\eta_t = \eta_{\max} \cdot \min(t / T_w, 1)
$$

$t \in [0, T_w]$: linear ramp $0 \to \eta_{\max}$.
$t \geq T_w$: $\eta = \eta_{\max}$ (or decay).

### 정의 1.2 — Cosine Decay (after warmup)

$$
\eta_t = \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2} \left(1 + \cos\!\left(\pi \frac{t - T_w}{T - T_w}\right)\right)
$$

$t \in [T_w, T]$: cosine 으로 $\eta_{\max} \to \eta_{\min}$.

### 정의 1.3 — Inverse Square Root (Vaswani 2017)

$$
\eta_t = d^{-0.5} \min(t^{-0.5}, t \cdot T_w^{-1.5})
$$

$t \leq T_w$: linear in $t$, otherwise $1/\sqrt{t}$.

원래 Transformer paper 의 schedule.

### 정의 1.4 — Warmup Ratio

$$
\rho_w = T_w / T
$$

- 작은 모델: $\rho_w \approx 5\text{-}10\%$
- 큰 모델 (LLaMA-2): $\rho_w \approx 0.3\%$ (2000 / 600000 steps)

### 정의 1.5 — Effective LR Schedule

$$
\eta_t = \eta_{\max} \cdot \text{schedule}(t)
$$

with schedule ∈ {linear-cosine, inverse-sqrt, constant-after-warmup}.

---

## 🔬 정리와 증명

### 정리 1.1 — Warmup 의 정당성 (Stability Argument)

큰 LR $\eta$ + 큰 초기 gradient $g_0$ → step $\eta \|g_0\|$ 가 너무 큼 → loss 발산.

Warmup: 작은 $\eta$ → 작은 step → gradient 가 stabilize 할 시간.

수학적으로: $\eta_t \|g_t\|$ 의 보존이 stability 의 sufficient condition (Lyapunov-like).

### 정리 1.2 — Vaswani 2017 의 Schedule 분석

$\eta_t = d^{-0.5} \min(t^{-0.5}, t \cdot T_w^{-1.5})$:

- $t = T_w$: $\eta = d^{-0.5} T_w^{-0.5}$ (peak)
- $t > T_w$: $\eta = d^{-0.5} t^{-0.5}$ (decay)

**의미**: Peak LR 이 $d^{-0.5} T_w^{-0.5}$ — width 와 warmup 둘 다에 의존. 큰 모델 (큰 $d$) 또는 긴 warmup (큰 $T_w$) 시 peak 작음.

### 정리 1.3 — Linear Scaling Rule 과의 호환

Goyal 2017 의 linear scaling rule: large batch $B \to kB$ 시 $\eta \to k\eta$.

Warmup 도 함께 scale: $T_w \to T_w$ (unchanged in steps) 또는 $T_w \to T_w/k$ (in tokens).

**Modern**: token 기준으로 warmup tokens 고정 (e.g., 10B tokens warmup), batch 와 무관.

### 정리 1.4 — Cosine vs Linear Decay

Cosine decay 가 linear decay 보다 우수:
- 처음 (peak 직후): 빠르게 감소 (cosine 의 derivative 큼)
- 끝 (small LR): 천천히 감소 (cosine plateau)
- → 학습 후반의 fine-tuning effect

Linear decay 는 simpler 그러나 약간 inferior.

### 정리 1.5 — Pre-LN 도 Warmup 권장 (Large Scale)

Ch2-03 정리 1.1: Pre-LN 의 gradient 가 $O(1)$ bounded — warmup 이론적으로 불필요.

그러나 실증:
- 큰 모델 (GPT-3+) 에서 warmup 없이 instability 가능
- Adam 의 second moment estimate 가 초기에 부정확 — warmup 으로 안정
- Common practice: 모든 큰 모델에 warmup

### 정리 1.6 — Warmup 이 없는 Initialization 기법

**T-Fixup** (Huang 2020): Post-LN 의 weight 초기화를 careful 하게 → warmup 불필요.

**Admin** (Liu 2020): 매 layer 의 sub-layer output norm 을 학습 가능 scaling 으로 → adaptive depth normalization.

**DeepNorm** (Wang 2022): residual scaling $\alpha = (2L)^{1/4}$ → 1000-layer warmup-free.

이 모두 academic, modern production 은 simpler 한 Pre-LN + warmup.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — 다양한 LR Schedule 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_warmup_cosine(t, T_w, T_total, eta_max, eta_min=0):
    if t < T_w:
        return eta_max * (t / T_w)
    progress = (t - T_w) / (T_total - T_w)
    return eta_min + (eta_max - eta_min) * 0.5 * (1 + np.cos(np.pi * progress))

def linear_warmup_constant(t, T_w, eta_max):
    if t < T_w:
        return eta_max * (t / T_w)
    return eta_max

def vaswani_inv_sqrt(t, T_w, d_model):
    return d_model ** -0.5 * min((t+1) ** -0.5, (t+1) * T_w ** -1.5)

T_total = 1000; T_w = 100
ts = np.arange(T_total)

eta_lin_cos = [linear_warmup_cosine(t, T_w, T_total, 1e-3) for t in ts]
eta_lin_const = [linear_warmup_constant(t, T_w, 1e-3) for t in ts]
eta_vaswani = [vaswani_inv_sqrt(t, T_w, 512) for t in ts]

plt.figure(figsize=(11, 4))
plt.plot(ts, eta_lin_cos,   label='Linear warmup + Cosine decay')
plt.plot(ts, eta_lin_const, label='Linear warmup + Constant')
plt.plot(ts, eta_vaswani,   label='Vaswani inverse-sqrt')
plt.xlabel('step'); plt.ylabel('learning rate')
plt.title('Common LR schedules')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

### 실험 2 — Warmup 효과 시뮬레이션

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers, d, h, mode='post'):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = nn.MultiheadAttention(d, h, batch_first=True)
            ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            ln1, ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.layers.append(nn.ModuleDict({'attn': attn, 'ffn': ffn, 'ln1': ln1, 'ln2': ln2}))
        self.mode = mode
    
    def forward(self, x):
        for layer in self.layers:
            if self.mode == 'pre':
                x = x + layer['attn'](layer['ln1'](x), layer['ln1'](x), layer['ln1'](x))[0]
                x = x + layer['ffn'](layer['ln2'](x))
            else:
                x = layer['ln1'](x + layer['attn'](x, x, x)[0])
                x = layer['ln2'](x + layer['ffn'](x))
        return x

def train_with_schedule(mode, schedule_fn, lr_max, total_steps=300):
    torch.manual_seed(0)
    model = SimpleTransformer(8, 64, 4, mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    losses = []
    for step in range(total_steps):
        lr = schedule_fn(step)
        for g in opt.param_groups:
            g['lr'] = lr
        x = torch.randn(2, 10, 64)
        target = torch.randn(2, 10, 64)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

# Post-LN with vs without warmup
no_warmup = lambda t: 1e-3
with_warmup = lambda t: linear_warmup_cosine(t, 30, 300, 1e-3)

l_post_no  = train_with_schedule('post', no_warmup, 1e-3)
l_post_yes = train_with_schedule('post', with_warmup, 1e-3)
l_pre_no   = train_with_schedule('pre', no_warmup, 1e-3)
l_pre_yes  = train_with_schedule('pre', with_warmup, 1e-3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(l_post_no, label='no warmup'); axes[0].plot(l_post_yes, label='warmup')
axes[0].set_title('Post-LN'); axes[0].legend(); axes[0].set_yscale('log')
axes[1].plot(l_pre_no, label='no warmup'); axes[1].plot(l_pre_yes, label='warmup')
axes[1].set_title('Pre-LN'); axes[1].legend(); axes[1].set_yscale('log')
for ax in axes: ax.set_xlabel('step'); ax.set_ylabel('loss')
plt.tight_layout(); plt.show()
# Post-LN no-warmup: spike / 발산 가능
# Post-LN warmup: stable
# Pre-LN: 둘 다 OK 그러나 warmup 이 약간 안정
```

### 실험 3 — 다양한 Warmup Ratio 비교

```python
losses_by_ratio = {}
for ratio in [0.01, 0.05, 0.10, 0.30]:
    T_w = int(ratio * 300)
    schedule = lambda t, tw=T_w: linear_warmup_cosine(t, tw, 300, 1e-3)
    losses_by_ratio[ratio] = train_with_schedule('pre', schedule, 1e-3)

plt.figure(figsize=(10, 4))
for ratio, losses in losses_by_ratio.items():
    plt.plot(losses, label=f'warmup ratio={ratio}')
plt.xlabel('step'); plt.ylabel('loss'); plt.yscale('log')
plt.title('Effect of warmup ratio')
plt.legend(); plt.show()
```

### 실험 4 — PyTorch 의 표준 scheduler

```python
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# 사용 예
torch.manual_seed(0)
model = SimpleTransformer(4, 32, 4, mode='pre')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=20, num_training_steps=200)

lrs = []
for step in range(200):
    lrs.append(opt.param_groups[0]['lr'])
    # ... actual training step ...
    scheduler.step()

plt.plot(lrs); plt.xlabel('step'); plt.ylabel('LR')
plt.title('PyTorch schedule with warmup'); plt.show()
```

### 실험 5 — Gradient Norm 의 Warmup 효과

```python
# 학습 시 gradient norm 추적
def train_track_grad(mode, schedule_fn, lr_max):
    torch.manual_seed(0)
    model = SimpleTransformer(8, 32, 4, mode=mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr_max)
    grad_norms = []
    for step in range(150):
        lr = schedule_fn(step)
        for g in opt.param_groups:
            g['lr'] = lr
        x = torch.randn(2, 10, 32)
        target = torch.randn(2, 10, 32)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        grad_norms.append(gn.item())
        opt.step()
    return grad_norms

g_post_no  = train_track_grad('post', lambda t: 1e-3, 1e-3)
g_post_yes = train_track_grad('post', lambda t: linear_warmup_cosine(t, 30, 150, 1e-3), 1e-3)

plt.figure(figsize=(9, 4))
plt.plot(g_post_no,  label='Post-LN no warmup', color='red', alpha=0.7)
plt.plot(g_post_yes, label='Post-LN warmup',   color='blue', alpha=0.7)
plt.xlabel('step'); plt.ylabel('gradient norm'); plt.yscale('log')
plt.title('Gradient norm during training')
plt.legend(); plt.show()
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 Warmup Recipe

- **GPT-3**: 375M tokens warmup (0.1% of 300B), peak LR $1.2 \times 10^{-4}$ → cosine to $0.1 \times$
- **LLaMA-2**: 2000 step warmup (~0.3%), peak LR $3 \times 10^{-4}$ → cosine to 10%
- **Chinchilla**: 1.5% warmup, more careful schedule

### 2. Warmup Schedule 의 변형

- **Linear** + **Cosine**: 가장 흔함
- **Linear** + **Linear decay**: simple
- **Inverse sqrt** (Vaswani): legacy
- **Constant** after warmup: fine-tuning, RLHF

### 3. Fine-tuning 에서의 Warmup

Pre-trained 모델 fine-tuning:
- Warmup ratio 작게 (0.5-1%)
- Peak LR 작게 ($10^{-5}$ 정도)
- Cosine decay 또는 constant

### 4. Cosine 의 final LR

- 0 까지 decay: 학습 종료가 명확
- $\eta_{\min} > 0$: 추가 학습 가능 (continued pretraining)
- LLaMA: $0.1 \times$ peak — partial decay

### 5. RLHF 의 schedule

PPO 등 RL 학습:
- Constant LR + linear decay (no cosine)
- Smaller LR (~$10^{-6}$)
- Short warmup or none

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Single peak LR | Multi-cycle (cosine restart) 도 가능 |
| Fixed total steps | 학습 중 결정 시 schedule 재계산 |
| Linear warmup | Exponential, polynomial 등 변형 |
| Cosine decay | Linear, exponential 도 가능 (큰 차이 X) |

---

## 📌 핵심 정리

$$\boxed{\text{Linear warmup + cosine decay 가 modern Transformer 의 표준 schedule}}$$

| Schedule | 식 | 사용 |
|----------|-----|------|
| Linear warmup | $\eta_t = \eta_{\max} \min(t/T_w, 1)$ | 모든 모델 |
| Cosine decay | $\eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot 0.5(1 + \cos(\pi p))$ | GPT, LLaMA |
| Inverse sqrt | $d^{-0.5} \min(t^{-0.5}, t T_w^{-1.5})$ | Vaswani 2017 |
| Constant | $\eta_{\max}$ | Fine-tuning |

| 모델 | Warmup ratio | Peak LR |
|------|--------------|---------|
| BERT | 1% | $5 \times 10^{-5}$ |
| GPT-3 | 0.1% | $1.2 \times 10^{-4}$ |
| LLaMA-2 | 0.3% | $3 \times 10^{-4}$ |
| Fine-tune | 0.5-3% | $10^{-5}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Total step 100,000, peak LR $10^{-3}$, warmup ratio 5% 인 linear-cosine schedule 의 step 1,000, 5,000, 50,000, 99,000 에서의 LR 을 계산하라. ($\eta_{\min} = 0$)

<details>
<summary>해설</summary>

$T_w = 5,000$ (5% of 100,000).

**Step 1000** (warmup 중): $\eta = 10^{-3} \cdot 1000/5000 = 2 \times 10^{-4}$.

**Step 5000** (warmup 끝): $\eta = 10^{-3}$ (peak).

**Step 50000** (cosine 중): $p = (50000-5000)/(100000-5000) = 45000/95000 \approx 0.474$. $\eta = 0.5 \cdot 10^{-3} \cdot (1 + \cos(0.474 \pi)) \approx 0.5 \cdot 10^{-3} \cdot (1 + \cos(85.4°)) \approx 0.5 \cdot 10^{-3} \cdot (1 + 0.082) \approx 5.4 \times 10^{-4}$.

**Step 99000** (cosine 끝): $p = (99000-5000)/95000 \approx 0.989$. $\eta = 0.5 \cdot 10^{-3} \cdot (1 + \cos(0.989\pi)) \approx 0.5 \cdot 10^{-3} \cdot (1 + (-0.999)) \approx 5 \times 10^{-7}$.

**Step 100000** (끝): $\eta \approx 0$ ($\eta_{\min} = 0$). $\square$

</details>

**문제 2** (심화): Warmup 이 Adam optimizer 의 어떤 결함과 직접 연결되는가? Adam 의 second moment estimate $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ 의 초기 부정확함이 warmup 의 동기 중 하나임을 분석하라.

<details>
<summary>해설</summary>

**Adam 의 second moment 초기 문제**:

$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ with $\beta_2 = 0.999$ (typical).

초기 ($t = 1$): $v_1 = 0.001 g_1^2$ — 매우 작음.

Bias correction: $\hat{v}_t = v_t / (1 - \beta_2^t)$:
$$
\hat{v}_1 = v_1 / (1 - 0.999) = v_1 / 0.001 = g_1^2
$$

→ **이론상** corrected, 그러나 **실제** $g_1$ 한 sample 의 squared gradient — **분산 큼**.

**Adam update**:
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

초기에 $\hat{v}_t$ 가 부정확 → effective LR 이 noisy.

**Warmup 의 역할**:

- 작은 $\eta$ 가 부정확한 $\hat{v}$ 의 영향을 absorb
- $\hat{v}_t$ 가 stabilize 할 시간 (typical $T_w \approx 1/(1-\beta_2) = 1000$ steps 정도)
- 그 후 정상 LR 적용

**RAdam (Liu 2019)**: variance estimate 의 confidence 까지 고려해 자동 warmup. 이론적으로 warmup-free, 그러나 표준 Adam + warmup 이 simpler 하고 충분.

**LAMB / Adafactor**: 큰 batch 에서의 추가 변형.

**결론**: Warmup 은 **Post-LN gradient + Adam variance** 두 문제를 동시에 mitigate. Pre-LN + RAdam 이라면 이론적으로 warmup 불필요하지만 production 은 simpler 표준. $\square$

</details>

**문제 3** (논문 비평): "Cosine decay to zero vs to nonzero" — 학습 끝 LR 이 0 인 경우와 $0.1 \times$ peak 인 경우의 trade-off 를 분석하라. Continued pretraining 또는 fine-tuning 의 관점에서?

<details>
<summary>해설</summary>

**Cosine decay to 0**:

장점:
- 학습 종료가 mathematically natural
- Final 모델이 converged state — 평가 명확
- "Final" 모델 명시

단점:
- 추가 학습 불가 (LR=0 에서 시작은 의미 없음)
- Continued pretraining 시 LR schedule 재시작 필요

**Cosine decay to $0.1 \times$ (10%)**:

장점:
- Continued pretraining 자연 — 같은 LR 로 학습 계속
- Multi-stage training (long context 추가, instruction tuning) 시 유연
- LLaMA-2 가 10% 채택

단점:
- "Final" 정의 모호
- 평가 시 어떤 step 이 best 인지 불명확

**Modern Practice**:

1. **Pre-training**:
   - LLaMA-2: 10% (continued pretraining 가능)
   - GPT-3: 10% (legacy, but generally common)
   - Chinchilla: 10x decay (i.e., to 10%)

2. **Fine-tuning**:
   - Constant LR 또는 linear decay
   - Cosine 보다 단순한 schedule

3. **Continued pretraining**:
   - Original schedule 의 final LR 에서 시작
   - 또는 새 cosine cycle (warm restart)

**Warm Restart (Loshchilov 2017)**:

$\eta$ 가 $0$ 에 도달하면 갑자기 peak 로 복귀 → 새 cosine cycle.

장점:
- Local minima 에서 escape 가능
- Multiple "restarts" 로 ensemble-like 효과

단점:
- 학습 안정성 저해 가능
- 실증적으로 일관된 이점 없음

**Linear Decay 비교**:

Linear decay to 0:
- Simpler — implementation 쉬움
- Cosine 과 거의 같은 final 성능
- Hyperparameter 적음

→ 일부 모델은 linear decay 채택 (T5).

**결론**:

- Cosine to 0: 학습 종료가 명확, single-stage
- Cosine to 10%: 유연성, multi-stage training
- Linear: Simpler alternative
- Warm restart: Academic, rarely production

Modern dominance: **Cosine to 10%** (LLaMA, Mistral, GPT-3 lineage). 유연성 우선. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch3-positional-encoding/05-rope-alibi.md) | [📚 README](../README.md) | [다음 ▶](./02-adamw.md)

</div>
