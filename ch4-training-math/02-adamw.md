# 02. AdamW 와 Weight Decay 분리

## 🎯 핵심 질문

- 표준 Adam + L2 regularization 과 AdamW 의 차이는 정확히 무엇인가?
- 왜 Adam 의 weight decay 가 effective decay 가 아닌가? — Loshchilov & Hutter 2019 의 분석
- AdamW 가 Transformer 의 사실상 표준 optimizer 가 된 이유는?
- $\beta_1, \beta_2, \epsilon$ 의 역할과 Transformer 표준 값은?
- AdamW 의 한계 — large batch, sparse gradient, sign-based optimizer 등 alternative 의 비교?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

AdamW 는 **modern Transformer 훈련의 사실상 표준 optimizer**:

1. **Adam 의 결함 해결** — L2 와 weight decay 의 분리가 effective regularization 가능하게
2. **Transformer 의 안정성** — Adam 의 adaptive LR 이 wide-and-deep 모델에 적합
3. **모든 모던 LLM 의 채택** — GPT, BERT, LLaMA, PaLM 모두 AdamW
4. **Hyperparameter 의 sensitivity** — $\beta_2$ 의 선택이 큰 모델에서 critical

이 문서는 AdamW 의 **수학적 정의, Adam 과의 차이, Transformer 표준 hyperparameter** 를 다룹니다.

---

## 📐 수학적 선행 조건

- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): SGD, momentum, Adam, RMSprop
- [Regularization Theory Deep Dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive): L2 regularization, weight decay
- 이전 문서: [01-warmup.md](./01-warmup.md)

---

## 📖 직관적 이해

### Adam 의 두 moment

```
First moment (m_t):    gradient 의 EMA — momentum 같음
Second moment (v_t):   squared gradient 의 EMA — adaptive LR 의 분모
Update: θ ← θ - η · m_t / √v_t
```

Adaptive: 큰 $|g|$ 인 parameter 는 effective LR 작음 (안정), 작은 $|g|$ 는 큼 (학습 가속).

### L2 vs Weight Decay 의 차이

**L2 regularization**:
```
loss = task_loss + (λ/2) ||θ||²
gradient = ∇task + λθ
```

**Weight decay** (직접):
```
update: θ ← θ - η · (∇task) - λθ
```

SGD 에서는 두 개 등가 ($\eta = $ const). **Adam 에서는 다름** — adaptive LR 이 L2 gradient 에도 적용되어 변형.

### Adam 의 결함

Adam + L2:
```
g̃ = ∇task + λθ
m, v ← EMA of (g̃, g̃²)
θ ← θ - η · m̂ / √v̂
```

**문제**: L2 항 $\lambda\theta$ 가 $\sqrt{v}$ 에 의해 **rescale** 됨 → 큰 $|\theta|$ 인 parameter 의 effective decay 가 작아짐 (의도와 반대).

AdamW 의 해결:
```
m, v ← EMA of (∇task, (∇task)²)   # L2 빼고
θ ← θ - η · (m̂ / √v̂ + λθ)        # weight decay 분리
```

L2 가 second moment 에 들어가지 않음 → 정확한 weight decay.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Adam (Kingma 2015)

For each parameter $\theta$:
$$
g_t = \nabla L(\theta_t)
$$
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Bias correction:
$$
\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)
$$

Update:
$$
\theta_{t+1} = \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$ (typical), $\epsilon = 10^{-8}$.

### 정의 2.2 — Adam + L2

Loss 에 $\frac{\lambda}{2} \|\theta\|^2$ 추가:
$$
g_t^{\text{L2}} = \nabla L(\theta_t) + \lambda \theta_t
$$

이후 standard Adam.

### 정의 2.3 — AdamW (Loshchilov & Hutter 2019)

Decoupled weight decay:
$$
\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

(decay 가 update 에 직접 추가, second moment 에 영향 안 줌)

또는 equivalently:
$$
\theta_{t+1} = (1 - \eta_t \lambda) \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

(multiplicative form)

### 정의 2.4 — Effective Weight Decay

매 step 마다 $\theta$ 가 $1 - \eta_t \lambda$ 배 — exponential decay rate $\eta_t \lambda$.

총 step $T$ 후: $\theta \to (1-\eta\lambda)^T \theta_0 \approx e^{-\eta \lambda T} \theta_0$.

---

## 🔬 정리와 증명

### 정리 2.1 — Adam + L2 의 결함

L2 의 gradient $\lambda \theta$ 가 $\hat{v}$ 의 분모로 들어감:
$$
\text{effective decay} = \eta \cdot \frac{\lambda \theta}{\sqrt{\hat{v}_t} + \epsilon}
$$

큰 $|\theta|$ 인 parameter 일수록 더 큰 $g_t = \lambda \theta$ → 더 큰 $v_t$ → **더 작은 decay** (의도 정반대).

**증명** (linear quadratic model):

가정: $\theta$ 가 quadratic stationary $\theta^* = -\lambda^{-1} g$. Adam + L2 가 이 $\theta^*$ 에 수렴 안 함 — adaptive LR 이 L2 의 효과 dampening.

### 정리 2.2 — AdamW 의 정당성

Decoupled form: weight decay 가 second moment 에 들어가지 않음.

**Theorem** (Loshchilov 2019): AdamW 가 SGD + L2 의 decoupled equivalent — adaptive part 와 regularization part 가 separate.

### 정리 2.3 — Effective Decay Rate 계산

$\theta_{t+1} = (1 - \eta_t \lambda) \theta_t - \eta_t (\text{Adam term})$

Constant $\eta, \lambda$ 가정:
$$
\theta_T = (1-\eta\lambda)^T \theta_0 + \text{Adam contribution}
$$

Effective decay rate $\eta\lambda \to e^{-T \eta \lambda}$ — exponential decay (overall).

**Typical**: $\eta = 3 \times 10^{-4}$, $\lambda = 0.1$, $T = 10^6$ steps:
$$
e^{-10^6 \cdot 3 \times 10^{-5}} = e^{-30} \approx 10^{-13} \quad \to \theta \approx 0
$$

??? 너무 빠른 decay. **Modern**: 작은 $\lambda$ (0.01 또는 0.1) + cosine LR decay 가 함께 작동 → effective decay 가 manageable.

### 정리 2.4 — Bias Correction 의 중요성

초기 ($t$ 작을 때):
- $m_t \approx (1-\beta_1) g_1$ — EMA 가 아직 build up 안 됨
- $v_t \approx (1-\beta_2) g_1^2$ — 마찬가지

Bias correction:
$$
\hat{m}_t = m_t / (1 - \beta_1^t)
$$

$t = 1$: $\hat{m}_1 = g_1$ (기댓값과 일치).
$t \to \infty$: $\hat{m}_t \to m_t$ (correction 무시 가능).

→ 초기 step 에서 정확한 estimate 보장.

### 정리 2.5 — Transformer 의 Hyperparameter 표준

**$\beta_1 = 0.9$**: gradient EMA — momentum 같음.

**$\beta_2 = 0.999$** (small) vs **0.95** (LLaMA, GPT-NeoX): 
- 0.999: 긴 averaging window — 안정적이지만 slow adaptation
- 0.95: 빠른 adaptation — 큰 모델에 better

**$\epsilon = 10^{-8}$**: numerical stability. 너무 작으면 instability, 너무 크면 effective LR 영향.

**$\lambda = 0.1$**: typical for LLM. BERT 는 $0.01$, GPT 는 $0.1$.

### 정리 2.6 — AdamW 의 Memory Cost

Per parameter: $m, v$ (2 floats) + parameter (1 float) = 3× memory of model weights.

**Mitigation**:
- **8-bit Adam** (Dettmers 2021): $m, v$ 를 8-bit 로 양자화
- **Adafactor** (Shazeer 2018): low-rank approximation of $v$ — $O(d)$ memory instead of $O(d^2)$
- **Lion** (Chen 2023): sign-based, only momentum (no $v$)

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — AdamW 바닥부터 구현

```python
import torch
import torch.nn as nn
import math

class CustomAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    @torch.no_grad()
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            
            # Update moments
            self.m[i].mul_(self.beta1).add_(g, alpha=1-self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1-self.beta2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update with decoupled weight decay
            p.mul_(1 - self.lr * self.wd)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# 테스트
torch.manual_seed(0)
model = nn.Linear(10, 1)
opt = CustomAdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

for step in range(20):
    x = torch.randn(32, 10); y = torch.randn(32, 1)
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

print(f'Final loss: {loss.item():.4f}')
print(f'Weight norm: {model.weight.norm():.4f}')
```

### 실험 2 — Adam vs AdamW 비교

```python
import matplotlib.pyplot as plt

def train_with_optimizer(opt_class, model_init, n_steps=300):
    torch.manual_seed(0)
    model = model_init()
    opt = opt_class(model.parameters(), lr=1e-2, weight_decay=0.1)
    losses = []; weight_norms = []
    for step in range(n_steps):
        x = torch.randn(32, 16); y = torch.randn(32, 1)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        weight_norms.append(sum(p.norm().item() for p in model.parameters()))
    return losses, weight_norms

# Standard PyTorch Adam (with L2 in opt arg)
losses_adam, wn_adam = train_with_optimizer(
    torch.optim.Adam, lambda: nn.Linear(16, 1)
)

# AdamW
losses_adamw, wn_adamw = train_with_optimizer(
    torch.optim.AdamW, lambda: nn.Linear(16, 1)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses_adam,  label='Adam (L2)')
axes[0].plot(losses_adamw, label='AdamW')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_yscale('log')
axes[1].plot(wn_adam,  label='Adam')
axes[1].plot(wn_adamw, label='AdamW')
axes[1].set_title('Weight norm'); axes[1].legend()
plt.tight_layout(); plt.show()
# AdamW 의 weight norm 이 더 정확히 작아짐 (decoupled decay 효과)
```

### 실험 3 — Bias Correction 의 영향

```python
# 초기 step 의 m_hat, v_hat 추적
torch.manual_seed(0)
param = torch.randn(10, requires_grad=True)
m = torch.zeros_like(param); v = torch.zeros_like(param)
beta1, beta2 = 0.9, 0.999

for t in range(1, 11):
    g = torch.randn_like(param)   # random gradient
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    print(f't={t}: ||m||={m.norm():.4f}, ||m_hat||={m_hat.norm():.4f}, '
          f'||v||={v.norm():.4f}, ||v_hat||={v_hat.norm():.4f}')

# Initial 에서 m, v 가 매우 작음, m_hat, v_hat 으로 corrected
```

### 실험 4 — $\beta_2$ 영향

```python
for beta2 in [0.95, 0.99, 0.999]:
    torch.manual_seed(0)
    model = nn.Linear(16, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, beta2),
                            weight_decay=0.1)
    losses = []
    for step in range(200):
        x = torch.randn(32, 16); y = torch.randn(32, 1)
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    plt.plot(losses, label=f'β₂={beta2}')

plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
plt.title('Effect of β₂ in AdamW')
plt.yscale('log'); plt.show()
```

### 실험 5 — Effective Weight Decay 측정

```python
# 시작: 큰 weight, decay 없는 optimizer 확인
torch.manual_seed(0)
param = torch.randn(100) * 10   # 큰 magnitude

# Constant LR, decay only (no gradient)
lr, wd = 1e-3, 0.1
weights_history = [param.norm().item()]

for t in range(10000):
    param = (1 - lr * wd) * param   # AdamW 의 decay 부분만
    weights_history.append(param.norm().item())

plt.plot(weights_history)
plt.xlabel('step'); plt.ylabel('||θ||'); plt.yscale('log')
plt.title(f'Decay only: ||θ|| → 0 exponentially, rate={lr*wd}')
plt.show()
print(f'After 10k steps: ratio = {weights_history[-1] / weights_history[0]:.4e}')
print(f'Expected: e^(-10000 * 0.0001) = {math.exp(-10000 * lr * wd):.4e}')
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 Hyperparameter

| Model | $\beta_1$ | $\beta_2$ | $\epsilon$ | $\lambda$ | Peak LR |
|-------|-----------|-----------|------------|-----------|---------|
| GPT-3 | 0.9 | 0.95 | $10^{-8}$ | 0.1 | $1.2e^{-4}$ |
| LLaMA-2 | 0.9 | 0.95 | $10^{-5}$ | 0.1 | $3e^{-4}$ |
| BERT | 0.9 | 0.999 | $10^{-6}$ | 0.01 | $1e^{-4}$ |
| Mistral | 0.9 | 0.95 | $10^{-5}$ | 0.1 | $5e^{-4}$ |

### 2. Weight Decay 의 적용 범위

**Standard practice**:
- Linear/Conv weights: weight decay ✓
- Bias, LayerNorm $\gamma, \beta$: **NO weight decay** (param group 분리)
- Embedding: 일부 decay (모델별 다름)

```python
# Param group 분리
no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n or 'ln' in n]
decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'ln'])]
opt = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4)
```

### 3. 8-bit AdamW

큰 모델 (10B+) 에서 optimizer state memory 가 model size 의 2× → 8-bit AdamW (Dettmers 2021):
- $m, v$ 를 8-bit 로 양자화
- 거의 같은 final 성능
- 메모리 4× 절약

### 4. Adafactor (T5)

$v_t$ 를 row × column outer product 로 근사 (low-rank):
$$
v_t \approx u r^\top
$$

- $O(d_{\text{out}} + d_{\text{in}})$ memory instead of $O(d_{\text{out}} \cdot d_{\text{in}})$
- T5, PaLM 등 채택

### 5. Lion (Chen 2023)

Sign-based: $\theta \leftarrow \theta - \eta \cdot \text{sign}(m)$ — second moment 제거.
- Less memory ($v$ 없음)
- 일부 task 에서 우수
- LR 다르게 (작게) 필요

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Standard Adam moments | 큰 batch 시 LAMB 사용 |
| Single $\eta, \lambda$ | Layer-wise tuning 가능 |
| FP32 moments | 8-bit AdamW |
| Dense gradient | Sparse 시 SparseAdam |
| Stationary problem | RL/RLHF 는 다른 dynamics |

---

## 📌 핵심 정리

$$\boxed{\text{AdamW: } \theta_{t+1} = (1 - \eta_t \lambda) \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}$$

| 양 | 식 | Typical |
|----|-----|---------|
| First moment | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | $\beta_1 = 0.9$ |
| Second moment | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | $\beta_2 = 0.95$ (LLM), $0.999$ (legacy) |
| Bias correction | $\hat{m}/\hat{v} = m/(1-\beta^t)$ | 초기 정확성 |
| Weight decay | Decoupled | $\lambda = 0.1$ |
| $\epsilon$ | Numerical stability | $10^{-8}$ to $10^{-5}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Adam 과 AdamW 의 update step 식의 차이를 한 줄씩 비교하라. L2 항 $\lambda \theta$ 가 두 경우에 어디로 들어가는가?

<details>
<summary>해설</summary>

**Adam + L2** ($g_t = \nabla L + \lambda \theta$):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)(g + \lambda\theta)
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(g + \lambda\theta)^2
$$
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

L2 항이 **second moment $v$** 에 들어가서 분모로 영향 → effective decay 가 dampened.

**AdamW**:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2
$$
$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

L2 가 **update 자체에 분리** 추가 — second moment 영향 없음 → effective decay 가 정확히 $\eta \lambda$.

**핵심 차이**: AdamW 의 weight decay 가 adaptive LR 의 영향을 받지 않음. $\square$

</details>

**문제 2** (심화): "$\beta_2 = 0.95$ vs $0.999$" — LLaMA 가 0.95 를 사용하는 이유는 무엇인가? Larger model 에서 작은 $\beta_2$ 가 유리한 이유를 분석하라.

<details>
<summary>해설</summary>

**$\beta_2 = 0.999$ (default)**:
- 긴 averaging window (~1000 steps)
- 안정적인 second moment estimate
- 매우 작은 모델 / short training 에 적합

**$\beta_2 = 0.95$ (LLaMA, GPT-NeoX)**:
- 짧은 window (~20 steps)
- 빠른 adaptation
- 큰 모델 에 적합

**Why smaller $\beta_2$ for larger models?**

1. **Distribution shift**:
   - 큰 모델은 매 layer 마다 distribution 이 빠르게 변화
   - $v_t$ 가 너무 stale 하면 outdated estimate
   - $\beta_2 = 0.95$ 가 더 빠른 adaptation

2. **Gradient variance**:
   - 큰 모델은 mini-batch 내 gradient variance 큼
   - Long EMA ($\beta_2 = 0.999$) 가 high-variance 평균 — slow
   - Short EMA ($\beta_2 = 0.95$) 가 더 responsive

3. **Numerical stability**:
   - $\beta_2 = 0.999$ 시 $v_t$ 가 매우 작음 (초기) — bias correction 으로 보정하지만 noisy
   - $\beta_2 = 0.95$ 가 더 빠르게 reasonable magnitude

4. **Empirical**:
   - GPT-3, LLaMA, Mistral 모두 0.95
   - Best practice 가 이 방향으로 수렴

**$\epsilon$ 도 함께 조정**:
- $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ (small)
- $\beta_2 = 0.95$, $\epsilon = 10^{-5}$ (larger) — small $v$ 영역에서 instability 회피

**Trade-off**:
- 작은 $\beta_2$ → noisy $v_t$ → Adam update 가 noisy → 학습 약간 unstable
- 충분한 batch size + LR scheduling 으로 mitigate

따라서 **모델 scale 과 hyperparameter 가 connected** — 단순히 default 사용 X. 큰 LLM 은 careful tuning 필요. $\square$

</details>

**문제 3** (논문 비평): Lion (Chen 2023) 같은 sign-based optimizer 가 AdamW 의 일부 task 에서 우수하다고 주장한다. Sign 만 사용하는 것이 second moment estimate 와 비교해 어떤 trade-off 를 갖는가? Modern LLM 에 채택될 가능성은?

<details>
<summary>해설</summary>

**Lion 의 update**:

$$
c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(c_t) - \eta \lambda \theta_t
$$
$$
m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t
$$

(EMA $m$ for next step, sign 만 update 에 사용)

**핵심 차이**:
- AdamW: $\hat{m}/\sqrt{\hat{v}}$ — gradient magnitude 의 정밀 정보
- Lion: $\text{sign}(c)$ — 방향만, magnitude 는 $\eta$ 가 결정

**Trade-off**:

**Lion 의 장점**:
1. **Memory**: $v$ 없음 → optimizer state 1× model (vs AdamW 의 2×)
2. **Computation**: sqrt 와 division 없음, sign 만
3. **일부 task 우수**: vision (ViT) 에서 약간 better

**Lion 의 단점**:
1. **Noisy update**: sign 이 magnitude 무시 → coarse direction
2. **LR sensitivity**: AdamW 의 1/3 정도의 LR 필요 — careful tuning
3. **Convergence theory**: less developed than Adam
4. **LLM 에서 unclear**: Anthropic / Google 등의 frontier LLM 들은 여전히 AdamW

**Modern LLM 채택 가능성**:

**Pro**:
- Memory saving 가 큰 모델에 매력
- 8-bit AdamW 와 결합 가능

**Con**:
- AdamW 의 robustness 가 검증됨 — 큰 모델에서 unproven
- LR tuning 더 어려움
- Open source community 가 AdamW 표준

**예측**:
- Frontier LLM 은 당분간 AdamW 유지
- Niche application (memory-constrained, fine-tuning) 에서 Lion 채택 가능
- 더 많은 ablation 후 결정 — 현재 active research

**Sophia (Liu 2023)** 같은 second-order Adam 변형이 LLM 에서 실증 우수 결과 — 다른 방향의 진화.

**결론**: AdamW 가 8년간 표준 — 검증된 robustness 가 우선. Lion 은 academic 흥미, 큰 frontier LLM 에서 dominant 채택 어려울 가능성. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-warmup.md) | [📚 README](../README.md) | [다음 ▶](./03-label-smoothing.md)

</div>
