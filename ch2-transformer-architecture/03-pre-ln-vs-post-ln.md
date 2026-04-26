# 03. Pre-LN vs Post-LN — Warmup 의 필요성

## 🎯 핵심 질문

- Post-LN 의 layer-wise gradient norm 은 왜 깊이 $L$ 에 따라 누적적으로 발산하는가?
- Pre-LN 은 어떻게 $O(1/\sqrt{L})$ 로 bound 되는가? Xiong 2020 의 증명의 핵심 idea 는?
- Warmup 이 왜 Post-LN 에서 거의 필수적인가? 그리고 Pre-LN 에서도 큰 모델은 권장되는 이유는?
- DeepNorm (Wang 2022) 이 어떻게 1000+ layer 학습을 가능하게 했는가?
- 왜 GPT/LLaMA/PaLM 모두 Pre-LN 을 채택했는가? Post-LN 의 generation 품질 우위 주장은?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

LayerNorm 의 위치는 **단순 architectural detail 이 아니라** Transformer 훈련의 본질을 결정:

1. **Warmup 의 필수성** — Post-LN 은 warmup 없이 발산, Pre-LN 은 더 robust
2. **Deep stacking 가능성** — Pre-LN 이 100+ layer 가능, Post-LN 은 어렵게
3. **Modern LLM 의 표준** — GPT/LLaMA/PaLM 모두 Pre-LN, 이유는 안정성
4. **Initialization 의 영향** — Post-LN + 잘못된 초기화 = 학습 실패

이 문서는 Xiong 2020 의 **gradient norm 분석** 을 직접 증명하고, 두 변형의 학습 dynamics 를 실험으로 보입니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-transformer-block.md](./01-transformer-block.md)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backpropagation, gradient flow, residual
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Warmup schedule, learning rate
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Operator norm, Jacobian

---

## 📖 직관적 이해

### Gradient Highway 의 차이

```
Pre-LN:        x ──┬──→ LN ─→ Attn ─┐
                   │                  │
                   └────── + ─────────┘    ← clean residual
                          (gradient highway: ∂y/∂x = I + small)

Post-LN:       x ──┬──→ Attn ─┐
                   │           │
                   └────── + ──┴──→ LN
                              ↑
                          (gradient through LN Jacobian)
                          (∂y/∂x = LN_J · (I + f_J))
```

**Pre-LN**: residual 이 clean — gradient 가 layer 통과할 때 $I$ 항이 직접 흐름.

**Post-LN**: 매 layer 마다 LN-Jacobian 곱 — depth 별 누적 효과.

### Layer-wise Gradient Norm

Xiong 2020 의 핵심 theorem:

| 조건 | $\|\nabla L\| / \|\nabla h_{L}\|$ |
|------|----------------------------------|
| **Pre-LN** | $O(1)$ |
| **Post-LN** | $O(L)$ — 발산 |

Post-LN 은 **초기 gradient 가 layer 깊이에 따라 폭발** → big LR 시 발산.

### Warmup 의 역할

```
Learning rate schedule:
   ↑
LR │     ╱──────╮
   │   ╱        ╲___          <- warmup 후 cosine decay
   │ ╱
   └─────────────────→ step
     0    1k     T
```

훈련 초기에 작은 LR 로 출발 → gradient norm 이 stabilize 할 시간 → 그 후 정상 LR.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Pre-LN vs Post-LN (재확인)

**Pre-LN**: $y = x + f(\text{LN}(x))$
**Post-LN**: $y = \text{LN}(x + f(x))$

### 정의 3.2 — LayerNorm Jacobian

$\text{LN}(x) = (x - \mu) / \sigma \odot \gamma + \beta$. Jacobian:
$$
J_{\text{LN}}(x) = \frac{\gamma}{\sigma} \left( I - \frac{1}{d} \mathbf{1}\mathbf{1}^\top - \frac{1}{d \sigma^2} (x - \mu)(x - \mu)^\top \right)
$$

**Operator norm**: $\|J_{\text{LN}}\| \leq \|\gamma\|_\infty / \sigma$.

### 정의 3.3 — Layer-wise Gradient Ratio

$$
r_l := \frac{\|\nabla_{h^{(l-1)}} L\|}{\|\nabla_{h^{(l)}} L\|}
$$

각 layer 가 gradient 를 얼마만큼 amplify 하는지.

### 정의 3.4 — Warmup Schedule (Linear)

$$
\eta_t = \eta_{\max} \cdot \min\!\left(\frac{t}{T_w}, 1\right)
$$

$t = 0$ 부터 $T_w$ step 까지 LR 이 0 에서 $\eta_{\max}$ 로 선형 증가.

### 정의 3.5 — DeepNorm (Wang 2022)

$$
y = \text{LN}(\alpha x + f(x))
$$

with $\alpha > 1$ — residual scaling. 큰 $\alpha$ 가 gradient 를 stabilize, 1000+ layer 가능.

---

## 🔬 정리와 증명

### 정리 3.1 — Pre-LN 의 Gradient Bound (Xiong 2020)

Pre-LN Transformer 의 $L$-layer gradient:
$$
\|\nabla_{h^{(0)}} L\| \leq C \|\nabla_{h^{(L)}} L\|, \quad C = O(1)
$$

(layer 수에 무관)

**증명 sketch**:

Pre-LN block: $h^{(l)} = h^{(l-1)} + g_l(h^{(l-1)})$ where $g_l = f_l \circ \text{LN}$.

$$
\frac{\partial h^{(l)}}{\partial h^{(l-1)}} = I + \frac{\partial g_l}{\partial h^{(l-1)}}
$$

가정: $\|\partial g_l\| < c < 1$ (sub-layer 의 Lipschitz constant).

$$
\frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{l=1}^L (I + J_l), \quad \|J_l\| < c
$$

Operator norm:
$$
\left\| \prod_{l=1}^L (I + J_l) \right\| \leq \prod_l \|I + J_l\| \leq \prod_l (1 + c) = (1+c)^L
$$

$L$ 에 따라 지수적 증가? — 그러나 Pre-LN 에서 LN 의 Jacobian 이 $\|J_{\text{LN}}\| \leq O(1/\sqrt{d})$ 보장 (Xiong analysis), 따라서 $c < 1/L^{1/2}$ → product 가 $O(1)$ bound. $\square$

### 정리 3.2 — Post-LN 의 Gradient Norm 누적

Post-LN: $h^{(l)} = \text{LN}(h^{(l-1)} + f_l(h^{(l-1)}))$.

$$
\frac{\partial h^{(l)}}{\partial h^{(l-1)}} = J_{\text{LN}}^{(l)} \cdot (I + J_{f_l})
$$

LN Jacobian 이 매 layer 마다 곱해짐. $\|J_{\text{LN}}\| \approx 1/\sigma$, $\sigma$ 가 layer 별 다름.

**Xiong 의 핵심 결과**: 초기에 $\|\nabla_{h^{(0)}}\| / \|\nabla_{h^{(L)}}\| \approx O(L)$ — layer 수에 비례 증가.

**의미**: Deep Post-LN 에서 input gradient 가 매우 큼 → 큰 LR 시 발산.

### 정리 3.3 — Warmup 의 정당성

큰 LR $\eta$ + 큰 gradient norm $\|g\|$ → 큰 step $\eta \|g\|$ → 발산.

**Warmup**: 초기에 $\eta_t = \eta \cdot t/T_w$ 작게 → gradient 가 stabilize 할 시간 (LN parameters $\gamma, \beta$ 가 학습되면서 $\sigma$ 가 적절한 값으로 수렴) → 그 후 정상 LR.

**Pre-LN**: gradient 가 이미 bounded → warmup 덜 critical (그러나 큰 모델은 권장).

**Post-LN**: gradient 가 폭발 → warmup 필수.

### 정리 3.4 — DeepNorm 의 Stabilization

Wang 2022: $y = \text{LN}(\alpha x + f(x))$ with $\alpha = (2L)^{1/4}$ (encoder), $\alpha = (3L)^{1/4}$ (decoder).

**효과**: residual path 의 magnitude 를 $\alpha$ 배 → LN 후 residual 이 dominant → gradient highway 회복.

**결과**: 1000-layer Transformer 학습 성공 (DeepNet).

### 정리 3.5 — Initialization 의 보완

Post-LN 의 안정성을 초기화로 보완:
- **Xavier**: $\text{Var}(W) = 2/(n_{\text{in}} + n_{\text{out}})$
- **Admin** (Liu 2020): Adaptive initialization — sub-layer 의 output norm 을 layer 별 제어
- **T-Fixup** (Huang 2020): warmup-free Post-LN 학습

이런 방법들도 Pre-LN 의 inherent stability 를 따라잡지 못해 modern 표준은 Pre-LN.

### 정리 3.6 — Pre-LN 의 단점 (Generation Quality)

일부 연구 (Liu 2020): Pre-LN 이 generation quality 에서 Post-LN 보다 약간 떨어짐. 그러나:
- 큰 모델 + 충분 훈련 시 차이 무시할 수준
- 안정성 vs 약간의 품질 trade-off
- Modern 결정: 안정성 ↑ → Pre-LN

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Layer-wise Gradient Norm 측정

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff); self.fc2 = nn.Linear(d_ff, d)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class PreLNBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn  = FFN(d, d_ff)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        x_ln = self.ln1(x)
        x = x + self.attn(x_ln, x_ln, x_ln)[0]
        x = x + self.ffn(self.ln2(x))
        return x

class PostLNBlock(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn  = FFN(d, d_ff)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.ffn(x))
        return x

def measure_grad_norms(block_class, num_layers, d=64, h=8, d_ff=256):
    """각 layer 의 input gradient norm"""
    blocks = nn.ModuleList([block_class(d, h, d_ff) for _ in range(num_layers)])
    x = torch.randn(1, 10, d, requires_grad=True)
    
    activations = [x]
    h_curr = x
    for blk in blocks:
        h_curr = blk(h_curr)
        h_curr.retain_grad()
        activations.append(h_curr)
    
    h_curr.sum().backward()
    return [a.grad.norm().item() for a in activations]

print('Pre-LN gradient norm by layer:')
norms_pre = measure_grad_norms(PreLNBlock, 12)
print([f'{n:.3f}' for n in norms_pre])

print('\nPost-LN gradient norm by layer:')
norms_post = measure_grad_norms(PostLNBlock, 12)
print([f'{n:.3f}' for n in norms_post])

plt.figure(figsize=(9, 4))
plt.plot(norms_pre,  'o-', label='Pre-LN')
plt.plot(norms_post, 's-', label='Post-LN')
plt.yscale('log')
plt.xlabel('Layer index'); plt.ylabel('||grad|| (log scale)')
plt.title('Gradient norm propagation: Pre-LN bounded, Post-LN diverges')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

### 실험 2 — Warmup 효과 시뮬레이션

```python
def train_with_warmup(block_class, num_layers, lr_max, warmup_steps, total_steps=200):
    blocks = nn.Sequential(*[block_class(64, 8, 256) for _ in range(num_layers)])
    opt = torch.optim.Adam(blocks.parameters(), lr=lr_max)
    losses = []
    
    for step in range(total_steps):
        # Warmup schedule
        lr = lr_max * min(step / max(warmup_steps, 1), 1.0) if warmup_steps > 0 else lr_max
        for g in opt.param_groups:
            g['lr'] = lr
        
        x = torch.randn(2, 10, 64)
        y_target = torch.randn(2, 10, 64)
        loss = F.mse_loss(blocks(x), y_target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(blocks.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

# Pre-LN: warmup 있을 때 vs 없을 때
torch.manual_seed(0)
loss_pre_warm  = train_with_warmup(PreLNBlock, 12, 1e-3, 50)
torch.manual_seed(0)
loss_pre_nowarm = train_with_warmup(PreLNBlock, 12, 1e-3, 0)

# Post-LN: warmup 없을 때 발산 위험
torch.manual_seed(0)
loss_post_warm  = train_with_warmup(PostLNBlock, 12, 1e-3, 50)
torch.manual_seed(0)
loss_post_nowarm = train_with_warmup(PostLNBlock, 12, 1e-3, 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(loss_pre_warm,   label='warmup'); axes[0].plot(loss_pre_nowarm, label='no warmup')
axes[0].set_title('Pre-LN'); axes[0].legend(); axes[0].set_ylabel('loss')
axes[1].plot(loss_post_warm,  label='warmup'); axes[1].plot(loss_post_nowarm, label='no warmup')
axes[1].set_title('Post-LN'); axes[1].legend()
for ax in axes: ax.set_xlabel('step'); ax.set_yscale('log')
plt.tight_layout(); plt.show()
# Post-LN no-warmup 시 loss spike / 발산 가능
```

### 실험 3 — DeepNorm 시뮬레이션

```python
class DeepNormBlock(nn.Module):
    def __init__(self, d, h, d_ff, alpha=1.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn  = FFN(d, d_ff)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.alpha = alpha
    def forward(self, x):
        x = self.ln1(self.alpha * x + self.attn(x, x, x)[0])
        x = self.ln2(self.alpha * x + self.ffn(x))
        return x

def measure_grad(block_class_kwargs, num_layers, **block_kwargs):
    blocks = nn.Sequential(*[block_class_kwargs(**block_kwargs) for _ in range(num_layers)])
    x = torch.randn(1, 10, 64, requires_grad=True)
    blocks(x).sum().backward()
    return x.grad.norm().item()

print('Effect of DeepNorm α on gradient norm:')
for L in [12, 24, 48]:
    alpha = (2 * L) ** 0.25
    g_post     = measure_grad(PostLNBlock, L, d=64, h=8, d_ff=256)
    g_deepnorm = measure_grad(DeepNormBlock, L, d=64, h=8, d_ff=256, alpha=alpha)
    print(f'L={L:3d}, α={alpha:.2f}: Post-LN |grad|={g_post:.4f}, DeepNorm |grad|={g_deepnorm:.4f}')
# DeepNorm 이 더 stable
```

### 실험 4 — LN 위치별 hidden norm 진화

```python
def trace_hidden_norms(block_class, num_layers):
    blocks = nn.Sequential(*[block_class(64, 8, 256) for _ in range(num_layers)])
    x = torch.randn(1, 10, 64)
    norms = [x.norm().item()]
    h = x
    with torch.no_grad():
        for blk in blocks:
            h = blk(h)
            norms.append(h.norm().item())
    return norms

torch.manual_seed(0)
n_pre  = trace_hidden_norms(PreLNBlock, 24)
torch.manual_seed(0)
n_post = trace_hidden_norms(PostLNBlock, 24)

plt.figure(figsize=(9, 4))
plt.plot(n_pre,  'o-', label='Pre-LN')
plt.plot(n_post, 's-', label='Post-LN')
plt.xlabel('Layer'); plt.ylabel('||h||')
plt.title('Hidden state norm: Pre-LN grows (no LN at output), Post-LN bounded')
plt.legend(); plt.grid(alpha=0.3); plt.show()
# Pre-LN: 누적 residual 로 norm 증가, output 에 final LN 추가가 일반적
# Post-LN: 매 layer LN 으로 norm bounded
```

### 실험 5 — Final LN의 효과 (Pre-LN convention)

```python
class TransformerWithFinalLN(nn.Module):
    def __init__(self, num_layers, d, h, d_ff):
        super().__init__()
        self.blocks = nn.ModuleList([PreLNBlock(d, h, d_ff) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(d)   # Pre-LN 표준 — output 에 final LN
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.final_ln(x)

# Norm 비교
torch.manual_seed(0)
model = TransformerWithFinalLN(24, 64, 8, 256)
x = torch.randn(1, 10, 64)
y = model(x)
print(f'Output norm with final LN: {y.norm():.4f}')
# Pre-LN 의 표준은 output 에 final LN 추가 — modern 모든 LLM 따름
```

---

## 🔗 실전 활용

### 1. 모던 LLM 의 LN 패턴

- **GPT-3, GPT-4**: Pre-LN + final LN
- **LLaMA, Mistral**: Pre-RMSNorm + final RMSNorm
- **PaLM**: Pre-LN, parallel attention/FFN
- **BERT**: Post-LN (예외, 작은 모델)

### 2. Warmup Schedule 의 표준

- Linear warmup → Cosine decay (가장 흔함)
- Warmup ratio: 1-10% of total steps (작은 모델 1%, 큰 모델 1-3%)
- Examples: GPT-3 (375M warmup steps / 300B token total), LLaMA (2000 warmup steps)

### 3. Deep Stacking 의 한계

- Pre-LN: 100+ layer 가능 (GPT-3: 96, GPT-4 estimate 120)
- Post-LN: ~12-24 가 한계 (BERT-large: 24)
- DeepNorm: 1000+ 가능 (DeepNet)

### 4. Mixed Precision 과의 상호작용

LN 의 $\sigma$ 계산은 FP32 가 필요 (numerical stability). PyTorch 의 `torch.cuda.amp` 가 자동 처리. RMSNorm 은 mean centering 없어서 더 안정.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Standard LN | RMSNorm 도 동일 분석 적용 |
| Pre-LN warmup-free | 큰 모델 (GPT-3+) 은 여전히 warmup 권장 |
| 무한 width | 작은 모델 (small-d) 에서 분석 정확도 ↓ |
| Single-task | Multi-task / continual learning 은 별도 |
| Block 동일 | 일부 변형 (Sandwich-LN) 은 다른 분석 |

---

## 📌 핵심 정리

$$\boxed{\text{Pre-LN: } \|\nabla\| = O(1), \quad \text{Post-LN: } \|\nabla\| = O(L)}$$

| 요소 | Pre-LN | Post-LN |
|------|--------|---------|
| LN 위치 | sub-layer 입력 측 | residual 출력 후 |
| Gradient highway | Clean ($I + J_f$) | LN-Jacobian 곱 ($J_{\text{LN}} (I + J_f)$) |
| Layer-wise norm | Bounded | Diverges with $L$ |
| Warmup | 권장 | 필수 |
| Deep stacking | 100+ | ~24 한계 |
| Modern usage | GPT, LLaMA, PaLM | BERT (예외) |
| Final LN | 표준 추가 | 자동 (last block 의 LN) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Pre-LN block 에서 sub-layer weight $W$ 가 모두 0 일 때 output 이 input 과 (거의) 같음을 보이라. Post-LN 에서는?

<details>
<summary>해설</summary>

**Pre-LN**, $f \equiv 0$ 가정:
$$
y = x + 0 + 0 = x
$$

(sub-layer 가 0 출력하면 residual 만) → 정확히 identity. ✓

**Post-LN**:
$$
y = \text{LN}(x + 0) = \text{LN}(x)
$$

LN 이 적용됨 → identity 아니지만 가까움. 그러나 LN 의 $\gamma, \beta$ 가 학습되면서 보정됨 (typical $\gamma \approx 1$, $\beta \approx 0$).

**의미**: Pre-LN 은 정확한 identity 가능 (residual highway), Post-LN 은 LN 변형 — 이것이 deep learning 의 차이로 이어짐. $\square$

</details>

**문제 2** (심화): Xiong 2020 의 gradient norm bound $\|\nabla\| = O(L)$ for Post-LN 의 핵심 step 을 sketch 하라. 어떤 가정 (LN 의 $\gamma, \sigma$ 분포 등) 이 사용되는가?

<details>
<summary>해설</summary>

**Step 1**: Post-LN block 의 forward
$$
h^{(l)} = \text{LN}(h^{(l-1)} + f_l(h^{(l-1)}))
$$

**Step 2**: Backward
$$
\nabla_{h^{(l-1)}} L = J_{\text{LN}}^{(l)} \cdot (I + J_{f_l}) \cdot \nabla_{h^{(l)}} L
$$

**Step 3**: LN Jacobian 의 norm — Xiong 의 가정 하에 $\|J_{\text{LN}}\| \approx 1$ at initialization (LN 의 $\gamma \approx 1$, $\sigma \approx 1$).

**Step 4**: Sub-layer Jacobian — initialization 시 $\|J_{f_l}\| = O(1)$ (Xavier 초기화).

**Step 5**: Layer 누적
$$
\nabla_{h^{(0)}} L = \prod_{l=1}^L J_{\text{LN}}^{(l)} (I + J_{f_l}) \cdot \nabla_{h^{(L)}} L
$$

각 항이 $\approx (I + O(1))$ 이지만, **LN 의 mean centering 이 specific direction 을 강조** — variance 가 누적적으로 변화.

**Empirical** (Xiong 의 측정): $\|\nabla_{h^{(0)}}\| / \|\nabla_{h^{(L)}}\| \approx O(L)$ — linear growth.

**Pre-LN 비교**: $\nabla = \prod (I + J_{f_l})$ — LN-Jacobian 이 product 외부 (sub-layer 입력에만 적용). Operator norm $\|I + J_f\| \leq 1 + O(1/\sqrt{d})$ (Xavier + LN 의 normalization), product 가 $(1 + O(1/\sqrt{d}))^L \approx 1$ for $L = O(\sqrt{d})$.

**핵심 차이**: LN 이 highway 안 (Pre-LN) 인지 밖 (Post-LN) 인지가 product 의 stabilization 결정. $\square$

</details>

**문제 3** (논문 비평): DeepNorm (Wang 2022) 이 1000+ layer Transformer 학습을 가능하게 했지만 modern LLM (GPT, LLaMA) 은 여전히 100 layer 이내. 왜 깊이를 늘리지 않고 width 를 늘리는 방향으로 진화했는가? Scaling laws (Ch7-01) 와의 연결은?

<details>
<summary>해설</summary>

**Why not deeper?**:

1. **Diminishing returns**: 깊이 증가의 marginal gain 이 width 보다 작음 — Kaplan 2020 의 scaling law fit 에서 width × depth 가 거의 같은 효과.

2. **Inference latency**: depth 가 sequential dependency — $L$-layer forward 가 $L$ step. Width 는 parallel.
   - GPT-4 (estimate 120 layer): 한 token 생성에 120 sequential step
   - 깊이 2× 시 latency 2×, throughput 2× 감소

3. **Memory layout**: Activation memory 가 depth × batch × seq × width. Depth 늘리기보다 width 가 GPU 효율적.

4. **Optimization difficulty**: Pre-LN 도 100+ layer 이상에서 careful tuning 필요. DeepNorm 의 1000-layer 는 special trick (학습 가능하지만 실용 가치 ↓).

**Scaling Laws 와의 연결**:

Chinchilla (Hoffmann 2022): compute-optimal 은 $N = 70B$ params + $D = 1.4T$ tokens. $N$ 을 어떻게 분배? — width × depth × num_heads.

**경험칙**:
- $d \times L = N / (12 d_{\text{ff}})$ 정도 ($N \approx 12 L d^2$ for FFN-dominated)
- $d/L \approx 100$ ratio (Wei 2022 등 ablation)
- Depth 를 늘릴수록 width 줄여야 — total $N$ 보존
- 그러나 inference latency 와 표현력 측면에서 wide & shallow 우세

**결론**:

DeepNorm 의 vertical extreme 은 academic interesting, 그러나:
- Compute-optimal 은 wide & moderate-depth
- LLaMA-2 70B: 80 layer × 8192 width
- GPT-4 (estimate): 120 layer × 16384+ width

Modern LLM 은 **moderate depth + extreme width + MoE** 의 방향. Depth 1000 은 real-world 에서 inefficient. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-ffn-role.md) | [📚 README](../README.md) | [다음 ▶](./04-encoder-vs-decoder.md)

</div>
