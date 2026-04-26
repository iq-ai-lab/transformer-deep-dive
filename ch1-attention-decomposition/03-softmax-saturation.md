# 03. Softmax Saturation 과 Gradient Vanishing

## 🎯 핵심 질문

- Softmax 의 gradient (Jacobian) 는 무엇이며, 어떤 입력 영역에서 0 으로 수렴하는가?
- "Saturation" 이 logit 의 어떤 통계적 조건에서 발생하고, $\sqrt{d_k}$ scaling 없이는 왜 거의 항상 발생하는가?
- 한 번 saturated 된 attention 을 학습으로 다시 desaturate 시킬 수 있는가? Why or why not?
- Saturation 이 deep transformer 에서 layer-wise 어떻게 누적되는가?
- 이 분석이 RNN 의 gradient vanishing 과 어떻게 다르고 같은가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Softmax saturation 은 **훈련 실패의 가장 대표적 원인** 입니다. $\sqrt{d_k}$ scaling 이 없거나 잘못된 초기화 시:

1. **첫 forward 부터 attention 이 one-hot** — 모든 token 이 단 하나의 token 에만 attend
2. **Gradient 가 즉시 0** — softmax Jacobian 이 0 → backprop 이 멈춤
3. **학습 회복 불가** — 가중치가 변하지 않으니 saturation 영구화

이 문서는 softmax 의 미분 분석으로 이 현상을 **수학적으로 정량화** 하고, $\sqrt{d_k}$ 가 어떻게 saturation 을 회피하는지 보입니다. Ch1-02 의 분산 분석이 입력 측 정규화였다면, 이 문서는 출력 측 (gradient) 분석.

---

## 📐 수학적 선행 조건

- 미적분: Multivariate chain rule, Jacobian
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Outer product, rank-1 matrix
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backpropagation, gradient flow
- 이전 문서: [02-sqrt-dk-scaling.md](./02-sqrt-dk-scaling.md) — 분산 분석

---

## 📖 직관적 이해

### Softmax 의 두 영역

```
입력 logit z              softmax(z)              gradient
모두 ≈ 0:                  ≈ 균등 (1/n)             정상 (sensitive)
한 entry 가 ≫ 다른:        ≈ one-hot                ≈ 0 (saturated)  🚨
```

Softmax 는 입력 차이가 클수록 한쪽으로 수렴 — 그러나 그 영역에서 미분이 0.

### Why saturation kills training

Backprop 의 chain rule:
$$
\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial \text{Attn}} \cdot \frac{\partial \text{Attn}}{\partial S} \cdot \frac{\partial S}{\partial Q}
$$

$\partial \text{Attn} / \partial S$ 가 softmax Jacobian. Saturated 시 이 값이 0 → 전체 gradient 가 0 → $Q$ 가 업데이트 안 됨 → saturation 지속.

### RNN 의 vanishing 과 비교

- **RNN vanishing**: 시간 축으로 gradient 가 $\prod_t \sigma'(\cdot)$ 누적, $T$ 가 클 때 0
- **Softmax saturation**: 한 layer 안에서 발생, $d_k$ 가 클 때 그 layer 의 gradient 가 0

둘 다 "scale" 이 잘못 통제되었을 때 미분이 0 으로 수렴하는 본질적으로 같은 현상.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Softmax Function

$z = (z_1, \ldots, z_n) \in \mathbb{R}^n$ 에 대해:
$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \in (0, 1), \quad \sum_i \sigma(z)_i = 1
$$

### 정의 3.2 — Softmax Jacobian

$$
J_{ij} := \frac{\partial \sigma(z)_i}{\partial z_j} = \sigma(z)_i (\delta_{ij} - \sigma(z)_j)
$$

행렬 형태:
$$
J = \text{diag}(\sigma) - \sigma \sigma^\top
$$

### 정의 3.3 — Saturation

Softmax 가 **$\epsilon$-saturated** 이려면:
$$
\max_i \sigma(z)_i \geq 1 - \epsilon
$$

(some entry 가 1 에 가까움)

### 정의 3.4 — Effective Gradient Norm

Attention layer 의 gradient 척도:
$$
\|\nabla_Q L\|_F = \left\| \frac{\partial L}{\partial \text{Attn}} J \frac{\partial S}{\partial Q} \right\|_F
$$

---

## 🔬 정리와 증명

### 정리 3.1 — Softmax Jacobian 의 명시적 형태

$$
J = \text{diag}(\sigma) - \sigma \sigma^\top
$$

**증명**: 정의 3.2 에서 i = j 일 때 $J_{ii} = \sigma_i (1 - \sigma_i)$, $i \neq j$ 일 때 $J_{ij} = -\sigma_i \sigma_j$. 행렬로 쓰면 위 식 $\square$.

### 정리 3.2 — Saturation 시 Jacobian 의 norm

만약 $\sigma$ 가 $\epsilon$-saturated (예: $\sigma_1 = 1-\epsilon$, 나머지 $\epsilon/(n-1)$) 라면:
$$
\|J\|_F^2 \leq 2\epsilon
$$

**증명**:

$$
\|J\|_F^2 = \sum_{i,j} J_{ij}^2 = \sum_i \sigma_i^2 (1 - \sigma_i)^2 + \sum_{i \neq j} \sigma_i^2 \sigma_j^2
$$

Saturated case 에서 $\sigma_1 \approx 1-\epsilon$, $\sigma_{j \neq 1} \approx \epsilon/(n-1)$:

- $i = 1$ 항: $(1-\epsilon)^2 \epsilon^2 \approx \epsilon^2$
- $i \neq 1$ 항: $(\epsilon/(n-1))^2 \cdot (1 - \epsilon/(n-1))^2 \approx (\epsilon/(n-1))^2$
- 합: $\epsilon^2 + (n-1) \cdot (\epsilon/(n-1))^2 = \epsilon^2 (1 + 1/(n-1)) \leq 2\epsilon^2$

Cross terms $(i \neq j)$ 도 비슷하게 $O(\epsilon^2)$.

따라서 $\|J\|_F = O(\epsilon)$ — saturated 시 Jacobian 이 거의 0 $\square$.

### 정리 3.3 — 분산이 큰 logit 의 saturation 확률

$z_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d., $n$ 개 entry 에서 max 의 기댓값:
$$
\mathbb{E}[\max_i z_i] \approx \sigma \sqrt{2 \log n}
$$

(Extreme value of Gaussian)

따라서 $\mathbb{E}[\sigma(z)_{\max}] \approx \frac{e^{\sigma \sqrt{2 \log n}}}{(n-1) + e^{\sigma \sqrt{2 \log n}}}$

**No-scale ($\sigma = \sqrt{d_k}$, $d_k = 64$, $n = 100$)**:
$$
\mathbb{E}[z_{\max}] \approx 8 \sqrt{2 \log 100} \approx 8 \times 3.03 \approx 24
$$
$$
\mathbb{E}[\sigma_{\max}] \approx \frac{e^{24}}{99 + e^{24}} \approx 1.0 - 99 e^{-24} \approx 1.0
$$

→ 거의 완전 saturated.

**Scaled ($\sigma = 1$, $n = 100$)**:
$$
\mathbb{E}[z_{\max}] \approx \sqrt{2 \log 100} \approx 3.03
$$
$$
\mathbb{E}[\sigma_{\max}] \approx \frac{e^{3}}{99 + e^{3}} \approx \frac{20}{119} \approx 0.17
$$

→ 정상 범위 $\square$.

### 정리 3.4 — Saturation 의 self-reinforcement (학습 시)

Saturated state 에서 시작하면 gradient 가 0 → weight 변화 0 → state 유지. **Lipschitz argument**: gradient norm $\|J\| < \delta$ 면 SGD step $\theta_{t+1} = \theta_t - \eta g$ 이 거의 무효.

**증명 sketch**: $\theta_t \to \theta^*$ 가 saturated stationary point 라면 $\nabla L(\theta^*) = 0$ 일 필요는 없지만, $\|\nabla L \cdot J\| \approx 0$ 이므로 SGD 가 escape 어려움 — saddle / plateau 의 일종 $\square$.

### 정리 3.5 — Layer-wise saturation 누적

Deep Transformer 에서 각 layer 가 saturation 을 강화한다고 가정 (실증 관찰). $L$-layer 후 effective gradient:
$$
\|\nabla\| \approx \prod_{l=1}^L \epsilon_l
$$

각 $\epsilon_l < 1$ 이면 지수적 감쇠. 이것이 RNN vanishing 과 같은 mechanism — Transformer 도 attention saturation + Pre/Post-LN 잘못 시 같은 함정 (Ch2-03).

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Softmax Jacobian 계산

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def softmax_jacobian(z):
    """Softmax Jacobian J = diag(σ) - σ σ^T"""
    s = F.softmax(z, dim=-1)
    return torch.diag_embed(s) - s.unsqueeze(-1) * s.unsqueeze(-2)

# 정상 영역
z_normal = torch.tensor([0.5, -0.2, 0.1])
J_normal = softmax_jacobian(z_normal)
print('Normal case:')
print(J_normal)
print(f'||J||_F = {J_normal.norm():.4f}')

# Saturated 영역
z_sat = torch.tensor([5.0, 0.0, -5.0])
J_sat = softmax_jacobian(z_sat)
print('\nSaturated case:')
print(J_sat)
print(f'||J||_F = {J_sat.norm():.4f}')   # 매우 작음
```

**예상 출력**:
```
Normal: ||J||_F ≈ 0.62
Saturated: ||J||_F ≈ 0.013
```

### 실험 2 — Logit Variance 별 Jacobian Norm

```python
n = 50
ns_norms, s_norms = [], []
for sigma in np.linspace(0.5, 10, 20):
    z = torch.randn(1000, n) * sigma
    s = F.softmax(z, dim=-1)
    J_norms = []
    for si in s:
        J = torch.diag(si) - si.unsqueeze(-1) @ si.unsqueeze(0)
        J_norms.append(J.norm().item())
    ns_norms.append(np.mean(J_norms))
    s_norms.append(sigma)

plt.figure(figsize=(8, 4))
plt.plot(s_norms, ns_norms, 'o-')
plt.xlabel('Logit std σ')
plt.ylabel('||Jacobian||_F')
plt.title('Softmax Jacobian collapses as logit variance grows')
plt.axvline(1.0, color='green', linestyle='--', label='scaled (σ=1)')
plt.axvline(8.0, color='red', linestyle='--', label='no-scale (d_k=64)')
plt.legend(); plt.show()
```

→ $\sigma$ 가 5 이상에서 Jacobian norm 이 빠르게 0 으로 ✓

### 실험 3 — 실제 Attention Layer Gradient

```python
def attention_grad_norm(d_k, T=50, scale=True, n_trials=20):
    grads = []
    for _ in range(n_trials):
        Q = torch.randn(T, d_k, requires_grad=True)
        K = torch.randn(T, d_k); V = torch.randn(T, d_k)
        scores = Q @ K.T
        if scale:
            scores = scores / np.sqrt(d_k)
        out = F.softmax(scores, dim=-1) @ V
        out.sum().backward()
        grads.append(Q.grad.norm().item())
    return np.mean(grads)

print(f'{"d_k":>6} | {"|grad| no-scale":>16} | {"|grad| scaled":>14}')
print('-' * 42)
for d_k in [8, 32, 128, 512, 2048]:
    g_ns = attention_grad_norm(d_k, scale=False)
    g_s  = attention_grad_norm(d_k, scale=True)
    print(f'{d_k:6d} | {g_ns:16.6f} | {g_s:14.6f}')
```

**예상**: scaled 는 $d_k$ 무관 일정, no-scale 은 $d_k$ 증가에 따라 0 으로 빠르게 수렴.

### 실험 4 — Saturation 의 self-reinforcement 시뮬레이션

```python
torch.manual_seed(42)
T, d_k = 30, 64
Q = torch.randn(T, d_k, requires_grad=True)
K = torch.randn(T, d_k, requires_grad=True)
V = torch.randn(T, d_k, requires_grad=True)
opt = torch.optim.SGD([Q, K, V], lr=0.1)

losses_ns, losses_s = [], []

# No-scale 학습
for step in range(100):
    scores = Q @ K.T   # NO scaling
    out = F.softmax(scores, dim=-1) @ V
    target = torch.randn_like(out)
    loss = ((out - target)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses_ns.append(loss.item())

# Reset, scaled 학습
torch.manual_seed(42)
Q = torch.randn(T, d_k, requires_grad=True)
K = torch.randn(T, d_k, requires_grad=True)
V = torch.randn(T, d_k, requires_grad=True)
opt = torch.optim.SGD([Q, K, V], lr=0.1)

for step in range(100):
    scores = Q @ K.T / np.sqrt(d_k)
    out = F.softmax(scores, dim=-1) @ V
    target = torch.randn_like(out)   # 같은 seed 라서 동일
    loss = ((out - target)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses_s.append(loss.item())

plt.plot(losses_ns, label='no scale', color='red')
plt.plot(losses_s,  label='scaled (√d)', color='blue')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
plt.title('Saturation prevents learning')
plt.show()
```

→ No-scale 은 loss 가 plateau (학습 안 됨), scaled 는 정상 감소 ✓

### 실험 5 — Layer-wise Saturation 누적

```python
def deep_attention(d_k, T, L, scale=True):
    """L 개 attention layer 통과 후 Jacobian norm"""
    h = torch.randn(T, d_k, requires_grad=True)
    h0 = h
    for _ in range(L):
        Q, K, V = h, h, h
        scores = Q @ K.T
        if scale: scores = scores / np.sqrt(d_k)
        h = F.softmax(scores, dim=-1) @ V
    h.sum().backward()
    return h0.grad.norm().item()

for L in [1, 4, 12]:
    for scale in [False, True]:
        g = deep_attention(64, 30, L, scale=scale)
        tag = 'scaled' if scale else 'no-scale'
        print(f'L={L:2d}, {tag:9s}: |grad to input| = {g:.4e}')
```

→ Deep + no-scale 시 gradient 가 지수적으로 0 — saturation 누적 ✓

---

## 🔗 실전 활용

### 1. Numerical Stability 의 log-sum-exp Trick

Softmax 직접 계산 시 $e^{z}$ overflow 가능. 표준 trick:
$$
\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

PyTorch 의 `F.softmax` 가 자동 처리. Custom 구현 시 주의.

### 2. Soft Saturation 의 의도적 사용

Distillation 에서 **temperature softmax** (Hinton 2015): teacher 의 soft label 에 $T > 1$ 적용해 의도적으로 saturation 완화 — student 가 dark knowledge 학습.

### 3. Sparsemax (Martins 2016)

Softmax 의 sparse 변형 — 일부 entry 를 정확히 0 으로. Saturation 없이 sparse attention 가능, 그러나 미분 가능성 trade-off.

### 4. Attention Dropout 의 회피

Softmax 후 dropout 은 saturation 을 randomly 완화 — regularization 효과. 그러나 dropout 자체가 attention 분포 왜곡, recent practice 는 attention dropout 안 쓰기도.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| i.i.d. logit 가정 | 실제로는 학습 후 dependency 생김, 분석은 초기 동작 설명 |
| 정규분포 logit | CLT 가정, 작은 $d_k$ 에서 부정확 |
| Saturation 이 불가역 | 학습 진행 시 일부 회복 가능 (residual + LN 도움) |
| Stationary analysis | 동적 학습 dynamics 별도 |
| $\sqrt{d_k}$ 만 충분 | LN, 초기화 등 다른 normalization 도 필수 (Ch2-03, Ch4-01) |

---

## 📌 핵심 정리

$$\boxed{J = \text{diag}(\sigma) - \sigma \sigma^\top, \quad \text{saturated} \Rightarrow \|J\|_F = O(\epsilon)}$$

| 영역 | $\sigma_{\max}$ | $\|J\|_F$ | Gradient | 학습 |
|------|---------|----------|----------|------|
| Uniform | $\approx 1/n$ | $O(1)$ | 정보 부족 | OK |
| Normal | $\approx 0.1{-}0.3$ | $O(1)$ | sensitive | ✓ |
| Saturated | $\approx 1 - \epsilon$ | $O(\epsilon)$ | 거의 0 | 실패 🚨 |

| 조건 | 결과 |
|------|------|
| $\sqrt{d_k}$ scaling 없음, $d_k = 64$ | $\sigma_{\max} \approx 1.0$ — saturated |
| $\sqrt{d_k}$ scaling, 임의 $d_k$ | $\sigma_{\max} \approx 0.1$ — 정상 |
| Layer 누적 | 지수적 gradient vanishing |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $z = (10, 0, -10)$ 의 softmax 값과 Jacobian norm 을 손으로 계산하라. 무엇이 saturated 됐는지?

<details>
<summary>해설</summary>

$$
\sigma(z) = \frac{1}{e^{10} + e^0 + e^{-10}} (e^{10}, 1, e^{-10}) \approx (1, 5e\text{-}5, 2e\text{-}9)
$$

$\sigma_1 \approx 1 - 5e\text{-}5$, 나머지는 $\approx 0$.

Jacobian $J = \text{diag}(\sigma) - \sigma \sigma^\top$:
- $J_{11} = \sigma_1(1-\sigma_1) \approx 5e\text{-}5$
- 다른 diagonal: $\approx 0$
- Off-diagonal: $-\sigma_1 \sigma_2 \approx -5e\text{-}5$

$\|J\|_F^2 \approx 2 \times (5e\text{-}5)^2 \approx 5e\text{-}9$, $\|J\|_F \approx 7e\text{-}5$.

→ saturated, gradient 거의 0 $\square$

</details>

**문제 2** (심화): Layer Normalization 이 attention layer 직전에 위치하면 (Pre-LN) saturation 이 어떻게 완화되는가? Post-LN 과 비교 분석하라.

<details>
<summary>해설</summary>

**Pre-LN**: $\text{LN}(x)$ 후 $W_Q, W_K$ 적용 → $Q, K$ 의 entry 가 unit variance.

**효과**:
1. **분산 안정** — Ch1-02 의 가정 ($Q, K$ unit variance) 보장
2. **Layer 누적 방지** — 매 layer 의 attention 입력이 동일한 normalization 영역
3. **$\sqrt{d_k}$ scaling 의 정확성** — i.i.d. 가정 보존

**Post-LN**: residual + attention 후 LN. 이때 attention 입력이 unit variance 보장 안 됨 → saturation 위험 더 큼.

**결론**: Pre-LN + $\sqrt{d_k}$ scaling 이 saturation 회피의 보완적 mechanism. Post-LN 은 warmup + careful 초기화 필수 (Ch2-03, Ch4-01).

이는 GPT/LLaMA 가 모두 Pre-LN 채택한 직접적 동기. $\square$

</details>

**문제 3** (논문 비평): Sparse attention (Longformer 등, Ch5-04) 은 attention matrix 의 일부만 계산한다. 이것이 saturation 의 위험을 어떻게 변화시키는가? 또한 Linear Attention (Ch5-02) 이 softmax 를 제거할 때 이 분석이 어떻게 무관해지고, 대신 어떤 새로운 stability 문제가 발생하는가?

<details>
<summary>해설</summary>

**Sparse Attention 의 saturation 영향**:

- **줄어든 $n$**: softmax 가 적은 entry 위에 적용 → max value 가 더 쉽게 dominant → saturation 위험 **증가** 가능
- **그러나 logit 분포는 동일** → $\sqrt{d_k}$ scaling 으로 mitigate
- **Local + global 구조** (Longformer): global token 이 모든 위치 attend → 그쪽이 saturate 되면 critical
- **실증**: Longformer 도 표준 $\sqrt{d_k}$ scaling 으로 충분

**Linear Attention 의 변화**:

$\text{softmax}$ 제거 시:
$$
\text{Attn}_{\text{lin}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
$$

- **Saturation 무관** — softmax 가 없으니 분석 불필요
- **새로운 문제 1: Numerical instability** — Denominator $\phi(Q) \phi(K)^\top \mathbf{1}$ 이 0 에 가까울 때 explode
- **새로운 문제 2: Sharpness 부족** — softmax 의 sharp peak 가 사라짐, 모든 token 균등 attend 경향, 표현력 ↓
- **새로운 문제 3: Feature map 선택의 민감성** — $\phi(x) = \text{ELU}(x) + 1$ 의 양수성 보장이 학습 안정성에 critical

**Performer (Ch5-03)** 는 random feature 로 softmax 를 다시 도입 → saturation 분석 다시 유효. 즉 trade-off:
- Linear: saturation 없음 vs 표현력 ↓
- Performer: 표현력 보존 vs saturation 위험 다시
- Standard + Flash: 표현력·saturation 모두 OK, 단 $O(T^2)$ FLOP

각 방법은 stability vs expressivity 의 다른 점에서 작동. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-sqrt-dk-scaling.md) | [📚 README](../README.md) | [다음 ▶](./04-attention-as-kernel.md)

</div>
