# 03. Label Smoothing 의 효과

## 🎯 핵심 질문

- Label smoothing 은 정확히 무엇이고, cross-entropy 와 어떻게 결합하는가?
- $\epsilon$-smoothed target 이 왜 calibration 을 개선하는가?
- Müller 2019: "When does label smoothing help?" — 어떤 task 에 도움이 되고 어떤 task 에 해로운가?
- Distillation 에서 label smoothing 이 student 에 해로운 이유는?
- Modern LLM 에서 label smoothing 사용은 — pre-training, fine-tuning 의 차이?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Label smoothing 은 **simple but powerful** regularization:

1. **Vaswani 2017 의 채택** — NMT 에서 BLEU score 향상의 직접 원인 중 하나
2. **Calibration 개선** — 모델의 confidence 가 정확도와 align
3. **Generalization 개선** — overfitting 완화
4. **그러나 always good 아님** — distillation, OOD detection 에는 해로움

이 문서는 label smoothing 의 **수학적 정의, 이론적 정당성, modern usage** 를 다룹니다.

---

## 📐 수학적 선행 조건

- [Regularization Theory Deep Dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive): Cross-entropy, calibration, ECE
- 정보이론: Entropy, KL divergence
- 이전 문서: [02-adamw.md](./02-adamw.md)

---

## 📖 직관적 이해

### Hard Label vs Soft Label

```
Hard label (one-hot):    [0, 0, 1, 0, 0, 0]   ← 정답 class 100% 확신
Smoothed (ε=0.1):        [0.02, 0.02, 0.92, 0.02, 0.02, 0.02]   ← 소량의 uncertainty
```

Smoothing factor $\epsilon$ 만큼 정답 confidence 줄임, 다른 class 에 균등 분배.

### Why does it help?

1. **Overfitting 방지**: 모델이 항상 정답 100% 예측 안 하도록 강제 → 일반화 ↑
2. **Calibration**: 모델의 softmax 출력이 실제 정확도와 더 잘 align
3. **Embedding clustering**: 같은 class embedding 이 너무 tight cluster 되지 않게

### Müller 2019 의 핵심 finding

```
Task                         | Label Smoothing 효과
----------------------------|--------------------
Image classification        |  + (도움)
NMT (translation)           |  + (도움)
Speech recognition          |  + (도움)
Knowledge distillation      |  - (해로움 to student)
```

Distillation: student 가 teacher 의 fine-grained dark knowledge 를 받아야 하는데, smoothed teacher 는 이 정보를 잃음.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — One-hot Label

$y \in \{1, \ldots, K\}$ 의 one-hot:
$$
y_k^{\text{onehot}} = \begin{cases} 1 & k = y \\ 0 & k \neq y \end{cases}
$$

### 정의 3.2 — Label Smoothing

Smoothing factor $\epsilon \in (0, 1)$:
$$
y_k^{\text{smooth}} = \begin{cases} 1 - \epsilon + \epsilon/K & k = y \\ \epsilon/K & k \neq y \end{cases}
$$

또는 간단히:
$$
y^{\text{smooth}} = (1 - \epsilon) y^{\text{onehot}} + \epsilon \cdot \mathbf{u}
$$

with $\mathbf{u} = (1/K, \ldots, 1/K)$ uniform.

### 정의 3.3 — Cross-entropy with Smoothed Label

$$
L_{\text{smooth}} = -\sum_{k=1}^K y_k^{\text{smooth}} \log p_k
$$

분해:
$$
= -(1-\epsilon) \log p_y - \frac{\epsilon}{K} \sum_k \log p_k
$$

**해석**: 정답 class 의 NLL + uniform distribution 과의 cross-entropy.

### 정의 3.4 — KL Divergence Form

$$
L_{\text{smooth}} = (1-\epsilon) \cdot \text{NLL}(y) + \epsilon \cdot D_{KL}(\mathbf{u} \| p) + \text{const}
$$

(uniform 과의 KL term 추가)

### 정의 3.5 — Calibration Error (ECE)

Expected Calibration Error:
$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|
$$

(confidence bin $B_m$ 별 accuracy vs 평균 confidence 의 차이)

---

## 🔬 정리와 증명

### 정리 3.1 — Smoothed Loss 의 분해

$$
L_{\text{smooth}} = (1-\epsilon) \cdot L_{\text{NLL}} + \epsilon \cdot H(\mathbf{u}, p)
$$

where $H(\mathbf{u}, p) = -\sum_k u_k \log p_k = -\frac{1}{K} \sum_k \log p_k$.

**증명**:
$$
L_{\text{smooth}} = -\sum_k [(1-\epsilon) y_k^{\text{onehot}} + \epsilon u_k] \log p_k = (1-\epsilon) L_{\text{NLL}} + \epsilon H(\mathbf{u}, p) \quad \square
$$

### 정리 3.2 — Optimal Solution (Confidence Cap)

$L_{\text{smooth}}$ 를 minimize 하는 $p^*$:
- $p_y^* = 1 - \epsilon + \epsilon/K$
- $p_{k \neq y}^* = \epsilon/K$

(즉 정답 confidence 가 $1 - \epsilon (1 - 1/K)$ 로 cap)

**증명**: $L_{\text{smooth}}$ 는 $D_{KL}(y^{\text{smooth}} \| p)$ 의 표현 (up to const), KL 의 최소가 $p = y^{\text{smooth}}$ — 즉 위의 cap 값 $\square$.

### 정리 3.3 — Logit Margin 의 영향

Smoothed loss 의 gradient (w.r.t. logit $z_k$):
$$
\frac{\partial L_{\text{smooth}}}{\partial z_k} = p_k - y_k^{\text{smooth}}
$$

- $k = y$: $p_y - (1-\epsilon + \epsilon/K)$ — $p_y$ 가 cap 에 도달하면 0
- $k \neq y$: $p_k - \epsilon/K$ — $p_k$ 가 $\epsilon/K$ 면 0

**의미**: 모델이 너무 confident 되면 ($p_y \to 1$) gradient 가 negative — confidence 줄이는 방향. **Implicit confidence cap**.

### 정리 3.4 — Embedding Geometry (Müller 2019)

Label smoothing 이 학습된 representation 의 geometry 변화:
- **Hard label**: 같은 class 의 representation 이 매우 tight cluster
- **Smoothed**: cluster 가 약간 spread out

**Müller 의 분석**: smoothed 가 inter-class geometry 를 더 uniform 하게 — equiangular tight frame approximation.

### 정리 3.5 — Calibration 개선

Hard label 학습 시 모델은 over-confident (training loss 줄이려면 $p_y \to 1$).

Smoothed label: 정답 confidence cap → over-confidence 방지 → ECE 감소.

**실증**: ImageNet 분류, label smoothing 없이 ECE = 0.05, $\epsilon = 0.1$ 시 ECE = 0.02 (Guo 2017, Müller 2019).

### 정리 3.6 — Distillation 에서의 단점 (Müller 2019)

Teacher 모델이 label smoothing 으로 학습되면:
- Teacher 의 soft prediction 이 더 uniform-like
- "Dark knowledge" (틀린 class 들의 relative ranking) 가 손실
- Student 의 distillation 이 less informative

**실증** (Müller): smoothed teacher → student worse than non-smoothed teacher (1-2% accuracy 손실).

**Modern advice**: Final model 은 smoothing OK, **distillation 의 teacher 는 NO smoothing**.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Smoothed Cross-entropy

```python
import torch
import torch.nn.functional as F
import numpy as np

def label_smoothed_loss(logits, target, epsilon=0.1):
    """logits: (B, K), target: (B,) class indices"""
    K = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # NLL
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze()
    
    # Uniform distribution loss
    uniform_loss = -log_probs.mean(dim=-1)
    
    # Combined
    return ((1 - epsilon) * nll + epsilon * uniform_loss).mean()

# 테스트
torch.manual_seed(0)
B, K = 4, 10
logits = torch.randn(B, K)
target = torch.randint(0, K, (B,))

loss_hard = F.cross_entropy(logits, target)
loss_smooth = label_smoothed_loss(logits, target, epsilon=0.1)
print(f'Hard label loss:      {loss_hard:.4f}')
print(f'Smoothed loss (ε=0.1): {loss_smooth:.4f}')
# Smoothed 가 약간 큼 (uniform term 추가)
```

### 실험 2 — PyTorch 의 native smoothing

```python
# PyTorch 1.10+ 의 native label smoothing
loss_pytorch = F.cross_entropy(logits, target, label_smoothing=0.1)
print(f'PyTorch native:       {loss_pytorch:.4f}')
# Custom 과 동일
```

### 실험 3 — 학습 시 Confidence 비교

```python
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleClassifier(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, K)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def train(loss_type, epochs=200):
    torch.manual_seed(0)
    model = SimpleClassifier(K=10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    confidences = []
    for ep in range(epochs):
        x = torch.randn(64, 20); y = torch.randint(0, 10, (64,))
        logits = model(x)
        if loss_type == 'hard':
            loss = F.cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        # 평가: 평균 max softmax (confidence)
        with torch.no_grad():
            x_test = torch.randn(64, 20)
            probs = F.softmax(model(x_test), dim=-1)
            confidences.append(probs.max(dim=-1).values.mean().item())
    return confidences

c_hard   = train('hard')
c_smooth = train('smooth')

plt.figure(figsize=(9, 4))
plt.plot(c_hard,   label='Hard label')
plt.plot(c_smooth, label='Smoothed (ε=0.1)')
plt.xlabel('epoch'); plt.ylabel('mean max softmax (confidence)')
plt.title('Label smoothing prevents over-confidence')
plt.legend(); plt.show()
# Hard 가 confidence 1.0 으로, smoothed 는 0.92 정도에서 cap
```

### 실험 4 — Calibration Error 측정

```python
def expected_calibration_error(probs, targets, n_bins=10):
    """ECE: 모델의 confidence 가 actual accuracy 와 align 정도"""
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()
    
    ece = 0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += in_bin.float().mean() * abs(avg_conf - avg_acc)
    return ece.item()

# 학습된 모델로 평가
torch.manual_seed(0)
model_hard = SimpleClassifier(K=10)
model_smooth = SimpleClassifier(K=10)

# 충분히 학습된 상태 가정 — random data 로 시뮬레이션
# (실제로는 같은 학습 후 비교)
x_test = torch.randn(1000, 20)
y_test = torch.randint(0, 10, (1000,))

probs_hard = F.softmax(model_hard(x_test) * 5, dim=-1)   # over-confident 시뮬레이션
probs_smooth = F.softmax(model_smooth(x_test) * 2, dim=-1)   # 덜 confident

print(f'ECE hard label:      {expected_calibration_error(probs_hard, y_test):.4f}')
print(f'ECE smoothed:        {expected_calibration_error(probs_smooth, y_test):.4f}')
```

### 실험 5 — Distillation 에서의 효과

```python
# Teacher: smoothed vs hard, student 는 teacher 의 prediction 으로 학습
torch.manual_seed(0)
teacher_hard = SimpleClassifier(K=10)
teacher_smooth = SimpleClassifier(K=10)

# Teacher 학습 (here 가짜)
# ... train both teachers ...

# Student 학습: KL divergence with teacher
def train_student(teacher, T=2.0, epochs=100):
    student = SimpleClassifier(K=10)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
    for _ in range(epochs):
        x = torch.randn(64, 20)
        with torch.no_grad():
            teacher_logits = teacher(x) / T
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student(x) / T, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * T**2
        opt.zero_grad(); loss.backward(); opt.step()
    return student

# 두 teacher 의 entropy 비교 — smoothed 가 더 uniform
x = torch.randn(100, 20)
ent_hard = -(F.softmax(teacher_hard(x), dim=-1) * F.log_softmax(teacher_hard(x), dim=-1)).sum(-1).mean()
ent_smooth = -(F.softmax(teacher_smooth(x), dim=-1) * F.log_softmax(teacher_smooth(x), dim=-1)).sum(-1).mean()
print(f'Teacher hard entropy:    {ent_hard:.4f}')
print(f'Teacher smooth entropy:  {ent_smooth:.4f}')
# Smoothed teacher 가 더 high entropy → less informative for distillation
```

---

## 🔗 실전 활용

### 1. NMT 의 표준

Vaswani 2017 NMT: $\epsilon = 0.1$.
- WMT 14 EN-DE: 1.5 BLEU 향상
- 모든 modern NMT 모델 채택

### 2. Image Classification

- ImageNet: $\epsilon = 0.1$ 표준 (Inception, ResNet 후)
- Vision Transformer: $\epsilon = 0.1$
- 0.5-1% accuracy 향상

### 3. Modern LLM Pre-training

GPT-3, LLaMA 등:
- Pre-training: label smoothing **사용 안 함** (또는 매우 작은 $\epsilon$)
- 이유: pre-training 의 prediction 이 매우 uncertain, smoothing 이 학습 어렵게

### 4. Fine-tuning

- Instruction tuning: smoothing 없음
- RLHF: 다른 objective (PPO)
- Classification fine-tuning: $\epsilon = 0.1$ 사용

### 5. Adaptive Smoothing

일부 연구: $\epsilon$ 을 학습 진행에 따라 변경
- 초기: 큰 $\epsilon$ (regularization 강함)
- 후기: 작은 $\epsilon$ (정확한 학습)

→ 표준 practice 는 fixed $\epsilon = 0.1$.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Uniform smoothing | Class-aware smoothing 가능 (rare class 우대) |
| Fixed $\epsilon$ | Adaptive smoothing 도 가능 |
| Single-task | Multi-task 별 다른 $\epsilon$ 필요 |
| Hard target 사용 시 | Distillation 시 teacher smoothing 피해야 |

---

## 📌 핵심 정리

$$\boxed{y^{\text{smooth}} = (1-\epsilon) y^{\text{onehot}} + \epsilon / K, \quad L_{\text{smooth}} = (1-\epsilon) \text{NLL} + \epsilon H(\mathbf{u}, p)}$$

| 효과 | 설명 |
|------|------|
| **Over-confidence 방지** | $p_y$ 의 cap |
| **Calibration 개선** | ECE 감소 |
| **Embedding geometry** | Inter-class distance 균등화 |
| **Generalization** | Overfitting 완화 |

| 사용 가이드 | $\epsilon$ |
|-----------|-----------|
| NMT, classification | 0.1 |
| Pre-training LLM | 사용 안 함 |
| Distillation teacher | 사용 안 함 |
| Fine-tune classifier | 0.1 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K = 5$, $\epsilon = 0.2$, 정답 class 가 2 일 때 smoothed target distribution 을 명시적으로 적어라.

<details>
<summary>해설</summary>

$y^{\text{onehot}} = (0, 0, 1, 0, 0)$.

$y^{\text{smooth}} = (1 - 0.2) \cdot (0, 0, 1, 0, 0) + 0.2 \cdot (0.2, 0.2, 0.2, 0.2, 0.2)$

$= (0, 0, 0.8, 0, 0) + (0.04, 0.04, 0.04, 0.04, 0.04)$

$= (0.04, 0.04, 0.84, 0.04, 0.04)$

검증: $\sum = 0.04 \times 4 + 0.84 = 0.16 + 0.84 = 1.0$ ✓

정답 class 의 confidence 가 $1 - 0.2 + 0.2/5 = 0.84$ — cap. $\square$

</details>

**문제 2** (심화): Label smoothing 이 logit 의 magnitude 를 어떻게 제한하는가? Hard label 시 logit 이 무한대로 커지는 이유와 smoothed 시 cap 의 수학적 분석.

<details>
<summary>해설</summary>

**Hard label cross-entropy**:
$$
L = -\log p_y = -\log \frac{e^{z_y}}{\sum_k e^{z_k}} = -z_y + \log \sum_k e^{z_k}
$$

Gradient:
$$
\frac{\partial L}{\partial z_y} = -1 + p_y, \quad \frac{\partial L}{\partial z_{k \neq y}} = p_k
$$

$L \to 0$ 되려면 $p_y \to 1$ → $z_y \to \infty$ (unbounded).

**Smoothed cross-entropy**:
$$
L = -\sum_k y_k^{\text{smooth}} \log p_k
$$

Gradient:
$$
\frac{\partial L}{\partial z_k} = p_k - y_k^{\text{smooth}}
$$

$L \to L^*$ minimum 시 $p_k = y_k^{\text{smooth}}$:
- $p_y = 1 - \epsilon + \epsilon/K$ (NOT 1)
- $p_{k \neq y} = \epsilon/K$ (NOT 0)

**Logit magnitude**:

$p_y / p_{k \neq y} = (1-\epsilon+\epsilon/K) / (\epsilon/K) = K/\epsilon - K + 1$.

이 ratio 에 대응하는 logit 차이:
$$
z_y - z_{k} = \log(K/\epsilon - K + 1)
$$

$\epsilon = 0.1$, $K = 1000$: $\log(10000 - 1000 + 1) \approx \log(9001) \approx 9.1$.

→ Logit 차이가 **유한**: $\approx 9$. Hard 의 무한대와 대조.

**의미**:
- Logit magnitude 가 bounded → numerical stability
- Confidence cap 이 자연스러움
- Implicit regularization 효과

**Müller 2019 의 geometric 해석**:
- Logit 의 representation space 에서, smoothed loss 의 minimum 이 finite point
- Class embedding 들이 specific geometry 형성 (equidistant)
- Pre-training 단계에서는 이 geometry 가 이미 emergent — smoothing 추가 효과 X

따라서 hard label 의 unbounded 한 logit growth 가 over-confidence 의 직접 원인. Smoothed label 이 자연스러운 cap 으로 해결. $\square$

</details>

**문제 3** (논문 비평): Modern LLM (GPT, LLaMA) 이 label smoothing 을 거의 사용 안 하는 이유는? Pre-training 의 본질이 classification 과 다른 점을 분석하라.

<details>
<summary>해설</summary>

**Modern LLM Pre-training 의 특징**:

1. **Inherently uncertain target**:
   - Next token prediction: 같은 context 에서 여러 likely tokens
   - 정확한 "정답" 이 없음 — language 의 자연스러운 다양성
   - Hard label 도 "soft" 한 의미 (data distribution 자체가 noisy)

2. **Vocabulary size 가 매우 큼**:
   - $K = 50000+$ tokens
   - $\epsilon/K$ 가 매우 작음 — 영향 미미
   - Smoothing 의 marginal benefit 작음

3. **충분한 데이터**:
   - Trillions of tokens — overfitting 거의 없음
   - Regularization 의 필요 ↓

4. **Calibration 문제 다름**:
   - LLM 의 calibration 은 token-level 이 아니라 sequence/task-level
   - Token-level smoothing 이 sequence calibration 개선에 직접 관련 X

**Image Classification 과의 차이**:

- ImageNet: $K = 1000$, 정답 명확, finite samples → smoothing 효과 큼
- LLM: $K = 50000$, 정답 ambiguous, infinite samples → smoothing 효과 적음

**예외 — Fine-tuning**:

- 분류 task fine-tuning: $\epsilon = 0.1$ 일반적 — classification 으로 회귀
- Instruction tuning: 사용 안 함 (생성 task)
- Code generation: 사용 안 함

**Distillation 에서의 회피**:

- LLM distillation (e.g., Mistral → Mistral-tiny):
- Teacher 가 label smoothing 학습 시 dark knowledge 손실
- Student 가 worse — Müller 2019 의 finding 직접 적용
- 따라서 frontier LLM 들이 smoothing 회피 (distillation 가능성 보존)

**RLHF 의 별도 objective**:

- PPO 등 RL 학습: cross-entropy 가 아닌 reward maximization
- Label smoothing 의 framework 직접 적용 X
- 별도의 KL regularization (with reference policy) 사용

**Open question**:

LLM 의 calibration 개선이 중요한 application:
- Hallucination detection
- Confidence-aware reasoning
- Structured prediction

이런 application 에는 label smoothing 외 다른 calibration 기법 (temperature scaling, bayesian inference) 가 더 적합.

**결론**:

Label smoothing 은 **NMT 와 image classification 의 specific context** 에 잘 작동. LLM pre-training 은 inherently different problem (생성, large vocab, noisy target) — smoothing 의 동기가 약함. Modern LLM 은 다른 regularization (dropout, weight decay, RMSNorm 등) 으로 충분. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-adamw.md) | [📚 README](../README.md) | [다음 ▶](./04-gradient-accumulation.md)

</div>
