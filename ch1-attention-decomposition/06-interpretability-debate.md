# 06. Attention 의 해석 가능성 논쟁

## 🎯 핵심 질문

- Attention weight 가 모델의 "결정 근거" 라고 부를 수 있는가? — Jain & Wallace 2019 의 비판은 무엇인가?
- 같은 출력을 만드는 다른 attention 분포가 가능한가 (counterfactual)?
- Wiegreffe & Pinter 2019 의 반박 — context 와 task 에 따라 attention 이 explanation 이 될 수 있는 조건은?
- Probing classifier 와 causal mediation analysis 가 attention 의 역할을 어떻게 더 정확히 측정하는가?
- Mechanistic interpretability (Anthropic 2021) 가 attention 을 "circuit" 으로 보는 framework 의 의미는?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Attention weight 가 **explanation 인가 아닌가** 의 논쟁은 Transformer 시대의 가장 중요한 ML 논쟁 중 하나입니다:

1. **모델 신뢰성** — 의료·법률·금융 등에서 "왜 이 결정?" 이 필수
2. **Debugging 도구** — 모델 실패 시 attention map 으로 hypothesis
3. **Mechanistic understanding** — Transformer 가 "무엇을" 학습하는가의 과학적 질문
4. **AI safety** — alignment, deception detection 등의 출발점

이 문서는 attention 해석 가능성에 대한 **양면 (찬성/반대) 의 주장** 을 분석하고, mechanistic interpretability 의 현대적 framework 를 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-scaled-dot-product.md](./01-scaled-dot-product.md), [05-multi-head.md](./05-multi-head.md)
- Causal inference: counterfactual, intervention
- 통계학: correlation vs causation, faithfulness

---

## 📖 직관적 이해

### Attention map 이 "본 것" 을 보여주는가?

```
Input: "The cat sat on the mat"
Question: "Where did the cat sit?"

Attention map (decoder query 'cat' on encoder):
The   cat   sat   on   the   mat
0.05  0.10  0.15  0.20  0.10  0.40   ← 'mat' 에 가장 높은 weight
```

직관적으로 "모델이 'mat' 을 봤다" 고 해석. 그러나:
- **다른 attention 으로도 같은 답** 가능?
- **Attention 이외의 path** (residual, FFN) 가 결정한다면?

### Counter-example: Same output, different attention

Jain & Wallace 2019 의 실험:
- 학습된 attention 분포 $A$ 가 있을 때, 다른 분포 $A'$ 로 force 해도 같은 prediction 가능
- → Attention 이 **유일한** explanation 아님

### Mechanistic view

```
Input ─→ Embedding ─→ [Attn + FFN] × L ─→ Output
                         ↑
                  여기서 어떤 "회로" 가 결정?
```

Attention 은 **information flow 의 한 path** 일 뿐, residual 과 FFN 도 동등한 path. 전체 **circuit** 분석이 필요.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Attention as Explanation Hypothesis

"Attention weight $\alpha_{ij}$ 가 $j$-th token 이 $i$-th 출력에 미친 **인과적 기여도** 를 측정한다"

### 정의 6.2 — Faithfulness

Explanation method $E$ 의 **faithful**:
- Removing high-$E$ feature → 큰 prediction 변화
- Removing low-$E$ feature → 작은 prediction 변화

### 정의 6.3 — Counterfactual Attention

원래 attention $A$ 의 row $A_{i,:}$ 를 다른 분포 $A'_{i,:}$ 로 대체:
$$
\text{out}'_i = \sum_j A'_{ij} v_j
$$

같은 prediction $f(\text{out}') = f(\text{out})$ 시 $A$ 가 unique explanation 아님.

### 정의 6.4 — Probing Classifier

Layer 별 hidden state $h^{(l)}$ 에 작은 classifier 를 붙여 어떤 정보가 인코딩됐는지 측정. Attention 과 별개의 분석 도구.

### 정의 6.5 — Mechanistic Circuit

Anthropic 2021: model 의 일부 weight 와 activation 을 묶어 "회로" 로 해석. 예: induction head — copy task 를 수행하는 specific attention head 조합.

---

## 🔬 정리와 증명

### 정리 6.1 — Attention 은 Generally Not Faithful

**Jain & Wallace 2019 의 주장**: 학습된 모델에서 다른 attention $A' \neq A$ 가 같은 prediction 을 만들 수 있다.

**증명 sketch**: Optimization 문제:
$$
A' = \arg\min_{A'} \|f(A' V) - f(A V)\|^2 + \lambda \cdot D_{\text{KL}}(A' \| \text{uniform})
$$

(같은 출력 + uniform 에 가까운 attention)

대부분의 모델/task 에서 nontrivial 한 $A' \neq A$ 발견 가능. $\square$

### 정리 6.2 — Wiegreffe & Pinter 2019 의 반박

**제한된 조건**: $A'$ 가 "natural" 분포 (학습 가능, smooth 등) 인지 검증 필요. Adversarial $A'$ 는 jain & wallace 가 만들 수 있지만, **자연스럽게 학습되지 않음**.

**Test**: $A'$ 로 학습 시 trained $A$ 와 같은 성능? 아니면 lower? → Lower 라면 $A$ 가 informative.

**일부 task 에서 attention 이 explanation**:
- Sentiment analysis: attention 이 sentiment-bearing word 에 집중 ✓
- 일반 classification: 덜 명확

### 정리 6.3 — Probing 의 정보 분리

Layer $l$ 의 representation 이 task $T$ 정보 인코딩 ↔ 작은 classifier 가 $T$ 예측 가능. 그러나 **encoded ≠ used** — 정보가 있어도 모델이 사용 안 할 수 있음.

**대응**: amnesic probing (Elazar 2021) — 정보 제거 후 모델 성능 측정으로 사용 여부 검증.

### 정리 6.4 — Causal Mediation Analysis

Pearl 의 causal framework 적용:
- **Intervention**: $A_{ij}$ 를 강제 변경
- **Mediation**: 변경이 출력에 미친 직접 효과 vs 간접 효과
- Vig 2020, Geiger 2021: attention 의 causal contribution 정량화

### 정리 6.5 — Induction Head (Anthropic 2021)

**Mechanistic finding**: 두 head 의 조합이 in-context learning 의 simple pattern (copy "A B ... A → B") 을 구현.

**구조**:
- **Previous-token head** (layer 1): 각 token 이 이전 token 정보 복사
- **Induction head** (layer 2): 같은 token 을 찾아 그 다음 token 을 출력

이는 attention 이 **specific computation 을 수행** 한다는 mechanistic 증거. Explanation 이 transparent 한 케이스.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Counterfactual Attention 검증

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 작은 학습된 sentiment model 가정 (random weight)
torch.manual_seed(0)
d, T = 16, 10
W = nn.Linear(d, d, bias=False)
classifier = nn.Linear(d, 2)

# 입력
x = torch.randn(T, d)
Q, K, V = W(x), W(x), W(x)

# 원래 attention
scores = Q @ K.T / np.sqrt(d)
A = F.softmax(scores, dim=-1)
out = A @ V
pred = classifier(out.mean(0))   # mean-pool + classify

# Counterfactual: random attention
A_cf = F.softmax(torch.randn(T, T), dim=-1)
out_cf = A_cf @ V
pred_cf = classifier(out_cf.mean(0))

print(f'Original prediction: {pred.argmax().item()}, scores: {pred.softmax(0)}')
print(f'Counterfactual:      {pred_cf.argmax().item()}, scores: {pred_cf.softmax(0)}')
# 다를 수 있고 같을 수도 — 모델 학습 후에는 더 robust
```

### 실험 2 — Attention Map 시각화 (의도적 pattern)

```python
import matplotlib.pyplot as plt

# 의도적으로 'cat' (idx 1) 에 attend 하는 query, 'mat' (idx 5) 에도 attend
T = 6
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
attn_manual = torch.zeros(T, T)
for i in range(T):
    attn_manual[i, 1] = 0.3   # everyone attends to 'cat'
    attn_manual[i, 5] = 0.4   # also to 'mat'
    attn_manual[i, i] = 0.3   # self
attn_manual = F.softmax(attn_manual, dim=-1)

plt.figure(figsize=(6, 5))
plt.imshow(attn_manual.numpy(), cmap='Blues')
plt.xticks(range(T), tokens, rotation=45); plt.yticks(range(T), tokens)
plt.xlabel('Key (attended to)'); plt.ylabel('Query (attending from)')
plt.colorbar(); plt.title('Manually-set attention pattern')
plt.tight_layout(); plt.show()
```

### 실험 3 — Adversarial Attention (Jain & Wallace 스타일)

```python
def find_adversarial_attention(V, original_out, max_iter=500, lr=0.01):
    """원래 출력을 만드는 다른 attention 찾기"""
    A_adv = torch.randn(T, T, requires_grad=True)
    opt = torch.optim.Adam([A_adv], lr=lr)
    
    for step in range(max_iter):
        A_norm = F.softmax(A_adv, dim=-1)
        out_adv = A_norm @ V
        # 같은 출력 강제
        loss = ((out_adv - original_out) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    
    return F.softmax(A_adv, dim=-1).detach()

V = torch.randn(T, d)
original_A = F.softmax(torch.randn(T, T), dim=-1)
original_out = original_A @ V

A_adv = find_adversarial_attention(V, original_out)
out_adv = A_adv @ V

# 검증
diff_output = (out_adv - original_out).abs().max()
diff_attention = (A_adv - original_A).abs().max()
print(f'Output difference: {diff_output:.6f} (≈ 0 → 같은 출력)')
print(f'Attention diff: {diff_attention:.4f} (큼 → 다른 attention)')
```

### 실험 4 — Probing Classifier 시뮬레이션

```python
# 가정: layer-wise hidden state h_l 에서 task T 정보 인코딩
# 작은 classifier 로 task 예측

class ProbeClassifier(nn.Module):
    def __init__(self, d, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(d, num_classes)
    def forward(self, h):
        return self.classifier(h.mean(0))   # pool + classify

# 가짜 학습 (random data)
probe = ProbeClassifier(d=16)
opt = torch.optim.Adam(probe.parameters(), lr=0.01)

for _ in range(100):
    h = torch.randn(T, 16)
    label = torch.tensor([1])
    loss = F.cross_entropy(probe(h).unsqueeze(0), label)
    opt.zero_grad(); loss.backward(); opt.step()

# Probe 정확도 측정 (random data 라 chance level)
acc = 0
for _ in range(100):
    h = torch.randn(T, 16)
    label = torch.tensor([1])
    pred = probe(h).argmax()
    acc += (pred == label).item()
print(f'Probe accuracy: {acc/100:.2f}')
```

### 실험 5 — Induction Head 시뮬레이션

```python
# 간단한 copy task: "A B X A" → "B"
# 두 head 조합으로 작동

def simulate_induction_head(seq, h_prev, h_induct):
    """
    h_prev: previous-token attention pattern
    h_induct: induction (find same token, attend to its successor)
    """
    T = len(seq)
    # Previous-token: token i가 token i-1 정보 가져옴
    prev_pattern = torch.eye(T, T)
    for i in range(1, T):
        prev_pattern[i, i-1] = 1.0
        prev_pattern[i, i] = 0.0
    prev_pattern = prev_pattern / prev_pattern.sum(dim=-1, keepdim=True)
    
    # Induction: 마지막 token 과 같은 token 찾고 그 다음으로 attend
    last_tok = seq[-1]
    induction_target = []
    for i, tok in enumerate(seq[:-1]):
        if tok == last_tok and i + 1 < T - 1:
            induction_target.append(i + 1)
    
    return induction_target

seq = ['A', 'B', 'X', 'A']
target = simulate_induction_head(seq, None, None)
print(f'Sequence: {seq}, last token: {seq[-1]}')
print(f'Induction head should attend to position(s): {target}')
print(f'Expected next token: {seq[target[0]]}')   # 'B'
```

---

## 🔗 실전 활용

### 1. Attention Visualization Tools

- **BertViz** (Vig 2019): BERT/GPT attention head 별 시각화
- **Captum** (PyTorch): attention 외에도 saliency, integrated gradients
- **Anthropic Interpretability**: circuit-level analysis

### 2. Diagnostic 도구로서의 Attention

Explanation 으로는 약하지만 **debugging** 에 유용:
- 모델이 padding 에 attend → tokenization 문제
- 첫 token 에 모두 attend → "no attention" sink (StreamingLLM)
- Loop 에 stuck → repetition penalty 필요

### 3. Mechanistic Interpretability Research

- **Induction heads**: in-context learning 의 origin
- **Path patching** (Conmy 2023): circuit identification 자동화
- **Sparse autoencoders** (Anthropic 2024): superposition decomposition
- AI safety 의 핵심 도구로 발전 중

### 4. Faithful Explanation 의 Alternatives

- **Integrated Gradients**: input 의 contribution 정량화
- **SHAP**: game-theoretic attribution
- **Causal mediation**: 직접 인과 분석

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Attention = explanation | Generally not — counterfactual 가능 |
| Single head 분석 | Multi-head + residual 모두 고려 필요 |
| Attention 이 충분한 정보 | FFN, residual 도 동등 path |
| Static analysis | Token-by-token 동적 동작 별도 |
| 인과적 해석 | Correlation 만 측정, intervention 필요 |

---

## 📌 핵심 정리

| 입장 | 주장 | 증거 |
|------|------|------|
| **Attention ≠ Explanation** (Jain & Wallace 2019) | Counterfactual attention 가능 | Adversarial $A'$ 로 같은 출력 |
| **조건부 Explanation** (Wiegreffe & Pinter 2019) | Natural training 에서 informative | $A'$ 학습 시 lower 성능 |
| **Mechanistic** (Anthropic 2021) | Specific circuit 은 transparent | Induction heads 등 |

| 도구 | 측정 |
|------|------|
| Attention map | Correlational, not faithful in general |
| Probing | 정보 인코딩 여부 (사용 여부 ≠) |
| Causal mediation | 인과적 contribution |
| Circuit analysis | Mechanistic understanding |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Attention map 이 "model 의 결정 근거" 라고 단순 해석할 때 발생할 수 있는 오류 3가지를 들어라.

<details>
<summary>해설</summary>

1. **Counterfactual 가능성**: 같은 output 을 만드는 다른 attention 분포 존재 → unique explanation 아님
2. **다른 path 무시**: Residual connection 과 FFN 이 정보 흐름의 동등 path. Attention 만으로 결정 안 됨
3. **Layer-wise 누적**: 한 layer 의 attention 이 다음 layer 의 입력만 결정, deep 에서는 layer 별 attention 이 직접적 인과 아님

추가:
- **Heads 간 redundancy**: 한 head 의 attention 이 정보를 인코딩해도 다른 head 가 backup
- **Encoded ≠ Used**: 정보가 attention 에 인코딩돼도 downstream 이 안 쓸 수 있음

따라서 attention 은 **diagnostic** 으로 유용하지만 **explanation** 으로 단순 사용 위험. $\square$

</details>

**문제 2** (심화): Wiegreffe & Pinter 2019 의 논점 — "natural training 에서 학습된 $A$ 가 informative" — 가 attention 의 explanation 가치를 어떻게 부분 회복하는가? Empirical test 의 설계는?

<details>
<summary>해설</summary>

**핵심 주장**: Adversarial 한 $A'$ 는 만들 수 있지만, 같은 모델 architecture 로 학습 시 그런 $A'$ 가 자연스럽게 발생 안 함. 따라서 학습된 $A$ 는 모델이 **선택한** explanation.

**Empirical Test 설계**:

1. **Frozen attention experiment**: $A$ 를 random fixed → 학습 → 성능 측정
   - 만약 random fixed $A$ 도 같은 성능 → attention 이 informative 아님
   - Lower 성능 → attention 학습이 의미 있음

2. **Adversarial 학습 가능성**: $A'$ 가 같은 출력을 만들지만 학습으로 도달 가능한가?
   - 적절한 hyperparameter 로 학습 시 도달하는가?
   - 도달 못 한다면 $A$ 가 모델이 선택한 (preferred) explanation

3. **Attention 분포의 entropy**: low-entropy ($A$ 가 sharp) vs high-entropy
   - Low: 명확한 attention pattern, more interpretable
   - High: uniform-like, less informative

**결과**:
- Sentiment analysis: 학습된 $A$ 가 sentiment-bearing word 에 집중, frozen 보다 우수 → informative
- Long-document classification: 차이 적음 → attention 이 less critical

**Wiegreffe 의 conclusion**: "Attention 이 explanation 인지" 는 task-dependent 한 empirical question. 모든 case 에 일반화 안 됨. $\square$

</details>

**문제 3** (논문 비평): Anthropic 의 mechanistic interpretability (induction heads, sparse autoencoders) 는 attention 의 해석 가능성 논쟁을 어떻게 변화시키는가? "Attention 은 explanation 인가" 에서 "Attention 이 어떤 computation 을 수행하는가" 로의 paradigm shift 의 의미는?

<details>
<summary>해설</summary>

**Paradigm Shift**:

**Old**: "Attention map 이 high 인 token 이 모델 결정에 기여한 token" (단순 input-output correlation)

**New (Mechanistic)**: "Attention 의 어떤 head + circuit 이 어떤 specific computation 을 수행하는가" (algorithmic decomposition)

**Anthropic 의 핵심 발견**:

1. **Induction heads (2022)**: Two-layer attention 조합이 in-context learning 의 simple pattern (copy) 구현. Specific weight 패턴이 specific algorithm 을 implement.

2. **Path patching (2023)**: Specific neuron/head 의 contribution 을 intervention 으로 측정. Causal 분석.

3. **Sparse autoencoders (2024)**: Hidden activation 의 superposition 을 분리, monosemantic feature 발견.

**의미의 변화**:

- **Faithful explanation**: Specific circuit 은 transparent — "이 computation 이 발생함" 을 mechanistic 으로 증명
- **Generalization 너머**: ICL, CoT 같은 emergent capability 의 internal mechanism 이해
- **AI safety**: Deception, manipulation 같은 적대적 행동 detect 가능 (이론)

**한계 / 진행 중**:

- **Scale**: Frontier model 은 너무 복잡, 전체 circuit 파악 불가
- **Generality**: 발견된 circuit 은 specific task — 일반 inference 의 mechanism 은?
- **Polysemanticity**: Single neuron 이 여러 의미 인코딩, 분리 어려움

**큰 그림**:

"Attention 이 explanation 인가" 의 binary 질문에서 "Transformer 의 internal computation 을 어디까지 mechanistic 하게 이해할 수 있는가" 의 질적 질문으로. 이는 **AI 의 과학화** — 단순 black box 에서 분석 가능한 system 으로의 전환. AI safety, capability evaluation, robust deployment 의 토대. $\square$

</details>

---

<div align="center">

[◀ 이전](./05-multi-head.md) | [📚 README](../README.md) | [다음 ▶](../ch2-transformer-architecture/01-transformer-block.md)

</div>
