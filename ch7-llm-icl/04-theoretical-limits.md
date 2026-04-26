# 04. Transformer 의 이론적 한계

## 🎯 핵심 질문

- Pérez 2019: Infinite-precision attention 은 Turing-complete 하지만 finite-precision 의 한계는?
- Hahn 2020: Bounded depth Transformer 가 counting, parity 같은 simple task 에 weak 한 이유?
- Compositional generalization 의 어려움 — SCAN, COGS 의 의미?
- Mamba (Gu 2023), RWKV (Peng 2023) 가 Transformer 의 어떤 fundamental limit 을 우회하는가?
- Universal Transformer 와 Looping Transformer 의 idea?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Transformer 의 한계 이해는 **next-generation architecture 의 토대**:

1. **이론적 ceiling** — scale 만으로 풀 수 없는 문제
2. **Architecture innovation 의 motivation** — Mamba, Hyena, retentive networks
3. **AI 의 future** — 단순 LLM 이 AGI 의 path 가 아닐 가능성
4. **Hybrid systems 의 정당성** — Transformer + symbolic + tools

이 문서는 Transformer 의 **이론적 한계와 alternative architectures** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 1, 2, 7 (이전 문서들)
- (선택) Computability theory: Turing machines, complexity classes
- 이전 문서: ICL (Ch7-02), CoT (Ch7-03)

---

## 📖 직관적 이해

### Theoretical Power vs Practical Limit

```
Pérez 2019: "Transformer can simulate Turing machine"
  → 이론적으로 universal computation
  → 그러나 infinite precision + 무한 layer 가정

Hahn 2020: "Bounded-depth Transformer cannot count or check parity"
  → 실제 모델의 fundamental limit
  → Specific simple tasks 에 weak
```

### Counting Limit (Hahn 2020)

```
Task: count occurrences of 'a' in "aaab" → 3
       
Bounded-depth Transformer:
  - 짧은 sequence: OK
  - Long sequence: degrades — exponential precision loss
  - Theoretical: depth $O(\log T)$ minimum
```

### Parity Limit

```
Task: is the number of 1s even or odd in "1011001"?
       
Hahn 2020: Depth-bounded Transformer 가 parity 못 풀음 (limit)
  - Each layer 만으로는 cumulative XOR 불가
  - $O(\log T)$ depth 필요 — unbounded T 에 어려움
```

### Compositional Limit

```
SCAN benchmark:
  Train: "jump twice", "walk twice", "run thrice"
  Test:  "jump thrice"  ← novel composition

Transformer 가 surprisingly poor — memorize 보다 generalize 어려움
```

### Why Mamba/RWKV Help

```
Mamba: state space model, $O(T)$ inference
  - Linear scan algorithm
  - Long context efficient
  - Some tasks (selective copying) 더 잘함

RWKV: linear attention with time decay
  - RNN-form generation
  - $O(d^2)$ per step
  - Counting 같은 specific task 에 better?
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Computational Power Hierarchy

- **Regular language**: finite automaton (FA)
- **Context-free**: pushdown automaton (PDA)
- **Context-sensitive**: linear bounded automaton (LBA)
- **Recursively enumerable**: Turing machine (TM)

### 정의 4.2 — Pérez 2019: Transformer Turing-completeness

**Theorem**: Transformer with arbitrary depth + infinite precision can simulate any Turing machine.

**Caveat**: requires
- Infinite precision (unbounded number of significant digits)
- Recurrent looping (or unbounded depth)
- Not finite Transformer

→ **Theoretical only**.

### 정의 4.3 — Hahn 2020: Depth-Bounded Limit

**Theorem (informal)**: Fixed-depth Transformer 가 다음 task 못 풀음:
- **Parity**: $\bigoplus_{i=1}^T x_i$ for $T$ unbounded
- **Counting**: count exact occurrences (not just majority)
- **Boolean expression evaluation**: nested parentheses

**Proof sketch**: communication complexity argument — bounded depth attention 의 information aggregation 의 bound.

### 정의 4.4 — Compositional Generalization

**SCAN** (Lake 2018): compositional sequence-to-sequence
- Primitives: "jump", "walk"
- Modifiers: "twice", "thrice"
- Composition: "jump twice" → "JUMP JUMP"

**COGS** (Kim 2020): grammar-based compositional generalization

Modern Transformer: surprisingly poor compositional generalization (without specific training).

### 정의 4.5 — Universal Transformer (Dehghani 2018)

Recurrent Transformer:
$$
h^{(l+1)} = \text{Block}(h^{(l)}), \quad l = 1, \ldots, ?
$$

with **dynamic depth** — model 이 stop 할 때까지 loop.

→ Transformer + RNN-like recurrence.

### 정의 4.6 — Mamba (Gu & Dao 2023)

State Space Model (SSM):
$$
h_{t+1} = A h_t + B x_{t+1}, \quad y_t = C h_t
$$

with **selective** $A, B, C$ (input-dependent — Mamba's key innovation).

- $O(T)$ training (parallel scan)
- $O(d^2)$ inference per step
- Long context 효율

---

## 🔬 정리와 증명

### 정리 4.1 — Pérez 2019: Theoretical Universal

Sufficient depth Transformer 가 Turing machine simulate.

**Construction**:
- Tape: encoded in token sequence
- State: 일부 token 의 representation
- Head movement: attention 으로 cell selection
- Transition: FFN 으로 next state

**한계**:
- Real model 의 depth bounded
- Real precision finite (FP16/32/64)
- Real context length bounded

→ **Theoretical capability ≠ practical capability**.

### 정리 4.2 — Hahn 2020: Bounded-Depth Limit

$L$-layer Transformer with embedding dim $d$:
- Cannot solve parity for $T > 2^{O(L)}$
- Cannot solve counting for $T > c \cdot d \cdot L$
- $O(\log T)$ depth required for these tasks

**Implication**:
- Long sequence + simple cumulative tasks 어려움
- Math (large numbers): partial limit
- Logical proofs (deeply nested): partial limit

### 정리 4.3 — Compositional Generalization Failure

Transformer 가 SCAN, COGS 에 weak:
- 학습: "jump twice", "walk twice"
- 테스트: "jump thrice" — 학습 안 한 composition
- Performance: 50-70% (vs human 100%)

**Why**:
- Memorization > generalization
- Compositional structure 의 explicit representation 부족
- Inductive bias for composition 약함

**Mitigation**:
- Curriculum learning
- Specific architectures (Compositional Attention)
- Symbolic + neural hybrid

### 정리 4.4 — Mamba 의 Theoretical Power

SSM 이 Transformer 의 일부 limit 우회:
- **Selective copying**: input-dependent state — Transformer 의 issue 회피
- **Long-range dependency**: $O(T)$ instead of $O(T^2)$
- **Counting / parity**: claim improvement (research ongoing)

**However**:
- Mamba 도 fundamental limit (Turing machine 시뮬레이션)
- Practical advantage 있지만 theoretical ceiling 비슷

### 정리 4.5 — Hybrid Architectures

Transformer + alternatives:
- **Jamba** (AI21 2024): Mamba + Transformer + MoE
- **RetNet** (Sun 2023): retentive network
- **Hyena** (Poli 2023): convolutional + gated

→ Single architecture 의 한계 인정, hybrid 가 frontier.

### 정리 4.6 — Need for Tools / Hybrid Systems

Theoretical 한계 + practical reasoning challenges:
- **Math**: calculator, theorem prover
- **Code**: code execution
- **Knowledge**: retrieval (RAG)
- **Long reasoning**: search (ToT, MCTS)

→ LLM alone 의 ceiling — hybrid systems 가 future.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Counting Task 의 Difficulty

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Toy: count 'a' in sequence
def make_counting_data(T, vocab_size=10):
    """Sequences with random tokens, target = count of token 0 ('a')"""
    seq = torch.randint(0, vocab_size, (T,))
    count = (seq == 0).sum().item()
    return seq, count

# Tiny Transformer
class TinyT(nn.Module):
    def __init__(self, d=64, h=4, L=2, vocab_size=10, max_T=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pe = nn.Embedding(max_T, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, max_T+1)   # predict count
    
    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pe(pos).unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1))   # pool

# Train
torch.manual_seed(0)
model = TinyT(d=64, L=4, max_T=64)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for step in range(2000):
    seqs = []; targets = []
    for _ in range(8):
        T = np.random.randint(8, 32)
        seq, count = make_counting_data(T)
        # Pad to max
        seq = torch.cat([seq, torch.zeros(64-T, dtype=torch.long)])
        seqs.append(seq); targets.append(count)
    
    seqs = torch.stack(seqs)
    targets = torch.tensor(targets)
    logits = model(seqs)
    loss = F.cross_entropy(logits, targets)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

print(f'Final loss: {losses[-1]:.4f}')

# Test on longer sequences (out of distribution)
correct_short = 0
correct_long = 0
for _ in range(100):
    seq_short, target_short = make_counting_data(20)
    seq_short = torch.cat([seq_short, torch.zeros(64-20, dtype=torch.long)])
    pred = model(seq_short.unsqueeze(0)).argmax(-1).item()
    if pred == target_short: correct_short += 1
    
    seq_long, target_long = make_counting_data(60)
    seq_long = torch.cat([seq_long, torch.zeros(64-60, dtype=torch.long)])
    pred = model(seq_long.unsqueeze(0)).argmax(-1).item()
    if pred == target_long: correct_long += 1

print(f'Short (T=20): {correct_short}%')
print(f'Long  (T=60): {correct_long}%')
# Long sequence 에서 성능 하락 — Hahn 2020 의 prediction
```

### 실험 2 — Parity Task

```python
def parity_data(T):
    seq = torch.randint(0, 2, (T,))
    parity = seq.sum().item() % 2
    return seq, parity

torch.manual_seed(0)
model_parity = TinyT(d=32, L=2, vocab_size=2, max_T=32)
opt = torch.optim.AdamW(model_parity.parameters(), lr=1e-3)

# Modify head for binary output
model_parity.head = nn.Linear(32, 2)

for step in range(2000):
    seqs = []; targets = []
    for _ in range(16):
        T = np.random.randint(8, 16)
        seq, p = parity_data(T)
        seq = torch.cat([seq, torch.zeros(32-T, dtype=torch.long)])
        seqs.append(seq); targets.append(p)
    seqs = torch.stack(seqs)
    targets = torch.tensor(targets)
    logits = model_parity(seqs)
    loss = F.cross_entropy(logits, targets)
    opt.zero_grad(); loss.backward(); opt.step()

# Test
correct = 0
for _ in range(100):
    seq, p = parity_data(np.random.randint(8, 16))
    seq = torch.cat([seq, torch.zeros(32-len(seq), dtype=torch.long)])
    pred = model_parity(seq.unsqueeze(0)).argmax(-1).item()
    if pred == p: correct += 1
print(f'Parity accuracy (in-distribution): {correct}%')
# Hahn 2020: depth-bounded model 은 long parity 못 풀음
# Small model 이 short parity 도 어려움 가능
```

### 실험 3 — Compositional Generalization (Toy SCAN)

```python
# Mini SCAN-like task
primitives = {'jump': 'JUMP', 'walk': 'WALK', 'run': 'RUN'}
modifiers = {'twice': 2, 'thrice': 3, 'four times': 4}

def make_scan_pair():
    prim = np.random.choice(list(primitives.keys()))
    mod = np.random.choice(list(modifiers.keys()))
    cmd = f'{prim} {mod}'
    out = ' '.join([primitives[prim]] * modifiers[mod])
    return cmd, out

# Train: only "twice" examples
# Test: "thrice", "four times" — compositional generalization

train_data = []
for _ in range(100):
    prim = np.random.choice(list(primitives.keys()))
    cmd = f'{prim} twice'
    out = ' '.join([primitives[prim]] * 2)
    train_data.append((cmd, out))

print(f'Train examples: {train_data[:3]}')
print(f'Test (compositional): "jump thrice" → ?')

# Standard Transformer 가 여기 어려움 — memorization 만, generalization 못
# Mitigation: explicit decomposition, larger data, specific architecture
```

### 실험 4 — Mamba 의 Selective Copying

```python
# Mamba 가 우수한 task: selective copying
# "Copy specific tokens (with markers) to output"

def selective_copy_data(T, vocab=10, marker=9):
    """Tokens marked with `marker` should be copied to output"""
    seq = torch.randint(0, vocab-1, (T,))
    marker_positions = torch.rand(T) < 0.2
    seq[marker_positions] = marker
    
    # Output: sequence of marker tokens (in order)
    output = seq[seq == marker].tolist()
    if not output: output = [0]
    return seq, output

# Standard Transformer 가 long T 에서 어려움
# Mamba (or RWKV) 가 선택적 retention 가능

# Toy demonstration
seq, out = selective_copy_data(50)
print(f'Sequence: {seq.tolist()[:20]}...')
print(f'Markers found: {len(out)}')
```

### 실험 5 — Architecture 비교 (개념)

```python
# Transformer vs Mamba vs RWKV 의 conceptual computation
# 각각 다른 strength

architectures = {
    'Transformer': {
        'training': 'O(T² d) parallel',
        'inference': 'O(T d) per step (KV cache)',
        'long_context': 'Limited (quadratic)',
        'reasoning': 'Strong (with scale)',
        'parity_counting': 'Weak (depth-bounded)',
    },
    'Mamba (SSM)': {
        'training': 'O(T d²) parallel (scan)',
        'inference': 'O(d²) per step',
        'long_context': 'Strong (linear)',
        'reasoning': 'Moderate (research ongoing)',
        'parity_counting': 'Better (selective state)',
    },
    'RWKV': {
        'training': 'O(T d) parallel',
        'inference': 'O(d) per step',
        'long_context': 'Strong',
        'reasoning': 'Moderate',
        'parity_counting': 'Better',
    },
}

for arch, props in architectures.items():
    print(f'\n{arch}:')
    for k, v in props.items():
        print(f'  {k}: {v}')
```

---

## 🔗 실전 활용

### 1. Hybrid Architectures 의 Rise

- **Jamba** (AI21 2024): Mamba + Transformer + MoE
- **Striped Hyena** (Together AI): hybrid
- 단일 architecture 의 한계 인정

### 2. Tool Use 의 Necessity

- **Calculator** for math
- **Code execution** for logic
- **Search** for knowledge
- **External memory** for long-term

→ LLM alone 의 한계를 외부 도구로 보완.

### 3. Reasoning Models (o1, etc.)

Architecture 의 한계를 test-time compute 로:
- Long internal CoT
- Search through reasoning paths
- Self-verification

### 4. Symbolic + Neural Hybrid

- **Neural-symbolic**: ML + symbolic reasoning
- **Differentiable theorem provers**: Lean + LLM
- **Program synthesis**: code generation + verification

### 5. Future Architectures (Research)

- **Liquid Neural Networks** (Hasani 2021): continuous-time
- **DNCs** (Differentiable Neural Computers): explicit memory
- **Compositional / modular**: explicit composition
- **Energy-based models**: different optimization

---

## ⚖️ 가정과 한계

| Theoretical Claim | Practical Implication |
|------------------|---------------------|
| Pérez 2019: Turing-complete | Theoretical only — finite precision |
| Hahn 2020: depth limit | Real models do struggle with long count/parity |
| Compositional limit | Mitigated by data + curriculum |
| Architecture innovation needed | Mamba etc. emerging |

---

## 📌 핵심 정리

| Limit | Source | Mitigation |
|-------|--------|------------|
| **Counting unbounded $T$** | Hahn 2020 | Tools, explicit counting |
| **Parity** | Hahn 2020 | RNN/SSM, multiple steps |
| **Compositional gen** | Empirical | Better training, specific arch |
| **Long-range computation** | Depth-bounded | Test-time compute, search |
| **Symbolic reasoning** | Inductive bias | Tool use, hybrid |

| Architecture | Strength | Weakness |
|--------------|----------|----------|
| Transformer | General reasoning, ICL | Quadratic, depth limit |
| Mamba | Long context, efficiency | Reasoning still developing |
| RWKV | Inference efficiency | Quality vs Transformer |
| Hybrid (Jamba) | Best of both | Complexity |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Pérez 2019 의 Turing-complete vs Hahn 2020 의 depth-bounded limit 의 정확한 차이는?

<details>
<summary>해설</summary>

**Pérez 2019 — Theoretical Universal**:

**Claim**: Transformer can simulate any Turing machine.

**Required conditions**:
- Arbitrary depth (or recurrent looping)
- Infinite precision (real-valued operations)
- Unbounded context length

**Implication**:
- Mathematically capable of any computation
- 그러나 **Idealized** — real implementation 불가

**Hahn 2020 — Practical Limit**:

**Claim**: Fixed-depth Transformer with finite precision cannot solve certain simple tasks.

**Specific examples**:
- Parity (T unbounded)
- Counting (large T)
- Boolean evaluation (deeply nested)

**Required conditions**:
- Bounded depth $L$
- Bounded precision (FP16/32)
- Specific tasks with cumulative dependency

**Implication**:
- Real models have **practical limits**
- Specific tasks 어려움

**Reconciliation**:

Both theorems coexist:
- Pérez: theoretical ceiling (high)
- Hahn: practical floor (limited)

**Real models = somewhere between**:
- Practically limited (Hahn-style)
- Not theoretically maxed (Pérez-style)
- Engineering: how close to theoretical with practical constraints?

**Modern Implications**:

1. **Test-time compute** (CoT) 가 effective depth 증가:
   - Bounded depth model + long CoT = effective deeper computation
   - Hahn's limit 의 partial workaround

2. **Hybrid architectures**:
   - Different limits for different architectures
   - Mamba 가 specific Hahn limits 회피

3. **Tool use**:
   - External calculator solves Hahn's counting/parity
   - Composition with Transformer = practical universality

→ **Theoretical limits matter 그러나 not fundamental ceiling** — engineering 으로 우회 가능. $\square$

</details>

**문제 2** (심화): Compositional generalization 이 LLM 의 fundamental weakness 인가? Scale + better data 로 mitigate 가능한가?

<details>
<summary>해설</summary>

**Compositional Generalization 의 어려움**:

SCAN, COGS 같은 task:
- Train: limited compositions
- Test: novel compositions of seen primitives
- Standard Transformer: 50-70% accuracy (human 100%)

**Why Hard?**

1. **No explicit composition**:
   - Transformer 의 attention 이 implicit composition
   - Symbolic structure 의 명시적 representation 없음

2. **Memorization tendency**:
   - Pre-training data 의 compositions 만 학습
   - Novel composition 에 weak

3. **Syntactic rigidity**:
   - 학습한 specific syntactic pattern 만
   - 새로운 syntax composition 어려움

**Mitigation Approaches**:

1. **More Data**:
   - Larger corpus 가 더 많은 compositions cover
   - 그러나 combinatorial explosion 어려움
   - Marginal improvement

2. **Curriculum Learning**:
   - Simple → complex
   - 각 stage 에서 composition 능력 학습
   - Some improvement

3. **Specific Architectures**:
   - Compositional Attention Networks
   - Modular networks
   - Better but specific to task

4. **Specialized Training**:
   - Augmented data (synthetic compositions)
   - Multi-task learning
   - Significant improvement on benchmarks

5. **Tool Use / Hybrid**:
   - Symbolic interpreter + LLM
   - Code generation for composition
   - Practical solution

**Modern LLM 의 Status**:

- GPT-4: better than GPT-3 on compositional tasks
- Claude 3.5: similar improvement
- 그러나 still not human-level

**Specific Examples**:

1. **Math word problems**:
   - Compositional (multiple operations)
   - LLM with CoT: significant improvement
   - Still struggles with novel composition

2. **Code**:
   - Highly compositional
   - Modern LLM 가 reasonable
   - Tool use (REPL) 가 mitigate

3. **Logical reasoning**:
   - Multi-step composition
   - CoT 가 도움
   - Verification 어려움

**Scale 의 영향**:

```
1B model: 30% compositional
10B model: 50%
100B model: 70%
1T model (estimate): 80%?
```

**Diminishing returns** — scale 만으로는 100% 못 도달.

**Future**:

1. **Architecture Innovation**:
   - Modular networks
   - Symbolic + neural
   - Compositional inductive bias

2. **Tool Use**:
   - Calculator, code, search
   - External composition

3. **Specialized Training**:
   - Compositional data augmentation
   - Synthetic composition examples

**근본 통찰**:

Compositional generalization 은 **partially fundamental, partially trainable**:
- Pure scaling 으로 mitigate 가능 but not solve
- Architecture innovation 필요
- Tool use + hybrid 가 practical solution

LLM 이 "AGI" 가 되려면 compositional generalization 의 fundamental advance 필요. **Scale alone 은 not the answer** — Sutton's "bitter lesson" 의 limit.

미래: **Compositional + Modular + Tool-using** 의 결합. $\square$

</details>

**문제 3** (논문 비평): Mamba 가 Transformer 의 limits 를 우회한다면, 미래 5년 안에 Transformer 가 dominant architecture 에서 dethrone 될 가능성? 또는 hybrid 가 dominant?

<details>
<summary>해설</summary>

**Current Status (2026)**:

- Transformer: dominant frontier (GPT-4, Claude 3.5, Gemini)
- Mamba: research + small/medium models
- Hybrid (Jamba): emerging
- RWKV, RetNet: niche

**Mamba 의 advantages**:

1. **Efficiency**: $O(T)$ vs $O(T^2)$
2. **Long context**: 1M+ tokens efficient
3. **Selective state**: input-dependent retention
4. **Inference**: $O(d^2)$/step (no KV cache growth)

**Mamba 의 disadvantages**:

1. **Reasoning**: less established than Transformer
2. **ICL**: emergent capability question
3. **Ecosystem**: fewer pre-trained models, tools
4. **Scale**: largest Mamba ~300B (vs Transformer 1T+)
5. **Implementation**: specialized scan algorithm

**Hybrid Trajectory**:

**Jamba (AI21 2024)** 의 approach:
- Most layers: Mamba (efficiency)
- Some layers: Transformer (reasoning)
- MoE for capacity

**Predictions (2026-2030)**:

**Frontier models**:
- 2026: still Transformer-dominant (GPT-5, Claude 4)
- 2027: hybrid 시도 (Anthropic, Google)
- 2028: Mamba/SSM-style 의 first frontier model?
- 2030: 한 architecture 의 dominance 끝남

**Reasoning models** (o1-like):
- Transformer 가 reasoning 에 잘 맞음
- 계속 Transformer-style 가능
- Test-time scaling 이 architecture 의 lid

**Long context**:
- 1M+ context: Mamba/hybrid 가 efficient
- Frontier 가 hybrid 채택 가능
- Pure Transformer 는 cost prohibitive

**Edge / Mobile**:
- Mamba/RWKV 의 inference efficiency 가 advantage
- 작은 모델 with reasoning capability
- 2027+ standard

**Why not faster transition?**

1. **Training cost**:
   - Transformer 의 hyperparameters, recipe well-known
   - New architecture 의 large-scale training 위험
   - $100M+ failure 어려움

2. **Pre-trained model ecosystem**:
   - HuggingFace, vLLM 등이 Transformer 중심
   - Migration cost 큼

3. **Quality gap**:
   - Mamba 가 Transformer 와 동등 quality 미입증 (frontier)
   - 점진적 catch-up

4. **Conservatism**:
   - Production system 의 risk-aversion
   - Proven architecture 우선

**Specific Predictions**:

- **2027 frontier**: GPT-5/Claude 4 — Transformer + reasoning training (still Transformer)
- **2028**: First frontier hybrid (Mamba + Transformer + MoE)
- **2029-2030**: Architecture diversity — different models for different use cases

**Open Source Trajectory**:

- LLaMA-4, Mistral large: Transformer + likely some hybrid features
- Mistral exploring Mamba blocks
- Open community 가 hybrid first

**근본 통찰**:

Transformer 의 dethrone 은 **gradual** 가 likely:
- Sudden replacement: unlikely (ecosystem, recipe known)
- Gradual hybrid: most likely
- Niche dominance (long context, edge): Mamba 우세

**Architecture 의 미래**:

"Transformer everywhere" 의 시대 → "right architecture for right job":
- Reasoning: Transformer (current strength)
- Long context: Mamba/hybrid
- Edge: efficient SSM
- Specialized: domain-specific architecture

**Hybrid 가 dominant**: 90% likely by 2030. **Pure Transformer dethrone**: 60% likely by 2032. **AGI architecture**: open question — possibly entirely new.

**LLM era 의 진화**:

```
2017-2020: Transformer 독점 시대
2020-2024: Transformer + 모든 augmentation
2024-2027: Architecture innovation emerging (Mamba, hybrid)
2027-2030: Diverse architectures dominant
2030+: 새로운 paradigm 등장 (가능성)
```

Transformer 는 historical milestone, 그러나 not eternal. **Continuous innovation** 이 AI 의 future. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-chain-of-thought.md) | [📚 README](../README.md) | 🎉 **레포 완료!**

</div>
