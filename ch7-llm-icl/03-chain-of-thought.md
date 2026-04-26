# 03. Chain-of-Thought 와 Reasoning (Wei 2022)

## 🎯 핵심 질문

- Chain-of-Thought (CoT) prompting 의 핵심 — "Let's think step by step" 의 효과는 무엇인가?
- Wei 2022 의 emergent finding — 왜 100B+ 모델에서만 CoT 가 효과적?
- Self-consistency, Tree of Thoughts 가 CoT 를 어떻게 발전시키는가?
- Process supervision (Lightman 2023) 의 의미 — outcome vs process reward?
- Modern reasoning model (o1, Claude 3.5) 이 CoT 를 어떻게 발전?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

CoT 는 **LLM reasoning 의 game changer**:

1. **Math, logic, complex reasoning 가능** — pre-CoT GPT-3 가 못한 task
2. **Emergent capability** — scale 의 직접적 결과
3. **Modern reasoning model 의 토대** — o1, Claude 3.5 Sonnet
4. **Test-time compute** — inference 시 더 많은 compute 로 better answer

이 문서는 CoT 의 **mechanism, emergence, modern variants** 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: ICL (Ch7-02), Scaling Laws (Ch7-01)
- Chapter 6: GPT (Ch6-02) — emergence

---

## 📖 직관적 이해

### Without vs With CoT

```
Without CoT:
  Q: A bat and a ball cost $1.10. Bat is $1 more than ball. How much is the ball?
  A: $0.10  (틀림! 정답: $0.05)

With CoT:
  Q: ...
  A: Let's think step by step.
     If ball costs x, bat costs x + 1.
     Total: x + (x+1) = 2x + 1 = 1.10
     2x = 0.10, x = 0.05
     The ball costs $0.05.
```

→ Step-by-step reasoning 이 정확도 dramatically 향상.

### CoT Prompt Pattern

```
Few-shot CoT:
  Q: <example question>
  A: <reasoning steps> ... So the answer is X.
  
  Q: <example 2>
  A: <reasoning> ... answer is Y.
  
  Q: <test question>
  A:
```

Model 이 reasoning chain 도 generate.

### Why does this work?

1. **More compute per problem**: 더 많은 token 생성 = 더 많은 forward computation
2. **Decomposition**: complex → simple subproblems
3. **Memory of intermediate**: each step builds on previous (in context)
4. **Self-correction**: model 이 step-by-step verify

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Chain-of-Thought Prompting

Prompt format:
$$
\text{Q: } \text{question} \\ \text{A: Let's think step by step.}
$$

Model generates reasoning steps + final answer.

### 정의 3.2 — Few-shot CoT

$$
\text{Examples: } \{(Q_i, R_i, A_i)\}_{i=1}^k
$$

with $R_i$ = reasoning, $A_i$ = answer. Test에 reasoning 학습 적용.

### 정의 3.3 — Zero-shot CoT (Kojima 2022)

Just append "Let's think step by step." to question — no examples.

### 정의 3.4 — Self-Consistency (Wang 2023)

다수 reasoning chain sampling 후 majority vote:
$$
\hat{A} = \arg\max_A \sum_{i=1}^N \mathbb{1}[A_i = A]
$$

with $N$ sampled reasoning paths.

### 정의 3.5 — Tree of Thoughts (Yao 2023)

Multi-branch reasoning:
- Each step 의 multiple alternatives 생성
- Search through tree (BFS, DFS)
- Backtrack + explore

### 정의 3.6 — Process Supervision (Lightman 2023)

Step 별 reward (vs outcome only):
$$
R_{\text{process}} = \sum_t r_t(\text{step}_t) \quad (\text{vs } R_{\text{outcome}} = r(\text{final}))
$$

각 step 의 quality 학습.

---

## 🔬 정리와 증명

### 정리 3.1 — CoT 의 Emergence (Wei 2022)

**Empirical Observation**:
- GPT-3 small/medium: CoT 가 도움 X 또는 hurt
- GPT-3 175B+: CoT 가 dramatically 도움 (GSM8K: 17% → 57%)

**Why emergent**:

1. **Reasoning ability emerges**:
   - 작은 모델은 reasoning step 자체 생성 못함
   - 큰 모델만이 coherent step generation
   - Required scale: 100B+

2. **Implicit Multi-step Capability**:
   - Pre-training data 에 reasoning chains 가 있음
   - 큰 모델만이 이 pattern 학습
   - 작은 모델은 surface pattern 만

3. **Compute per token 의 marginal value**:
   - 작은 모델: 추가 token 의 marginal info 작음
   - 큰 모델: each token 의 deep computation
   - Test-time compute 가 useful 한 scale 필요

### 정리 3.2 — Test-Time Compute Scaling

$N_{\text{tokens}}$ 의 reasoning generated 시 effective compute:
$$
C_{\text{test}} = 2 N_{\text{model}} \cdot N_{\text{tokens}}
$$

**Insight**: forward pass 가 each generated token 마다 더 많은 compute. CoT 가 이를 leverage.

→ Modern: o1 같은 reasoning model 이 test-time compute 를 explicit scale.

### 정리 3.3 — Self-Consistency 의 정당성

Multiple reasoning paths $\{R_1, \ldots, R_N\}$ generated. Majority vote 가 single answer 보다 우수:

**근거**:
- Different paths 가 different errors
- Consistent answers across paths = high confidence
- Wisdom of crowd

**Empirical**: GSM8K 에서 single CoT 57% → self-consistency (N=40) 74%.

### 정리 3.4 — Tree of Thoughts 의 Search

Sequential reasoning (CoT) 대비:
- BFS / DFS through reasoning tree
- 각 node 의 evaluation
- Backtrack from dead ends

**Improvement** on hard puzzles (Game of 24): 4% (CoT) → 74% (ToT).

→ Search 가 reasoning power 증가시킴.

### 정리 3.5 — Process Supervision 의 우위

**Outcome supervision**: final answer 만 reward.
**Process supervision**: each step 의 correctness reward.

Lightman 2023 의 finding:
- Process > Outcome (math task 에서)
- 더 reliable, interpretable
- Step-level error catch

**Cost**: process annotation 더 expensive (각 step labeled).

### 정리 3.6 — Modern Reasoning (o1, Claude 3.5)

Beyond CoT:
- **Internal reasoning**: hidden CoT (사용자에 안 보임)
- **Long CoT**: 수천 token 의 reasoning
- **Tool use**: code execution, calculator
- **Reflection**: 자기 critique 후 revise

→ Emergent ability 가 explicit reasoning capability 로 진화.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — CoT Prompt with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Toy math problem
problem = "If 2 + 3 = 5, what is 4 + 7?"

# Without CoT
prompt_no_cot = problem + " Answer: "
inputs = tokenizer(prompt_no_cot, return_tensors='pt')
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print('Without CoT:', tokenizer.decode(output[0]))

# With CoT (zero-shot)
prompt_cot = problem + " Let's think step by step. "
inputs = tokenizer(prompt_cot, return_tensors='pt')
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print('With CoT:', tokenizer.decode(output[0]))

# GPT-2 small (124M) 은 CoT 효과 약함 — emergent in larger
```

### 실험 2 — Few-shot CoT Pattern

```python
import torch

few_shot_cot = """Q: Roger has 5 tennis balls. He buys 2 cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each = 6 more balls. 5 + 6 = 11. The answer is 11.

Q: There were 9 computers in the server room. 5 more computers were installed each day. How many computers are now in the server room after 4 days?
A:"""

inputs = tokenizer(few_shot_cot, return_tensors='pt')
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
print(tokenizer.decode(output[0]))

# 패턴: explicit reasoning steps
```

### 실험 3 — Self-Consistency Simulation

```python
import numpy as np
from collections import Counter

def sample_reasoning_chains(prompt, n_samples=10, temperature=0.7):
    """Sample multiple reasoning paths"""
    answers = []
    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, 
                                    do_sample=True, temperature=temperature)
        text = tokenizer.decode(output[0])
        # Extract answer (toy: last number)
        numbers = [int(s) for s in text.split() if s.isdigit()]
        if numbers:
            answers.append(numbers[-1])
    return answers

# Self-consistency
prompt = "Q: 2 + 3 = ?\nA: Let's think. 2 + 3 = "
answers = sample_reasoning_chains(prompt, n_samples=5)
print(f'Sampled answers: {answers}')

if answers:
    most_common = Counter(answers).most_common(1)[0][0]
    print(f'Self-consistency answer: {most_common}')
```

### 실험 4 — Tree of Thoughts 의 idea (Pseudocode)

```python
def tree_of_thoughts(problem, max_depth=3, branching=3):
    """
    Pseudo-implementation of ToT
    각 node 에서 multiple thoughts 생성, 평가, 선택
    """
    def expand(state, problem):
        """Generate multiple next thoughts from current state"""
        # In real impl: prompt model with current state, sample multiple
        thoughts = [f'thought_{i} from {state}' for i in range(branching)]
        return thoughts
    
    def evaluate(state):
        """Heuristic evaluation of state quality"""
        # In real impl: prompt model to score state
        return np.random.rand()
    
    # BFS with pruning
    states = [(problem, 0)]   # (state, depth)
    for d in range(max_depth):
        new_states = []
        for state, depth in states:
            if depth < max_depth:
                children = expand(state, problem)
                for c in children:
                    new_states.append((c, depth + 1))
        # Keep top-K (pruning)
        new_states.sort(key=lambda x: evaluate(x[0]), reverse=True)
        states = new_states[:branching]   # keep top branching
    
    return states[0][0]   # best state

result = tree_of_thoughts("Solve 24 with [3, 5, 7, 8]")
print(f'ToT result: {result}')
# 실제 ToT 는 model evaluation + backtracking
```

### 실험 5 — Process Reward Simulation

```python
# Process supervision: each step 별 correctness label
def process_reward_simulation(reasoning_steps):
    """
    Each step 의 quality 측정
    Real impl: trained reward model 으로 step-level scoring
    """
    rewards = []
    for step in reasoning_steps:
        # Toy: contains math symbol → likely good step
        is_math = any(c in step for c in '+-*/=')
        rewards.append(1.0 if is_math else 0.5)
    return rewards

# Example
reasoning = [
    "Let me set up the equation",
    "x + (x+1) = 1.10",
    "2x = 0.10",
    "x = 0.05",
    "The answer is $0.05"
]

rewards = process_reward_simulation(reasoning)
print(f'Step-level rewards: {rewards}')
print(f'Total: {sum(rewards):.2f}')
print(f'Outcome reward only would not distinguish good vs bad steps')
```

---

## 🔗 실전 활용

### 1. Modern LLM 의 CoT

- **GPT-4**: native CoT, often hidden internally
- **Claude 3.5 Sonnet**: explicit step-by-step
- **Gemini 1.5**: long context CoT (multi-document reasoning)
- **o1 (OpenAI)**: extreme CoT — long internal reasoning

### 2. CoT Benchmarks

- **GSM8K**: 8K grade school math
- **MATH**: competition math
- **BIG-Bench Hard**: diverse reasoning
- **MMLU**: multi-subject knowledge

CoT 가 모든 benchmark 에서 직접 성능 향상.

### 3. Production Pattern

```python
# Standard CoT prompt
prompt = """You are a helpful assistant. Think step by step before answering.

Question: {question}

Reasoning:"""

# Or with explicit format
prompt = """Question: {question}
Let's break this down:
1. ...
2. ...
3. ...
Answer: ..."""
```

### 4. Reasoning Models 의 Architecture

o1, Claude 3.5 등의 reasoning model:
- **Internal reasoning**: hidden chain (not shown to user)
- **Thinking time**: longer inference
- **Search**: multiple reasoning paths
- **Verification**: self-check before final answer

→ "Test-time compute" 가 frontier 의 새 axis.

### 5. Limitations

- **Hallucination**: confident but wrong reasoning
- **Compounding errors**: bad early step → all later wrong
- **Cost**: long CoT = many tokens = expensive
- **Latency**: real-time application 에 어려움

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Step-by-step decomposition 가능 | Some tasks not decomposable |
| 큰 model | Small model 은 CoT 효과 적음 |
| Reasoning chain 정확 | Model hallucinate 가능 |
| Final answer 명확 | Open-ended 어려움 |
| 적절한 prompt | Format sensitive |

---

## 📌 핵심 정리

$$\boxed{\text{CoT: explicit step-by-step reasoning, emergent at } \sim 100B+ \text{ params}}$$

| Method | Idea | Improvement |
|--------|------|-------------|
| **Zero-shot CoT** | "Let's think step by step" | Strong baseline |
| **Few-shot CoT** | Examples with reasoning | More structured |
| **Self-consistency** | Multiple paths + majority | +15-20% |
| **Tree of Thoughts** | Tree search + eval | +50% on hard tasks |
| **Process Supervision** | Step-level reward | More reliable |

| Modern | Innovation |
|--------|-----------|
| o1 | Long internal CoT |
| Claude 3.5 | Reasoning + tools |
| Gemini 1.5 | Long-context reasoning |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Why does GSM8K (math) benefit from CoT 17% → 57% (175B GPT-3)? 어떤 step-by-step 이 도움?

<details>
<summary>해설</summary>

**GSM8K Example**:

Problem: "Roger has 5 tennis balls. He buys 2 cans, each containing 3 balls. How many balls does Roger have?"

**Without CoT**:
- Direct generation: "11" (correct) or "8" (incorrect — 5 + 3)
- Single token generation 으로 multi-step computation 어려움

**With CoT**:
- "Roger started with 5 balls."
- "2 cans of 3 balls each = 6 balls"
- "Total: 5 + 6 = 11"
- "The answer is 11."

**Why CoT helps**:

1. **Multi-step computation made explicit**:
   - Final answer 가 multiple intermediate values 의 함수
   - Direct generation 은 모든 step 을 한 token 안에 — 어려움
   - CoT 가 each step 별 token 으로 분리

2. **Working memory in context**:
   - 5, 2, 3 등의 number 가 context 에 남음
   - Model 이 reference 가능
   - Self-attention 으로 retrieve

3. **Pattern matching with training data**:
   - Math textbook 의 solutions 가 step-by-step
   - Pre-training 시 이 pattern 학습
   - CoT prompt 가 이 pattern activate

4. **Compute per problem 증가**:
   - 매 reasoning token 마다 forward pass
   - 더 많은 token = 더 많은 computation
   - 작은 모델은 이 추가 compute 활용 못함

**Specific Step Importance**:

GSM8K problems 의 typical structure:
- Identify quantities (5, 2, 3)
- Identify operations (multiplication, addition)
- Compute intermediate (2 × 3 = 6)
- Compute final (5 + 6 = 11)

CoT 가 each step 을 explicit — error free more likely.

**Without CoT 의 failure mode**:
- Often correct numbers but wrong operation
- 또는 first step 만 (5 + 2 = 7 → "7")
- Decomposition 없이 final answer 추측

**Therefore**:

CoT 의 효과는 **compute distribution**:
- Same total compute, distributed over steps
- Each step 별 cleaner reasoning
- Reduced compounding errors

GSM8K 같은 multi-step problem 에 본질적 fit. 단일-step problem 에는 CoT 가 마이너 또는 무관. $\square$

</details>

**문제 2** (심화): "Test-time compute" 가 새로운 frontier 라면, training compute vs test compute 의 trade-off 는? o1 같은 모델이 어떻게 이를 활용하는가?

<details>
<summary>해설</summary>

**Training vs Test Compute Trade-off**:

**Traditional**:
- Big training compute (one-time, $\$10M+$)
- Small test compute (per-query, $\$0.01$)
- Total cost = train + N × inference

**With Test-Time Scaling** (CoT, search):
- Same training compute
- Larger test compute (longer CoT, more sampling)
- Total cost = train + N × (inference × scaling factor)

**When is this favorable?**

For **hard problems** (math, coding, reasoning):
- Training으로 quality 높이려면 더 큰 모델 필요 — exponential cost
- Test-time scaling: linear in compute, dramatic quality gain
- Better quality per compute$

**o1 의 Approach**:

OpenAI o1 (2024):
- Native reasoning model
- Long internal CoT (수천 token 의 hidden reasoning)
- Test-time compute scaling: dial up inference budget for harder problems
- Trained to reason effectively (not just CoT prompted)

**Architecture Hypothesis (o1)**:
- Standard LLM + reasoning training (with reward for good reasoning)
- 또는 dual-system (reasoner + verifier)
- 또는 search at inference (multiple CoT + selection)

**Inference Compute Scaling**:

```
Easy problem: ~100 tokens reasoning (~$0.01)
Hard problem: ~5000 tokens reasoning (~$0.50)
Math olympiad: ~50000 tokens (~$5)
```

→ User 또는 system 이 "thinking time" 결정.

**Trade-off Calculation**:

For specific hard problem:
- Option A: $1B$ params, 100 token CoT, 60% accuracy
- Option B: $100B$ params, 100 token CoT, 80% accuracy (10× training cost)
- Option C: $1B$ params, 5000 token CoT, 80% accuracy

C 와 B 가 동등한 quality, but C 는 training cost 1/10. Inference cost C 가 더 큼 (50× more tokens), 그러나:
- One-time training >> inference per query
- Hard problems 에 가끔만 long CoT 필요

→ Net win for test-time scaling.

**Future Implications**:

1. **Scaling laws revision**:
   - Compute-optimal recipe 가 변화 (test-time included)
   - Smaller training model + more inference compute

2. **Frontier capability**:
   - o1 → GPT-5 class
   - Hard reasoning tasks (math, science) 의 새 frontier

3. **Cost structure**:
   - Premium tier (lots of inference compute) for hard tasks
   - Standard tier for easy queries
   - Dynamic compute allocation

4. **Specialization**:
   - Reasoning model (long CoT) for math/code
   - Conversation model (short generation) for chat

**Open Question**:

- Test-time scaling 의 ceiling? — quality gain 이 saturate?
- Reasoning quality 가 verification 에 의존? — 누가 verify?
- Inference cost 가 user 에 transparent?

**근본 통찰**:

LLM evolution 의 새 axis:
- 2020-2023: train compute scaling (GPT-3, GPT-4)
- 2024-: test compute scaling (o1, Claude reasoning)
- 미래: balanced, dynamic compute allocation

이는 **architecture 의 진화** 가 아니라 **deployment 의 진화** — 같은 model, different inference. $\square$

</details>

**문제 3** (논문 비평): CoT 가 emergent capability 라면 왜 작은 모델 (1B-10B) 에서도 reasoning 학습 가능한 architecture (Mamba, 또는 specialized reasoner) 가 가능할까? Architecture innovation vs scale 의 future?

<details>
<summary>해설</summary>

**현재 Status**:

- CoT 는 100B+ 에서 emergent
- 작은 모델 (1B) 은 CoT 시도해도 성능 저하 또는 marginal gain
- Architecture innovation 이 작은 모델에 reasoning 추가 가능?

**Approaches to Small-Model Reasoning**:

1. **Specialized Reasoning Architecture**:
   - **Toolformer** (Schick 2023): small model + tool use (calculator, etc.)
   - **PAL** (Gao 2023): code generation for math
   - 작은 모델 + external tool = reasoning capability
   - Not architecture change, but capability augmentation

2. **Self-Improvement / Distillation**:
   - **STaR** (Zelikman 2022): bootstrap reasoning data from model
   - **Reflexion** (Shinn 2023): self-critique and improve
   - Larger model 의 reasoning 을 smaller model 에 distill

3. **Better Training Recipe**:
   - **OpenAI o1**: reasoning-specific training
   - **Reasoning corpus**: math textbooks, code with explanations
   - Smaller model with reasoning-focused training

4. **Architecture Innovation**:
   - **Mamba**: efficient state spaces — same ICL/reasoning capability per param?
   - **Hyena**: long context efficient
   - 그러나 reasoning 의 emergence 가 scale + data dominant

**Why Scale Matters**:

Reasoning emergence 의 기여:

1. **Pattern complexity**: small model 은 simple patterns, large model 은 abstract
2. **Implicit knowledge**: 큰 corpus 에서 학습한 다양한 reasoning examples
3. **Compute capacity**: each step 의 deep computation 가능
4. **Memory**: long context 의 effective use

Architecture 가 이 모두를 substitute 가능?

**Specific Examples**:

- **Mistral 7B** (2023): GPT-3 175B 와 비슷한 일반 quality, but reasoning 약함
- **Phi-3** (Microsoft, 2024): small but high-quality data → competitive
- **Gemma**, **LLaMA-3 8B**: small + good training → competitive on benchmarks

→ **High-quality data + better training** 가 scale 의 일부 substitute.

**Future Trajectories (2025-2027)**:

1. **Architecture diversity**:
   - Mamba/Hybrid models 가 long context reasoning 의 efficiency
   - Transformer + reasoning module (specialized)
   - MoE for selective expert (reasoning expert)

2. **Better Training**:
   - Reasoning-specific corpus (math, code, logical proofs)
   - Synthetic data (model-generated reasoning)
   - Process supervision

3. **Test-Time Compute**:
   - Small model + lots of search/sampling
   - Tree of Thoughts, MCTS
   - Self-consistency

4. **Hybrid Systems**:
   - Small model + tool use + retrieval
   - Specialized reasoners + generalists
   - Multi-agent collaboration

**Predictions**:

- **Frontier**: still big models (1T+), with reasoning training
- **Production**: 7B-30B with strong reasoning (via training + tools)
- **Edge**: tiny models with specific tools

**Architecture vs Scale**:

- **Both matter**: architecture innovation accelerates scaling, scaling enables architecture
- **2024 reality**: scale dominant, architecture marginal
- **2026 prediction**: architecture more important — diminishing returns of pure scale

**Specific Predictions**:

- **Mamba-like reasoning**: 2027 즈음 가능 (research direction)
- **Specialized reasoner**: niche, complement frontier
- **Tool use + small model**: production standard

**근본 통찰**:

CoT 의 emergence 는 **scale 의 surprising property** — 그러나 **scale 만이 path 아님**. Architecture, training, tools, search 의 결합이 reasoning 의 future.

작은 모델로 reasoning 가능? **Yes, with right combination**:
- Architecture: efficient (Mamba)
- Training: reasoning-focused
- Tools: external compute
- Search: test-time scaling

미래 LLM = **carefully orchestrated hybrid systems**, not single big model. CoT 는 emergent capability 의 prototype, future 는 explicit reasoning system. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-in-context-learning.md) | [📚 README](../README.md) | [다음 ▶](./04-theoretical-limits.md)

</div>
