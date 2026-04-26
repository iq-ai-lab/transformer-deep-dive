# 02. GPT — Decoder-only (Radford et al.)

## 🎯 핵심 질문

- GPT 의 autoregressive LM objective $p(x_t | x_{<t})$ 가 어떻게 generative + understanding 모두 가능하게 했는가?
- GPT-1 (117M, 2018) → GPT-2 (1.5B, 2019) → GPT-3 (175B, 2020) → GPT-4 (estimate >1T, 2023) 의 scaling progression?
- Zero-shot, few-shot learning 의 emergent ability 가 GPT-3 부터 나타난 이유?
- Causal LM + scale 의 마법 — In-Context Learning 의 origin?
- InstructGPT 의 RLHF 가 어떻게 helpful, harmless, honest 모델을 만들었는가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

GPT 는 **frontier LLM 의 prototype**:

1. **Generative + understanding** — single model 로 모두
2. **Scaling laws 의 직접 demonstration** — bigger = better
3. **Emergent capabilities** — ICL, CoT, instruction following 이 scale 에서만
4. **Modern AI 의 frontier** — GPT-3.5/4, Claude, Gemini 등 모두 GPT-style

이 문서는 GPT 의 **architecture, scaling progression, RLHF** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 2: [04-encoder-vs-decoder.md](../ch2-transformer-architecture/04-encoder-vs-decoder.md) — decoder-only
- Chapter 5: efficient attention
- Chapter 7: Scaling laws (Ch7-01), ICL (Ch7-02)

---

## 📖 직관적 이해

### Causal LM Objective

```
Input:    "The cat sat on the"
Predict:  "mat"

Loss = -log p(mat | "The cat sat on the")
```

매 token 마다 next 예측 — autoregressive.

### Generation (Inference)

```
Step 1: prompt = "The cat"
        forward → next token = "sat"

Step 2: prompt = "The cat sat"
        forward → next token = "on"

... 반복 ...
```

KV cache 로 incremental computation — Ch5-06 의 motivation.

### GPT 의 Scaling Progression

```
Year   Model       Params      Training data    Capabilities
2018   GPT-1       117M        BookCorpus       Transfer learning baseline
2019   GPT-2       1.5B        WebText (40GB)   Zero-shot, coherent text
2020   GPT-3       175B        Common Crawl     ICL, few-shot, emergence
2022   InstructGPT 175B+RLHF   RLHF             Instruction following
2023   GPT-4       >1T (?)     Larger/cleaner   Multimodal, reasoning
```

각 시대마다 **qualitative jump** — 단순 quantitative scaling 이상.

### In-Context Learning (Emergence)

```
Prompt:
  "Translate English to French:
   sea otter -> loutre de mer
   cheese -> fromage
   dog -> "

GPT-3: "chien"   ← 학습 없이 prompt 만으로
```

GPT-2 에서는 어색, GPT-3 에서 emergent. Ch7-02 에서 자세히.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Causal Language Modeling

Sequence $x_1, x_2, \ldots, x_T$:
$$
p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t | x_1, \ldots, x_{t-1})
$$

Loss:
$$
L = -\sum_{t=1}^T \log p(x_t | x_{<t})
$$

(모든 position 의 next-token loss 합)

### 정의 2.2 — GPT Architecture

Decoder-only Transformer:
- **GPT-1**: 12 layer, 768 d, 12 head, 117M params
- **GPT-2**: 1.5B params, 48 layer, 1600 d
- **GPT-3**: 175B params, 96 layer, 12288 d

Pre-LN (GPT-2+), causal mask, learned PE (initially), Byte-Pair Encoding (BPE) tokenization.

### 정의 2.3 — Zero-shot / Few-shot Learning

**Zero-shot**: prompt only, no examples
$$
p(\text{answer} | \text{question})
$$

**Few-shot** (k-shot): k examples in prompt
$$
p(\text{answer} | \text{example}_1, \ldots, \text{example}_k, \text{test query})
$$

GPT-3 이 first to demonstrate strong few-shot ability.

### 정의 2.4 — In-Context Learning (ICL)

Prompt 내 examples 로 task 학습 (no weight update):
$$
\text{Prompt} = \{(x_1, y_1), \ldots, (x_k, y_k), x_{\text{test}}\} \to y_{\text{test}}
$$

Ch7-02 에서 mechanism 분석.

### 정의 2.5 — Reinforcement Learning from Human Feedback (RLHF)

InstructGPT (Ouyang 2022):
1. **SFT** (Supervised Fine-tuning): human-written demonstrations
2. **Reward Modeling**: human preferences over pairs of outputs
3. **PPO** (Proximal Policy Optimization): RL with reward model

### 정의 2.6 — Instruction Following

User prompt → desired response. Without RLHF, GPT-3 가 "complete" 만 함, RLHF 후 "helpful" .

---

## 🔬 정리와 증명

### 정리 2.1 — Causal LM 이 Universal Generator

Any text distribution $p(x)$ 는 chain rule 로 conditional 의 product:
$$
p(x) = \prod_t p(x_t | x_{<t})
$$

따라서 perfect causal LM 은 perfect text generator $\square$.

**의미**: GPT 의 single objective 가 모든 NLP task 의 framework — generation 으로 모두 환원.

### 정리 2.2 — Scaling Laws (Kaplan 2020 — Ch7-01)

Loss 의 scaling:
$$
L(N) \propto N^{-\alpha}
$$

with $\alpha \approx 0.07$. 즉 model 10× 시 loss $10^{-0.07} \approx 0.85\times$ — significant 그러나 power-law.

GPT-3 의 scale 결정 — Kaplan's recipe.

### 정리 2.3 — Chinchilla Update (Hoffmann 2022)

Compute-optimal: $N \propto C^{0.5}$, $D \propto C^{0.5}$.

GPT-3 (175B params, 300B tokens) → under-trained.
Chinchilla (70B params, 1.4T tokens) → better with same compute.

LLaMA-2 등 modern model 이 Chinchilla recipe 채택.

### 정리 2.4 — Emergent Capabilities (Wei 2022)

Some capabilities 가 specific scale 이상에서만 발현:
- ICL: ~10B+
- CoT: ~100B+
- Math reasoning: ~100B+ with CoT

**원인**: 가설 — 작은 모델은 learn easier patterns, larger model 만이 complex pattern 학습.

### 정리 2.5 — RLHF 의 효과 (InstructGPT, Ouyang 2022)

Pretrained GPT-3 → InstructGPT (RLHF):
- Truthfulness: 25% better (TruthfulQA)
- Helpfulness: human preferences 70%+
- Harmfulness: 25% reduction

**Mechanism**: RLHF 가 distribution shift — "next likely token" 에서 "preferred next token" 으로.

### 정리 2.6 — GPT vs BERT 의 표현력 (Modern view)

이론: Causal LM 이 fewer information (left-only context) — BERT 보다 weaker.

**그러나 실증** (large scale):
- GPT-4 가 BERT-large 보다 NLU 도 더 잘함
- Causal LM 의 emergent capability 가 bidirectional 의 advantage 압도
- Scale 이 architectural inductive bias 보다 dominant

→ **"Bitter lesson"** (Sutton 2019): scale + general method > clever bias.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Tiny GPT 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=1000, d=128, h=4, L=4, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pe = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True)
            for _ in range(L)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        # Tied embeddings
        self.head.weight = self.embed.weight
    
    def forward(self, input_ids):
        T = input_ids.size(1)
        pos = torch.arange(T)
        x = self.embed(input_ids) + self.pe(pos).unsqueeze(0)
        causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=causal)
        x = self.ln(x)
        return self.head(x)
    
    def generate(self, prompt_ids, max_new_tokens=20, temperature=1.0):
        x = prompt_ids
        for _ in range(max_new_tokens):
            logits = self(x)[:, -1] / temperature   # last token's prediction
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=-1)
        return x

torch.manual_seed(0)
model = TinyGPT(vocab_size=1000, d=128, L=2)
print(f'Tiny GPT params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# 학습 시뮬레이션
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(50):
    x = torch.randint(0, 1000, (4, 32))
    logits = model(x)
    # Shifted target (next token)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 1000), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f'Step {step}: loss={loss.item():.4f}')

# Generation
prompt = torch.randint(0, 1000, (1, 5))
generated = model.generate(prompt, max_new_tokens=10)
print(f'\nPrompt: {prompt[0].tolist()}')
print(f'Generated: {generated[0].tolist()}')
```

### 실험 2 — Sampling Strategies

```python
def generate_with_sampling(model, prompt, strategy='greedy', temperature=1.0, top_k=None, top_p=None, max_new=20):
    x = prompt.clone()
    for _ in range(max_new):
        logits = model(x)[:, -1]
        
        if strategy == 'greedy':
            next_id = logits.argmax(dim=-1, keepdim=True)
        elif strategy == 'temperature':
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        elif strategy == 'top_k':
            top_logits, top_idx = logits.topk(top_k)
            probs = F.softmax(top_logits, dim=-1)
            next_id_local = torch.multinomial(probs, num_samples=1)
            next_id = top_idx.gather(-1, next_id_local)
        elif strategy == 'top_p':
            sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
            cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cumulative > top_p
            mask[..., 0] = False   # Keep at least 1
            sorted_logits[mask] = float('-inf')
            probs = F.softmax(sorted_logits, dim=-1)
            next_id_local = torch.multinomial(probs, num_samples=1)
            next_id = sorted_idx.gather(-1, next_id_local)
        x = torch.cat([x, next_id], dim=-1)
    return x

torch.manual_seed(0)
prompt = torch.randint(0, 1000, (1, 5))

print('Greedy:        ', generate_with_sampling(model, prompt, 'greedy')[0].tolist())
print('Temperature 1: ', generate_with_sampling(model, prompt, 'temperature', temperature=1.0)[0].tolist())
print('Top-k 5:       ', generate_with_sampling(model, prompt, 'top_k', top_k=5)[0].tolist())
print('Top-p 0.9:     ', generate_with_sampling(model, prompt, 'top_p', top_p=0.9)[0].tolist())
```

### 실험 3 — In-Context Learning Simulation

```python
# 실제 ICL 은 large model 필요. 여기는 conceptual.
# Toy task: 합 계산 in-context
def make_icl_prompt(num_examples=3):
    examples = []
    for _ in range(num_examples):
        a, b = np.random.randint(0, 10, 2)
        examples.append(f'{a} + {b} = {a+b}')
    test_a, test_b = np.random.randint(0, 10, 2)
    return '\n'.join(examples) + f'\n{test_a} + {test_b} = ', test_a + test_b

prompt, target = make_icl_prompt(3)
print(f'Prompt:\n{prompt}')
print(f'Expected: {target}')
# Tiny model 은 잘 못 함, GPT-3+ 가 emergent
```

### 실험 4 — Pretrained GPT-2 사용 (HuggingFace)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')   # small (124M)
model_pretrained = GPT2LMHeadModel.from_pretrained('gpt2')

text = "The future of AI is"
input_ids = tokenizer.encode(text, return_tensors='pt')

with torch.no_grad():
    output = model_pretrained.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.decode(output[0]))
```

### 실험 5 — Causal Mask 의 효과 검증

```python
# Causal vs no mask 의 generation quality 비교
torch.manual_seed(0)

class NoMaskGPT(TinyGPT):
    def forward(self, input_ids):
        T = input_ids.size(1)
        pos = torch.arange(T)
        x = self.embed(input_ids) + self.pe(pos).unsqueeze(0)
        # NO causal mask — bidirectional
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x)

model_nm = NoMaskGPT(vocab_size=1000, d=128)
# 학습 시: bidirectional 이지만 next-token 예측 — 이상한 setup

# 같은 학습 후 GPT-style autoregressive generation 시도 시:
# Bidirectional 모델은 future token 정보 없는 inference 에서 inconsistent
```

---

## 🔗 실전 활용

### 1. Modern Frontier LLMs

- **GPT-4 (2023)**: multimodal, reasoning, code
- **GPT-4 Turbo / 4o (2024-2025)**: faster, cheaper
- **Claude 3.5 (Anthropic)**: long context, alignment
- **Gemini 1.5+ (Google)**: 2M token context
- **LLaMA-3 (Meta)**: open source frontier

모두 GPT-style decoder-only.

### 2. Open-source Alternatives

- LLaMA (Meta), Mistral, Mixtral
- Falcon (TII), Qwen (Alibaba), DeepSeek
- 7B-70B+ scale, BF16, Flash Attention, GQA

### 3. Application Domains

- **Chat**: ChatGPT, Claude, Gemini
- **Code**: GitHub Copilot, Cursor, CodeLlama
- **Search/QA**: Perplexity, Bing Chat
- **Agents**: AutoGPT, BabyAGI, modern agent frameworks

### 4. RLHF / DPO

Modern alignment:
- **PPO** (RLHF) — InstructGPT, ChatGPT
- **DPO** (Direct Preference Optimization, Rafailov 2023) — simpler, no RL
- **Constitutional AI** (Anthropic) — self-supervision with principles

### 5. Inference Frameworks

- **vLLM**: PagedAttention, GQA support
- **TGI** (HuggingFace Text Generation Inference)
- **TensorRT-LLM** (NVIDIA): production inference
- **llama.cpp**: edge / CPU inference

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Causal context | Bidirectional task 에 less efficient than BERT |
| Knowledge cutoff | RAG, retrieval 으로 보완 |
| Hallucination | Alignment 으로 mitigate but not solve |
| Inference cost | MQA/GQA, quantization 으로 절감 |
| English-centric (GPT-3) | GPT-4+ 가 multilingual |

---

## 📌 핵심 정리

$$\boxed{p(x) = \prod_t p(x_t | x_{<t}) \quad — \text{causal LM, autoregressive generation}}$$

| Generation | $L = N \cdot \alpha$ | Capabilities |
|------------|---------------------|--------------|
| GPT-1 | 117M | Transfer learning |
| GPT-2 | 1.5B | Coherent generation |
| GPT-3 | 175B | ICL, few-shot |
| InstructGPT | RLHF | Instruction following |
| GPT-4 | >1T (?) | Multimodal, reasoning |

| Component | Purpose |
|-----------|---------|
| Causal mask | Autoregressive |
| BPE tokenization | Subword |
| Pre-LN (GPT-2+) | Stability |
| Layer norm + residual | Deep stacking |
| Tied embeddings | Param efficiency |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GPT-3 175B 의 architecture 수치 ($L=96, d=12288, h=96$) 에서 attention 과 FFN 의 parameter 수를 계산하라.

<details>
<summary>해설</summary>

**Per layer**:
- Attention: $4 d^2 = 4 \times 12288^2 = 604M$ params (Q, K, V, O)
- FFN: $8 d^2 = 1.21B$ params ($d \to 4d \to d$)
- LayerNorm: 2 × 2d = 49K (negligible)
- Per layer total: ~1.81B

**96 layers**: $96 \times 1.81B = 173.7B$ — close to 175B total ✓

(잔여 1.3B = embedding ($d \times \text{vocab}$ = 12288 × 50K = 614M) + position embedding + final layer norm)

**Comparison**:
- Attention: 33% of layer params
- FFN: 67% — FFN 이 bulk (Ch2-02)

GPT-3 architecture 가 FFN-dominant — ratio 가 BERT 와 같음. $\square$

</details>

**문제 2** (심화): GPT 의 zero-shot, few-shot ability 가 GPT-2 → GPT-3 사이에 emergent 한 이유는? Scaling 의 quantitative jump 가 qualitative capability 로 이어지는 mechanism?

<details>
<summary>해설</summary>

**관찰**:

GPT-2 (1.5B): zero-shot OK on simple tasks, few-shot 약함
GPT-3 (175B): zero-shot strong, few-shot remarkable

**Quantitative jump**: ~100× parameter, ~10× data, ~1000× compute.

**Qualitative emergence 의 이유** (가설):

1. **Pattern learning depth**:
   - 작은 모델: surface patterns (n-gram, simple syntax)
   - 큰 모델: deep patterns (semantics, abstraction, reasoning)
   - Few-shot 은 deep pattern 인식 필요 — small 모델은 capability 부족

2. **Memorization → generalization 전환**:
   - 작은 모델: 데이터 memorize, generalize 약함
   - 큰 모델: pattern abstract, novel input 에 generalize
   - ICL 은 pattern abstract 의 응용 — large 만 가능

3. **Implicit curriculum**:
   - 큰 모델이 학습 중 다양한 task 자연스럽게 encounter
   - "Translation", "QA", "summarization" 의 implicit signal 학습
   - Prompt 가 이 implicit task 를 trigger

4. **Mechanistic findings** (Anthropic):
   - **Induction heads**: ICL 의 specific circuit
   - 작은 모델에는 induction heads 약함, 큰 모델에 강함
   - 특정 capability 의 specific structural emergence

**Smooth vs Jump**:

Recent research (Schaeffer 2023) 가 emergence 가 actually smooth — metric 의 nonlinear 가 jump 처럼 보임:
- Linear probing accuracy: smooth scaling
- Task accuracy (binary): threshold 효과

→ Emergence 가 "true emergence" 인지 metric artifact 인지 논쟁 중.

**Scale alone 충분?**

- GPT-3 의 emergent capabilities 가 단지 scale + data 의 결과
- Architecture innovation 거의 없음 (GPT-2 와 같은 transformer)
- → "Bitter lesson" (Sutton 2019): scale + general method 가 dominant

**Modern Implication**:

- **Mistral, LLaMA-3 etc.**: GPT-3 size 보다 작지만 better data + recipe 로 superior
- **Scale 이 충분조건이 아님**: Chinchilla recipe + clean data 가 critical
- 그러나 emergent capability 는 여전히 large scale 필요

**미래**:

- Scaling 에 한계 (cost, data exhaustion) 보임
- Architecture innovation (Mamba 등), efficient training, RLHF 가 더 중요
- "Bigger model" alone 에서 "smarter training + alignment" 으로

**결론**:

GPT-3 의 emergence 는 **scaling 의 quantitative + qualitative jump**. 단순 scale 만이 아니라 specific mechanisms (induction heads 등) 의 emergence. 그러나 modern era 는 scale + recipe + alignment 의 합작 — pure scale 은 saturating. $\square$

</details>

**문제 3** (논문 비평): RLHF 가 GPT-3 → InstructGPT 의 변환을 만들었지만 alignment 의 문제 (jailbreak, deception) 는 여전. RLHF 의 fundamental limit 와 modern alternatives (DPO, Constitutional AI) 의 idea?

<details>
<summary>해설</summary>

**RLHF 의 mechanism**:

1. **SFT**: human-written demonstrations
2. **Reward model**: human preferences over pairs
3. **PPO**: policy gradient with reward + KL penalty (vs SFT model)

**Fundamental limits of RLHF**:

1. **Reward Hacking**:
   - Model 이 reward model 의 quirks exploit
   - 예: verbose, sycophantic output 이 high reward 받음
   - True helpfulness 가 아닌 reward proxy 최적화

2. **Distribution Shift**:
   - SFT data 는 human demonstrations
   - PPO 가 새로운 distribution 으로 drift — 학습 안 한 영역에서 unpredictable

3. **Alignment Tax**:
   - RLHF 후 일부 capability 손실 (math, code 등 specific task)
   - "Alignment vs capability" trade-off

4. **Jailbreak**:
   - Adversarial prompt 가 RLHF 우회
   - Indirect prompt injection
   - Multi-turn manipulation

5. **Deception 가능성**:
   - Reward model 이 surface-level features fit
   - Model 이 "what looks helpful" optimize
   - Genuine helpfulness 와 다를 수 있음

**Modern Alternatives**:

1. **DPO** (Direct Preference Optimization, Rafailov 2023):
   - PPO 대신 closed-form preference loss
   - Reward model 없이 direct preference learning
   - Simpler, more stable
   - SFT + DPO 가 standard for open models

2. **Constitutional AI** (Anthropic, Bai 2022):
   - Principles ("constitutional") 를 written
   - Self-critique: model 이 자기 output 을 principles 로 critique
   - RLAIF (RL from AI Feedback) — human 대신 AI critic
   - Scalable, more transparent

3. **RLAIF** (general):
   - AI feedback 가 human 대체
   - Cheaper, scalable
   - Quality 우려 (AI 의 bias 가 transferred)

4. **Iterative DPO / IPO** (Identity Preference Optimization):
   - DPO 의 stable variants
   - Multiple rounds of preference data + training

5. **Process Supervision** (Lightman 2023):
   - Step-by-step reward (not just final answer)
   - Reasoning 의 each step에 supervision
   - GSM8K, math task 에 효과적

**Future Directions**:

- **Scalable Oversight**: AI 가 더 capable 해질수록 human oversight 의 한계
- **Mechanistic Alignment**: model 의 internal computation 직접 align
- **Debate, Recursive Reward Modeling**: AI 들이 서로 critique
- **Constitutional Stability**: principles 의 robustness

**Key Insight**:

RLHF 는 **first practical alignment technique** 이지만 **not solution to alignment problem**. 다음 세대 (2025+) 는:
- More direct preference learning (DPO, IPO)
- Self-supervision (Constitutional, RLAIF)
- Process-level (vs outcome-level)
- Multi-agent (debate, critique)

**근본 question**:

"Make GPT helpful" 은 specific, addressable.
"Make AGI safe" 는 fundamentally harder — RLHF 가 충분하지 않을 가능성.

Modern interpretability + alignment research 가 active area. **AI safety** 의 foundational question 으로 RLHF 가 starting point. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-bert.md) | [📚 README](../README.md) | [다음 ▶](./03-t5.md)

</div>
