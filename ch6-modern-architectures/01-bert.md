# 01. BERT — Encoder-only (Devlin 2019)

## 🎯 핵심 질문

- BERT 의 **Masked Language Modeling** 이 어떻게 bidirectional context 학습을 가능하게 했는가?
- Next Sentence Prediction (NSP) 의 역할과 RoBERTa 가 NSP 를 제거한 이유?
- BERT 의 fine-tuning paradigm — pre-training + task-specific head 의 의미?
- BERT-base (110M), BERT-large (340M) 의 architecture 차이?
- WordPiece tokenization, [CLS], [SEP] token 의 디자인?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

BERT 는 **NLP transfer learning paradigm 의 시작**:

1. **Pre-training + Fine-tuning** — single pre-trained model 을 다양한 task 에 fine-tune
2. **Bidirectional context** — MLM 으로 양쪽 context 학습 (vs unidirectional GPT-1)
3. **GLUE 9-task SOTA** — single architecture 가 all NLU benchmark 정복
4. **Encoder-only 의 정의** — generation 못하지만 NLU 에 강력

이 문서는 BERT 의 **objective, architecture, fine-tuning recipe** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 2: [04-encoder-vs-decoder.md](../ch2-transformer-architecture/04-encoder-vs-decoder.md) — encoder block
- Chapter 3: PE
- Chapter 4: 훈련 recipe

---

## 📖 직관적 이해

### Bidirectional Pre-training

```
Sentence: "The cat sat on the mat"

Mask 15% randomly:
Input:    "The cat [MASK] on the mat"
Target:   predict "sat" using BOTH "The cat" (left) AND "on the mat" (right)
```

→ 양쪽 context 사용 — 정확한 word prediction.

### MLM vs Causal LM

```
Causal (GPT):
  P(x_t | x_1, ..., x_{t-1})    ← left context only

MLM (BERT):
  P(x_t | x_1, ..., x_{t-1}, x_{t+1}, ..., x_T)    ← both contexts

→ MLM 이 NLU 에 더 강력한 representation
```

### Pre-training + Fine-tuning Paradigm

```
Stage 1: Pre-training
  - Large unlabeled corpus (BookCorpus + Wikipedia, ~3B words)
  - MLM + NSP objective
  - Output: pre-trained BERT

Stage 2: Fine-tuning  
  - Small labeled task data (GLUE, etc.)
  - Add task-specific head (classification, NER, QA)
  - Fine-tune entire model
```

### Special Tokens

```
[CLS] My dog [SEP] My cat [SEP]
  ↑              ↑      ↑
classification    sentence boundary
token (gathers info)
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Masked Language Modeling

Random 15% of tokens:
- 80%: replace with `[MASK]`
- 10%: replace with random token
- 10%: keep original

Loss:
$$
L_{\text{MLM}} = -\sum_{i \in M} \log p(x_i | \tilde{x})
$$

(masked positions $M$, $\tilde{x}$ = corrupted input)

### 정의 1.2 — Next Sentence Prediction (NSP)

Two sentences $A, B$:
- 50%: $B$ follows $A$ in original text → label = 1
- 50%: $B$ random sentence → label = 0

Use `[CLS]` token's embedding for binary classification:
$$
p(\text{IsNext}) = \sigma(W_{\text{NSP}} h_{[CLS]})
$$

### 정의 1.3 — BERT Architecture

- **BERT-base**: $L = 12$, $d = 768$, $h = 12$, params = 110M
- **BERT-large**: $L = 24$, $d = 1024$, $h = 16$, params = 340M
- Post-LN block (Vaswani 2017 original)
- Learned absolute PE, max_len = 512

### 정의 1.4 — WordPiece Tokenization

Subword tokenization (subset of BPE-like):
- Common words → single token
- Rare words → subword pieces

예: "playing" → "play", "##ing" (##: continuation marker)

Vocabulary size: 30,522 (BERT-base/large).

### 정의 1.5 — Input Representation

$$
\text{Input}_t = \text{Token}_t + \text{Segment}_t + \text{Position}_t
$$

- Token embedding (WordPiece)
- Segment embedding (sentence A vs B)
- Position embedding (learned)

### 정의 1.6 — Fine-tuning Heads

Task-specific small head on top:
- **Classification**: $h_{[CLS]} \to \text{Linear} \to \text{class}$
- **NER**: per-token $h_t \to \text{Linear} \to \text{tag}$
- **QA (SQuAD)**: per-token $h_t \to \text{Linear}_2 \to (\text{start}, \text{end})$ logits

---

## 🔬 정리와 증명

### 정리 1.1 — Bidirectional vs Unidirectional 의 표현력

**Claim**: MLM 이 causal LM 보다 stronger representation 학습 (NLU task 기준).

**근거**:
1. **Information**: 양쪽 context 가 더 많은 정보
2. **Disambiguation**: 동음이의어, 모호 표현이 양쪽 context 로 명확
3. **NLU 에 적합**: 분류, NER 은 generation 아니라 representation quality

**실증**: BERT 가 ELMo, GPT-1 (causal) 대비 GLUE 평균 5%+ 향상.

### 정리 1.2 — MLM 의 Discrepancy 문제

Pre-training 에 [MASK] 가 있지만, fine-tuning 에는 없음 → discrepancy.

**해결**:
- 80% [MASK], 10% random, 10% original
- 모델이 [MASK] 만 의존하지 않게
- Downstream task 에 더 robust

### 정리 1.3 — NSP 의 효과 vs 무용 (RoBERTa 발견)

Devlin 2019: NSP 가 sentence-pair task 에 도움.

**RoBERTa** (Liu 2019): NSP 제거 + 더 많은 데이터 + dynamic masking → BERT 보다 우수.

**결론**: NSP 가 weak signal — 더 좋은 sentence-level objective 필요. 이후 모델 (ALBERT, ELECTRA 등) 도 다양한 sentence-level objective.

### 정리 1.4 — [CLS] Token 의 정당성

[CLS] token 이 sequence 의 첫 위치에 있고, attention 으로 모든 token 정보 모음:
- Classification 시 $h_{[CLS]}$ 가 sequence-level representation
- Pre-training 의 NSP 가 [CLS] 학습을 명시적으로 supervise
- Fine-tuning 시 작은 head 만 추가하면 됨

### 정리 1.5 — Fine-tuning vs Feature-extraction

Two strategies for using BERT:
- **Fine-tuning** (Devlin 2019 권장): entire model 학습 — best performance
- **Feature extraction** (frozen BERT + task-head): cheaper, slightly worse

**Modern trend**: parameter-efficient fine-tuning (LoRA, adapters) — middle ground.

### 정리 1.6 — BERT 의 한계

1. **Generation 불가**: bidirectional 이라 autoregressive 못함
2. **Max length 512**: long document 어려움
3. **Computation**: bidirectional 이 causal 보다 expensive (attention 의 full computation)
4. **Static representation**: pre-training 후 변화 없음 — knowledge cutoff 문제

→ GPT-style decoder + RLHF 등 modern paradigm 으로 대체.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — MLM 의 Pre-training 시뮬레이션

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 작은 BERT-like model
class TinyBERT(nn.Module):
    def __init__(self, vocab_size=1000, d=64, h=4, L=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pe = nn.Embedding(64, d)   # learned positional
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True)
            for _ in range(L)
        ])
        self.lm_head = nn.Linear(d, vocab_size)
    
    def forward(self, input_ids):
        T = input_ids.size(1)
        pos = torch.arange(T)
        x = self.embed(input_ids) + self.pe(pos).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

# MLM training simulation
torch.manual_seed(0)
vocab_size = 1000; T = 16
model = TinyBERT(vocab_size, d=64)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

mask_token_id = vocab_size - 1   # last vocab id reserved as [MASK]

for step in range(100):
    # Random sentence
    input_ids = torch.randint(0, vocab_size-1, (4, T))
    target_ids = input_ids.clone()
    
    # Mask 15%
    mask_mask = (torch.rand(4, T) < 0.15)
    masked_ids = input_ids.clone()
    masked_ids[mask_mask] = mask_token_id
    
    # Forward
    logits = model(masked_ids)
    
    # Loss only on masked positions
    loss = F.cross_entropy(logits[mask_mask], target_ids[mask_mask])
    
    opt.zero_grad(); loss.backward(); opt.step()
    
    if step % 20 == 0:
        print(f'Step {step}: loss={loss.item():.4f}')
```

### 실험 2 — Fine-tuning a Classification Head

```python
class BERTClassifier(nn.Module):
    def __init__(self, bert, num_classes):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.embed.embedding_dim, num_classes)
    
    def forward(self, input_ids):
        T = input_ids.size(1)
        pos = torch.arange(T)
        x = self.bert.embed(input_ids) + self.bert.pe(pos).unsqueeze(0)
        for layer in self.bert.layers:
            x = layer(x)
        # Use CLS token (position 0)
        cls = x[:, 0]
        return self.classifier(cls)

# Fine-tuning
torch.manual_seed(0)
clf = BERTClassifier(model, num_classes=3)
opt_ft = torch.optim.AdamW(clf.parameters(), lr=1e-4)

for step in range(50):
    input_ids = torch.randint(0, vocab_size, (8, T))
    labels = torch.randint(0, 3, (8,))
    
    logits = clf(input_ids)
    loss = F.cross_entropy(logits, labels)
    opt_ft.zero_grad(); loss.backward(); opt_ft.step()
    
    if step % 10 == 0:
        print(f'FT Step {step}: loss={loss.item():.4f}')
```

### 실험 3 — Bidirectional vs Causal Attention 비교

```python
class CausalBERT(TinyBERT):
    def forward(self, input_ids):
        T = input_ids.size(1)
        pos = torch.arange(T)
        x = self.embed(input_ids) + self.pe(pos).unsqueeze(0)
        causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=causal)
        return self.lm_head(x)

# 같은 task (next-token prediction) 로 비교
torch.manual_seed(0)
bidir = TinyBERT(vocab_size)
causal = CausalBERT(vocab_size)
# (실제로는 별도 학습 필요 — 여기는 conceptual)

# Bidirectional 의 [MASK] 위치 예측 vs Causal 의 next-token
print('BERT (bidirectional MLM): masked position 의 양쪽 context 활용')
print('GPT (causal LM): left context 만 활용 → less information')
```

### 실험 4 — HuggingFace Transformers 사용 (실전)

```python
# pip install transformers
from transformers import BertTokenizer, BertForMaskedLM

# Pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# MLM 추론
text = "The cat [MASK] on the mat."
inputs = tokenizer(text, return_tensors='pt')
print(f'Input ids: {inputs.input_ids}')
print(f'Tokens: {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}')

with torch.no_grad():
    outputs = model(**inputs)

# Mask token 위치 찾기
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predictions = outputs.logits[0, mask_idx]
top_5 = predictions.topk(5)
top_tokens = [tokenizer.decode([t]) for t in top_5.indices[0]]
print(f'Top-5 predictions for [MASK]: {top_tokens}')
# 예상: 'sat', 'lay', 'sleeps', 'is', 'was'
```

### 실험 5 — Tokenization 의 효과

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

words = ['playing', 'tokenization', 'unbelievable', 'COVID-19', 'antidisestablishmentarianism']
for word in words:
    tokens = tokenizer.tokenize(word)
    print(f'{word:30s} -> {tokens}')

# 'playing' -> ['playing']
# 'tokenization' -> ['token', '##ization']
# 'unbelievable' -> ['un', '##bel', '##iev', '##able']
# 'COVID-19' -> ['co', '##vid', '-', '19']
# subword 의 예
```

---

## 🔗 실전 활용

### 1. BERT 의 변형 모델

- **RoBERTa** (Liu 2019): no NSP, more data, dynamic masking — better than BERT
- **DistilBERT** (Sanh 2019): 40% smaller, 60% faster, 97% performance
- **ALBERT** (Lan 2020): factorized embedding, cross-layer sharing — fewer params
- **DeBERTa** (He 2020): disentangled attention + relative PE
- **ELECTRA** (Clark 2020): replaced token detection (vs MLM) — more efficient

### 2. Multilingual BERT

- mBERT: 104 languages — single model
- XLM-R: 100 languages, RoBERTa-style — better multilingual

### 3. Domain-Specific BERT

- BioBERT (biomedical), SciBERT (scientific), FinBERT (financial)
- Continued pre-training on domain corpus
- 전문 분야의 vocabulary, terminology 학습

### 4. Modern Replacement

BERT 시대 이후 (2022+):
- Decoder-only LLM (GPT-3.5, LLaMA, etc.) 가 dominant
- BERT-style 은 specific task (분류, NER) 에서 cost-effective
- 그러나 frontier 는 generative model

### 5. 실제 활용 Pattern

```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Fine-tune on labeled data (sentiment, etc.)
# 작은 head + entire BERT 학습 — quick adaptation
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Max length 512 | Longformer, ETC 등 long-doc 변형 |
| Static (no generation) | T5, BART 등 encoder-decoder |
| English-centric | mBERT, XLM-R 으로 multilingual |
| Pre-training corpus 의 cutoff | Continued pre-training 으로 update |
| Computation 큼 | Distillation (DistilBERT, TinyBERT) |

---

## 📌 핵심 정리

$$\boxed{\text{BERT: bidirectional Transformer encoder + MLM + NSP, fine-tuned per task}}$$

| Component | Role |
|-----------|------|
| MLM | Bidirectional context learning |
| NSP | Sentence-level (later removed) |
| [CLS] | Sequence representation |
| [SEP] | Sentence boundary |
| WordPiece | Subword tokenization |
| Encoder block | Transformer encoder (post-LN) |
| Fine-tuning head | Task-specific |

| Variant | Innovation |
|---------|-----------|
| BERT-base | $L=12, d=768$, 110M |
| BERT-large | $L=24, d=1024$, 340M |
| RoBERTa | No NSP, more data |
| ALBERT | Factorized + shared |
| ELECTRA | Replaced token detection |

---

## 🤔 생각해볼 문제

**문제 1** (기초): BERT-base 의 attention layer 1개의 parameter 수를 계산하라. ($d=768, h=12$)

<details>
<summary>해설</summary>

각 attention layer:
- $W_Q, W_K, W_V \in \mathbb{R}^{768 \times 768}$: 각 $768^2 = 589,824$ params
- $W_O$: $589,824$
- Total attention: $4 \times 589,824 = 2,359,296 \approx 2.4M$

$d_k = d/h = 64$ — single head 의 W_Q 가 $\mathbb{R}^{768 \times 64}$, 12 heads concatenated = $\mathbb{R}^{768 \times 768}$ ✓

12-layer BERT-base: $12 \times 2.4M = 28.8M$ for attention.
+ FFN ($8 d^2 = 4.7M$ per layer × 12 = 56.6M)
+ Embedding ($30522 \times 768 = 23.4M$) + Position embedding (512 × 768 = 0.4M)
= ~110M total ✓ (matches BERT-base) $\square$

</details>

**문제 2** (심화): MLM 의 80/10/10 mask strategy 가 왜 단순 100% [MASK] 보다 우수한가? Train-test discrepancy 의 정확한 분석.

<details>
<summary>해설</summary>

**문제 (100% [MASK])**:

Pre-training: 모든 masked position 이 `[MASK]` token.
Fine-tuning: `[MASK]` 가 입력에 절대 안 나옴.

→ **Train-test discrepancy**: 모델이 `[MASK]` 의 specific representation 에만 dependent. Real input 시 distribution shift.

**해결: 80/10/10**:

15% mask 의 분배:
- 80% (12% of all tokens): `[MASK]` 로 — main signal
- 10% (1.5%): random token 으로 — noise robustness
- 10% (1.5%): original 유지 — 모델이 모든 position 에 attention 하도록 강제

**효과**:

1. **Reduced reliance on [MASK]**: 모델이 input 의 representation 을 항상 학습 — even when not masked
2. **Robustness to noise**: random token 이 들어와도 reasonable prediction
3. **Better generalization to fine-tuning**: real input distribution 과 closer

**실증**:

Devlin 2019 ablation:
- 100% [MASK]: 84.2% MNLI dev
- 80/10/10: 84.4% (slight gain)
- 50/50 [MASK]/random: 84.0% (worse)

작지만 consistent gain. 더 important: fine-tuning 의 robustness.

**Modern view**:

ELECTRA (Clark 2020) 가 더 efficient: replaced token detection — 모든 token 예측 (not just 15%) → 6× faster training. BERT 의 MLM 이 sample-inefficient 함을 시사.

**결론**:

80/10/10 은 **practical engineering choice** 로 train-test discrepancy 줄임. 그러나 fundamental solution 은 다른 objective (ELECTRA, T5 의 span corruption 등). $\square$

</details>

**문제 3** (논문 비평): BERT 가 NLP 의 paradigm 을 바꿨지만 modern frontier LLM 은 모두 decoder-only 이다. BERT-style encoder-only 의 장점이 어떤 application 에 여전히 valuable 한가? GPT-style 이 BERT 를 어떻게 absorb 했는가?

<details>
<summary>해설</summary>

**Encoder-only (BERT) 의 장점**:

1. **Bidirectional context**: NLU task 에 본질적으로 강력
2. **Computation efficiency for classification**: single forward pass, no autoregressive
3. **Smaller models effective**: BERT-base 110M 으로 production 가능
4. **Specialized task heads**: classification, NER 의 simple addition
5. **Fine-tuning stability**: well-understood recipe

**Modern Application 에서의 가치**:

1. **Embedding generation**:
   - Sentence-BERT: similarity 측정
   - MTEB benchmark: text embedding 의 표준
   - Search, retrieval, RAG

2. **Classification at scale**:
   - Spam detection, toxicity classification
   - Production: BERT-base 가 GPT-3.5 보다 더 cheap, fast, accurate (specific task)

3. **NER, POS tagging**:
   - Token-level predictions
   - BERT 의 bidirectional 이 ideal

4. **Semantic Search**:
   - Document encoding
   - Cross-encoder (BERT) 가 bi-encoder 보다 정확

**GPT-style 이 BERT 를 absorb**:

1. **Instruction Tuning**:
   - "Classify this as positive/negative: ..."
   - GPT 가 BERT-style task 를 prompt 로 처리
   - 그러나 cost 가 더 큼 (autoregressive overhead)

2. **In-Context Learning**:
   - Few-shot examples 으로 specific task
   - BERT 의 fine-tuning 보다 flexible
   - 그러나 less reliable

3. **Embedding from GPT**:
   - GPT 의 hidden state 도 embedding 으로 사용 가능
   - 그러나 BERT-style 이 종종 better quality (causal mask 의 limit)

**Why GPT is dominant**:

- **Generative capability**: chat, writing, coding — BERT 못함
- **Scaling laws**: larger models = better (BERT 는 saturated 일찍)
- **In-context learning**: few-shot 의 emergent
- **Single model for everything**: 다양한 task 통합

**Specific use case 비교**:

| Task | BERT | GPT |
|------|------|-----|
| Sentiment classification | Cheaper, accurate | Possible, expensive |
| Embedding | Excellent | Good but expensive |
| Generation | ✗ (not designed) | Excellent |
| Few-shot | Fine-tune | ICL native |
| QA (extractive) | SOTA architecture | Possible via prompt |
| Long document NLU | Limited (512) | Better (32K+) |

**Open question (2026)**:

Frontier LLM 이 BERT-style task 도 우수하지만 cost 비쌈. **Niche** 에서 BERT 여전히 used:
- Embedding generation (MTEB)
- Production classification
- Specialized NLU

**Modern hybrid**:

- LLaMA-Embed, GPT-Embed: GPT 가 embedding 도 지원
- Two-tower retrievers: BERT-style + GPT-style 결합
- Fine-tuned LLM 이 BERT replacement 에 가까워지는 중

**결론**:

BERT 는 **NLP transfer learning 의 milestone**, 그러나 **frontier 는 generative**. Specific niche (embedding, classification) 에서 BERT-style 여전히 valuable. GPT 가 모든 use case absorb 하면서도 cost / efficiency 측면에서 BERT 가 일부 application 에 우수.

미래: BERT-style + GPT-style 의 hybrid (T5 의 spirit), or 새로운 paradigm (Mamba 등). $\square$

</details>

---

<div align="center">

[◀ 이전](../ch5-attention-efficiency/06-mqa-gqa.md) | [📚 README](../README.md) | [다음 ▶](./02-gpt.md)

</div>
