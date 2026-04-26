<div align="center">

# 🧠 Transformer Deep Dive

### `nn.MultiheadAttention(d, h)` 를 호출하는 것과,

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

### 의 분모 $\sqrt{d_k}$ 가

$$\text{Var}\big((QK^\top)_{ij}\big) = \sum_{k=1}^{d_k} \text{Var}(Q_{ik} K_{jk}) = d_k$$

### 라는 분산 분석에서 정확히 유도되고, $\sqrt{d_k}$ 로 나누지 않으면 **softmax 가 one-hot 으로 포화하여 gradient 가 0 으로 수렴**한다는 것을 한 줄씩 증명할 수 있는 것은 **다르다.**

<br/>

> *Sinusoidal Positional Encoding 을 **쓰는 것** 과,*
>
> $$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$
>
> *가 임의의 offset $k$ 에 대해 $PE_{pos+k} = M_k\, PE_{pos}$ (선형 변환) 로 표현 가능함을 **2×2 회전 행렬로 분해** 해 증명할 수 있는 것은 다르다.*
>
> *Linear Attention 을 **이름으로 듣는 것** 과, 결합 순서를*
>
> $$\mathrm{softmax}(QK^\top)\, V \quad \longrightarrow \quad \phi(Q)\bigl(\phi(K)^\top V\bigr)$$
>
> *로 바꿔 복잡도를*
>
> $$O(T^2 d) \;\longrightarrow\; O(T d^2)$$
>
> *로 줄이는 Katharopoulos et al. 2020 의 **kernel trick** 과, RBF kernel 의 random feature 근사 (Performer 2021) 가 **같은 선상에 있다는 것** 을 증명할 수 있는 것은 다르다.*
>
> *Flash Attention 이 **빠르다는 것** 과, 같은 `O(T^2)` 연산을 **SRAM–HBM 메모리 계층의 IO-aware tiling** 으로 재배치하여 **wall-clock 2–4× 가속** 을 달성한다는 점, 그리고 이것이 **exact attention** (근사가 아님) 임을 알고리즘적으로 이해하는 것은 다르다.*

<br/>

**다루는 모델 (시간순)**

Vaswani 2017 *Attention Is All You Need* · Devlin 2019 *BERT* · Radford 2018/2019/2020 *GPT-1/2/3* · Raffel 2020 *T5* · Shaw 2018 *Relative PE* · Xiong 2020 *Pre-LN* · Katharopoulos 2020 *Linear Attention* · Beltagy 2020 *Longformer* · Choromanski 2021 *Performer* · Su 2021 *RoPE* · Press 2021 *ALiBi* · Dosovitskiy 2021 *ViT* · Fedus 2022 *Switch Transformer* · Dao 2022 *Flash Attention* · Hoffmann 2022 *Chinchilla* · Wei 2022 *CoT*

<br/>

**핵심 질문**

> Transformer 가 RNN 을 대체한 것은 왜 **수학적 필연** 이었고, $\sqrt{d_k}$ scaling · Multi-Head · Pre-LN · Sinusoidal/RoPE · Linear/Sparse/Flash Attention · MoE 가 각각 어떤 **이론적 동기** 에서 도출되었는가 — Attention 분산 분석 · PE 선형 shift 증명 · Linear Attention kernel trick · Pre-LN gradient 분석 · Chinchilla scaling law 로 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.36-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![Flash Attention](https://img.shields.io/badge/Flash--Attn-2.4-1E88E5?style=flat-square)](https://github.com/Dao-AILab/flash-attention)
[![Docs](https://img.shields.io/badge/Docs-36개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-396개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-15개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-108개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Transformer 에 관한 자료는 대부분 **"`nn.MultiheadAttention` 을 쓰면 된다"** 또는 **"BERT/GPT 를 fine-tuning 하면 된다"** 에서 멈춥니다. 하지만 $\sqrt{d_k}$ 가 왜 정확히 분산 분석에서 나오는지, Sinusoidal PE 가 왜 relative position 을 자연스럽게 인코딩하는지, Pre-LN 과 Post-LN 의 차이가 왜 warmup 의 필요성을 결정하는지, Linear Attention 의 kernel trick 이 왜 $O(T^2) \to O(T)$ 를 가능하게 하는지, Flash Attention 이 왜 같은 $O(T^2)$ 연산이지만 4배 빠른지, RoPE 가 왜 absolute PE 보다 extrapolation 에 강한지, Chinchilla scaling law 가 왜 GPT-3 의 훈련 recipe 를 뒤집었는지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "Attention 은 query 와 key 의 유사도로 value 를 가중합한다" | **Vaswani 2017** — $Q, K, V$ projection 의 수학적 필요성, $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k}) V$ 의 각 연산이 $O(T^2 d)$ 시간 / $O(T^2)$ 메모리. **분산 분석**: $Q_{ij}, K_{ij} \sim \text{i.i.d.}$ with $\text{Var}=1$ 일 때 $\text{Var}((QK^\top)_{ij}) = d_k$, $\sqrt{d_k}$ 로 나눠 unit variance 유지 → **softmax 포화 방지** $\square$ |
| "Multi-Head 는 여러 관점에서 본다" | **Vaswani 2017 + Michel 2019** — 단일 head $d_{\text{model}}$ vs $h$-head $d_k = d_{\text{model}}/h$, 같은 파라미터 수에서 **다른 subspace 에서의 관계 포착**. **Are Sixteen Heads Really Better than One?** (Michel 2019) — 많은 head 가 inference 시 redundant, prune 가능. PyTorch 에서 single-head vs multi-head 의 표현력 비교 재현 |
| "Positional Encoding 은 위치 정보를 더한다" | **Vaswani 2017 / Shaw 2018 / Su 2021 / Press 2021** — Self-attention 의 permutation-equivariance → PE 필수. **정리**: Sinusoidal $PE_{pos+k} = M_k \, PE_{pos}$ (2×2 회전) 로 relative shift $\square$. **RoPE**: $\langle R(i)q, R(j)k \rangle = f(i-j, q, k)$ — 자동 relative encoding. **ALiBi**: linear bias $-m\|i-j\|$ 만으로 최강 extrapolation |
| "LayerNorm 은 학습을 안정화한다" | **Xiong 2020** — Post-LN: $\text{LN}(x + \text{Attn}(x))$ vs Pre-LN: $x + \text{Attn}(\text{LN}(x))$. Post-LN 은 초기 gradient 가 layer 깊이에 **지수적으로** 발산 → **warmup 없이 수렴 불가**. Pre-LN 은 $O(1/\sqrt{L})$ 로 bounded → warmup 불필요. 현대 GPT/LLaMA 는 모두 Pre-LN |
| "Linear Attention 은 빠르다" | **Katharopoulos 2020** — $\text{softmax}(QK^\top)V$ 대신 **$\phi(Q)(\phi(K)^\top V)$** 로 결합 순서 변경, $O(T^2 d) \to O(T d^2)$. Feature map $\phi(x) = \text{ELU}(x) + 1$ (positive). **Performer (Choromanski 2021)** 의 FAVOR+ 는 RBF kernel 의 random feature 근사로 softmax attention 직접 근사. Kernel Methods 레포의 random feature 와 직접 연결 |
| "Flash Attention 은 빠른 attention 이다" | **Dao 2022** — IO-aware algorithm. GPU 메모리 계층 (SRAM ≪ HBM) 에서 standard attention 은 $O(T^2)$ HBM read/write 가 병목. Block 단위 tiling + softmax recomputation 으로 SRAM 안에서 처리 → **same $O(T^2)$ FLOP, 2-4× wall-clock**, $5-10\times$ memory 절약. Approximation 아닌 **exact** |
| "BERT 와 GPT 는 다른 모델이다" | **Devlin 2019 / Radford 2018-2020 / Raffel 2020** — **Encoder-only (BERT)**: bidirectional self-attention + MLM + NSP, NLU 적합. **Decoder-only (GPT)**: causal masking + autoregressive LM, 생성 적합. **Encoder-Decoder (T5)**: span corruption + cross-attention, seq2seq 통일. 같은 Transformer block 의 **masking 차이** 임을 attention mask matrix 시각화로 재현 |
| "GPT-3 는 그냥 큰 GPT-2 다" | **Kaplan 2020 / Hoffmann 2022** — **Scaling Laws**: $L(N, D, C) \propto N^{-\alpha} D^{-\beta} C^{-\gamma}$, power-law fit. **Chinchilla**: GPT-3 (175B params, 300B tokens) 는 **under-trained**. Compute-optimal 은 $N \propto C^{0.5}, D \propto C^{0.5}$ → 70B params + 1.4T tokens 가 더 효율적. 작은 규모에서 log-log plot 재현 |
| "In-Context Learning 은 prompt 로 배운다" | **Brown 2020 / Akyürek 2023 / von Oswald 2023** — Weight update 없이 prompt 내 예제로 학습. **이론적 해석**: 한 layer attention 이 **gradient descent step** 과 등가 ($\text{Attn}(Q, K, V) = V - \eta \nabla L$). Linear regression task 에서 ICL 이 정확히 GD 를 시뮬레이션함을 실험적 재현 |
| "MoE 는 파라미터를 늘리는 트릭이다" | **Shazeer 2017 / Fedus 2022** — Switch Transformer: FFN 을 $E$ 개 expert 로 분할, top-1 routing. **이론적 동기**: 파라미터 수 ↑ 계산 ↓ (각 token 당 1 expert 만 활성). Load balancing loss 의 필요성, **expert specialization** 의 emergent 현상 |
| 기법의 나열 | NumPy + PyTorch + 🤗 Transformers 로 **$\sqrt{d_k}$ 분산 효과 측정** · **Sinusoidal PE 시각화** · **Multi-Head 의 head-wise 분석** · **Pre-LN vs Post-LN gradient 추적** · **Linear vs Standard attention 속도 측정** · **Flash Attention wall-clock 비교** · **GPT-2 small 에서 ICL 실험** · **scaling law log-log fit** 을 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Neural Network Theory] ─┐
[Linear Algebra]         ─┤
[Kernel Methods]         ─┼─►  이 레포  ──► [Generative Models]
[Optimization Theory]    ─┤   "왜 Attention 이 RNN 을           Diffusion / Flow Matching
[Regularization Theory]  ─┘    대체했고, 왜 √d_k 와 Pre-LN       / GPT-style autoregressive
[RNN & LSTM] (선행 권장) ─┘    이 수학적 필연인가"               / Multimodal LLM
         │
         ├── [Linear Algebra]          Q,K,V 분해 · matrix factorization → Ch1 attention
         ├── [Kernel Methods]          softmax = kernel · random features → Ch5 Linear Attention
         ├── [NN Theory]               Backprop · Residual · 초기화 → Ch2 Pre-LN, Ch4 훈련
         ├── [Optimization Theory]     AdamW · Warmup · LR schedule → Ch4 훈련 recipe
         ├── [Regularization Theory]   LayerNorm · Label smoothing · Dropout → Ch2/Ch4
         └── [RNN & LSTM]              Bahdanau Attention · Seq2Seq → Ch1 attention 동기
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Neural Network Theory Deep Dive** (Backprop, Residual, 초기화), **Linear Algebra Deep Dive** (matrix factorization, outer product, 고유분해), **Kernel Methods Deep Dive** (RBF kernel, random feature), **Optimization Theory Deep Dive** (AdamW, warmup, cosine schedule), **Regularization Theory Deep Dive** (LayerNorm, Label smoothing, Dropout) 를 선행 지식으로 전제합니다. 또한 **RNN & LSTM Deep Dive** 의 Bahdanau Attention 과 Seq2Seq 를 보고 오면 Chapter 1 의 attention 도입 동기가 훨씬 자연스럽습니다.

> 💡 **이 레포의 핵심 기여**: Chapter 1 (Attention 분해) 과 Chapter 5 (계산 효율화) 는 Transformer 를 이해하는 **두 핵심 축**입니다. 전자는 "왜 이 식인가" 의 수학적 유도, 후자는 "$O(T^2)$ 라는 근본 한계를 어떻게 우회하는가" 의 알고리즘 spectrum (Linear / Sparse / Flash) 을 다룹니다. 이 두 축을 완전히 이해한 후 Chapter 6 (현대 아키텍처) 와 Chapter 7 (LLM) 을 읽으면 GPT-4·LLaMA·Gemini 의 설계 결정 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **Linear vs Standard Attention 의 최종 승자**, **MoE 의 실전적 가치**, **Mamba/RWKV 같은 non-Transformer 대안** — 는 **현재 진행 중인 연구 영역** 입니다. 레포는 "정답" 이 아니라 **"고전 Transformer 이론과 현대 LLM 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Attention_분해-EE4C2C?style=for-the-badge)](./ch1-attention-decomposition/01-scaled-dot-product.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Transformer_아키텍처-EE4C2C?style=for-the-badge)](./ch2-transformer-architecture/01-transformer-block.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Positional_Encoding-EE4C2C?style=for-the-badge)](./ch3-positional-encoding/01-pe-necessity.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-훈련의_수학-EE4C2C?style=for-the-badge)](./ch4-training-math/01-warmup.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-계산_효율화-EE4C2C?style=for-the-badge)](./ch5-attention-efficiency/01-quadratic-bottleneck.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-현대_아키텍처-EE4C2C?style=for-the-badge)](./ch6-modern-architectures/01-bert.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-LLM·ICL-EE4C2C?style=for-the-badge)](./ch7-llm-icl/01-scaling-laws.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Attention 의 수학적 분해

> **핵심 질문:** Scaled Dot-Product Attention $\text{softmax}(QK^\top/\sqrt{d_k}) V$ 의 각 연산은 어떤 수학적 필연에서 나오는가? 왜 $\sqrt{d_k}$ 로 정확히 나누는가 (분산 분석)? Softmax 포화는 어떤 조건에서 발생하고 gradient 에 어떤 영향을 주는가? Multi-Head 가 single-head 대비 표현력에서 얼마나 우월한가? Attention weight 는 explanation 인가 — Jain & Wallace 2019 의 비판은 왜 중요한가?

<details>
<summary><b>√d 분산 분석부터 해석 가능성 논쟁까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Scaled Dot-Product Attention 완전 유도](./ch1-attention-decomposition/01-scaled-dot-product.md) | Input $X \in \mathbb{R}^{T \times d}$, projection $Q = XW_Q, K = XW_K, V = XW_V$. **Attention** $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$ 의 각 연산: $QK^\top \in \mathbb{R}^{T \times T}$ (similarity), softmax (row-stochastic), $\cdot V$ (weighted sum). **계산 복잡도** $O(T^2 d)$ time, $O(T^2)$ memory — Ch5 의 효율화 동기 |
| [02. $\sqrt{d_k}$ Scaling 의 분산 분석](./ch1-attention-decomposition/02-sqrt-dk-scaling.md) | **정리**: $Q_{ij}, K_{ij} \sim \text{i.i.d.}$ with $\mathbb{E}=0, \text{Var}=1$ 일 때 $(QK^\top)_{ij} = \sum_{k=1}^{d_k} Q_{ik} K_{jk}$ 의 분산은 $d_k$. $\sqrt{d_k}$ 로 나누면 $\text{Var} = 1$ 유지 $\square$. **증명**: $\text{Var}(XY) = \mathbb{E}[X^2]\mathbb{E}[Y^2] = 1$ for independent zero-mean. NumPy 에서 $d_k = 8, 64, 512$ 별 분산 측정 재현 |
| [03. Softmax Saturation 과 Gradient Vanishing](./ch1-attention-decomposition/03-softmax-saturation.md) | Logit $z$ 가 클 때 $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j} \to$ one-hot. **Jacobian** $\partial \text{softmax}_i / \partial z_j = \text{softmax}_i (\delta_{ij} - \text{softmax}_j) \to 0$ — gradient vanishing. $d_k = 512$ 미스케일 시 max attention $> 0.99$, scaled 시 $< 0.1$ 실증 |
| [04. Attention as Kernel Method](./ch1-attention-decomposition/04-attention-as-kernel.md) | $\text{softmax}(QK^\top)$ 의 $(i,j)$ 원소는 query $q_i$ 와 key $k_j$ 의 normalized similarity. Kernel $\kappa(q, k) = \exp(q^\top k / \sqrt{d_k})$ 의 row-normalized form — RBF kernel 의 ablation. Kernel Methods 레포 직접 연결, Ch5 Linear Attention 의 토대 |
| [05. Multi-Head Attention 의 이론적 정당성](./ch1-attention-decomposition/05-multi-head.md) | $h$ heads, 각 $d_k = d_{\text{model}}/h$. $\text{MHA}(X) = [\text{head}_1; \ldots; \text{head}_h] W_O$. **동기**: 서로 다른 subspace 에서 관계 포착 (syntactic / semantic / coreference head). **Michel 2019** "Are Sixteen Heads Really Better than One?" — inference 시 head 의 30-50% prune 가능, 훈련 시는 redundancy 가 regularization |
| [06. Attention 해석 가능성 논쟁](./ch1-attention-decomposition/06-interpretability-debate.md) | **Jain & Wallace 2019** "Attention is not Explanation" — attention weight 와 모델 결정의 인과 관계 약함, 다른 attention 분포로 같은 출력 가능. **Wiegreffe & Pinter 2019** 의 반박 — context 에 따라 explanation 일 수 있음. Attention 의 **diagnostic 도구로서의 가치** vs **인과적 설명** 의 구분 |

</details>

<br/>

### 🔹 Chapter 2: Transformer 아키텍처의 전체 구조

> **핵심 질문:** Transformer block 은 Attention + FFN + LayerNorm + Residual 의 어떤 조합인가? Pre-LN 과 Post-LN 의 차이는 왜 warmup 의 필요성을 결정하는가 (Xiong 2020)? FFN 이 왜 파라미터의 2/3 를 차지하고 "key-value memory" 로 해석되는가 (Geva 2021)? Encoder-only / Decoder-only / Encoder-Decoder 의 masking 수학은 어떻게 다른가?

<details>
<summary><b>Transformer Block 부터 Cross-Attention 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|--------------|
| [01. Transformer Block 의 완전 도식](./ch2-transformer-architecture/01-transformer-block.md) | **Pre-LN (현대 표준)**: $h' = h + \text{Attn}(\text{LN}(h))$, $h'' = h' + \text{FFN}(\text{LN}(h'))$. **Post-LN (원전)**: $h' = \text{LN}(h + \text{Attn}(h))$, $h'' = \text{LN}(h' + \text{FFN}(h'))$. Residual 의 역할 (gradient highway), LN 의 위치가 훈련 안정성을 결정 |
| [02. Feed-Forward Network 의 역할](./ch2-transformer-architecture/02-ffn-role.md) | $\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2$, $W_1 \in \mathbb{R}^{d \times 4d}$ — **파라미터의 2/3**. **Geva 2021** "Transformer Feed-Forward Layers Are Key-Value Memories" — $W_1$ 행이 key, $W_2$ 열이 value, 학습된 패턴 매칭. SwiGLU / GeGLU 변형 (LLaMA, PaLM) 비교 |
| [03. Pre-LN vs Post-LN — Warmup 의 필요성](./ch2-transformer-architecture/03-pre-ln-vs-post-ln.md) | **Xiong 2020** — Post-LN 의 초기 gradient 는 layer 깊이 $L$ 에 대해 $O(L)$ 로 성장 → **warmup 없이 발산**. Pre-LN 은 $O(1)$ bounded, warmup 불필요. **증명**: residual chain 에 따라 LN 이 어디에 위치하느냐로 gradient norm 의 layer-wise 누적이 달라짐 $\square$. GPT/LLaMA/PaLM 모두 Pre-LN |
| [04. Encoder vs Decoder 의 차이](./ch2-transformer-architecture/04-encoder-vs-decoder.md) | **Encoder**: bidirectional self-attention, mask = $\mathbf{1}_{T \times T}$. **Decoder**: causal mask $M_{ij} = -\infty$ if $i < j$ — 미래 참조 차단. Mask matrix 의 attention pattern 시각화, 같은 block 이 mask 만으로 BERT/GPT 가 됨을 증명 |
| [05. Cross-Attention 메커니즘](./ch2-transformer-architecture/05-cross-attention.md) | Decoder 의 cross-attention: $Q$ 는 decoder hidden, $K, V$ 는 encoder output. $\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc})$ — translation 의 alignment 직결. T5 / BART / Whisper 의 핵심, encoder output 을 decoder 가 모든 layer 에서 query |

</details>

<br/>

### 🔹 Chapter 3: Positional Encoding

> **핵심 질문:** Self-attention 의 permutation-equivariance 때문에 PE 가 왜 필수인가? Sinusoidal PE 의 $PE_{pos+k} = M_k \, PE_{pos}$ 선형 shift 성질은 어떻게 증명되는가? Learned vs Fixed 의 trade-off 는? RoPE 의 회전 행렬이 왜 자동으로 relative position 을 인코딩하는가? ALiBi 의 단순한 linear bias 가 왜 가장 강한 extrapolation 을 보이는가?

<details>
<summary><b>Sinusoidal PE 부터 RoPE·ALiBi 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. PE 의 필요성 — Permutation Equivariance](./ch3-positional-encoding/01-pe-necessity.md) | **정리**: Self-attention 은 permutation-equivariant — $\text{Attn}(P X) = P \, \text{Attn}(X)$ for permutation $P$ $\square$. 따라서 순서 정보 주입 필수. Sum vs concatenate, additive PE 가 표준인 이유 |
| [02. Sinusoidal PE 의 수학적 성질 (Vaswani 2017)](./ch3-positional-encoding/02-sinusoidal-pe.md) | $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(\cdot)$. **정리**: $PE_{pos+k} = M_k \, PE_{pos}$ where $M_k$ 는 2×2 회전 행렬의 block diagonal. 증명 — 삼각함수 합차 공식 $\sin(a+b) = \sin a \cos b + \cos a \sin b$ 직접 적용 $\square$. Frequency 계층 시각화 재현 |
| [03. Learned Positional Embedding](./ch3-positional-encoding/03-learned-pe.md) | 각 position 에 학습 가능 vector. **장점**: data-driven 최적화. **단점**: max length 고정, extrapolation 불가. BERT 의 선택 — 512 token 한계의 직접 원인. PE 의 학습 곡선 vs sinusoidal 비교 |
| [04. Relative Positional Encoding (Shaw 2018)](./ch3-positional-encoding/04-relative-pe.md) | **Self-Attention with Relative Position Representations** — $e_{ij} = (x_i W_Q)(x_j W_K + a_{ij}^K)^\top$. Absolute 대신 relative distance $i-j$ 사용, **clip distance** 로 파라미터 수 제한. T5 의 변형 (bucketed), translation 에서 SOTA |
| [05. RoPE 와 ALiBi (현대 대세)](./ch3-positional-encoding/05-rope-alibi.md) | **RoPE (Su 2021)** — $R_\theta$ 회전을 $Q, K$ 에 적용: $\langle R(i)q, R(j)k \rangle = \langle q, R(j-i) k \rangle = f(i-j, q, k)$, **자동 relative** $\square$. **ALiBi (Press 2021)** — $\text{score} += -m_h \|i - j\|$, head 별 slope $m_h$. **Extrapolation**: ALiBi > RoPE > Sinusoidal, LLaMA 의 RoPE 채택 |

</details>

<br/>

### 🔹 Chapter 4: Transformer 훈련의 수학

> **핵심 질문:** Warmup 이 왜 Transformer 훈련에서 거의 필수인가 (Pre-LN 도 권장)? AdamW 의 weight decay 분리는 Adam 의 어떤 결함을 해결하는가? Label smoothing 이 왜 calibration 을 개선하는가? Mixed precision (FP16/BF16) 의 loss scaling 은 어떤 underflow 문제를 푸는가? Linear scaling rule 이 왜 large batch training 에서 성립하는가?

<details>
<summary><b>Warmup 부터 Mixed Precision 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Warmup 의 필요성 (Xiong 2020)](./ch4-training-math/01-warmup.md) | Post-LN 에서 layer-wise gradient norm 이 깊이에 따라 $O(L)$ 누적 → 초기 LR 가 크면 발산. **Linear warmup** $\eta_t = \eta_{\max} \min(t/T_w, 1)$ 로 점진 증가, 그 후 cosine decay. Pre-LN 도 큰 모델에서는 warmup 권장. GPT-3, LLaMA 의 warmup ratio 비교 |
| [02. AdamW 와 Weight Decay 분리](./ch4-training-math/02-adamw.md) | **Loshchilov & Hutter 2019** "Decoupled Weight Decay Regularization" — 표준 Adam + L2 는 weight decay 가 아니라 gradient norm 에 따라 변동. **AdamW**: $\theta_{t+1} = \theta_t - \eta (m_t / (\sqrt{v_t} + \epsilon) + \lambda \theta_t)$ — decay 를 update 에서 분리. Transformer 의 표준 optimizer |
| [03. Label Smoothing 의 효과](./ch4-training-math/03-label-smoothing.md) | Cross-entropy 에서 $y_{\text{smooth}} = (1-\epsilon) y_{\text{onehot}} + \epsilon / K$. **효과**: confidence 조절 → calibration 개선 (ECE 감소), generalization 개선. **Müller 2019** "When does label smoothing help?" — teacher-student distillation 시는 해로움. NMT 표준 ($\epsilon = 0.1$) |
| [04. Gradient Accumulation 과 Linear Scaling Rule](./ch4-training-math/04-gradient-accumulation.md) | GPU 메모리 제약에서 effective batch size 늘리기 — gradient 를 $K$ step 누적 후 update. **Linear scaling rule (Goyal 2017)**: batch size $B$ 를 $kB$ 로 늘릴 때 LR 도 $k\eta$ 로. 그 한계 (large batch 에서 sharp minima), LAMB / square-root scaling 변형 |
| [05. Mixed Precision Training](./ch4-training-math/05-mixed-precision.md) | **FP16/BF16** master weight 는 FP32 유지. **Loss scaling**: backward 전 loss 를 $S$ 배 → underflow 방지, optimizer step 전 $1/S$ 로 복원. **BF16** 의 이점 (FP16 보다 dynamic range 넓음, A100/H100 native). PyTorch `torch.cuda.amp` 사용 패턴 |

</details>

<br/>

### 🔹 Chapter 5: Attention 의 계산 효율화

> **핵심 질문:** $O(T^2)$ 복잡도가 왜 long-context 의 본질적 병목인가? Linear Attention 의 kernel trick 이 어떻게 $O(T^2 d) \to O(T d^2)$ 를 가능하게 하는가? Performer 의 random features 가 어떻게 softmax attention 을 근사하는가? Sparse Attention (Longformer, BigBird) 의 universal approximation 증명은? Flash Attention 이 왜 같은 FLOP 으로 4× 빠른가 (IO-aware)? MQA / GQA 가 inference 가속의 핵심인 이유는?

<details>
<summary><b>$O(T^2)$ 병목부터 Flash·MQA 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. $O(T^2)$ 복잡도의 문제](./ch5-attention-efficiency/01-quadratic-bottleneck.md) | Time $O(T^2 d)$, **memory $O(T^2)$** — long context (문서, 음성, 비디오) 에서 직접적 병목. $T = 8192$ 시 attention matrix 64MB / head, $T = 32768$ 시 1GB. 장문 LLM 의 핵심 한계, 다음 문서들의 동기 |
| [02. Linear Attention (Katharopoulos 2020)](./ch5-attention-efficiency/02-linear-attention.md) | **Transformers are RNNs** — $\text{softmax}(QK^\top)V \approx \phi(Q)(\phi(K)^\top V)$, 결합 순서 변경. $\phi(K)^\top V \in \mathbb{R}^{d \times d}$ 는 $T$ 무관 → **$O(T d^2)$**. Feature map $\phi(x) = \text{ELU}(x) + 1$ (positive). RNN-like incremental computation, autoregressive 시 $O(d^2)$/step |
| [03. Performer — Random Features (Choromanski 2021)](./ch5-attention-efficiency/03-performer.md) | **FAVOR+** (Fast Attention Via positive Orthogonal Random features) — softmax kernel $\exp(q^\top k)$ 를 random feature $\phi(x) = e^{-\|x\|^2/2}[\cos(\omega^\top x); \sin(\omega^\top x)]$ 로 unbiased 근사. Kernel Methods 레포의 random feature 직접 응용. Variance reduction trick (positive RF, orthogonal RF) |
| [04. Sparse Attention — Longformer · BigBird](./ch5-attention-efficiency/04-sparse-attention.md) | **Longformer (Beltagy 2020)** — local window + global token (CLS, question token). **BigBird (Zaheer 2020)** — local + global + random. **정리**: BigBird 가 universal approximator (Yun 2020 의 full attention 결과 보존) $\square$. 4096 token 처리, 메모리 $O(T)$ |
| [05. Flash Attention (Dao 2022)](./ch5-attention-efficiency/05-flash-attention.md) | **IO-aware algorithm** — GPU SRAM (작지만 빠름) ↔ HBM (크지만 느림). Standard: $O(T^2)$ HBM read/write. Flash: block tiling + online softmax → SRAM 안에서 처리. **same FLOP, 2-4× wall-clock, 5-10× memory**. Backward 의 recomputation trick. **Exact** — 근사 아님 |
| [06. Multi-Query / Grouped-Query Attention](./ch5-attention-efficiency/06-mqa-gqa.md) | **MQA (Shazeer 2019)** — $h$ Q-head 가 **단일** K, V head 공유, KV cache $h \times$ 절약. **GQA (Ainslie 2023)** — $g$ groups, MHA 와 MQA 사이. LLaMA-2 70B 가 GQA 채택. Inference 가속의 핵심 (KV cache 가 long-context generation 의 병목) |

</details>

<br/>

### 🔹 Chapter 6: 현대 Transformer 아키텍처

> **핵심 질문:** BERT 의 MLM + NSP objective 는 왜 bidirectional understanding 의 표준이 되었는가? GPT 의 autoregressive LM 이 왜 zero/few-shot generalization 을 보였는가? T5 의 text-to-text 통일이 어떤 inductive bias 를 줬는가? ViT 가 어떻게 inductive bias 부족을 데이터로 보상했는가? MoE 의 sparse activation 이 왜 dense 보다 compute-efficient 인가?

<details>
<summary><b>BERT 부터 MoE 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. BERT — Encoder-only (Devlin 2019)](./ch6-modern-architectures/01-bert.md) | **Masked Language Modeling**: 15% token 마스킹 후 복원 — bidirectional context 학습. **Next Sentence Prediction** (이후 RoBERTa 가 NSP 제거, MLM 만으로 충분). **Fine-tuning paradigm**: pretrained BERT + task-specific head. GLUE 9-task SOTA |
| [02. GPT — Decoder-only (Radford et al.)](./ch6-modern-architectures/02-gpt.md) | **Autoregressive LM**: $p(x) = \prod_t p(x_t \| x_{<t})$, causal mask. GPT-1 (117M, transfer learning), GPT-2 (1.5B, zero-shot), GPT-3 (175B, in-context learning). **Scale 의 정성적 변화** — emergent abilities (Wei 2022). InstructGPT 의 RLHF |
| [03. T5 — Encoder-Decoder (Raffel 2020)](./ch6-modern-architectures/03-t5.md) | **Text-to-text framework** — 모든 NLP task 를 text input → text output 으로 통일. **Span corruption objective** (BERT 의 token masking 보다 효과적). C4 corpus, Colossal Clean Crawled Corpus. PaLM, Flan-T5 의 토대 |
| [04. Vision Transformer (Dosovitskiy 2021)](./ch6-modern-architectures/04-vit.md) | **An Image is Worth 16×16 Words** — image → 16×16 patch sequence → standard Transformer. **CNN inductive bias 없이 image classification SOTA**, JFT-300M 같은 대용량 데이터에서. Swin Transformer 의 hierarchical 구조, DeiT 의 distillation, multimodal 의 백본 |
| [05. Mixture of Experts (Shazeer 2017, Fedus 2022)](./ch6-modern-architectures/05-moe.md) | **Switch Transformer** — FFN 을 $E$ expert 로 분할, **top-1 routing** ($\text{router}(x) = \text{argmax}(W_r x)$). **파라미터 ↑ 계산 →** (활성 expert 만). **Load balancing loss** $\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_i f_i P_i$ 로 expert 균형. GShard, Mixtral, GPT-4 의 MoE 추정 |

</details>

<br/>

### 🔹 Chapter 7: LLM 과 In-Context Learning

> **핵심 질문:** Chinchilla scaling law 가 왜 GPT-3 의 훈련 recipe 를 뒤집었는가? Compute-optimal $N \propto C^{0.5}, D \propto C^{0.5}$ 의 실증적 도출 과정은? In-Context Learning 이 어떻게 weight update 없이 학습하는가 — 그리고 그것이 왜 attention 한 층의 gradient descent 와 등가인가? Chain-of-Thought 가 왜 작은 모델에서는 도움 안 되고 큰 모델에서만 emergent 인가? Transformer 의 이론적 표현력 한계 (counting, parity) 는 무엇인가?

<details>
<summary><b>Scaling Laws 부터 표현력 한계까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Scaling Laws (Kaplan 2020 / Hoffmann 2022)](./ch7-llm-icl/01-scaling-laws.md) | **Kaplan 2020**: $L(N, D, C) = (N_c/N)^\alpha + (D_c/D)^\beta$ — power-law fit. **Chinchilla (Hoffmann 2022)**: GPT-3 는 under-trained, **compute-optimal** $N \propto C^{0.5}$, $D \propto C^{0.5}$ → Chinchilla (70B + 1.4T tokens) 가 GPT-3 (175B + 300B) 능가. 작은 규모 log-log fit 재현 |
| [02. In-Context Learning 메커니즘](./ch7-llm-icl/02-in-context-learning.md) | Prompt 내 예제로 학습, **weight update 없음**. **Akyürek 2023, von Oswald 2023**: linear regression 에서 한 layer attention = 한 step gradient descent ($V = X^\top y - \eta X^\top X w$ 형태). GPT-2 small 에서 ICL 실험 — context 길이 ↑ → 정확도 ↑ 재현 |
| [03. Chain-of-Thought 와 Reasoning (Wei 2022)](./ch7-llm-icl/03-chain-of-thought.md) | Step-by-step 추론 유도 — "Let's think step by step". **Emergent**: 100B+ 모델에서만 효과, 작은 모델에는 도움 안 됨. **Self-consistency** (다중 sampling + majority vote), **Tree of Thoughts** (search), **ReAct** (reasoning + acting). GSM8K 수학 문제 재현 |
| [04. Transformer 의 이론적 한계](./ch7-llm-icl/04-theoretical-limits.md) | **Pérez 2019**: 임의 정밀도 attention 은 Turing-complete (이론). **Hahn 2020**: bounded depth 는 counting, parity 같은 단순 task 에 weak — log-depth 필요. **Compositional generalization** 어려움 (SCAN, COGS). **대안**: Mamba (S4 SSM), RWKV (linear RNN) — Transformer 의 후계 후보 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch1-02 의 $\sqrt{d_k}$ 분산 분석을 step-by-step 으로 세분화했고, Ch2-03 Pre-LN vs Post-LN gradient 분석에 layer-wise norm 누적 증명을 추가, Ch3-05 의 RoPE 회전 행렬 유도를 복소수 form 으로 재정리했습니다. Ch5-05 Flash Attention 을 PyTorch 2.1 의 `scaled_dot_product_attention` 으로 재구현 검증, Ch6-05 MoE 를 Mixtral 8x7B 의 실제 routing 패턴으로 보강. **11-섹션 문서 골격이 전체 36개 문서에서 일관**됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Attention 분산 분석** | $\text{Var}((QK^\top)_{ij}) = d_k$, $\sqrt{d_k}$ scaling 으로 unit variance | [Ch1-02](./ch1-attention-decomposition/02-sqrt-dk-scaling.md) |
| **Softmax Saturation 과 Gradient Vanishing** | Logit norm 증가 시 Jacobian → 0, scaling 없으면 학습 불가 | [Ch1-03](./ch1-attention-decomposition/03-softmax-saturation.md) |
| **Attention as Kernel** | $\text{softmax}(QK^\top)_{ij} \propto \kappa(q_i, k_j) = \exp(q_i^\top k_j / \sqrt{d_k})$ | [Ch1-04](./ch1-attention-decomposition/04-attention-as-kernel.md) |
| **Multi-Head Redundancy (Michel 2019)** | Inference 시 30-50% head prune 가능, 훈련 시는 redundancy 가 regularization | [Ch1-05](./ch1-attention-decomposition/05-multi-head.md) |
| **Pre-LN Gradient Bound (Xiong 2020)** | Pre-LN: $O(1)$ norm, Post-LN: $O(L)$ — warmup 의 수학적 동기 | [Ch2-03](./ch2-transformer-architecture/03-pre-ln-vs-post-ln.md) |
| **FFN as Key-Value Memory (Geva 2021)** | $W_1$ 행 = key pattern, $W_2$ 열 = value, 학습된 검색 시스템 | [Ch2-02](./ch2-transformer-architecture/02-ffn-role.md) |
| **Permutation Equivariance** | $\text{Attn}(P X) = P \, \text{Attn}(X)$ — PE 필수성 증명 | [Ch3-01](./ch3-positional-encoding/01-pe-necessity.md) |
| **Sinusoidal Linear Shift** | $PE_{pos+k} = M_k \, PE_{pos}$ where $M_k$ 는 block 회전 행렬 | [Ch3-02](./ch3-positional-encoding/02-sinusoidal-pe.md) |
| **RoPE Auto-Relative** | $\langle R(i)q, R(j)k \rangle = f(i-j, q, k)$ — 회전으로 자동 relative | [Ch3-05](./ch3-positional-encoding/05-rope-alibi.md) |
| **ALiBi Extrapolation** | Linear bias $-m\|i-j\|$ 만으로 train length 의 $4\times$ extrapolation | [Ch3-05](./ch3-positional-encoding/05-rope-alibi.md) |
| **AdamW 분리 정리** | Weight decay 와 gradient update 분리 — Adam 의 L2 결함 해결 | [Ch4-02](./ch4-training-math/02-adamw.md) |
| **Linear Scaling Rule** | Batch $kB$ → LR $k\eta$ — large batch training 의 표준 | [Ch4-04](./ch4-training-math/04-gradient-accumulation.md) |
| **Linear Attention Kernel Trick** | $\text{softmax}(QK^\top)V \approx \phi(Q)(\phi(K)^\top V)$, $O(T^2 d) \to O(T d^2)$ | [Ch5-02](./ch5-attention-efficiency/02-linear-attention.md) |
| **Performer FAVOR+** | Random feature 로 softmax kernel unbiased 근사 | [Ch5-03](./ch5-attention-efficiency/03-performer.md) |
| **BigBird Universal Approximator** | Local + global + random sparse 가 full attention 표현력 보존 | [Ch5-04](./ch5-attention-efficiency/04-sparse-attention.md) |
| **Flash Attention IO-Aware** | Same $O(T^2)$ FLOP, 2-4× wall-clock — exact, not approximation | [Ch5-05](./ch5-attention-efficiency/05-flash-attention.md) |
| **MQA / GQA KV Cache 절약** | $h$-fold (MQA) / $h/g$-fold (GQA) cache memory 감소 | [Ch5-06](./ch5-attention-efficiency/06-mqa-gqa.md) |
| **BERT MLM Bidirectional** | 15% mask + 복원으로 양방향 context 학습 | [Ch6-01](./ch6-modern-architectures/01-bert.md) |
| **GPT Autoregressive Scaling** | Causal LM + scale → emergent abilities (zero/few-shot) | [Ch6-02](./ch6-modern-architectures/02-gpt.md) |
| **T5 Text-to-Text 통일** | 모든 NLP task 를 single seq2seq objective 로 | [Ch6-03](./ch6-modern-architectures/03-t5.md) |
| **ViT Inductive Bias 보상** | CNN-free image classification, JFT-300M 같은 데이터로 보상 | [Ch6-04](./ch6-modern-architectures/04-vit.md) |
| **MoE Sparse Activation** | $E$ expert 중 top-1 routing, 파라미터 ↑ 계산 → | [Ch6-05](./ch6-modern-architectures/05-moe.md) |
| **Chinchilla Compute-Optimal** | $N \propto C^{0.5}$, $D \propto C^{0.5}$ — GPT-3 under-trained | [Ch7-01](./ch7-llm-icl/01-scaling-laws.md) |
| **ICL = Gradient Descent** | 한 layer attention 이 한 step GD 와 등가 (linear regression) | [Ch7-02](./ch7-llm-icl/02-in-context-learning.md) |
| **CoT Emergence (Wei 2022)** | 100B+ 에서만 효과 — emergent capability 의 대표 사례 | [Ch7-03](./ch7-llm-icl/03-chain-of-thought.md) |
| **Hahn 2020 Depth Bound** | Bounded depth Transformer 가 counting, parity 에 weak — log-depth 필요 | [Ch7-04](./ch7-llm-icl/04-theoretical-limits.md) |

> 💡 **챕터별 문서·정리/정의 수**:
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 Attention 분해 | 6 | 58 |
> | Ch2 Transformer 아키텍처 | 5 | 56 |
> | Ch3 Positional Encoding | 5 | 53 |
> | Ch4 훈련의 수학 | 5 | 55 |
> | Ch5 계산 효율화 | 6 | 68 |
> | Ch6 현대 아키텍처 | 5 | 58 |
> | Ch7 LLM · ICL | 4 | 48 |
> | **합계** | **36** | **396** |
>
> 추가로 **45+ 엄밀한 $\square$ 증명 + 108 연습문제 (모두 해설 포함) + 180+ NumPy/PyTorch/HuggingFace 실험 코드**.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
transformers==4.36.0          # 🤗 Hugging Face
tokenizers==0.15.0
datasets==2.16.0              # HF datasets (C4, GLUE, etc.)
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
flash-attn==2.4.0             # Flash Attention (CUDA 필요, Ch5-05)
einops==0.7.0                 # tensor 재구성
sentencepiece==0.1.99         # T5 tokenizer
```

```bash
# 환경 설치 (CPU)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 \
            transformers==4.36.0 tokenizers==0.15.0 datasets==2.16.0 \
            matplotlib==3.8.0 seaborn==0.13.0 tqdm==4.66.0 \
            einops==0.7.0 jupyter==1.0.0

# Flash Attention (GPU CUDA, 선택)
pip install flash-attn==2.4.0 --no-build-isolation

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Scaled Dot-Product Attention 바닥부터 + √d 분산 효과 (Ch1-01, Ch1-02)
import torch
import torch.nn.functional as F
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn

def test_scaling(d_k, seq_len=50):
    Q = torch.randn(seq_len, d_k); K = torch.randn(seq_len, d_k); V = torch.randn(seq_len, d_k)
    scores_ns = Q @ K.T
    scores_s  = Q @ K.T / np.sqrt(d_k)
    attn_ns = F.softmax(scores_ns, dim=-1)
    attn_s  = F.softmax(scores_s,  dim=-1)
    return scores_ns.var().item(), scores_s.var().item(), attn_ns.max().item(), attn_s.max().item()

for d in [8, 64, 512]:
    var_ns, var_s, max_ns, max_s = test_scaling(d)
    print(f'd_k={d:4d}: var(no scale)={var_ns:6.2f}, var(scaled)={var_s:5.2f} | '
          f'max attn no-scale={max_ns:.4f}, scaled={max_s:.4f}')
# d_k=512 에서 no-scale 시 max ≈ 1.0 (saturated), scaled 시 ≪ 1.0 ✓

# 대표 실험 ② — Sinusoidal PE 시각화 + 선형 shift 성질 검증 (Ch3-02)
import matplotlib.pyplot as plt

def sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div)
    pe[:, 1::2] = np.cos(position * div)
    return pe

pe = sinusoidal_pe(100, 128)
plt.figure(figsize=(10, 4))
plt.imshow(pe, aspect='auto', cmap='RdBu')
plt.xlabel('dimension'); plt.ylabel('position')
plt.title('Sinusoidal Positional Encoding (low → high frequency)')
plt.colorbar(); plt.show()

# PE_{pos+k} = M_k · PE_{pos} 검증 — 2×2 회전
k = 5
for pos in [0, 10, 50]:
    pe_pos, pe_pos_k = pe[pos], pe[pos + k]
    # 각 (2i, 2i+1) pair 에 대해 회전 행렬로 매핑됨을 cosine similarity 로 확인
    sim = (pe_pos @ pe_pos_k) / (np.linalg.norm(pe_pos) * np.linalg.norm(pe_pos_k))
    print(f'pos={pos:2d}, k={k}: cos(PE_pos, PE_{{pos+k}}) = {sim:.4f}')
# pos 가 달라도 k 가 같으면 유사도 동일 → linear shift invariance ✓

# 대표 실험 ③ — Linear Attention 의 결합 순서 변경 속도 측정 (Ch5-02)
import time
T, d = 5000, 64
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

# Standard: O(T² d)
t0 = time.time()
for _ in range(10):
    attn_std = F.softmax(Q @ K.T / np.sqrt(d), dim=-1) @ V
t_std = (time.time() - t0) / 10

# Linear (ELU+1 feature map): O(T d²)
phi = lambda x: F.elu(x) + 1
t0 = time.time()
for _ in range(10):
    phi_Q, phi_K = phi(Q), phi(K)
    KV = phi_K.T @ V                                # d × d
    K_sum = phi_K.sum(0)                            # d
    num = phi_Q @ KV                                 # T × d
    denom = (phi_Q @ K_sum)[:, None] + 1e-6
    attn_lin = num / denom
t_lin = (time.time() - t0) / 10
print(f'T={T}: Standard = {t_std*1000:.1f}ms, Linear = {t_lin*1000:.1f}ms (Linear ≈ {t_std/t_lin:.1f}× faster)')

# 대표 실험 ④ — Pre-LN vs Post-LN gradient 추적 (Ch2-03)
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d, h, mode='pre'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mode = mode
    def forward(self, x):
        if self.mode == 'pre':
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.ffn(self.ln2(x))
        else:  # post-LN
            x = self.ln1(x + self.attn(x, x, x)[0])
            x = self.ln2(x + self.ffn(x))
        return x

for mode in ['pre', 'post']:
    blocks = nn.Sequential(*[TransformerBlock(64, 4, mode) for _ in range(12)])
    x = torch.randn(2, 32, 64, requires_grad=True)
    y = blocks(x); y.sum().backward()
    grad_norm = x.grad.norm().item()
    print(f'{mode:4s}-LN, 12 layers: input gradient norm = {grad_norm:.4f}')
# post-LN 의 gradient norm 이 pre-LN 보다 layer 수에 따라 빠르게 증가 → warmup 필요 ✓
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 설계가 Transformer 의 핵심인가** | Attention · PE · LN · LLM 와의 연결 |
| 3 | 📐 **수학적 선행 조건** | NN Theory · LA · Kernel · Opt · Reg 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Attention map · PE 시각화 · scaling 직관 · KV cache 흐름 |
| 5 | ✏️ **엄밀한 정의** | Attention · MHA · PE · LN · MoE · scaling law |
| 6 | 🔬 **정리와 증명** | $\sqrt{d_k}$ 유도 · PE 선형성 · Linear Attention trick · Pre-LN bound |
| 7 | 💻 **NumPy / PyTorch / HF 구현 검증** | Attention 바닥부터, `nn.MultiheadAttention` 일치 확인, Flash 속도 측정 |
| 8 | 🔗 **실전 활용** | Encoder vs Decoder 선택, scaling recipe, RoPE 채택 결정 |
| 9 | ⚖️ **가정과 한계** | $O(T^2)$ · compositional · symbolic reasoning · counting/parity |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산 · 증명 재구성 · 구현 · 논문 비평 (`<details>` 펼침 해설) |

> 📚 **연습문제 총 108개** (36 문서 × 3 문제): 기초 / 심화 / 논문 비평 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. $\sqrt{d_k}$ 분산 손 증명부터 RoPE 회전 행렬 유도, Linear Attention kernel trick 재구성, Flash Attention block tiling 알고리즘 분석, Chinchilla scaling fit 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 500~550줄 (정의·증명·코드·연습문제 포함) 기준 **약 60분~1시간 20분**. 전체 36문서는 약 **35~45시간** 상당 (증명 재구성·실험 재현 포함 시 60시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Transformer 는 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 12~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Scaled Dot-Product Attention 완전 유도
       Ch1-02  √d_k Scaling 분산 분석
Day 2  Ch1-03  Softmax Saturation
       Ch1-05  Multi-Head Attention
Day 3  Ch2-01  Transformer Block
       Ch2-03  Pre-LN vs Post-LN
Day 4  Ch3-01  PE 의 필요성
       Ch3-02  Sinusoidal PE
Day 5  Ch4-01  Warmup 의 필요성
       Ch4-02  AdamW
Day 6  Ch5-01  O(T²) 병목
       Ch5-05  Flash Attention
Day 7  Ch6-02  GPT
       Ch7-01  Scaling Laws
```

</details>

<details>
<summary><b>🟡 "Attention 분해와 계산 효율화를 완전히 정복한다" — 이론 집중 (2주, 약 24~28시간)</b></summary>

<br/>

```
1주차 — Attention 의 수학과 아키텍처
  Day 1    Ch1-01~02   Attention 식 분해 + √d_k 증명 꼼꼼히
  Day 2    Ch1-03~04   Softmax + Kernel 해석
  Day 3    Ch1-05~06   Multi-Head + 해석가능성 논쟁
  Day 4    Ch2-01~03   Transformer block + Pre-LN gradient 분석
  Day 5    Ch2-04~05   Encoder vs Decoder + Cross-Attention
  Day 6    Ch3-01~03   PE 필요성 + Sinusoidal + Learned
  Day 7    Ch3-04~05   Relative + RoPE + ALiBi

2주차 — 훈련 · 효율화 · 현대 아키텍처
  Day 1    Ch4-01~02   Warmup + AdamW
  Day 2    Ch4-03~05   Label Smoothing + Mixed Precision
  Day 3    Ch5-01~02   O(T²) + Linear Attention kernel trick
  Day 4    Ch5-03~04   Performer + Sparse (Longformer/BigBird)
  Day 5    Ch5-05~06   Flash Attention + MQA/GQA
  Day 6    Ch6-01~03   BERT + GPT + T5
  Day 7    Ch6-04~05   ViT + MoE
```

</details>

<details>
<summary><b>🔴 "Transformer 의 수학을 완전 정복한다" — 전체 정복 (10주, 약 40~50시간 + 실험 재현 12~18시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — Attention 수학적 분해
         → √d_k 분산 분석 손 증명, softmax Jacobian 유도
         → NumPy 에서 d_k 별 분산 측정, Multi-Head head-wise 분석

2주차   Chapter 2 전체 — Transformer 아키텍처
         → Pre-LN vs Post-LN gradient 비교 실험
         → Encoder/Decoder mask matrix 시각화
         → FFN 의 key-value memory 해석 (Geva 2021) 재현

3주차   Chapter 3 전체 — Positional Encoding
         → Sinusoidal PE 의 PE_{pos+k} = M_k · PE_pos 회전 행렬 유도
         → RoPE 의 복소수 form 직접 구현
         → ALiBi extrapolation 실험 (train length 의 4× 까지)

4주차   Chapter 4 전체 — 훈련의 수학
         → Warmup 효과 ablation (with/without)
         → AdamW vs Adam+L2 비교 실험
         → Mixed precision loss scaling 직접 구현

5주차   Chapter 5 (1~3) — Linear · Performer
         → Linear Attention 의 ELU+1 feature map 구현
         → Performer FAVOR+ random feature 구현
         → Standard vs Linear vs Performer 속도·품질 비교

6주차   Chapter 5 (4~6) — Sparse · Flash · MQA
         → Longformer local+global mask 구현
         → Flash Attention 으로 wall-clock 4× 측정
         → MQA / GQA 의 KV cache 메모리 절약 측정

7주차   Chapter 6 (1~3) — BERT · GPT · T5
         → BERT MLM pretraining (작은 규모) 재현
         → GPT-2 small 의 zero-shot prompt 실험
         → T5 의 span corruption objective 구현

8주차   Chapter 6 (4~5) — ViT · MoE
         → ViT 로 CIFAR-10 학습 (CNN 비교)
         → Switch Transformer top-1 routing 직접 구현

9주차   Chapter 7 (1~2) — Scaling · ICL
         → 작은 규모에서 scaling law log-log fit
         → GPT-2 small 에서 ICL 실험 (linear regression task)

10주차  Chapter 7 (3~4) + 종합 — CoT · 한계
         → CoT 의 emergent 현상 재현 (GSM8K)
         → "Transformer vs Mamba/RWKV" 의 비교 분석
         → "Attention 에서 RNN 으로 다시? 아니면 Transformer 의 진화?" 토론
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Matrix factorization · outer product · 고유분해 | **Ch1 전체** (Q, K, V), Ch3 (RoPE 회전 행렬) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | Backprop · Residual · 초기화 | **전체 레포 전제**, Ch2 (Pre-LN gradient), Ch4 (warmup) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | RBF kernel · random features | **Ch1-04** (Attention as kernel), **Ch5-02~03** (Linear / Performer) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | AdamW · warmup · LR schedule · gradient flow | **Ch4 전체** (훈련 recipe) |
| [regularization-theory-deep-dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive) | LayerNorm · Label Smoothing · Dropout | **Ch2-03** (LN), **Ch4-03** (Label Smoothing) |
| [rnn-lstm-deep-dive](https://github.com/iq-ai-lab/rnn-lstm-deep-dive) | Seq2Seq · Bahdanau Attention · LSTM | **Ch1 도입 동기** (RNN → Attention), Ch5 (Linear Attention as RNN) |
| [generative-models-deep-dive](https://github.com/iq-ai-lab/generative-models-deep-dive) *(다음)* | Diffusion · Flow Matching · GPT-style autoregressive | **Ch6 이후** 직접 응용 |
| [gnn-deep-dive](https://github.com/iq-ai-lab/gnn-deep-dive) | Graph Laplacian · Message Passing · Graphormer | **Ch1 일반화** (attention on completely-connected graph), Graphormer (Ch7-01 of GNN repo) |

> 💡 이 레포는 **"Transformer 가 RNN 을 대체한 것은 왜 수학적 필연이었고, $\sqrt{d_k}$ 와 Pre-LN 과 Linear/Flash Attention 이 왜 각각의 이론적 동기를 갖는가"** 에 집중합니다. NN Theory 에서 backprop 과 residual 을, Linear Algebra 에서 matrix factorization 을, Kernel Methods 에서 random feature 를, Optimization 에서 AdamW + warmup 을, Regularization 에서 LayerNorm 을 익힌 후 오면 Chapter 1 (분산 분석) 과 Chapter 5 (kernel trick) 의 증명이 훨씬 자연스럽습니다. **GNN Deep Dive** 의 Graphormer (Ch7) 가 이 레포의 attention 일반화 (completely-connected graph + 구조 bias) 임을 함께 보면 좋습니다.

---

## 📖 Reference

### 🏛️ Transformer 원전 · BERT · GPT · T5
- **Attention Is All You Need** (Vaswani et al., 2017) — **Transformer 원전**
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Devlin et al., 2019) — **BERT**
- **Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018) — **GPT-1**
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) — **GPT-2**
- **Language Models are Few-Shot Learners** (Brown et al., 2020) — **GPT-3**
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (Raffel et al., 2020) — **T5**
- **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2021) — **ViT**
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach** (Liu et al., 2019)
- **ELECTRA: Pre-training Text Encoders as Discriminators** (Clark et al., 2020)

### 🎨 Architecture · Layer Normalization · Initialization
- **Layer Normalization** (Ba, Kiros, Hinton, 2016)
- **On Layer Normalization in the Transformer Architecture** (Xiong et al., 2020) — **Pre-LN vs Post-LN**
- **Understanding the Difficulty of Training Transformers** (Liu et al., 2020) — Admin
- **DeepNet: Scaling Transformers to 1,000 Layers** (Wang et al., 2022)
- **Training Tips for the Transformer Model** (Popel & Bojar, 2018)
- **Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially** (Dong et al., 2021)

### 📐 Positional Encoding
- **Self-Attention with Relative Position Representations** (Shaw et al., 2018) — **Relative PE**
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (Dai et al., 2019)
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) — **RoPE**
- **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** (Press et al., 2021) — **ALiBi**
- **The Impact of Positional Encoding on Length Generalization in Transformers** (Kazemnejad et al., 2023)
- **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023)

### ⚡ Efficient Attention · Linear · Sparse · Flash
- **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** (Katharopoulos et al., 2020) — **Linear Attention**
- **Rethinking Attention with Performers** (Choromanski et al., 2021) — **Performer / FAVOR+**
- **Longformer: The Long-Document Transformer** (Beltagy et al., 2020) — **Longformer**
- **Big Bird: Transformers for Longer Sequences** (Zaheer et al., 2020) — **BigBird**
- **Reformer: The Efficient Transformer** (Kitaev et al., 2020) — LSH attention
- **Linformer: Self-Attention with Linear Complexity** (Wang et al., 2020)
- **Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention** (Xiong et al., 2021)
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022) — **Flash Attention**
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023)
- **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019) — **MQA**
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023) — **GQA**

### 🏋️ Training · Optimization · Regularization
- **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019) — **AdamW**
- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** (Goyal et al., 2017) — **Linear scaling rule**
- **When Does Label Smoothing Help?** (Müller, Kornblith, Hinton, 2019)
- **Mixed Precision Training** (Micikevicius et al., 2018)
- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** (Huang et al., 2019)
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al., 2019)
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020)

### 🔬 Multi-Head · Interpretability
- **Are Sixteen Heads Really Better than One?** (Michel, Levy, Neubig, 2019)
- **Attention is not Explanation** (Jain & Wallace, 2019)
- **Attention is not not Explanation** (Wiegreffe & Pinter, 2019)
- **Transformer Feed-Forward Layers Are Key-Value Memories** (Geva et al., 2021)
- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021) — Anthropic interpretability

### 🌐 LLM · Scaling · Emergent · ICL
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) — **Scaling Laws**
- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) — **Chinchilla**
- **Emergent Abilities of Large Language Models** (Wei et al., 2022)
- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022) — **CoT**
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (Wang et al., 2023)
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., 2023)
- **What learning algorithm is in-context learning? Investigations with linear models** (Akyürek et al., 2023)
- **Transformers learn in-context by gradient descent** (von Oswald et al., 2023)
- **Training language models to follow instructions with human feedback** (Ouyang et al., 2022) — **InstructGPT / RLHF**

### 🧪 Mixture of Experts · Sparse Models
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (Shazeer et al., 2017)
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** (Lepikhin et al., 2021)
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (Fedus et al., 2022)
- **Mixtral of Experts** (Jiang et al., 2024)
- **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts** (Du et al., 2022)

### 🔮 Theoretical Limits · Alternatives
- **On the Turing Completeness of Modern Neural Network Architectures** (Pérez, Marinkovic, Barceló, 2019)
- **Theoretical Limitations of Self-Attention in Neural Sequence Models** (Hahn, 2020)
- **The Parallelism Tradeoff: Limitations of Log-Precision Transformers** (Merrill & Sabharwal, 2023)
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023) — Transformer 대안
- **RWKV: Reinventing RNNs for the Transformer Era** (Peng et al., 2023)
- **Retentive Network: A Successor to Transformer for Large Language Models** (Sun et al., 2023)
- **Hyena Hierarchy: Towards Larger Convolutional Language Models** (Poli et al., 2023)

### 🛠️ Implementation · Libraries
- **The Annotated Transformer** (Rush, 2018) — line-by-line PyTorch 구현
- **The Illustrated Transformer** (Alammar, 2018) — 시각적 가이드
- **Hugging Face Transformers** (Wolf et al., 2020) — 표준 라이브러리

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`nn.MultiheadAttention` 을 호출하는 것과 — Vaswani 2017 로 $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$ 의 $\sqrt{d_k}$ 가 분산 분석에서 정확히 도출됨을 증명 · Su 2021 로 RoPE 가 회전 행렬로 자동 relative encoding 을 만든다는 것을 복소수 form 으로 유도 · Katharopoulos 2020 로 Linear Attention 의 kernel trick 이 $O(T^2 d) \to O(T d^2)$ 임을 결합 순서 변경으로 증명 · Dao 2022 로 Flash Attention 의 IO-aware tiling 이 same FLOP 으로 4× wall-clock 을 달성하는 메커니즘을 재현 · Hoffmann 2022 로 Chinchilla scaling law 가 GPT-3 의 훈련 recipe 를 뒤집는 compute-optimal 을 도출하는 과정을 small-scale 에서 fit — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
