# 01. $O(T^2)$ 복잡도의 문제

## 🎯 핵심 질문

- Self-attention 의 $O(T^2 d)$ time + $O(T^2)$ memory 가 long context 에서 정확히 어디서 병목인가?
- $T = 8192$, $d = 4096$ 인 LLaMA-2 의 attention matrix 가 메모리에서 차지하는 크기는?
- KV cache 가 long generation 에서 또 다른 병목인 이유는?
- 이 한계를 우회하는 4가지 family — Linear, Sparse, Flash, Approximation — 의 idea?
- Long context 가 important 한 application 의 예 — 왜 efficiency 가 critical 한가?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

$O(T^2)$ 복잡도는 Transformer 의 **fundamental architectural limit** 입니다:

1. **Long context 의 본질적 병목** — 32K, 128K, 1M token context 에서 quadratic blow-up
2. **다양한 우회 방법** — Linear (Ch5-02), Sparse (Ch5-04), Flash (Ch5-05), MQA/GQA (Ch5-06)
3. **Inference 의 KV cache** — generation 시 추가 병목
4. **Modern LLM 의 필수 도구** — 모든 long-context 모델이 이 문제 다룸

이 문서는 $O(T^2)$ 의 **정확한 분석과 다양한 mitigation 방향** 을 정리합니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md)
- 시간/공간 복잡도, GPU memory hierarchy

---

## 📖 직관적 이해

### Attention Matrix 의 크기

```
Sequence length T:        8192     32768     131072
Attention matrix T×T:     67M      1.07B     17B    elements
Memory (FP32):            256MB    4GB       64GB
Memory (BF16):            128MB    2GB       32GB
```

→ 32K context 에서 attention matrix 자체가 **수 GB**. GPU 메모리 (40GB - 80GB) 에 fit 어려움.

### Time Complexity

```
T = 1024:    ~1M FLOPs per attention head
T = 8192:    ~67M FLOPs (64×)
T = 32768:   ~1B FLOPs (1000×)
T = 131072:  ~17B FLOPs (16000×)

Quadratic blow-up — sequence 길이 4× 시 시간 16× 
```

### KV Cache 의 별도 문제

Generation 시 매 step 마다 새 token 의 K, V 를 캐시:
```
Per step new K, V:  d_k × num_heads = d
Cache size at T:    T × d × num_layers
```

GPT-3 175B (96 layer, $d = 12288$) 의 KV cache:
- $T = 2048$: $2048 \times 12288 \times 96 \times 2 = 5GB$ (BF16, K + V)
- $T = 32768$: $80GB$ — 한 모델 메모리 차지

### Long Context 의 Application

- **Long document QA**: 100+ page PDFs, books
- **Code understanding**: large codebases (10K+ lines)
- **Audio/video transcription**: hours-long content
- **Complex agent reasoning**: multi-step planning with full context

→ 모두 $T = 32K - 1M$ 필요.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Self-Attention Complexity

Sequence length $T$, embedding dim $d$, head dim $d_k$:

**Time**:
- $QK^\top$: $O(T^2 d_k)$
- Softmax: $O(T^2)$
- $A V$: $O(T^2 d_v)$
- Total: $O(T^2 d)$

**Memory**:
- $Q, K, V$: $O(T d)$
- Attention matrix $A$: $O(T^2)$
- Output: $O(T d)$
- **Dominant**: $O(T^2)$

### 정의 1.2 — Multi-Head Attention 의 Total

$h$ heads with $d_k = d/h$:
- Time: $h \cdot O(T^2 d/h) = O(T^2 d)$
- Memory: $O(h T^2)$ — 각 head 의 attention matrix

### 정의 1.3 — KV Cache

Autoregressive generation 에서 누적되는 K, V:
$$
\text{KV cache size} = 2 \cdot L \cdot T \cdot d_{\text{model}}
$$

(L layers, K and V each $d_{\text{model}}$, $T$ tokens)

### 정의 1.4 — Effective Context Length

학습 시 max context 가 inference 의 hard limit (extrapolation 별도).

---

## 🔬 정리와 증명

### 정리 1.1 — Time Complexity

Standard attention $\text{softmax}(QK^\top/\sqrt{d_k}) V$:

$$
T_{\text{attn}} = O(T^2 d_k) + O(T^2) + O(T^2 d_v) = O(T^2 d)
$$

**증명**: 각 matrix 곱이 $O(T \cdot T \cdot d)$ — naive matmul. $\square$

### 정리 1.2 — Space Complexity

Attention matrix $A \in \mathbb{R}^{T \times T}$ (or $\mathbb{R}^{h \times T \times T}$ for MHA):

$$
\text{Memory} = O(h T^2) \quad \text{plus} \quad O(T d) \text{ for Q, K, V, output}
$$

$T \gg d$ 시 $T^2$ 가 dominant.

### 정리 1.3 — KV Cache Size

Autoregressive generation 의 cumulative cache:
$$
\text{KV cache} = 2 \cdot L \cdot T \cdot d_{\text{model}} \cdot \text{bytes}
$$

- LLaMA-2 70B: $L = 80$, $d = 8192$, $T = 4096$ → $5.4GB$ (BF16)
- 32K context: $43GB$ — model size 에 비교

### 정리 1.4 — Wall-clock 의 추가 요인

Theoretical $O(T^2 d)$ 외 wall-clock 영향:
- **Memory bandwidth**: HBM ↔ SRAM 의 transfer 가 compute 보다 slower (특히 attention)
- **GPU underutilization**: small batch + long sequence
- **Communication**: distributed setting 의 all-reduce

→ Flash Attention (Ch5-05) 가 memory bandwidth 측면 직접 해결.

### 정리 1.5 — Long-context 의 Practical Limits

**Hardware constraint**:
- A100 80GB: $T \leq 16K$ for 7B model with full attention
- H100 80GB: $T \leq 32K$
- Multi-GPU sharding 으로 더 길게 가능 (with overhead)

**Training**:
- 32K context training: Flash Attention essential
- 128K+: Sparse attention 또는 RAG 등 hybrid 필요

### 정리 1.6 — Mitigation Family 의 Trade-offs

| Method | Time | Memory | Exact? |
|--------|------|--------|--------|
| Standard | $O(T^2 d)$ | $O(T^2)$ | Yes |
| Linear (Ch5-02) | $O(T d^2)$ | $O(T d)$ | No (approx) |
| Sparse (Ch5-04) | $O(T d \cdot s)$ | $O(T s)$ | No (limited) |
| Flash (Ch5-05) | $O(T^2 d)$ | $O(T)$ effective | Yes |
| MQA/GQA (Ch5-06) | $O(T^2 d)$ | $O(T h')$ | Yes |

각 method 의 strength 가 다름 — 종종 결합 사용.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Memory Usage 측정

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def attention_memory(T, d_k, num_heads=1, dtype=torch.float32):
    """Memory in MB for standard attention"""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    
    qkv_mem = 3 * T * d_k * num_heads * bytes_per_elem
    attn_mem = num_heads * T * T * bytes_per_elem
    out_mem = T * d_k * num_heads * bytes_per_elem
    
    total = qkv_mem + attn_mem + out_mem
    return total / 1024 / 1024   # MB

print(f'{"T":>8} | {"Q,K,V (MB)":>12} | {"Attn (MB)":>10} | {"Total (MB)":>10}')
print('-' * 50)
for T in [512, 2048, 8192, 32768, 131072]:
    bytes_per_elem = 4   # FP32
    qkv = 3 * T * 4096 * bytes_per_elem / 1e6
    attn = T * T * bytes_per_elem / 1e6
    total = qkv + attn
    print(f'{T:8d} | {qkv:12.2f} | {attn:10.2f} | {total:10.2f}')
```

**예상 출력** (FP32, $d = 4096$, single head):
```
       T |   Q,K,V (MB) |  Attn (MB) | Total (MB)
--------------------------------------------------
     512 |        25.17 |       1.05 |      26.21
    2048 |       100.66 |      16.78 |     117.44
    8192 |       402.65 |     268.44 |     671.09
   32768 |      1610.61 |    4294.97 |    5905.58
  131072 |      6442.45 |   68719.48 |   75161.93
```

→ T=131K 에서 attention matrix 만 70GB! ✗

### 실험 2 — Wall-clock Time 측정

```python
import time

def measure_attention_time(T, d_k, n_trials=3):
    Q = torch.randn(T, d_k); K = torch.randn(T, d_k); V = torch.randn(T, d_k)
    
    t0 = time.time()
    for _ in range(n_trials):
        scores = (Q @ K.T) / np.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
    return (time.time() - t0) / n_trials

times = {}
for T in [512, 1024, 2048, 4096, 8192]:
    if T <= 8192:
        t = measure_attention_time(T, 64)
        times[T] = t
        print(f'T={T:5d}: {t*1000:.2f}ms')

# Quadratic 확인
ts = list(times.keys())
plt.loglog(ts, [times[t] for t in ts], 'o-', label='Measured')
plt.loglog(ts, [(t/ts[0])**2 * times[ts[0]] for t in ts], '--', label='O(T²) reference')
plt.xlabel('T'); plt.ylabel('time (s)')
plt.legend(); plt.title('Attention time complexity'); plt.show()
```

### 실험 3 — KV Cache 누적 시뮬레이션

```python
# Generation 시 KV cache 가 누적되는 양 측정
def kv_cache_size(T, d_model, num_layers, dtype_bytes=2):
    """KV cache size in MB"""
    return 2 * num_layers * T * d_model * dtype_bytes / 1e6

# LLaMA-2 7B: L=32, d=4096
# LLaMA-2 70B: L=80, d=8192

for T in [2048, 8192, 32768, 131072]:
    s_7b = kv_cache_size(T, 4096, 32)
    s_70b = kv_cache_size(T, 8192, 80)
    print(f'T={T:6d}: KV cache LLaMA-2 7B = {s_7b/1024:.2f}GB, '
          f'LLaMA-2 70B = {s_70b/1024:.2f}GB')

# T=131K 에서 70B 는 168GB KV cache!
```

### 실험 4 — Naive Long-context Attention 의 Failure

```python
# T=8192 시 메모리 확인
T = 8192
d_k = 64
try:
    Q = torch.randn(T, d_k, device='cpu')
    K = torch.randn(T, d_k, device='cpu')
    V = torch.randn(T, d_k, device='cpu')
    scores = (Q @ K.T) / np.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    print(f'T={T}: Success, attention matrix = {attn.element_size() * attn.numel() / 1e6:.1f} MB')
except RuntimeError as e:
    print(f'T={T}: OOM — {e}')

# T=131072 시도
T = 131072
try:
    Q = torch.randn(T, d_k)
    K = torch.randn(T, d_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    print(f'T={T}: Success')
except (RuntimeError, MemoryError) as e:
    print(f'T={T}: Memory error')
```

### 실험 5 — Mitigation 의 Memory 비교

```python
def linear_attention_memory(T, d_k):
    """Linear attention: O(T d²) memory"""
    return 3 * T * d_k * 4 / 1e6 + d_k * d_k * 4 / 1e6   # Q,K,V + KV product

def sparse_attention_memory(T, d_k, sparsity_factor=0.1):
    """Sparse attention: O(T s d) where s = sparsity_factor × T"""
    s = int(sparsity_factor * T)
    return T * s * 4 / 1e6 + 3 * T * d_k * 4 / 1e6

print(f'{"T":>8} | {"Standard":>12} | {"Linear":>10} | {"Sparse":>10}')
print('-' * 48)
for T in [2048, 8192, 32768, 131072]:
    std = T * T * 4 / 1e6 + 3 * T * 64 * 4 / 1e6
    lin = linear_attention_memory(T, 64)
    spr = sparse_attention_memory(T, 64, 0.05)
    print(f'{T:8d} | {std:12.2f} | {lin:10.2f} | {spr:10.2f}')
# Linear, Sparse 가 standard 보다 훨씬 작음
```

---

## 🔗 실전 활용

### 1. Long-context Model 의 Architecture Choice

| Context Length | Recommended |
|---------------|-------------|
| 2-4K | Standard attention |
| 8-32K | Flash Attention |
| 32-128K | Flash + RoPE/ALiBi |
| 128K-1M | Sparse + Flash + KV cache 최적화 |
| 1M+ | Hybrid (RAG, sliding window, MoE) |

### 2. Inference Optimization

KV cache 최적화 techniques:
- **MQA/GQA**: Ch5-06
- **PagedAttention** (vLLM): KV cache 의 page-based 관리
- **Flash Attention 2**: KV cache 최적화 포함
- **Quantization**: KV cache 를 INT8, INT4 로 양자화

### 3. RAG vs Long Context

긴 정보를 다루는 두 방법:
- **Long context**: 모든 정보를 sequence 로
- **RAG**: 외부 검색 + 짧은 context

Trade-off:
- Long context: 직접 reasoning, 그러나 cost ↑
- RAG: cheap, 그러나 retrieval quality 의존

### 4. Application 별 Bottleneck

- **Document QA**: 긴 input, 짧은 output → prefill (long context attention) bottleneck
- **Long generation**: 짧은 input, 긴 output → decoding (KV cache) bottleneck
- **Both** (long doc summary): 둘 다 critical

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Naive matmul | Tiled, fused kernel 가능 (Flash) |
| Single GPU | Multi-GPU sharding 으로 확장 |
| Full attention | Sparse, sliding window |
| FP32 memory | BF16/FP16 로 절약 |
| Independent layers | Activation checkpointing 으로 memory ↓ |

---

## 📌 핵심 정리

$$\boxed{\text{Attention: } T = O(T^2 d), M = O(T^2). \quad \text{KV cache: } M = O(L T d)}$$

| Bottleneck | Source | Mitigation |
|-----------|--------|------------|
| Time | $T^2 d$ matmul | Linear, Sparse |
| Memory (attn matrix) | $T^2$ | Flash (block tiling) |
| Memory (KV cache) | $L T d$ | MQA, GQA, quantization |
| Bandwidth | HBM ↔ SRAM | Flash (IO-aware) |

| Modern Approach | Approach |
|-----------------|----------|
| Linear Attention | $O(T)$ via kernel trick (Ch5-02) |
| Performer | Random features (Ch5-03) |
| Sparse | Local + global (Ch5-04) |
| Flash | IO-aware exact (Ch5-05) |
| MQA / GQA | KV head 절약 (Ch5-06) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LLaMA-2 70B ($L=80, d=8192$, $h=64$, $d_k = 128$, GQA $g=8$) 의 16K context 에서 attention matrix 와 KV cache 의 크기를 BF16 로 계산하라.

<details>
<summary>해설</summary>

**Attention matrix** (per layer, single sample):
$h \times T \times T \times 2$ bytes = $64 \times 16384 \times 16384 \times 2 = 32GB$

전 layer 이 같이 메모리에 있을 필요는 없음 (forward 시 한 layer 씩). Per layer 32GB 가 issue — Flash Attention 필요.

**KV cache** (full context):
GQA 의 KV head = $g = 8$ (vs 64 Q heads).
$$
2 \times L \times T \times g \times d_k \times 2 = 2 \times 80 \times 16384 \times 8 \times 128 \times 2 = 5.4GB
$$

(GQA 가 $h/g = 8$ fold KV cache 절약 → 정상 MHA 면 43GB)

**의미**:
- Attention matrix per layer: 32GB → Flash 필수
- KV cache: 5.4GB → 10GB GPU 도 inference 가능
- GQA 의 효과: 43GB → 5.4GB, $8\times$ 절약 ✓

장기 generation 시 KV cache 가 dominant — GQA 의 직접 motivation. $\square$

</details>

**문제 2** (심화): "$O(T^2)$ 가 본질적인가?" — 다양한 mitigation (Linear, Sparse, Flash) 의 fundamental limit 를 분석하라. Linear attention 이 정확히 $O(T)$ 인 trade-off 는?

<details>
<summary>해설</summary>

**Standard attention 의 $O(T^2)$ 의 본질**:

$\text{Attn}(Q,K,V) = \text{softmax}(QK^\top) V$. 

$QK^\top$ 의 모든 $T \times T$ entry 를 계산한 후 softmax row-wise → 각 $i$ 가 모든 $j$ 와의 score 필요 → fundamental $O(T^2)$.

**Mitigations**:

1. **Linear Attention (Ch5-02)**:
   - $\phi(Q) (\phi(K)^\top V)$ — 결합 순서 변경
   - $\phi(K)^\top V \in \mathbb{R}^{d \times d}$ — $T$ 무관
   - Time: $O(T d^2)$, Memory: $O(d^2 + T d)$
   - **Trade-off**: softmax 의 sharp peak 손실 (Ch1-04 의 kernel decomposition 의 approximation)
   - 수학적으로 모든 attention 표현 못함 — **표현력 ↓**

2. **Sparse Attention (Ch5-04)**:
   - 일부 (i, j) pair 만 계산 — local + global
   - Time: $O(T \cdot s \cdot d)$ where $s$ = sparsity
   - **Trade-off**: 임의 token pair 의 long-range dependency 못 잡음 (선택된 pattern 만)
   - 수학적으로 universal approximator (BigBird) — 그러나 exact attention 아님

3. **Flash Attention (Ch5-05)**:
   - **Same** $O(T^2)$ FLOP — 근사 아님
   - Memory $O(T)$ (effective)
   - Wall-clock 2-4× faster (IO-aware)
   - **No trade-off** — exact

4. **MQA / GQA (Ch5-06)**:
   - $O(T^2 d_k)$ same time
   - Memory $T \times d_k$ (KV head 절약)
   - **Trade-off**: 약간의 표현력 (head 수 ↓)

**Fundamental Limit**:

- Exact attention 의 information-theoretic lower bound: 임의 token pair 정보 → $\Omega(T^2)$ time
- Approximation 시 lower bound 회피 가능
- Flash 는 same complexity but better constant — 이론적 한계 아님

**Linear attention 의 $O(T)$ 정당성**:

- $O(T d^2)$ — $T$ 에 linear, $d$ 에 quadratic
- Sequence length 1M with $d = 4096$: $4 \times 10^{12}$ vs standard $10^{12} \times 4 \times 10^3 = 4 \times 10^{15}$
- → Linear 가 1000× faster (이론)

**Linear 의 표현력 한계**:

- Softmax 의 nonlinear sharpness 손실
- Linear kernel ($\phi(x) = \text{ELU}(x) + 1$ 등) 가 exp inner product 의 표현력 일부 손실
- Performer (Ch5-03) 가 random feature 로 일부 회복

**현실의 결정**:

- 짧은 context: Standard + Flash 충분
- 중간 (32K): Standard + Flash + RoPE
- 긴 (128K+): Hybrid (sparse + Flash) 또는 RAG
- 매우 긴 (1M+): RNN/SSM 같은 alternative 또는 efficient kernel (Mamba)

→ **$O(T^2)$ 는 standard 의 한계, 그러나 다양한 mitigation 으로 practical 처리 가능**. $\square$

</details>

**문제 3** (논문 비평): Long context (1M tokens) LLM 이 점점 흔해지고 있다 (Gemini 1.5, Claude 3.5). 이런 모델이 architecture 적으로 어떤 mitigation 의 결합인가? 그리고 RAG 가 long context 의 alternative 가 되는 조건은?

<details>
<summary>해설</summary>

**1M token LLM 의 Architecture Mitigation**:

Frontier 1M context 모델 (Gemini 1.5, Claude 3.5) 의 estimate:

1. **Flash Attention 3** (or 후속): $O(T^2)$ but linear memory
2. **GQA / MQA**: KV cache 절약 ($8\times$+ )
3. **KV cache quantization**: INT8 또는 INT4 로
4. **Sparse attention** in some layers: local + global
5. **Paged attention**: KV cache 의 efficient memory management
6. **MoE**: parameter ↑ compute → (sparse activation)
7. **RoPE / ALiBi**: long-context PE
8. **Position interpolation / NTK-aware**: train length 너머 inference

**Multi-stage training**:
- Stage 1: short context (4K) pre-training (효율적)
- Stage 2: long context fine-tuning (32K, 128K, 1M)
- 각 stage 마다 PE 와 attention 조정

**Hardware**:
- Multi-GPU sharding (Tensor Parallelism, Pipeline Parallelism)
- Specialized inference servers (vLLM, TGI)
- INT4/INT8 quantization for production

**RAG 가 alternative 인 조건**:

**Long context 우위**:
- Retrieval target 이 명확 안 함 — 전체 context 필요
- Cross-document reasoning — 여러 문서 동시 reasoning
- Coding, analysis — full codebase 또는 dataset
- Complex multi-step query

**RAG 우위**:
- Retrieval target 명확 — specific question
- Knowledge base 매우 큼 (TB scale) — context 에 못 넣음
- Real-time / dynamic data — index 가 자주 update
- Cost 우선 — long context 비용 high

**Hybrid**:
- Modern RAG + LLM (Anthropic Sonnet, Gemini)
- Long context 으로 multiple retrieved docs 처리
- 각 strength 결합

**Trade-offs**:

| 측면 | Long Context | RAG |
|------|-------------|-----|
| Latency | High (long forward) | Low (retrieval + short forward) |
| Cost | High (token-proportional) | Low (mostly retrieval) |
| Reasoning | 자연스러운 cross-doc | Retrieval quality 의존 |
| Information freshness | Static (training) | Dynamic (live KB) |
| Implementation | Single LLM | LLM + retrieval system |

**미래 (2026+)**:

- Long context 더 cheap 해질 것 (efficient attention 발전)
- RAG 가 disappearing? — 일부 use case 에서 long context 가 충분
- Hybrid 가 dominant — 양쪽 strength 결합
- New paradigm: 학습 시 retrieval (REALM, RETRO 같은 아이디어 의 재부상)

**근본적 통찰**:

$O(T^2)$ 의 우회는 **architecture, training, hardware** 의 합작. Single technique 으로 1M context 불가능 — **system engineering** 의 결과.

RAG vs Long Context 는 **information access pattern** 의 차이 — fundamental architecture choice. 미래는 두 paradigm 의 hybrid + new approaches (Mamba 같은 state space models). $\square$

</details>

---

<div align="center">

[◀ 이전](../ch4-training-math/05-mixed-precision.md) | [📚 README](../README.md) | [다음 ▶](./02-linear-attention.md)

</div>
