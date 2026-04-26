# 04. Sparse Attention — Longformer · BigBird

## 🎯 핵심 질문

- Sparse attention 의 핵심 — 모든 (i, j) pair 가 아니라 sparse subset 만 계산?
- Longformer (Beltagy 2020) 의 local + global pattern 의 sparsity 설계?
- BigBird (Zaheer 2020) 의 local + global + random — 왜 random 이 필요한가?
- BigBird 가 universal approximator 임의 증명의 의미?
- Sparse attention 의 GPU 구현 어려움 — sparse matrix 가 dense 보다 느린 이유?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Sparse Attention 은 **long context 의 most explicit approach**:

1. **Linear $O(T s)$** — sparsity factor $s$ 로 cost 줄임
2. **Inductive bias** — local/global structure 를 architecture 에 baked-in
3. **이론적으로 universal** — BigBird 의 universal approximation 보장
4. **그러나 GPU 구현 어려움** — sparse pattern 이 dense 보다 종종 slower

이 문서는 sparse attention 의 **변형들과 GPU implementation challenges** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md)
- 이전 문서: [01-quadratic-bottleneck.md](./01-quadratic-bottleneck.md)
- 그래프 이론: connectivity, expander graph

---

## 📖 직관적 이해

### Sparse Pattern Examples

```
Standard (dense):           Local window:           Local + global:
[X X X X X X]              [X X . . . .]          [X X . . . G]
[X X X X X X]              [X X X . . .]          [X X X . . G]
[X X X X X X]              [. X X X . .]          [G G X X G G]   ← global token
[X X X X X X]              [. . X X X .]          [. . X X X G]
[X X X X X X]              [. . . X X X]          [. . . X X G]
[X X X X X X]              [. . . . X X]          [G G G G G X]   ← global

T² entries                  ~T·w entries (window)   ~T·w + T entries
```

### Longformer 의 Idea

```
1. Local sliding window:    각 token 이 ±w 이웃에만 attend
2. Global tokens:           특정 token (CLS, question 등) 이 모두 attend
                            모두가 그 token 에 attend (대칭)
                            
→ 대부분 local, 중요한 정보는 global
```

### BigBird 의 추가 idea

```
1. Local: ±w window
2. Global: 일부 token (대칭)
3. Random: 각 token 이 랜덤 r 개 token 에 attend

→ Random 이 "shortcut" 제공 → 모든 정보가 short path 로 도달 가능
```

### Universal Approximation (BigBird)

Local + global + random sparse 가 **여전히 full attention 의 표현력 보존** — Yun 2020 의 Transformer universal approximation 결과를 sparse 로 확장.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Attention Mask Pattern

$M \in \{0, 1\}^{T \times T}$ — $M_{ij} = 1$ 이면 attend 가능.

Sparse attention:
$$
A_{ij} = \begin{cases} \text{softmax}(s_{ij}) & M_{ij} = 1 \\ 0 & M_{ij} = 0 \end{cases}
$$

(softmax 는 $M_{ij} = 1$ 인 entry 만 normalize)

### 정의 4.2 — Local Attention (Sliding Window)

Window size $w$:
$$
M^{\text{local}}_{ij} = \mathbb{1}[|i - j| \leq w]
$$

Each row 에 $2w + 1$ entries (boundary 제외).

### 정의 4.3 — Global Attention

Global token set $\mathcal{G} \subseteq \{1, \ldots, T\}$:
$$
M^{\text{global}}_{ij} = \mathbb{1}[i \in \mathcal{G} \text{ or } j \in \mathcal{G}]
$$

(대칭 — global token 이 모든 token 과 connect)

### 정의 4.4 — Random Attention

각 token $i$ 에 random $r$ tokens $\mathcal{R}_i$:
$$
M^{\text{rand}}_{ij} = \mathbb{1}[j \in \mathcal{R}_i]
$$

### 정의 4.5 — BigBird Pattern

$$
M^{\text{BB}} = M^{\text{local}}(w) + M^{\text{global}}(\mathcal{G}) + M^{\text{rand}}(r)
$$

(union of three patterns)

### 정의 4.6 — Sparse Attention Computation

Sparse $A$ 의 efficient form (assume sparsity $s$ per row):
$$
S_{ij} = q_i^\top k_j / \sqrt{d_k} \quad \text{for } M_{ij} = 1
$$

Time: $O(T s d_k)$, Memory: $O(T s)$.

---

## 🔬 정리와 증명

### 정리 4.1 — Longformer Complexity

Longformer (window $w$ + global $|\mathcal{G}|$):
- Per-row attended entries: $2w + 1 + |\mathcal{G}|$
- Total: $T (2w + |\mathcal{G}|) + |\mathcal{G}| T$ (global rows)
- $\approx O(T w + T |\mathcal{G}|)$

For $w = 512$, $|\mathcal{G}| = O(1)$: linear in $T$.

### 정리 4.2 — BigBird 의 Path Length

Random graph 이론: 각 node 가 random $r$ neighbors 일 때 graph diameter $\approx \log T / \log r$.

BigBird 의 random + local 결합:
- Local: short-range
- Random: short-range + long-range shortcut
- Global: hub for long-range

→ 임의 token pair 의 path length $O(\log T)$.

### 정리 4.3 — BigBird Universal Approximation (Zaheer 2020)

**Theorem**: BigBird 가 sufficient depth 시 임의 continuous sequence-to-sequence function 을 근사.

**증명 sketch**:
- Local + random + global 의 합집합이 expander graph
- Expander 가 $O(\log T)$ depth 로 임의 information 전파
- Yun 2020 의 dense Transformer universal approx 를 expander 위로 확장

**의미**: Sparse 임에도 표현력 손실 없음 — full attention 의 efficient 대체.

### 정리 4.4 — Sparse vs Dense GPU 의 Efficiency

이론: sparse 가 $T^2/s$ 배 빠름 ($s$ = sparsity).

**그러나 GPU**:
- Dense matmul: highly optimized (cuBLAS, Tensor Core)
- Sparse matmul: irregular memory access, GPU underutilization
- Sparse 가 $1\%$ sparsity 에서도 dense 보다 slower 가능

**해결**: structured sparsity (block-sparse, 2:4 pattern) — GPU-friendly.

### 정리 4.5 — Block-Sparse Attention

Pattern 을 $b \times b$ block 단위로:
- Block 1 = active, block 0 = skip
- Each block 이 $O(b^2)$ FLOP — dense matmul (efficient)
- Total: $O(\text{# blocks} \cdot b^2)$

NVIDIA 의 native block-sparse support — Triton 으로 구현.

### 정리 4.6 — Information Bottleneck

Sparse 의 한계: 일부 (i, j) pair 의 직접 access 불가.
- Local + global: long-range 는 global hub 통해서만
- 정보 압축 필요 — 일부 detail 손실
- **표현력 = depth × sparsity** trade-off

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Sparse Pattern Visualization

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

T = 30

# Local
def local_mask(T, w):
    mask = torch.zeros(T, T)
    for i in range(T):
        for j in range(max(0, i-w), min(T, i+w+1)):
            mask[i, j] = 1
    return mask

# Global
def global_mask(T, global_indices):
    mask = torch.zeros(T, T)
    for g in global_indices:
        mask[g, :] = 1
        mask[:, g] = 1
    return mask

# Random
def random_mask(T, r, seed=0):
    torch.manual_seed(seed)
    mask = torch.zeros(T, T)
    for i in range(T):
        idx = torch.randperm(T)[:r]
        mask[i, idx] = 1
    return mask

# Combine
mask_local = local_mask(T, w=3)
mask_global = global_mask(T, global_indices=[0, 15])
mask_random = random_mask(T, r=2)
mask_bigbird = ((mask_local + mask_global + mask_random) > 0).float()

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, m, title in zip(axes, [mask_local, mask_global, mask_random, mask_bigbird],
                        ['Local (window=3)', 'Global (CLS, 15)', 'Random (r=2)', 'BigBird (combined)']):
    ax.imshow(m.numpy(), cmap='Blues')
    ax.set_title(title)
plt.tight_layout(); plt.show()

# Sparsity 계산
print(f'Dense:    {T*T} entries (100%)')
print(f'Local:    {int(mask_local.sum())} entries ({mask_local.mean()*100:.1f}%)')
print(f'BigBird:  {int(mask_bigbird.sum())} entries ({mask_bigbird.mean()*100:.1f}%)')
```

### 실험 2 — Sparse Attention Implementation (간단)

```python
def sparse_attention(Q, K, V, mask):
    """Mask = (T, T) binary, attend only where mask=1"""
    d = Q.size(-1)
    scores = Q @ K.T / np.sqrt(d)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V

torch.manual_seed(0)
T, d = 30, 16
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

# Standard (full attention)
out_full = sparse_attention(Q, K, V, torch.ones(T, T))

# Sparse
out_sparse = sparse_attention(Q, K, V, mask_bigbird)

print(f'Full attention output: {out_full[0, :5]}')
print(f'Sparse output:         {out_sparse[0, :5]}')
print(f'Difference:            {(out_full - out_sparse).abs().mean():.4f}')
# 다름 — sparse 의 information bottleneck
```

### 실험 3 — Longformer-style Implementation

```python
def longformer_attention(Q, K, V, window=8, global_idx=[0]):
    T, d = Q.size()
    mask = local_mask(T, window) + global_mask(T, global_idx)
    mask = (mask > 0).float()
    return sparse_attention(Q, K, V, mask)

torch.manual_seed(0)
out_lf = longformer_attention(Q, K, V, window=5, global_idx=[0, 15])
print(f'Longformer attention: {out_lf.shape}')
```

### 실험 4 — Path Length 측정 (BigBird Connectivity)

```python
def shortest_path_length(adj, src, dst):
    """BFS"""
    visited = {src}
    queue = [(src, 0)]
    while queue:
        node, dist = queue.pop(0)
        if node == dst:
            return dist
        for neighbor in range(adj.size(0)):
            if adj[node, neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1   # unreachable

T = 100
mask_bb = ((local_mask(T, w=3) + global_mask(T, [0]) + random_mask(T, r=2)) > 0).float()

# 임의 token pair 의 shortest path
import random
random.seed(0)
for _ in range(5):
    i, j = random.sample(range(T), 2)
    pl = shortest_path_length(mask_bb, i, j)
    print(f'Path {i} -> {j}: length = {pl}')

# 평균 path length
import numpy as np
total = 0; count = 0
for _ in range(100):
    i, j = random.sample(range(T), 2)
    pl = shortest_path_length(mask_bb, i, j)
    if pl > 0:
        total += pl; count += 1
print(f'\nAverage path length: {total/count:.2f}  (theoretical: O(log T) = ~7)')
```

### 실험 5 — Sparse vs Dense Speed (PyTorch)

```python
import time

# 표준 sparse 는 PyTorch 에서 dense 보다 느릴 수 있음
torch.manual_seed(0)

def benchmark(T, mask_fn, n_trials=10):
    Q = torch.randn(T, 64); K = torch.randn(T, 64); V = torch.randn(T, 64)
    mask = mask_fn(T)
    
    # Dense
    t0 = time.time()
    for _ in range(n_trials):
        out = sparse_attention(Q, K, V, torch.ones(T, T))
    t_dense = (time.time() - t0) / n_trials
    
    # Sparse (with mask)
    t0 = time.time()
    for _ in range(n_trials):
        out = sparse_attention(Q, K, V, mask)
    t_sparse = (time.time() - t0) / n_trials
    
    return t_dense, t_sparse

for T in [256, 1024, 4096]:
    mask_fn = lambda T_: ((local_mask(T_, w=64) + global_mask(T_, [0])) > 0).float()
    td, ts = benchmark(T, mask_fn)
    print(f'T={T:4d}: Dense={td*1000:.1f}ms, Sparse={ts*1000:.1f}ms (Note: PyTorch naive sparse 는 dense 보다 느릴 수 있음)')

# 현실: PyTorch 의 native sparse 는 GPU 에서 dense 보다 종종 slower
# → 진짜 효율은 specialized library (DeepSpeed sparse attention, Triton block-sparse) 사용
```

---

## 🔗 실전 활용

### 1. Longformer 의 채택

- Original Longformer: 4096 token context
- 학술 논문 분류 / QA: long document 처리
- 그러나 modern LLM 에 비해 작은 영향 (Flash Attention 의 등장)

### 2. BigBird 의 응용

- Genomics: protein/DNA sequences
- Long document classification
- 이론적 보장 — "sparse 가 정말 enough"

### 3. Sliding Window in Modern LLM

- Mistral 7B: sliding window attention with $w = 4096$
- Mistral-8x7B: same
- 큰 context (32K) 에서 일부 layer 만 sliding window

### 4. Sparse Attention 의 GPU Efficient 구현

- **DeepSpeed Sparse Attention**: structured block-sparse
- **Triton block-sparse**: NVIDIA H100 의 2:4 sparsity native
- **Flash-Attention 2**: sliding window 지원
- Naive sparse pattern 은 GPU 에서 비효율, structured block 필수

### 5. Hybrid Approaches

- Some layers: full Flash Attention
- Other layers: sliding window (Mistral 패턴)
- 일부 layer 만 long-range, 나머지는 local 으로 충분

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Predefined sparsity | Learned sparsity 도 가능 |
| Static mask | Dynamic (input-dependent) sparse 는 어려움 |
| Local + global + random | Other patterns 도 가능 (random walks, etc) |
| GPU efficiency | Block-sparse, Triton 으로 mitigate |
| Universal approximation | 이론, 실제 표현력 약간 손실 |

---

## 📌 핵심 정리

| Pattern | Per-row attend | Total | Inductive Bias |
|---------|---------------|-------|----------------|
| **Dense** | $T$ | $O(T^2)$ | None |
| **Local** | $2w+1$ | $O(T w)$ | Locality |
| **Global** | $|\mathcal{G}|$ | $O(T |\mathcal{G}|)$ | Hub-based |
| **Random** | $r$ | $O(T r)$ | Shortcut |
| **BigBird** | $w + |\mathcal{G}| + r$ | $O(T (w + r))$ | All combined |

| Property | Standard | Longformer | BigBird |
|----------|----------|------------|---------|
| Time | $O(T^2 d)$ | $O(T w d)$ | $O(T (w+r) d)$ |
| Memory | $O(T^2)$ | $O(T w)$ | $O(T (w+r))$ |
| Long-range | Direct | Via global | Via random |
| Approximation | Exact | Limited | Universal |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 4096$, sliding window $w = 512$ 인 Longformer 의 attention matrix 의 nonzero entry 수와 sparsity 를 계산하라.

<details>
<summary>해설</summary>

**Local sliding window**:
- Each row 의 attended entries: $2w + 1 = 1025$ (boundary 제외, edge effect ignore)
- Total: $T \times (2w+1) = 4096 \times 1025 \approx 4.2M$ entries

**Sparsity** (1 = active, 0 = inactive):
- Nonzero / Total = $4.2M / (4096^2) = 4.2M / 16.8M \approx 25\%$

→ Window 가 sequence 의 1/4 수준이면 75% sparsity.

**Memory savings** (FP32):
- Dense: $4096^2 \times 4 = 67MB$
- Sparse: $4.2M \times 4 = 17MB$ (3.9× savings)

**Practical**: Longformer 가 실제로 4K context 에서 attention matrix 가 67MB → 17MB. 큰 context 에서 더 큰 효과. $\square$

</details>

**문제 2** (심화): BigBird 의 universal approximation 의 의미 — local + random + global 이 dense attention 만큼 표현력을 갖는다는 증명의 핵심 idea 는?

<details>
<summary>해설</summary>

**BigBird 의 표현력 정리** (Zaheer 2020):

**Theorem (informal)**: BigBird with appropriate hyperparameters 가 임의 continuous sequence-to-sequence function 을 임의 정밀도로 근사.

**증명의 핵심 ideas**:

1. **Yun 2020 의 dense Transformer UAT**:
   - Sufficient depth + width 의 dense Transformer 가 universal approximator
   - 핵심: attention 이 임의 token-pair 정보 mixing 가능

2. **Sparse attention 으로의 확장**:
   - Dense 의 표현력을 sparse 로 simulating
   - 핵심: $O(\log T)$ layer 안에서 임의 information 전파 가능 + arbitrary computation

3. **Expander Graph 의 역할**:
   - **Expander graph**: every subset 이 sufficient neighbors
   - Local + random + global 의 union 이 expander
   - Expander 의 mixing time $O(\log T)$ — fast information propagation

4. **구체적 mechanism**:
   - **Local**: nearby token 즉시 mix
   - **Random**: long-range shortcut (small-world property)
   - **Global**: hub for centralized routing
   - 합쳐서 $O(\log T)$ depth 안에 모든 token 연결

5. **Layer 수**:
   - Dense: 1 layer 로 모든 pair 직접 attend
   - BigBird: $O(\log T)$ layer 로 multi-hop attention 통해 effective full coverage
   - $O(\log T)$ depth 는 manageable (e.g., $T = 4096$ 시 12 layer)

**이론적 의미**:

- **표현력 손실 없음**: BigBird 가 dense 와 same expressivity (with $\log T$ depth)
- **효율 gain**: $O(T)$ memory vs $O(T^2)$
- **trade-off**: depth ↑, 그러나 depth 의 cost 가 sequence cost 보다 작음

**실증 차이**:

이론은 universal but 실제로:
- BigBird 가 standard 보다 약간 worse (약 1-2% on benchmarks)
- 이론 vs practical 의 gap
- 이유: $O(\log T)$ depth 의 inefficient learning, hyperparameter tuning 어려움

**비교**:

| Method | Time | Memory | Theoretical | Practical |
|--------|------|--------|-------------|-----------|
| Dense | $O(T^2)$ | $O(T^2)$ | UAT | Standard quality |
| Longformer | $O(T w)$ | $O(T w)$ | Limited | OK |
| BigBird | $O(T (w + r))$ | $O(T (w+r))$ | UAT | Slightly worse than dense |
| Linear | $O(T)$ | $O(d^2)$ | Approximation | Worse |

따라서 BigBird 의 universal approximation 은 **이론적 reassurance** — sparse 가 fundamentally limited 가 아님. 실용 quality 는 implementation 과 hyperparameter 의존. $\square$

</details>

**문제 3** (논문 비평): Sparse attention 의 GPU 구현 어려움 — naive sparse pattern 이 dense 보다 종종 느린 이유와, structured (block-sparse) 가 어떻게 해결하는지 분석하라.

<details>
<summary>해설</summary>

**GPU 의 dense matmul 효율**:

Modern GPU (A100, H100):
- **Tensor Core**: dense FP16/BF16 matmul 에 highly optimized
- **Memory access pattern**: contiguous, predictable
- **Throughput**: ~300 TFLOPs (FP16) per A100

**Naive sparse 의 비효율**:

1. **Irregular memory access**:
   - Sparse pattern 이 random → memory load 가 scatter/gather
   - Cache miss 빈번 — bandwidth utilization ↓
   - GPU 의 SIMD execution model 과 mismatch

2. **Underutilization**:
   - Tensor Core 가 idle — sparse op 가 native 안 됨
   - FP32 fallback — 더 slow

3. **Overhead**:
   - Sparse format conversion (CSR, COO 등)
   - Mask check overhead

**예**: $T = 4096$ 에서 90% sparse:
- Theoretical: 10× speedup
- Naive PyTorch sparse: 0.5× (dense 보다 slower!)

**Structured Sparsity 의 해결**:

1. **Block-sparse**:
   - $b \times b$ block 단위로 sparsity
   - Active block 안은 dense matmul (efficient)
   - Total: $O(\text{# active blocks} \times b^2)$
   - Triton 같은 framework 가 block-sparse Tensor Core 사용

2. **2:4 Sparsity (NVIDIA Ampere+)**:
   - 매 4개 element 중 정확히 2개가 nonzero
   - Hardware-native sparsity
   - 이론적으로 2× speedup, 실제 1.5-2× (Tensor Core 의 sparse mode)

3. **Sliding Window 의 block 표현**:
   - Diagonal block 만 active
   - 매우 efficient memory access
   - Mistral 등이 채택

**Modern Implementation**:

- **DeepSpeed Sparse Attention**: block-sparse, custom CUDA
- **Triton block-sparse**: high-level DSL 로 GPU kernel 작성
- **Flash-Attention 2**: causal + sliding window 지원
- **xformers**: sparse attention 의 다양한 패턴

**예시 Speed**:

$T = 32K$, $d = 64$, sliding window $w = 4K$:

| Method | Time | Memory |
|--------|------|--------|
| Dense | OOM | OOM |
| Naive sparse mask | 5000ms | 16GB |
| Flash + sliding window | 50ms | 1GB |
| Block-sparse | 30ms | 1GB |

→ Implementation 이 critical. Naive 는 useless, structured + optimized 는 huge speedup.

**근본 통찰**:

Sparse attention 의 promise 는 **structured + hardware-aware** 구현으로만 realize. 이것이:

1. Mistral 같은 production model 이 sliding window 채택 (block-friendly)
2. Random sparsity 같은 BigBird 가 production 에서 적게 사용 (irregular, hard to optimize)
3. Sparse attention research 가 hardware 의 sparsity support (NVIDIA 2:4) 와 align

**미래**:
- H100 의 native sparsity support 더 강화
- Sparse attention 의 production adoption 증가
- 그러나 Flash Attention 의 dense efficiency 가 여전히 strong baseline

**결론**: Theoretical sparse efficiency vs practical GPU efficiency 의 gap. **Architecture-hardware co-design** 이 sparse attention 의 진정한 enabler. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-performer.md) | [📚 README](../README.md) | [다음 ▶](./05-flash-attention.md)

</div>
