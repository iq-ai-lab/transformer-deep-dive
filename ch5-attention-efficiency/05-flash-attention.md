# 05. Flash Attention (Dao 2022)

## 🎯 핵심 질문

- Flash Attention 의 핵심 idea — 같은 $O(T^2)$ FLOP 으로 어떻게 $4 \times$ 빠른가?
- IO-aware algorithm 의 의미 — GPU 의 SRAM 과 HBM memory hierarchy 활용?
- Block tiling + online softmax 의 어떤 mechanism?
- "Exact" attention — 근사 아닌 이유? 표현력 손실 없음?
- Flash Attention 2/3 의 추가 개선과 H100 FP8 활용?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

Flash Attention 은 **modern Transformer training 의 game changer**:

1. **Hardware-aware** — 알고리즘 자체가 GPU memory hierarchy 활용
2. **2-4× wall-clock 가속** — 같은 FLOP 으로 더 빠름
3. **5-10× memory 절약** — $O(T^2)$ 대신 $O(T)$ effective
4. **Exact** — approximation 아님, 표현력 손실 없음
5. **모든 modern LLM 의 표준** — GPT-4, LLaMA, Claude 등 모두 채택

이 문서는 Flash Attention 의 **block tiling + online softmax** algorithm 과 hardware 정당성을 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 1: [01-scaled-dot-product.md](../ch1-attention-decomposition/01-scaled-dot-product.md), [03-softmax-saturation.md](../ch1-attention-decomposition/03-softmax-saturation.md)
- 이전 문서: [01-quadratic-bottleneck.md](./01-quadratic-bottleneck.md)
- GPU memory hierarchy: SRAM, HBM, registers

---

## 📖 직관적 이해

### GPU Memory Hierarchy

```
Register:  ~10 KB per thread, ~1 cycle access
SRAM:      ~256 KB per SM,    ~1-10 cycles  ← Flash Attention 의 working set
L2 cache:  ~50 MB per GPU,    ~100 cycles
HBM:       ~80 GB,            ~400-800 cycles  ← Standard attention 의 bottleneck

→ HBM 접근이 SRAM 의 100x slower!
```

### Standard Attention 의 IO 문제

```
Standard:
  1. Compute QK^T (T × T) → write to HBM        [O(T²) HBM write]
  2. Read QK^T from HBM → compute softmax       [O(T²) HBM read]
  3. Write softmax(QK^T) to HBM                 [O(T²) HBM write]
  4. Read softmax + V → compute output          [O(T² + Td) HBM read]
  
HBM I/O: O(T²) — 메모리 transfer 가 bottleneck
```

### Flash Attention 의 Idea

```
Flash:
  1. Q, K, V 를 SRAM 에 fit 하는 block 으로 나눔
  2. 각 block pair 에 대해 SRAM 안에서 모든 op:
     - QK^T computation
     - Softmax (online)
     - Output 누적
  3. Output 만 HBM 에 write
  
HBM I/O: O(T·d) — sequence size × dim, no T² in HBM!
```

### Online Softmax

```
Standard softmax:
  m = max(scores)        ← T elements 모두 봐야 함
  exp_scores = exp(scores - m)
  softmax = exp_scores / sum(exp_scores)

Online (incremental):
  Process block-by-block, maintain (m, l) state:
    m_new = max(m_old, max(block))
    l_new = e^(m_old - m_new) · l_old + sum(e^(block - m_new))
  Final: divide by l_new
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Standard Attention (재확인)

$$
S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{T \times T}
$$
$$
A = \text{softmax}(S) \in \mathbb{R}^{T \times T}
$$
$$
O = AV \in \mathbb{R}^{T \times d_v}
$$

Memory: $S, A$ 가 $O(T^2)$ — HBM 저장.

### 정의 5.2 — IO Complexity

GPU 의 memory transfer 횟수를 IO 라 함. Wall-clock time 의 dominant factor 종종 IO.

Standard attention IO: $O(T^2)$ (writing $S, A$ to HBM).

### 정의 5.3 — Block Partitioning

$Q, K, V$ 를 size $B$ block 으로 나눔:
- $Q_i \in \mathbb{R}^{B \times d_k}$ for $i \in \{1, \ldots, T/B\}$
- 마찬가지로 $K_j, V_j$

### 정의 5.4 — Online Softmax State

블록 처리 중 maintain:
- $m_i$: 지금까지 본 max score (per row $i$)
- $\ell_i$: 지금까지 normalizer
- $O_i$: 지금까지 output (running sum)

새 block 처리 시 update.

### 정의 5.5 — Flash Attention Algorithm

```
Initialize O, m, ℓ in HBM (initial values)
For j in K, V blocks:
    Load K_j, V_j to SRAM
    For i in Q blocks:
        Load Q_i, m_i, ℓ_i, O_i to SRAM
        Compute S_ij = Q_i K_j^T / √d_k
        Compute m_ij = max(S_ij)
        m_new = max(m_i, m_ij)
        Compute P_ij = exp(S_ij - m_new)
        ℓ_ij = sum(P_ij)
        ℓ_new = e^(m_i - m_new) ℓ_i + ℓ_ij
        O_new = (1/ℓ_new) [e^(m_i - m_new) ℓ_i O_i + e^(m_ij - m_new) P_ij V_j]
        Write O_new, m_new, ℓ_new to HBM
```

(Dao 2022 의 Algorithm 1 simplified)

### 정의 5.6 — Block Size Choice

SRAM size $M$ 가정. Block size $B = M/(4 d_k)$ 로 $K, V$ block 이 SRAM 에 fit.

Total IO: $O(T \cdot d \cdot T/B) = O(T^2 d / B)$.

$B = O(M/d)$ → IO = $O(T^2 d^2 / M)$.

---

## 🔬 정리와 증명

### 정리 5.1 — Online Softmax 의 정확성

Block-by-block 처리한 softmax 가 standard 와 정확히 같음.

**증명**:

Block $A$ 와 block $B$ 의 union 의 softmax:
$$
\text{softmax}([A; B])_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}
$$

with $m = \max(s)$.

Online: $A$ 만 처리 후 $m_A, \ell_A = \sum_j e^{s_{Aj} - m_A}$.
$B$ 처리 시: $m_B, \ell_B$.
Combined: $m = \max(m_A, m_B)$, $\ell = e^{m_A - m} \ell_A + e^{m_B - m} \ell_B$.

이는 정확히 union 의 softmax denominator. 따라서 online 이 standard 와 같음 $\square$.

### 정리 5.2 — Flash 의 IO Complexity

Block size $B$:
- 각 $K_j, V_j$ block 이 모든 $Q_i$ 와 interact → $T/B$ 번 read
- Total HBM read of $K, V$: $T d \cdot T/B = T^2 d / B$

$B$ 가 SRAM 에 fit 하는 max → $B \propto M/d$ where $M$ = SRAM size.

$$
\text{IO} = O(T^2 d^2 / M)
$$

**Compared with standard $O(T^2)$**:

A100 SRAM ≈ 192KB per SM, $d = 128$ 시 $B \approx 100$ — practical block size.
$T = 4096$, $M = 192KB$, FP16: IO = $4096^2 \times 128^2 / 192KB \approx 14M$ vs standard $16M$.

→ 비슷하지만 **constant factor** 가 매우 다름 (HBM access pattern 효율).

### 정리 5.3 — Wall-clock Speedup

Theoretical FLOP 같음. 그러나:
- Standard: HBM transfer dominant (large $T^2$)
- Flash: SRAM 안에서 op, HBM transfer 적음

**Empirical**:
- $T = 1024$: Flash 1.7×
- $T = 4096$: Flash 2.4×
- $T = 16384$: Flash 4.0×

Long sequence 에서 advantage ↑.

### 정리 5.4 — Memory Savings

Standard: $O(T^2)$ memory for $S, A$.
Flash: $O(T \cdot d)$ memory (block buffer + output).

**Savings**: $T = 8192, d = 128$:
- Standard: $T^2 \cdot 4 = 256MB$
- Flash: $T \cdot d \cdot 4 = 4MB$
- 64× memory savings

### 정리 5.5 — Backward Pass

Flash attention 의 backward 도 같은 idea:
- Block tiling
- Online recomputation (forward 의 일부 결과 재계산)
- Memory: $O(T)$ vs standard $O(T^2)$

**Trade-off**: 일부 forward op 재계산 (~30% extra FLOPs) but 큰 memory 절약.

### 정리 5.6 — Exact Attention

Flash 가 approximation 아님 — 정확한 $\text{softmax}(QK^\top)V$ 계산.

**증명**: Online softmax 가 정확 (정리 5.1) + matrix multiplication 의 결합법칙으로 block-wise 가 sequential 과 같음 $\square$.

→ **표현력 손실 없음**, 단순히 memory access pattern 의 최적화.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Online Softmax Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np

def online_softmax(scores, block_size=64):
    """블록 단위로 softmax 계산 — Flash Attention 의 핵심"""
    T = scores.size(0)
    n_blocks = (T + block_size - 1) // block_size
    
    # State
    m = torch.tensor(float('-inf'))
    ell = torch.tensor(0.0)
    
    # Pass 1: compute m, ell
    for j in range(n_blocks):
        block = scores[j*block_size:(j+1)*block_size]
        m_j = block.max()
        m_new = max(m, m_j)
        ell = torch.exp(m - m_new) * ell + torch.exp(block - m_new).sum()
        m = m_new
    
    # Pass 2: compute softmax (이론적, Flash 는 fused)
    return torch.exp(scores - m) / ell

# 검증
torch.manual_seed(0)
T = 256
scores = torch.randn(T) * 5

p_standard = F.softmax(scores, dim=-1)
p_online = online_softmax(scores, block_size=32)

print(f'Max diff: {(p_standard - p_online).abs().max():.2e}')  # ≈ 0
```

### 실험 2 — Flash Attention Forward (간단한 implementation)

```python
def flash_attention_forward(Q, K, V, block_size_q=32, block_size_kv=32):
    """Flash Attention 의 forward — pedagogical implementation"""
    T, d = Q.size()
    O = torch.zeros(T, d)
    L = torch.zeros(T)   # row max
    M = torch.full((T,), float('-inf'))  # rolling max
    
    n_kv_blocks = (T + block_size_kv - 1) // block_size_kv
    n_q_blocks = (T + block_size_q - 1) // block_size_q
    
    for j in range(n_kv_blocks):
        K_j = K[j*block_size_kv:(j+1)*block_size_kv]
        V_j = V[j*block_size_kv:(j+1)*block_size_kv]
        
        for i in range(n_q_blocks):
            Q_i = Q[i*block_size_q:(i+1)*block_size_q]
            
            # Block 의 attention scores
            S_ij = Q_i @ K_j.T / np.sqrt(d)   # (B_q, B_kv)
            
            # Online softmax update
            m_block = S_ij.max(dim=-1).values
            m_new = torch.maximum(M[i*block_size_q:(i+1)*block_size_q], m_block)
            
            P = torch.exp(S_ij - m_new.unsqueeze(-1))
            l_block = P.sum(dim=-1)
            
            # Update O, L, M
            row_slice = slice(i*block_size_q, (i+1)*block_size_q)
            old_M = M[row_slice]; old_L = L[row_slice]; old_O = O[row_slice]
            
            l_new = torch.exp(old_M - m_new) * old_L + l_block
            O[row_slice] = (torch.exp(old_M - m_new).unsqueeze(-1) * old_L.unsqueeze(-1) * old_O + P @ V_j) / l_new.unsqueeze(-1)
            L[row_slice] = l_new
            M[row_slice] = m_new
    
    return O

# 검증
torch.manual_seed(0)
T, d = 64, 16
Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)

# Standard
out_std = F.softmax(Q @ K.T / np.sqrt(d), dim=-1) @ V

# Flash (block-wise)
out_flash = flash_attention_forward(Q, K, V, block_size_q=8, block_size_kv=8)

print(f'Standard output: {out_std[0, :5]}')
print(f'Flash output:    {out_flash[0, :5]}')
print(f'Max diff:        {(out_std - out_flash).abs().max():.2e}')
# 거의 0 — Flash 가 정확 ✓
```

### 실험 3 — Memory Comparison

```python
def measure_memory(impl, T, d):
    """Approximate memory usage"""
    if impl == 'standard':
        # Q, K, V, scores, attn, output
        return (3 * T * d + T * T + T * T + T * d) * 4 / 1e6  # MB
    elif impl == 'flash':
        # Q, K, V, output, block buffers (~B*d), state (T*small)
        return (3 * T * d + T * d + 2 * 64 * d + 2 * T) * 4 / 1e6

print(f'{"T":>8} | {"Standard":>12} | {"Flash":>10}')
print('-' * 35)
for T in [512, 2048, 8192, 32768]:
    s = measure_memory('standard', T, 64)
    f = measure_memory('flash', T, 64)
    print(f'{T:8d} | {s:10.1f}MB | {f:8.1f}MB')

# T=32K 시 Standard 4GB, Flash 8MB → 500× 절약
```

### 실험 4 — PyTorch native Flash Attention

```python
# PyTorch 2.0+ 의 native flash attention support
torch.manual_seed(0)
T, d = 1024, 64
Q = torch.randn(1, 1, T, d)   # (B, H, T, d_k)
K = torch.randn(1, 1, T, d)
V = torch.randn(1, 1, T, d)

# Native scaled_dot_product_attention (자동 Flash backend)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
print(f'Output: {out.shape}')

# Causal version
out_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f'Causal output: {out_causal.shape}')

# Backend 확인
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out_flash = F.scaled_dot_product_attention(Q, K, V)
print(f'Flash backend used (CUDA only): {out_flash.shape}')
```

### 실험 5 — Speed Comparison (PyTorch 2.0+)

```python
import time

def benchmark(T, d, use_flash=True):
    Q = torch.randn(2, 8, T, d, device='cuda' if torch.cuda.is_available() else 'cpu')
    K = torch.randn_like(Q); V = torch.randn_like(Q)
    
    # Warmup
    for _ in range(3):
        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                F.scaled_dot_product_attention(Q, K, V)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                F.scaled_dot_product_attention(Q, K, V)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                F.scaled_dot_product_attention(Q, K, V)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
                F.scaled_dot_product_attention(Q, K, V)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / 10

# Note: requires GPU for true Flash backend
# CPU 에서는 math fallback
```

---

## 🔗 실전 활용

### 1. PyTorch 2.0+ 의 표준

```python
# Modern PyTorch
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, attn_mask=None)
# 자동으로 Flash backend 선택 (GPU 환경)
```

### 2. 모든 Modern LLM 의 채택

- **GPT-3.5+**: Flash Attention 2
- **GPT-4**: Flash + custom optimizations
- **LLaMA-2/3**: Flash Attention
- **Claude**: Flash Attention variants
- **Gemini**: TPU 의 native attention (similar idea)

### 3. Flash Attention 2 (Dao 2023)

추가 개선:
- Better parallelism — work distribution 최적화
- Better partitioning — output 의 fewer non-matmul ops
- Causal-specific optimization

→ Flash 1 대비 1.5-2× 추가 가속.

### 4. Flash Attention 3 (2024)

H100 의 FP8, async 활용:
- 1.5-2× faster than Flash 2 on H100
- FP8 support — H100 의 native

### 5. Library Support

```bash
# pip install flash-attn
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| GPU only | CPU 에는 less benefit (memory hierarchy 다름) |
| SRAM size 알려짐 | Hardware-specific tuning |
| Standard mask patterns | Custom mask 는 추가 work |
| FP16/BF16 / FP8 | FP32 도 가능 but slower |
| Single GPU | Multi-GPU 는 추가 communication |

---

## 📌 핵심 정리

$$\boxed{\text{Flash: same } O(T^2) \text{ FLOP, but IO } O(T^2 d / B) \text{ → wall-clock 2-4× faster, memory } O(T)}$$

| Feature | Standard | Flash |
|---------|----------|-------|
| FLOP | $O(T^2 d)$ | $O(T^2 d)$ |
| HBM IO | $O(T^2)$ | $O(T^2 d^2/M)$ |
| SRAM usage | Minimal | Heavy (block) |
| Memory | $O(T^2)$ | $O(T)$ |
| Wall-clock | Baseline | 2-4× faster |
| Approximation | Exact | Exact |

| Modern Variant | Year | Improvement |
|----------------|------|-------------|
| Flash Attention | 2022 | Initial 2-4× |
| Flash Attention 2 | 2023 | Additional 1.5-2× |
| Flash Attention 3 | 2024 | H100 FP8 + async |

---

## 🤔 생각해볼 문제

**문제 1** (기초): A100 GPU 의 SRAM 192KB, $d_k = 64$ (FP16) 일 때 Flash Attention 의 block size $B$ 를 계산하라.

<details>
<summary>해설</summary>

SRAM 에 fit 해야 하는 것: $K_j, V_j$ block 한 쌍 (그리고 block-wise $Q_i$ tile, intermediate $S_{ij}$).

대략:
- $K_j, V_j$: $2 \times B \times d_k \times 2$ bytes (FP16)
- $Q_i$: $B \times d_k \times 2$
- $S_{ij}$: $B \times B \times 4$ (FP32 for accumulation)

Total: $B \cdot d_k \cdot 6 + B^2 \cdot 4 \leq M = 192KB = 196608$ bytes

For $d_k = 64$:
- Linear part: $B \cdot 64 \cdot 6 = 384B$
- Quadratic part: $4 B^2$

$4B^2 + 384B \leq 196608$
$B \leq \sqrt{196608/4} = 222$ (quadratic dominant for large B)

**Practical**: $B = 128$ (power of 2, leaves room for register spillover, output, etc.)

Flash Attention 의 actual implementation 은 careful tuning — Dao 2022 가 $B_q = B_{kv} = 64-128$ 추천. $\square$

</details>

**문제 2** (심화): Flash Attention 이 backward pass 도 efficient 하게 처리하는 mechanism 은? Recomputation trick 의 cost-benefit 분석.

<details>
<summary>해설</summary>

**Backward Pass 의 challenge**:

Standard backward of attention:
1. $\nabla O \to \nabla A$: needs $V$
2. $\nabla A \to \nabla S$: needs $A$ (softmax Jacobian)
3. $\nabla S \to \nabla Q, \nabla K$: needs $Q, K$

→ Need $A$ (the $T \times T$ attention matrix) for backward.

**Standard backward IO**:
- Forward 시 $A$ HBM 에 save → backward 시 read
- $O(T^2)$ memory for storing $A$

**Flash backward (recomputation trick)**:

Save 만:
- Output $O$
- LogSumExp $L = \log \sum e^{S}$ per row (size $T$)
- $m, \ell$ 의 forward 의 final state

Backward 시:
- Block-wise recompute $S_{ij} = Q_i K_j^T$
- LSE 정보로 $A_{ij} = \exp(S_{ij} - L_i)$ 즉시 계산
- 표준 chain rule 적용

**Cost**:
- Forward 의 ~30% extra FLOPs (recomputation)
- Memory: $O(T)$ for LSE vs $O(T^2)$ for $A$

**Benefit**:
- $T = 4K, d = 128$: 64MB vs 1MB — 64× memory savings
- Long sequence 에서 학습 가능 (메모리 bottleneck 해결)
- Wall-clock: 약간 slower per-step but enables larger context

**Why worth it**:

Modern LLM 학습은 **memory-bound**, not compute-bound. Memory 절약이 batch size, sequence length 의 직접 enabler.

**Comparison**:

| Method | Memory | FLOPs | Wall-clock |
|--------|--------|-------|------------|
| Save A | $O(T^2)$ | 1× | Baseline |
| Recompute (Flash) | $O(T)$ | 1.3× | Slightly slower |
| Gradient checkpointing (entire layer) | $O(T)$ | 2× (full re-forward) | Slower |

→ Flash 의 backward 가 sweet spot — memory 절약 + 적당한 compute overhead. $\square$

</details>

**문제 3** (논문 비평): Flash Attention 이 ML algorithm 과 hardware 의 co-design 의 milestone 이다. 이런 IO-aware approach 가 다른 ML primitives (matmul, softmax 등) 에도 적용 가능한가? 미래 hardware-software co-design 의 방향은?

<details>
<summary>해설</summary>

**Flash Attention 의 general principles**:

1. **Memory hierarchy aware**: SRAM ≪ HBM, transfers cost > compute
2. **Block tiling**: working set 을 fast memory 에
3. **Operator fusion**: 여러 op 을 single kernel 로
4. **Recomputation trade-off**: memory 절약 위해 compute 추가

**다른 ML primitive 에 적용**:

1. **Layer Normalization**:
   - Standard: 두 pass (mean, variance, then normalize)
   - Fused: single pass (Welford's algorithm, online statistics)
   - 이미 PyTorch 구현 안에서 fused

2. **Matrix Multiplication**:
   - 이미 highly optimized (cuBLAS)
   - Block tiling 의 oldest example
   - Tensor Core 가 hardware support

3. **Softmax + CE Loss**:
   - Fused: log-softmax + cross-entropy 한 번에
   - Memory 절약 + numerical stability

4. **Adam Optimizer Step**:
   - Fused multi-tensor apply (apex, PyTorch)
   - Per-parameter $m, v$ update + weight update single kernel

5. **Backward Operations**:
   - Backward of LN, GELU 등 fused 가능
   - Activation checkpointing 의 일반화

**미래 Hardware-Software Co-design**:

1. **NVIDIA Hopper / H100**:
   - **Transformer Engine**: FP8 mixed precision
   - **Async Tensor Core**: overlap compute and memory
   - **Distributed Shared Memory**: cross-SM SRAM access

2. **Specialized Accelerators**:
   - TPU v5: TPU pod 의 specialized matmul
   - **Cerebras Wafer-Scale**: 별도 paradigm
   - **Groq**: deterministic, low-latency LLM inference

3. **Algorithm-Hardware co-design**:
   - **Mamba**: scan algorithm 이 GPU-friendly
   - **MoE routing**: hardware-aware sparsity
   - **Speculative decoding**: 다양한 model 협력

4. **Memory Innovations**:
   - **HBM3, HBM4**: 더 fast HBM
   - **CXL**: GPU 간 memory pooling
   - **PCM, ReRAM**: 새 memory technology

5. **Training Optimizations**:
   - **Activation Recomputation**: Flash 의 일반화
   - **Pipeline Parallelism**: sequence 처리의 시간 nominal
   - **Mixture-of-Experts**: sparse compute

**Universal Principle**:

"Compute is cheap, memory access is expensive" — modern ML 의 본질.

Future:
- Models 가 "compute-optimal" 에서 "IO-optimal" 로 design
- Quantization (INT4, FP4) 가 memory bandwidth 의 직접 ↓
- Sparse + dense hybrid 가 both compute and memory 절약

**Flash Attention 의 lasting impact**:

Modern LLM 학습이 가능 — 32K, 128K, 1M context 가 Flash 없이 불가능. **Hardware-software co-design** 의 milestone.

Future LLM architecture 가 hardware constraint 더 직접 고려 — abstract ML algorithm 에서 system-aware design 으로 paradigm shift. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-sparse-attention.md) | [📚 README](../README.md) | [다음 ▶](./06-mqa-gqa.md)

</div>
