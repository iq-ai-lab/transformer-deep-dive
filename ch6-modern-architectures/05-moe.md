# 05. Mixture of Experts — Sparse Transformer (Shazeer 2017, Fedus 2022)

## 🎯 핵심 질문

- MoE 의 핵심 — FFN 을 $E$ 개 expert 로 분할하고 top-k routing 시 파라미터 ↑ 계산 → 가능한 이유?
- Switch Transformer 의 top-1 routing 이 이전 top-2 보다 efficient 한 이유?
- Load balancing loss 의 필요성 — 일부 expert 만 활성 (collapse) 방지?
- Sparse activation 의 emergent specialization — expert 별 다른 token 유형 학습?
- GShard, Switch, Mixtral 등의 evolution?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

MoE 는 **scale 의 새로운 paradigm**:

1. **파라미터 vs 계산 의 분리** — 1T param 모델이 7B param 의 compute 로 동작
2. **Mixtral, GPT-4 (estimate)** — modern frontier LLM 의 조용한 표준
3. **Conditional computation** — token 별 적합한 expert 활성화
4. **Specialization** — expert 별 다른 domain 자연스럽게 학습

이 문서는 MoE 의 **routing, load balancing, 표현력** 을 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 2: [02-ffn-role.md](../ch2-transformer-architecture/02-ffn-role.md) — FFN 의 key-value memory 해석
- 이전 문서: BERT, GPT, T5

---

## 📖 직관적 이해

### MoE 의 Idea

```
Standard FFN:        x → W_1 (d → 4d) → ReLU → W_2 (4d → d) → output
                     모든 token 이 같은 weight 사용

MoE FFN:             x → router → top-k expert 선택 → 그 expert 들만 활성화
                     E expert 중 k 개만 (sparse)
```

각 token 이 다른 expert 사용 — **conditional computation**.

### Routing

```
Router: x → softmax(W_r · x)        → expert probabilities
        Pick top-k                  → selected experts
        Weighted sum of expert outputs

Switch (top-1):  x → expert_e(x)    where e = argmax router(x)
```

### Why does this scale?

```
Standard FFN: 8d² params, 8d² FLOP/token
MoE (E=8 expert, top-1): 8 × 8d² = 64d² params, 8d² FLOP/token

→ 8× param, same FLOP — capacity ↑ without compute ↑
```

### Specialization (emergent)

```
Expert 1: 학습된 후 코드 token 처리에 specialize
Expert 2: 자연어
Expert 3: 수학
...

Router 가 token 별 적절한 expert 선택 — modular specialization
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — MoE Layer

$E$ experts, each $\text{FFN}_e: \mathbb{R}^d \to \mathbb{R}^d$.

Router: $g(x) = \text{softmax}(W_r x) \in \Delta^{E-1}$ (probability simplex).

**Output**:
$$
y = \sum_{e=1}^E g_e(x) \cdot \text{FFN}_e(x)
$$

### 정의 5.2 — Top-k Sparse Routing

오직 top-k expert 만 활성:
$$
y = \sum_{e \in \text{TopK}(g(x))} \tilde{g}_e(x) \cdot \text{FFN}_e(x)
$$

with $\tilde{g}$ renormalized over top-k.

### 정의 5.3 — Switch Transformer (Fedus 2022)

Top-1 (k=1):
$$
y = \text{FFN}_{e^*}(x), \quad e^* = \arg\max_e g_e(x)
$$

Single expert per token — simplest MoE.

### 정의 5.4 — Load Balancing Loss

Routing 이 specific expert 에 collapse 안 하도록:
$$
L_{\text{aux}} = \alpha \cdot E \sum_{e=1}^E f_e \cdot P_e
$$

with:
- $f_e$ = fraction of tokens routed to expert $e$
- $P_e$ = average router probability for expert $e$
- $\alpha = 0.01$ typical

→ $f_e \approx P_e \approx 1/E$ (uniform) 시 minimum.

### 정의 5.5 — Capacity Factor

각 expert 의 capacity (max tokens to process):
$$
C = \frac{T \cdot k \cdot \text{capacity\_factor}}{E}
$$

Capacity factor (typically 1.0-1.25) — "buffer" for load imbalance.

### 정의 5.6 — Token Dropping

Capacity 초과 시 일부 token 의 expert routing 무시 (drop):
- Skip 또는 next-best expert 로
- Loss: 약간의 information 손실 vs computational efficiency

---

## 🔬 정리와 증명

### 정리 5.1 — Parameter vs Compute 의 Decoupling

Standard FFN: params = compute = $8d^2$ per layer.

MoE with $E$ experts, top-k:
- Params: $E \cdot 8d^2$
- Compute: $k \cdot 8d^2$ per token

**Ratio**: $E/k \times$ more params for same compute.

Switch (top-1, E=8): 8× more params, same compute.

### 정리 5.2 — Expert Specialization

학습 후 expert 들이 다른 input 분포 처리:
- Empirical (Switch): expert 별 다른 token type, syntactic role 등 학습
- Implicit clustering of input space

**Mechanism**: routing 이 학습 가능 → "어떤 token 이 어떤 expert" 의 mapping 자연스럽게.

### 정리 5.3 — Load Balancing 의 필요성

Trivial routing (모두 expert 1): $L_{\text{aux}}$ = $E \cdot 1 \cdot 1 = E$ (max).
Uniform: $L_{\text{aux}}$ = $E \cdot (1/E) \cdot (1/E) \cdot E = 1$ (min).

→ Uniform routing 이 $L_{\text{aux}}$ 최소화. Joint optimization 이 specialization + balance 강제.

### 정리 5.4 — Switch (Top-1) 의 정당성

Shazeer 2017 의 original: top-2 routing.
Fedus 2022 (Switch): top-1 도 충분.

**근거**:
- Top-2 의 추가 compute 가 quality gain 작음
- Top-1 의 simplicity 이 implementation, scaling advantages

→ Modern MoE 는 top-1 또는 top-2 (Mixtral).

### 정리 5.5 — Communication Overhead

Distributed MoE 의 challenge:
- Token 이 다른 GPU 의 expert 로 routing → network communication
- All-to-all communication primitive
- Network bandwidth 가 bottleneck (특히 큰 batch)

**Mitigation**:
- Expert 분산 (각 GPU 가 few experts)
- Expert parallelism + Data parallelism 결합
- Communication-aware routing (locality)

### 정리 5.6 — Mixtral 의 Architecture

Mixtral 8x7B (Mistral AI):
- 8 experts, top-2 routing
- Per layer: $8 \times 13B$ FFN params
- Per token: 13B effective compute
- Total: 47B params, 13B active per token

**비교 with dense**:
- 47B dense: full forward 47B
- Mixtral: 13B forward — 약 4× faster

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — MoE Layer 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts (each is a FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """x: (B, T, d) or (B*T, d)"""
        original_shape = x.shape
        x = x.view(-1, self.d_model)   # (N, d)
        N = x.size(0)
        
        # Router: get top-k experts and their weights
        router_logits = self.router(x)                    # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_idx = router_probs.topk(self.top_k, dim=-1)
        # Renormalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (sequentially for clarity)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                # Tokens that have e as their k-th choice
                mask = (top_k_idx[:, k] == e)
                if mask.any():
                    expert_in = x[mask]
                    expert_out = self.experts[e](expert_in)
                    out[mask] += top_k_probs[mask, k:k+1] * expert_out
        
        # Auxiliary loss for load balancing
        # f_e = fraction of tokens to e, P_e = avg prob of e
        f_e = torch.zeros(self.num_experts, device=x.device)
        for e in range(self.num_experts):
            f_e[e] = (top_k_idx == e).any(dim=-1).float().mean()
        P_e = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * (f_e * P_e).sum()
        
        return out.view(original_shape), load_balance_loss

# 테스트
torch.manual_seed(0)
moe = MoELayer(d_model=64, d_ff=256, num_experts=8, top_k=2)
x = torch.randn(2, 10, 64)
y, lb_loss = moe(x)
print(f'MoE output: {y.shape}')
print(f'Load balance loss: {lb_loss.item():.4f}')

# Param 비교
moe_params = sum(p.numel() for p in moe.parameters())
dense_ffn_params = 64 * 256 + 256 * 64 + 256 + 64   # standard FFN
print(f'\nDense FFN: {dense_ffn_params}')
print(f'MoE 8 experts: {moe_params}  ({moe_params/dense_ffn_params:.1f}× larger)')
```

### 실험 2 — Switch Transformer (Top-1)

```python
class SwitchLayer(nn.Module):
    """Top-1 routing only — Switch Transformer"""
    def __init__(self, d_model, d_ff, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        gate, expert_idx = probs.max(dim=-1)   # (N,), (N,)
        
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (expert_idx == e)
            if mask.any():
                out[mask] = gate[mask].unsqueeze(-1) * self.experts[e](x[mask])
        
        # Load balance
        f_e = torch.tensor([float((expert_idx == e).sum() / x.size(0)) for e in range(self.num_experts)])
        P_e = probs.mean(dim=0)
        lb = self.num_experts * (f_e * P_e).sum()
        
        return out.view(original_shape), lb

torch.manual_seed(0)
switch = SwitchLayer(64, 256, num_experts=8)
y, lb = switch(x)
print(f'Switch output: {y.shape}, LB loss: {lb.item():.4f}')
```

### 실험 3 — Routing Distribution 시각화

```python
import matplotlib.pyplot as plt

# 학습 안 된 router — random routing
torch.manual_seed(0)
moe_test = MoELayer(64, 256, num_experts=8, top_k=2)
x_test = torch.randn(100, 64)
_, _ = moe_test(x_test)

router_logits = moe_test.router(x_test)
router_probs = F.softmax(router_logits, dim=-1)

# Per-expert usage
top1_assignments = router_probs.argmax(dim=-1)
expert_counts = torch.zeros(8)
for e in range(8):
    expert_counts[e] = (top1_assignments == e).float().mean()

plt.bar(range(8), expert_counts.numpy())
plt.xlabel('Expert'); plt.ylabel('Fraction of tokens')
plt.title('Routing distribution (uniform target)')
plt.axhline(1/8, color='r', linestyle='--', label='Uniform target')
plt.legend(); plt.show()
# 학습 후엔 일부 imbalance 있을 수 있음 — load balance loss 가 이를 mitigate
```

### 실험 4 — Expert Specialization 시뮬레이션

```python
# 가짜 expert specialization: 작은 toy model 학습
torch.manual_seed(0)
moe_train = MoELayer(64, 128, num_experts=4, top_k=1)
opt = torch.optim.AdamW(moe_train.parameters(), lr=1e-3)

# 두 종류 task: A (random task 1), B (random task 2)
for step in range(200):
    # Sample task type
    task = torch.randint(0, 2, (8,))
    x = torch.randn(8, 1, 64)
    if task[0] == 0:
        x = x + torch.tensor(0.5)   # task 0 의 distinguishing pattern
    target = torch.randn(8, 1, 64)
    
    out, lb_loss = moe_train(x)
    loss = ((out - target) ** 2).mean() + 0.01 * lb_loss
    opt.zero_grad(); loss.backward(); opt.step()

# 학습 후 expert 사용 패턴
with torch.no_grad():
    x_a = torch.randn(50, 1, 64) + 0.5   # task 0
    x_b = torch.randn(50, 1, 64)         # task 1
    
    routes_a = moe_train.router(x_a.view(-1, 64)).argmax(dim=-1)
    routes_b = moe_train.router(x_b.view(-1, 64)).argmax(dim=-1)

print(f'Task A routing: {torch.bincount(routes_a, minlength=4).tolist()}')
print(f'Task B routing: {torch.bincount(routes_b, minlength=4).tolist()}')
# 다른 task 가 다른 expert 에 — specialization
```

### 실험 5 — Compute 비교

```python
import time

T, d = 500, 256
x = torch.randn(T, d)

# Dense FFN
ffn_dense = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))
moe_layer = MoELayer(d, 4*d, num_experts=8, top_k=2)

# Time
t0 = time.time()
for _ in range(50):
    ffn_dense(x.view(1, T, d))
t_dense = (time.time() - t0) / 50

t0 = time.time()
for _ in range(50):
    moe_layer(x.view(1, T, d))
t_moe = (time.time() - t0) / 50

print(f'Dense FFN: {t_dense*1000:.2f}ms')
print(f'MoE (8 expert, top-2): {t_moe*1000:.2f}ms (overhead from routing + sequential)')
# Real efficient MoE 는 expert parallelism 으로 native 가능
```

---

## 🔗 실전 활용

### 1. Mixtral 8x7B (Mistral AI)

- 8 experts, top-2 routing
- 47B total params, 13B active
- LLaMA-2 70B 와 competitive on benchmarks
- Open source, locally runnable

### 2. Switch Transformer (Google)

- 1.6T params (max version), 7B active per token
- Top-1 routing
- T5-style encoder-decoder
- 4× faster than dense baseline

### 3. GPT-4 (estimated)

- Mixture of Experts (rumored)
- ~1.8T total params, ~280B active
- 다중 expert 가 different tasks specialize

### 4. Chinese open-source

- DeepSeek-MoE, Qwen-MoE
- 16-256 experts
- Modern variations

### 5. Implementation Frameworks

```python
# Megatron-DeepSpeed: efficient MoE
# pip install transformers
# from transformers import MixtralForCausalLM

# Native MoE with expert parallelism
# All-to-all communication for distributed
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Independent experts | Cross-expert interaction 가능 (LIMoE) |
| Load balancing 강제 | 일부 imbalance 자연스러움 (specialized expert) |
| Top-k routing | Soft routing (all experts) 도 가능 (Sparse MoE) |
| Static expert count | Dynamic (어떤 layer 만 MoE) 가능 |
| Communication overhead | Network hardware 의존 |

---

## 📌 핵심 정리

$$\boxed{\text{MoE: } y = \sum_{e \in \text{TopK}} \tilde{g}_e(x) \cdot \text{FFN}_e(x), \quad \text{params} \uparrow E\text{-fold, compute same}}$$

| Component | Role |
|-----------|------|
| Router | Per-token expert selection |
| Top-k | Sparse activation |
| Load balance loss | Prevent collapse |
| Capacity factor | Buffer for imbalance |

| Variant | E | top-k | Note |
|---------|-----|------|------|
| Switch (2022) | up to 2048 | 1 | T5 base |
| GShard (2021) | up to 600 | 2 | TPU |
| Mixtral 8x7B | 8 | 2 | 47B/13B active |
| GPT-4 (est.) | 16-128 | 2-4 | Frontier |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Mixtral 8x7B 의 active params (per token) 을 계산하라. 47B total 의 분배는?

<details>
<summary>해설</summary>

**Mixtral 8x7B**:
- 32 layers, $d = 4096$
- 각 layer 의 FFN: 8 expert, 각 expert $4096 \to 14336 \to 4096$ (SwiGLU)
- Per expert FFN params: ~$3 \times 4096 \times 14336 \approx 176M$
- Per layer: $8 \times 176M = 1.4B$ in FFN

**Total breakdown**:
- 32 layers × 1.4B FFN = 45B
- Attention: ~4 layers $d^2 \times 32 = 2B$ (GQA-4)
- Embedding: ~250M
- **Total ≈ 47B** ✓

**Active per token (top-2)**:
- FFN: 2 experts × 176M × 32 layers = 11.3B
- Attention: 2B (always active)
- Embedding: 250M
- **Active ≈ 13.5B per token** ✓

**Effective compute**:
- ~13B params 의 forward
- LLaMA-2 13B 와 비슷한 inference cost
- LLaMA-2 70B 와 비슷한 quality
- → MoE 의 명백한 advantage. $\square$

</details>

**문제 2** (심화): Load balancing loss 가 0 (perfectly balanced) 이 항상 optimal 인가? Specialization 과의 trade-off 분석.

<details>
<summary>해설</summary>

**Perfect balance 의 case**: $f_e = P_e = 1/E$ for all $e$.

**Load balance loss**:
$$
L_{\text{aux}} = E \sum_e f_e P_e = E \cdot E \cdot (1/E)^2 = 1
$$

(이 값이 minimum 인 등가 분포)

**그러나 perfect balance 가 항상 좋은가?**

1. **Specialization 의 자연스러운 imbalance**:
   - 어떤 expert 가 frequent pattern (예: punctuation, common words) specialize
   - 다른 expert 가 rare pattern (specific vocabulary)
   - Frequent expert 가 더 자주 사용 — natural imbalance

2. **Token frequency 의 imbalance**:
   - 자연어의 token frequency 가 Zipf's law (heavy tail)
   - Frequent token 의 expert 가 더 자주 routing — balance 깨짐 자연스러움
   - Forcing balance 시 specialization 손해

3. **Computation 의 efficient distribution**:
   - GPU 별 expert 분산 시 perfect balance 가 throughput optimal
   - 그러나 model quality 는 약간 다른 distribution 이 better

**Practical Trade-off**:

- $\alpha = 0.01$ (Switch Transformer): mild balancing
- $\alpha$ 클 시: forced uniform — specialization ↓, throughput ↑
- $\alpha$ 작을 시: free specialization — collapse 위험, throughput ↓

**Modern Approach**:

- **Soft balancing**: small $\alpha$, allow some imbalance
- **Capacity factor**: 일부 imbalance 허용 (1.0-1.25)
- **Token dropping**: capacity 초과 시 drop, but 5% 미만으로 설정

**Empirical**:

Switch Transformer ablation:
- $\alpha = 0$: collapse — 일부 expert 만 사용 (quality 매우 ↓)
- $\alpha = 0.01$: optimal balance + specialization
- $\alpha = 0.1$: forced balance, specialization 약화

**근본 통찰**:

Load balance 는 **enabling mechanism** — collapse 방지가 main goal. Perfect balance 가 아닌 **enough balance** that specialization 가능.

이는 attention 의 saturation 과 비슷 — 양 극단을 피하는 것이 key. $\square$

</details>

**문제 3** (논문 비평): MoE 가 frontier LLM (GPT-4, Mixtral) 의 secret sauce 라면 왜 OpenAI, Anthropic 이 명시적으로 발표 안 하는가? MoE 의 challenge 와 future direction?

<details>
<summary>해설</summary>

**Why MoE 가 dominant 그러나 explicit 발표 X**:

1. **Competitive secrecy**:
   - Frontier model 의 architecture 가 competitive advantage
   - GPT-4, Claude 의 details 가 비공개
   - Estimate (Mixture of Experts, 1.8T total) 만 leak

2. **Implementation challenges**:
   - MoE 의 efficient implementation 이 specific GPU + framework 필요
   - Communication, expert parallelism 의 know-how
   - Open source 시 reverse engineering 가능

3. **Quality 의 specific factors**:
   - MoE 의 quality 는 expert 수, routing, training recipe 에 sensitive
   - Mixtral 의 success 는 details 의 결과 — 그대로 따라가도 같은 결과 안 됨

**MoE 의 Challenges**:

1. **Training Instability**:
   - Routing 의 학습이 unstable (expert collapse, load imbalance)
   - 큰 hyperparameter sensitivity
   - Specific training recipes 필요

2. **Inference 복잡성**:
   - Token 별 다른 expert → batch processing 비효율
   - Communication overhead in distributed inference
   - vLLM, TGI 등 inference framework 가 MoE 지원

3. **Memory**:
   - Total params 가 매우 큼 (400B+)
   - GPU 메모리 한계 — multi-GPU 필수
   - Expert offloading (CPU/disk) 으로 single GPU MoE 시도

4. **Quality Gap**:
   - Mixtral 의 quality 가 LLaMA-2 70B 보다 우수 not 압도적
   - Per-active-param 으로 ratio 의 advantage
   - 그러나 frontier 와 비교 시 capability 차이 명확

**Future Directions**:

1. **MoE + RLHF**:
   - Instruction tuning 의 MoE-specific challenges
   - Each expert 가 different alignment target?

2. **Conditional Compute**:
   - MoE 의 generalization
   - Layer 별 다른 routing
   - Token 별 layer 수 다름 (early exit)

3. **Multi-modal MoE**:
   - Image 와 text 의 different experts
   - Vision MoE (V-MoE)

4. **Expert sharing across layers**:
   - Param 절약, communication 감소
   - Cross-layer expert routing

5. **MoE 의 alternative**:
   - **Mamba**: efficient state space, no MoE
   - **DeepSeekMoE**: 256 experts, fine-grained routing
   - 다양한 sparsity paradigm

**Modern Open Source MoE**:

- Mixtral 8x7B, 8x22B (Mistral)
- DeepSeek-MoE, Qwen-MoE
- 점점 standard architecture 가 됨

**근본 통찰**:

MoE 는 **scale 의 frontier** 의 거의 필수 도구. **Param efficiency (per FLOP)** 의 best paradigm.

그러나 MoE 가 **panacea 아님**:
- Implementation 어려움
- Quality 의 specific factors
- Inference 의 complexity

미래 LLM 은 **MoE + Mamba + 다른 efficient architecture 의 hybrid**. Single architecture (Transformer dense) 의 시대는 종료, **diverse architecture 의 시대**.

GPT-5, Claude 4 가 어떤 architecture 일지 — MoE + 새로운 idea 의 결합 가능성 큼. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-vit.md) | [📚 README](../README.md) | [다음 ▶](../ch7-llm-icl/01-scaling-laws.md)

</div>
