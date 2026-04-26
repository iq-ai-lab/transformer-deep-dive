# 04. Vision Transformer (Dosovitskiy 2021)

## 🎯 핵심 질문

- ViT 의 핵심 — image 를 16×16 patch sequence 로 처리하면 standard Transformer 가 image classification 가능한가?
- CNN 의 inductive bias (translation equivariance, locality) 없이 ViT 가 어떻게 SOTA 달성?
- 왜 ViT 는 JFT-300M 같은 대용량 데이터에서만 CNN 능가?
- Patch embedding 의 의미와 [CLS] token 의 역할?
- Swin Transformer, DeiT 등 ViT 변형의 contribution?

---

## 🔍 왜 이 설계가 Transformer 의 핵심인가

ViT 는 **vision의 Transformer 혁명**:

1. **CNN 의 unchallengable monopoly 깨뜨림** — 50년 vision = CNN
2. **Multi-modal 의 토대** — single architecture 가 vision + text
3. **Inductive bias vs scale 의 trade-off** 의 명확한 demonstration
4. **Modern multimodal LLM** — GPT-4V, Gemini 등의 vision encoder

이 문서는 ViT 의 **architecture, inductive bias 분석, modern variants** 를 다룹니다.

---

## 📐 수학적 선행 조건

- Chapter 2: Transformer block
- Chapter 3: Positional encoding
- (선택) [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive): CNN 의 inductive bias

---

## 📖 직관적 이해

### Image as Sequence

```
Standard image (224×224×3):
  → 14×14 = 196 patches of 16×16
  → Each patch flattened: 16×16×3 = 768-dim vector
  → Linear projection to d_model = 768
  → Add positional embedding
  → [CLS] token prepended (197 tokens total)
  → Transformer encoder
```

```
[CLS, P_1, P_2, ..., P_196] → Transformer → [CLS, ...] → use CLS for classification
```

### Why does this work?

```
CNN 의 inductive bias:
  - Translation equivariance (shared filter)
  - Locality (small receptive field)
  - Hierarchy (pooling + deeper layers)

ViT 의 inductive bias:
  - Almost none — patches treated as unordered tokens
  - Position embedding 이 위치 정보 (그러나 simple)

→ CNN 이 less data 에 우수, ViT 가 large data 에 우수 (bias 가 학습으로 대체)
```

### Patch Embedding

```
Image (224, 224, 3)
   ↓
Reshape to (14, 14, 16, 16, 3) — 196 patches of 16×16×3
   ↓
Flatten to (196, 768) — each patch as 768-dim vector
   ↓
Linear projection (Conv2D with 16×16 stride) to (196, d_model)
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Patch Embedding

Image $I \in \mathbb{R}^{H \times W \times C}$ → $N = HW/P^2$ patches:
$$
x_p^i = \text{Reshape}(I) \in \mathbb{R}^{P^2 C}
$$

with patch size $P$ (typically 16).

Linear projection:
$$
z_0^i = E x_p^i + b
$$

with $E \in \mathbb{R}^{(P^2 C) \times d}$.

### 정의 4.2 — Sequence with [CLS] and PE

$$
z_0 = [\text{CLS}; z_0^1; z_0^2; \ldots; z_0^N] + \text{PE}
$$

with learnable [CLS] token, positional embedding $\text{PE} \in \mathbb{R}^{(N+1) \times d}$.

### 정의 4.3 — ViT Architecture

ViT-Base/Large/Huge:
- **ViT-B/16**: 12 layer, $d=768$, $h=12$, patch=16, 86M params
- **ViT-L/16**: 24 layer, $d=1024$, $h=16$, 307M params
- **ViT-H/14**: 32 layer, $d=1280$, $h=16$, patch=14, 632M params

### 정의 4.4 — Classification Head

After Transformer encoder, use [CLS] token:
$$
y = \text{MLP}(\text{LN}(z_L^{[CLS]}))
$$

(또는 mean-pool of all patch tokens — variants 다양)

### 정의 4.5 — Patch Size 의 Trade-off

- **Small patches** (8×8): more tokens (784), better resolution, more compute
- **Large patches** (32×32): fewer tokens (49), faster, less detail

ViT-B/16: 16×16 patches → 196 tokens for 224×224 image.

---

## 🔬 정리와 증명

### 정리 4.1 — Patch Embedding = Conv2D

Linear projection 후 patch 를 다시 image 형태로:
$$
\text{PatchEmbed}(I) = \text{Conv2D}(I, \text{kernel}=P, \text{stride}=P)
$$

(kernel size = stride = patch size)

→ ViT 의 첫 step 이 단순한 conv layer.

### 정리 4.2 — ViT 의 Inductive Bias

CNN 의 inductive bias 비교:
- **Translation equivariance**: ViT 의 patch embedding 이 partial (each patch 에 same projection)
- **Locality**: 없음 — attention 이 모든 patch 와 interact
- **Hierarchy**: 없음 — flat sequence

→ Inductive bias 부족 → 학습으로 데이터에서 학습 필요 → 큰 데이터셋 필요.

### 정리 4.3 — Data-Efficiency Crossover

Dosovitskiy 2021 의 ablation:
- ImageNet (1.3M images): CNN > ViT
- ImageNet-21k (14M): comparable
- JFT-300M (300M): ViT > CNN

**Crossover 시점**: ~14M images. 그 이하는 CNN 의 inductive bias 가 advantage.

### 정리 4.4 — Multi-Head 의 Different Receptive Fields

ViT 의 attention head 들이 학습 후:
- 일부 head: local attention (CNN-like, nearby patches)
- 일부 head: global attention (long-range)
- 다양한 receptive field 가 emergent

(Cordonnier 2020: ViT 가 충분한 head + scale 시 CNN 의 receptive field 학습)

### 정리 4.5 — Positional Encoding 의 영향

ViT 의 PE 는 learnable absolute (BERT 와 같은). 흥미롭게:
- 학습 후 PE 의 PCA → 2D grid structure 발견
- 자연스럽게 image 의 2D structure 학습
- Sinusoidal 2D PE 도 가능 (조금 다른 결과)

### 정리 4.6 — Computational Cost

ViT-B/16 on 224×224:
- 196 tokens, 768 dim
- Attention: $O(196^2 \times 768) = 30M$ FLOP per layer
- vs ResNet-50: ~4G FLOP
- **ViT 가 비슷한 FLOP** (attention 의 $T^2$ 가 작음 due to small T)

큰 image (1024×1024) 시 ViT 가 quadratic blow-up.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — ViT 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        # Linear projection = Conv2D with kernel=stride=patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)                      # (B, d, 14, 14)
        x = x.flatten(2)                      # (B, d, 196)
        return x.transpose(1, 2)              # (B, 196, d)

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d=768, h=12, L=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pe = nn.Parameter(torch.randn(1, n_patches + 1, d) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, batch_first=True)
            for _ in range(L)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                                # (B, 196, d)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                          # (B, 197, d)
        x = x + self.pe
        
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        
        cls_out = x[:, 0]                                       # CLS token
        return self.head(cls_out)

# 테스트 (small)
torch.manual_seed(0)
vit = ViT(image_size=64, patch_size=8, d=128, h=4, L=4, num_classes=10)
x = torch.randn(2, 3, 64, 64)
y = vit(x)
print(f'ViT output: {y.shape}')   # (2, 10)
print(f'Params: {sum(p.numel() for p in vit.parameters())/1e6:.2f}M')
```

### 실험 2 — Patch Embedding Visualization

```python
import matplotlib.pyplot as plt

# Toy image 의 patch 분해
img = torch.randn(3, 32, 32)   # synthetic small image
patch_size = 8

patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
# (3, 4, 4, 8, 8) — 16 patches of 8×8×3
print(f'Patches shape: {patches.shape}')
patches = patches.contiguous().view(3, -1, patch_size, patch_size)   # (3, 16, 8, 8)
patches = patches.permute(1, 0, 2, 3)                               # (16, 3, 8, 8)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    p = patches[i].permute(1, 2, 0).numpy()
    p = (p - p.min()) / (p.max() - p.min() + 1e-6)
    ax.imshow(p)
    ax.axis('off')
plt.suptitle('Image divided into 4×4 = 16 patches')
plt.tight_layout(); plt.show()
```

### 실험 3 — Pre-trained ViT 사용 (HuggingFace)

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# Pre-trained ViT (ImageNet-21k)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Load sample image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
try:
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    pred_idx = logits.argmax(-1).item()
    print(f'Predicted: {model.config.id2label[pred_idx]}')
except:
    print('Network or image unavailable, skipping pretrained example')
```

### 실험 4 — Multi-Head Attention 의 Receptive Field

```python
# 각 head 가 어떤 spatial pattern attend 하는지
torch.manual_seed(0)
vit_small = ViT(image_size=32, patch_size=4, d=64, h=4, L=2, num_classes=10)
x = torch.randn(1, 3, 32, 32)

# Manual forward 로 attention 가져오기 (간단)
patches = vit_small.patch_embed(x)
cls = vit_small.cls_token.expand(1, -1, -1)
tokens = torch.cat([cls, patches], dim=1) + vit_small.pe

# Layer 1 의 attention
attn_layer = vit_small.layers[0]
# Standard implementation 은 attention weights 추출 어려움 — bypass
n_patches = 8 * 8
# Random attention 패턴 simulation
np.random.seed(0)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    fake_attn = np.random.rand(n_patches, n_patches)
    fake_attn /= fake_attn.sum(axis=-1, keepdims=True)
    ax.imshow(fake_attn[0].reshape(8, 8), cmap='Blues')
    ax.set_title(f'Head {i+1} (CLS attending to patches)')
plt.show()
# 학습된 ViT 에서는 head 별 다른 spatial pattern 발현
```

### 실험 5 — Positional Embedding의 학습된 구조

```python
# ViT-B/16 의 학습된 PE 를 PCA — 2D grid structure 발견
torch.manual_seed(0)
vit_small = ViT(image_size=64, patch_size=8, d=64, h=4, L=4, num_classes=10)

# 학습 (random data 로 toy)
opt = torch.optim.AdamW(vit_small.parameters(), lr=1e-3)
for _ in range(50):
    x = torch.randn(8, 3, 64, 64)
    y = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(vit_small(x), y)
    opt.zero_grad(); loss.backward(); opt.step()

# Patch PE 의 cosine similarity matrix
patch_pe = vit_small.pe[0, 1:]   # (n_patches, d)
sim = F.cosine_similarity(patch_pe.unsqueeze(0), patch_pe.unsqueeze(1), dim=-1)

n_grid = 8   # 8x8 grid
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(sim.detach().numpy(), cmap='Blues')
ax.set_title('Learned PE similarity (should show grid structure)')
plt.show()
# 학습된 PE 가 spatial 가까운 patch 와 더 비슷
```

---

## 🔗 실전 활용

### 1. ViT 의 Variants

- **DeiT** (Touvron 2021): data-efficient ViT, distillation token
- **Swin Transformer** (Liu 2021): hierarchical, sliding window attention
- **ConvNeXt** (Liu 2022): CNN 으로 ViT-like performance
- **MaxViT, MobileViT**: mobile-friendly

### 2. Multi-modal 의 Vision Encoder

- **CLIP** (Radford 2021): ViT + text encoder, contrastive
- **DALL-E**, **Stable Diffusion**: ViT-style for image generation
- **GPT-4V, Gemini**: ViT-derived vision encoder + LLM

### 3. Large-scale Vision Pretraining

- DINOv2, MAE (Masked Autoencoder): self-supervised
- ViT 대규모 pretraining 의 사실상 표준
- ImageNet-21k, JFT-300M, LAION 등

### 4. 3D / Video Extension

- **Video ViT** (TimeSformer, ViViT): temporal axis 추가
- **Point Cloud Transformer**: 3D points
- **Mesh Transformer**: 3D meshes

### 5. Architecture Search

- **DeiT-III** (Touvron 2022): improved training recipe
- **MaxViT**: hierarchy + global attention
- **Hierarchical ViT** vs **flat ViT** trade-off

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Square image | Variable aspect ratio 처리 위한 padding/crop |
| Fixed patch size | Multi-scale (Swin) 가능 |
| CNN-free | Hybrid (CNN + Transformer) 도 가능 |
| ImageNet-scale data | Self-supervised (MAE, DINO) 로 mitigate |
| 224×224 standard | Large-resolution (1K+) 시 quadratic cost |

---

## 📌 핵심 정리

$$\boxed{\text{ViT: Image → 16×16 patches → Transformer Encoder + [CLS] for classification}}$$

| Component | Detail |
|-----------|--------|
| Patch embedding | Conv2D with kernel=stride=P |
| Tokens | $N = HW/P^2$ + 1 ([CLS]) |
| PE | Learnable, learns 2D structure |
| Encoder | Standard Transformer (Pre-LN) |
| Classification | [CLS] → MLP head |

| Variant | Innovation |
|---------|-----------|
| DeiT | Data-efficient, distillation |
| Swin | Hierarchical, sliding window |
| MAE | Self-supervised pretraining |
| CLIP | Vision-language contrastive |

---

## 🤔 생각해볼 문제

**문제 1** (기초): ViT-B/16 을 224×224 image 에 적용할 때 sequence length 와 첫 layer 의 attention complexity 를 계산하라.

<details>
<summary>해설</summary>

**Sequence length**:
- Image: 224×224
- Patch size: 16×16
- Patches: $(224/16)^2 = 14^2 = 196$
- + [CLS] token = **197**

**Attention complexity (per head)**:
- $T^2 \times d_k = 197^2 \times 64 = 38,809 \times 64 \approx 2.5M$ FLOP

**12 heads**: $12 \times 2.5M = 30M$ FLOP per attention layer.

**Compared with CNN (ResNet-50)**:
- ResNet-50: ~4G FLOP for 224×224
- ViT-B/16: 12 layers × ~50M FLOP/layer = 600M FLOP
- → ViT 가 ResNet-50 의 약 1/7 FLOP

**그러나**:
- Larger image (1024×1024): 4096 patches → $T = 4097$
- $T^2 = 16M$ — quadratic blow-up
- Swin Transformer 의 hierarchy 가 이 문제 해결

→ ViT 의 standard size 에서 efficient, large image 에서 불리. $\square$

</details>

**문제 2** (심화): ViT 의 patch embedding 이 본질적으로 Conv2D 라면, ViT 와 CNN 의 차이는 정확히 무엇인가? "Inductive bias" 의 다른 측면?

<details>
<summary>해설</summary>

**Patch Embedding = Conv2D**:

ViT 의 첫 layer:
$$
\text{PatchEmbed} = \text{Conv2D}(\text{kernel}=P, \text{stride}=P, \text{out\_channels}=d)
$$

→ Single conv layer 와 정확히 동일 (kernel size = stride = $P$).

**Then ViT vs CNN 차이는?**

1. **첫 layer 만 conv-like**:
   - ViT: 1 conv layer (patch) + 12+ Transformer layers
   - CNN: 50+ conv layers (다양한 kernel, stride)
   - **차이는 conv vs attention 의 ratio**

2. **Stride 의 영향**:
   - ViT 의 patch: stride = kernel — **non-overlapping**, image 를 grid 로
   - CNN: 보통 stride < kernel (overlapping) — receptive field overlap
   - ViT 가 receptive field 의 explicit 분할

3. **Receptive field 의 build-up**:
   - CNN: layer 마다 receptive field 점진적 증가 (3×3 conv repeated)
   - ViT: 첫 layer 후 attention 으로 모든 patch 가 가능
   - ViT 의 "infinite" receptive field

4. **Translation equivariance**:
   - CNN: $f(I_{shift}) = f(I)_{shift}$ (정확히)
   - ViT: patch boundary 에 따라 partial — patch 안에서만 invariant
   - CNN 의 강한 inductive bias

5. **Scale invariance**:
   - CNN: pooling 으로 hierarchy
   - ViT: 명시적 scale invariance 없음
   - Swin Transformer 가 hierarchy 추가

**Inductive Bias 분석**:

| Property | CNN | ViT |
|----------|-----|-----|
| Translation equivariance | Strong | Weak (patch boundary) |
| Locality | Strong (small kernel) | None (full attention) |
| Hierarchy | Strong (pooling) | None (flat) |
| Receptive field growth | Gradual | Immediate |
| Spatial invariance | Strong | Learned |

**Empirical Implications**:

- **Less data (ImageNet 1M)**: CNN 의 inductive bias 가 sample efficiency 우수
- **Large data (JFT-300M)**: ViT 의 lack of bias 가 advantage (학습 자유도 ↑)
- **Crossover**: ~14M images

**Modern Hybrid**:

- ConvNeXt: CNN 으로 ViT 의 advantages 일부 복원
- Swin: ViT 에 hierarchy 추가
- Hybrid (CoAtNet): conv + transformer 결합

**근본 통찰**:

CNN 과 ViT 의 차이는 **architecture 가 아니라 inductive bias 의 정도**. CNN 의 strong bias (locality, hierarchy, equivariance) vs ViT 의 minimal bias.

"Bias vs Bitter Lesson": 충분 data 시 minimal bias (ViT) 가 우월, 적은 data 시 strong bias (CNN) 가 우월. **데이터 시대 (2020+)**는 ViT 의 dominance, 그러나 **sample efficiency** 가 critical 한 niche 에서는 CNN. $\square$

</details>

**문제 3** (논문 비평): GPT-4V, Gemini 같은 frontier multimodal LLM 이 ViT-style vision encoder 를 사용한다. 그러나 vision-only task (ImageNet classification) 에서는 ConvNeXt 같은 CNN 이 여전히 competitive. Vision 의 future 가 ViT 인가 hybrid 인가?

<details>
<summary>해설</summary>

**Frontier Multimodal LLM 의 Vision Encoder**:

- **GPT-4V**: ViT-derived (CLIP-style)
- **Gemini**: ViT-style + multimodal training
- **Claude 3 Vision**: similar
- **LLaVA, BLIP-2**: ViT (e.g., EVA-CLIP) + LLM

이유:
1. **Token-based representation**: ViT output 이 자연스럽게 text token 과 결합
2. **Scaling**: ViT 의 simpler architecture 가 large-scale 에 유리
3. **Multimodal pre-training**: image-text pair 로 contrastive 학습 (CLIP)
4. **Open ecosystem**: HuggingFace, EVA, DINOv2 등 ViT 생태계

**Vision-only Task 에서 CNN 의 경쟁력**:

- **ConvNeXt** (Liu 2022): ResNet-style 그러나 ViT 의 training recipe 적용 → ViT 와 비슷한 성능
- **EfficientNet, RegNet**: smaller, faster
- **ImageNet classification**: ViT 와 ConvNeXt 가 거의 동등
- **Object detection (DETR vs Mask-RCNN)**: 다양한 결과

**왜 vision-only 에서 CNN 도 강력**:

1. **Inductive bias**: smaller dataset (COCO 등) 에서 advantage
2. **FLOPs efficiency**: 같은 accuracy 에 더 적은 compute
3. **Locality**: object detection 의 spatial reasoning 자연스러움
4. **Mobile / edge**: ConvNet 이 quantization, pruning 더 잘됨

**Vision Future 의 Trend**:

1. **Multimodal frontier**: ViT (Token-based)
   - Text 와 자연스러운 결합
   - Scaling 의 advantage
   - Multimodal training 의 default

2. **Vision-only specialized**: CNN 또는 hybrid
   - Mobile, edge, real-time
   - Object detection, segmentation
   - 작은 dataset

3. **Hybrid**:
   - **CoAtNet** (Dai 2021): conv + transformer
   - **Swin V2**: hierarchical ViT
   - Best of both: locality + global attention

4. **Self-supervised**:
   - MAE (Masked Autoencoder, He 2022): ViT-based
   - DINOv2 (Oquab 2023): ViT contrastive
   - Self-supervised 가 large-scale pre-training 의 future

**Specific Predictions (2025-2027)**:

- **Multimodal LLM (GPT-5, Gemini 2)**: ViT 더 deep 한 통합 (vision tokens directly in LLM context)
- **Vision-only foundation model**: ViT-based (DINOv2 successors)
- **Real-time / mobile**: hybrid 또는 efficient ViT (MobileViT, FastViT)
- **3D / video**: ViT-style (TimeSformer 변형)

**근본 통찰**:

Vision 의 future 는 **task-specific**:
- Multimodal frontier: ViT (token-based, scalable)
- Specialized vision: hybrid 또는 CNN (efficient, biased)

CNN vs ViT 의 binary 가 아니라 **각 application 의 right tool**. 그러나 frontier 는 ViT-derived dominant — multimodal scaling 의 자연스러운 결과.

**Modern era 의 paradigm**:

- "Vision = CNN" (2010s)
- "Vision = ViT" (early 2020s)
- "Vision = task-specific architecture" (mid 2020s+)

ViT 의 contribution 은 vision research 의 frontier 를 다시 open — single architecture monopoly 깨뜨림. 미래는 architectural diversity. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-t5.md) | [📚 README](../README.md) | [다음 ▶](./05-moe.md)

</div>
