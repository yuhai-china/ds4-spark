# Transformer 原理详解

## 背景与动机

- RNN 的局限性：串行计算导致长程依赖难以捕捉，梯度消失/爆炸问题严重
- CNN 在 NLP 中感受野有限，难以建模全局依赖
- Transformer 提出**完全基于注意力机制**的架构，实现并行计算与全局建模
- 2017 年由 Vaswani 等人提出，成为 NLP 乃至多模态领域的基础范式

## 核心创新：自注意力机制

- 每个位置通过 **Q、K、V** 矩阵计算注意力分数
- 公式：$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
- **Q (Query)**：当前位置的查询向量
- **K (Key)**：所有位置的键向量，决定哪些位置被关注
- **V (Value)**：所有位置的值向量，加权聚合信息
- **缩放因子 $\sqrt{d_k}$**：防止点积结果过大导致 softmax 梯度消失

## 多头注意力机制

- 将 Q、K、V 拆分为 **h 个头部**，每个头部独立计算注意力
- 多头允许模型在不同子空间学习不同类型的关联模式
- 公式：$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$
- 每个头部：$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$
- 典型设置：h=8（8 个头部），每个头部维度 $d_k = d_{model}/h = 64$

## 位置编码

- 自注意力本身是**排列不变**的（对位置不敏感）
- 需要注入位置信息，让模型感知序列顺序
- **正弦位置编码**：
  - $PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$
  - $PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$
- 优点：可泛化到任意长度，无需额外参数
- 现代变体（如 GPT）改用**可学习的位置嵌入**

## Encoder-Decoder 结构

- **Encoder**：6 层，每层包含多头注意力和 FFN，用于编码输入序列
- **Decoder**：6 层，每层包含 Masked 多头注意力 + Cross-Attention + FFN
- **Masked 注意力**：防止 Decoder 看到未来信息（因果掩码）
- **Cross-Attention**：Decoder 的 Query 与 Encoder 的 Key/Value 交互
- 输出层：线性变换 + Softmax 生成词汇概率

## 关键组件详解

- **残差连接**：$Output = LayerNorm(X + Sublayer(X))$，解决深层退化问题
- **Layer Normalization**：对每个样本独立归一化，稳定训练
- **前馈网络 (FFN)**：两层线性变换 + ReLU，$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$
- **Dropout**：在残差连接和注意力计算后应用，防止过拟合
- 所有组件可微，端到端梯度反向传播

## 代码示例：核心注意力实现

```python
import torch
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, V)
```

## 关键对比：Transformer vs RNN

| 特性 | Transformer | RNN |
|------|------------|-----|
| 计算方式 | 并行（同时处理所有位置） | 串行（逐步处理） |
| 长程依赖 | 注意力直接连接任意位置 | 需通过隐藏状态传递 |
| 训练效率 | 高（可大规模并行） | 低（无法并行） |
| 位置感知 | 需额外位置编码 | 天然顺序处理 |
| 可扩展性 | 可堆叠至上百层 | 深层次易梯度问题 |

## 总结与展望

- Transformer 通过**自注意力 + 多头 + 位置编码**实现高效全局建模
- 成为 BERT、GPT、T5 等预训练模型的基石
- 后续改进：稀疏注意力（Longformer）、线性注意力（Performer）、RWKV 等
- 跨模态扩展：ViT（视觉）、Swin Transformer、Whisper（语音）
- 核心思想：**用注意力替代循环，用并行替代串行**

## 参考资料

> "Attention is all you need" — 论文核心贡献
> — Ashish Vaswani et al., NeurIPS 2017

> 理解 Transformer 的关键在于把握三个核心设计：自注意力、多头机制、位置编码
> — 技术社区共识
