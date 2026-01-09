```markdown
# CAFR-MIL: Context-Aware Feature Refinement with Orthogonal Regularization

> 论文 **"Context-Aware Feature Refinement with Orthogonal Regularization for Whole Slide Image Classification"** 官方实现  
> 轻量级、即插即用的 MIL 增强框架

---

## 🌟 核心特性
| 特性 | 一句话说明 |
|---|---|
| **上下文感知特征精炼 (CAFR)** | Bag-级全局原型动态校准实例特征，抑制背景噪声 |
| **正交正则化 (Orthogonal Regularization)** | 通道几何去相关，学习多样且不冗余的病理语义 |
| **即插即用** | 无缝嵌入 ABMIL / TransMIL / CLAM / ACMIL 等，仅增 **0.58% FLOPs** |
| **卓越性能** | Camelyon16/17、TCGA-NSCLC 三大基准一致提升 |

---

## 🏗️ 算法框架图
> ⚠️ 占位：请在此处插入 **框架总览图**（PDF/SVG 均可，推荐白底，宽度 ≤ 800 px）  
> 图示建议：左侧原始 MIL → 中间 CAFR 模块 → 右侧正交损失 → 输出。

---

## 🔬 可视化热图
> ⚠️ 占位：请在此处插入 **Camelyon16 可视化对比热图**（PNG/JPG，宽度 ≤ 800 px）  
> 图示建议：上方为原始 WSI + 标注，下方为 CAFR 增强后热图，左侧加颜色条。

---

## 📦 快速上手

### 1. 环境
```bash
python≥3.7  pytorch≥1.7.1
```

### 2. 安装
```bash
git clone https://github.com/yourrepo/CAFR-MIL.git
cd CAFR-MIL
pip install -r requirements.txt
```

### 3. 一行集成
把 `CAFR` 插在 **Aggregator 之前** 即可：

```python
from cafr import ContextAwareFeatureRefiner, OrthogonalLoss

refiner      = ContextAwareFeatureRefiner(input_dim=768)   # 与特征维度一致
ortho_loss   = OrthogonalLoss()

def training_step(feats, labels):
    refined = refiner(feats)               # [B, N, D] → [B, N, D]
    logits  = mil_backbone(refined)        # 任意 MIL 头
    cls     = F.cross_entropy(logits, labels)
    ortho   = ortho_loss(refined)
    return cls + 0.085 * ortho
```

---

## 📊 主要结果（3 跑平均）

| 骨干 | 数据集 | ACC | AUC |
|---|---|---|---|
| ACMIL | Camelyon16 | 0.9023 | 0.9297 |
| **ACMIL+CAFR** | **Camelyon16** | **0.9302** | **0.9436** |
| CLAM-SB | TCGA-NSCLC | 0.9172 | 0.9702 |
| **CLAM-SB+CAFR** | **TCGA-NSCLC** | **0.9569** | **0.9808** |

---

## 📝 引用
```bibtex
@article{zhou2026cafr,
  title={Context-Aware Feature Refinement with Orthogonal Regularization for Whole Slide Image Classification},
  author={Zhou, Shicheng and Wang, Zefeng and Yu, Jikai and Wu, Boyuan and Zhu, Jiayun},
  journal={The Visual Computer},
  year={2026}
}
```

## 📧 联系
- 周士程：2024388427@stu.zjhu.edu.cn  
- 王泽锋：zefeng.wang@zjhu.edu.cn
```
