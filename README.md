# MGR（Multi-view Graph-enhanced Retrieval）

本仓库用于实现基于 **SARAF** 的二次开发项目：  
在保留原始时间域检索思想的基础上，引入频域与统计域表征，并结合图结构增强样本检索，以提升时序预测效果。

---

## 1. 项目背景与定位

- **SARAF**：作为基础时序预测框架，提供时间域检索 + 预测融合的主干思路。
- **MGR**：在 SARAF 之上扩展多视图检索（Time / Frequency / Statistical）与图增强检索机制。

本项目目标是实现以下流程：

1. 保留 Time View（时间域）检索；
2. 增加 Frequency View（FFT 频域特征）；
3. 增加 Statistical View（统计特征，如 max/min/mean/var/autocorr 等）；
4. 利用频域和统计域特征构建样本图；
5. 多阶段检索（Top-M 种子 → K 邻居 → 候选池 → Re-ranking）；
6. 基于检索样本未来片段生成 Retrieval Prediction；
7. 与基础 Forecasting Model 做 1:1 融合，得到最终预测。

---

## 2. 当前仓库审计结果（2026-10-16）

> 重要说明：当前分支中仅包含非常精简的初始化内容，尚未看到题述中提到的核心代码文件。  
> 包括但不限于：`layers/Retrieval_mgr.py`、`models/MGR.py`、Dataset/DataLoader、训练测试脚本、实验脚本等。

### 当前目录结构

```text
MGR/
├── README.md
└── requirements.txt
```

### 已确认现状

- 远端当前仅可见分支：`copilot/organize-enhance-mgr-project`
- 仓库中尚无可执行训练/测试代码与配置文件
- 因缺少源码，无法对模型间 Tensor shape、数据流、参数配置进行代码级核验

---

## 3. 预期的 MGR 核心流程（设计说明）

以下为项目目标流程说明（用于对齐实现方向）：

1. **Time View Retrieval**：按 SARAF 既有机制进行时间窗口相似度检索。
2. **Frequency View Retrieval**：对历史窗口做 FFT，提取频域特征并计算相似度（参考 FreDF 思想）。
3. **Statistical View Retrieval**：提取统计特征并编码为检索向量（参考 tsfresh 思想）。
4. **Graph Construction**：基于频域+统计域构图，表达样本间结构关系。
5. **Multi-stage Retrieval**：Top-M seeds → K-neighbors → candidate pool → re-ranking。
6. **Retrieval Prediction**：聚合检索到的历史样本未来片段生成检索预测。
7. **Fusion**：`Final Prediction = 0.5 * Retrieval Prediction + 0.5 * Base Forecast`。

> 注：上述为目标行为定义，不代表当前仓库已经具备对应实现。

---

## 4. 环境依赖与安装

当前仓库未包含可执行 Python 训练代码，因此依赖暂无法从源码中精确反推。  
已提供 `requirements.txt` 作为最小占位说明，待代码补充后再固定版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. 训练与测试运行说明（当前状态）

当前仓库尚无训练/测试脚本，因此暂不可直接运行：

- `train.py` / `test.py`：未发现
- 实验脚本（shell/python）：未发现
- 配置文件（yaml/json/arg parser 入口）：未发现

待后续补充代码后，可在此 README 增加：

- 数据准备方式
- 训练命令
- 测试命令
- 关键参数说明（窗口长度、M/K、融合权重等）

---

## 6. 当前已实现与待完善项

### 当前已实现

- 项目目标与方法说明文档（本 README）
- 环境依赖占位文件（`requirements.txt`）

### 仍需进一步完善

1. 补充核心源码（`layers/`、`models/`、`data/`、`scripts/`）
2. 完成 MGR 多视图与图增强检索的可运行实现
3. 补充训练/测试入口与参数配置
4. 增加最小可复现实验脚本
5. 补充单元测试或集成测试（至少覆盖关键数据流与 shape）

---

## 7. 不确定性与风险标注

- 由于当前仓库缺少目标源码，无法验证：
  - 多视图检索具体实现是否存在
  - 图结构构建策略是否与设计一致
  - Tensor shape 与数据流是否正确
  - 实验结果与性能结论

为避免误导，本文档**未虚构任何实验结果或代码能力**。
