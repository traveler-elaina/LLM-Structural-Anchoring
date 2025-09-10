# 结构化锚定：一项关于上下文对大型语言模型输出影响的计算研究

**Structural Anchoring: A Computational Study on the Effect of Context on Large Language Model Outputs**

## 1. 项目概述 (Project Overview)

本项目旨在通过一项严谨的计算实验，探索和量化一种我们称之为**“结构化锚定”（Structural Anchoring）**的新现象。该现象描述了一个复杂的对话上下文，如何系统性地改变大型语言模型（LLM）后续文本输出的语言学特征。

我们开发了一套包含超过10个NLP指标的自动化分析流水线，用以精确测量LLM在不同上下文条件下的“有序性”、“复杂性”和“创造性”。本项目不仅证实了“结构化锚定”效应的普遍存在，还揭示了不同顶尖模型（如Google Gemini, DeepSeek）在该任务下独特的行为模式，为高级Prompt工程和AI认知研究提供了新的视角和量化工具。

## 2. 核心发现：“结构化锚定”效应

本研究的核心发现是，LLM的生成模式会受到其先前接触过的信息的**结构特性**的强烈影响，**即便这些信息的语义内容与当前任务无关**。

具体而言，当一个LLM预先接触一个复杂的（无论是否相关的）对话上下文后，它会从一个通用的“知识检索”模式，跃迁至一个“情境推理”模式。这个新模式的特征呈现出一种深刻的悖论：

- **更有序 (More Ordered):** 输出文本在主题上更聚焦，信息熵显著降低。
  
- **但更复杂与新颖 (More Complex & Novel):** 与此同时，输出的文本更出人意料（困惑度升高），也更具创造性（新颖度得分和词汇多样性增加）。
  

我们认为，这种效应的主要驱动力是上下文的 **“结构存在性”** （我们称之为“结构化锚定”），而非其 **“语义相关性”**。

## 3. 实验设计 (Experimental Design)

我们采用了一个包含三种对照条件的“共享指令”设计，以分离和检验“结构化锚定”的效应。

- **`Idle` (无上下文组):** 作为基线，LLM仅接收一个独立的、要求结构化输出的指令。
  
- **`Dialogue-Relevant` (相关上下文组):** 在接收相同指令前，先接触一段与指令语义相关的对话历史。
  
- **`Dialogue-Unrelated` (无关上下文组):** 在接收相同指令前，先接触一段与指令语义**无关**，但结构与`Relevant`组相似的对话历史（作为“活性安慰剂”）。
  

## 4. 主要结果 (Key Results)

我们对270个样本进行了完整的统计分析。所有核心指标在三组之间都表现出了极高的统计显著性差异（ANOVA, p < 0.001）。

#### 统计摘要 (Statistical Summary)

事后检验（Tukey's HSD）揭示了最关键的模式：`Dialogue-Relevant`组和`Dialogue-Unrelated`组的表现相似，但它们都与`Idle`组存在巨大差异。

| 指标 (Metric) | Idle组均值 | Relevant组均值 | Unrelated组均值 |
| --- | --- | --- | --- |
| **shannon_entropy** (熵 ↓) | 6.68 | 6.51 | 6.46 |
| **novelty_score** (新颖度 ↑) | 0.39 | 0.73 | 0.75 |


#### 可视化结果 (Visualization)

主成分分析（PCA）图直观地展示了三种条件下AI输出模式的系统性差异。`Idle`组（蓝色）与两个有上下文的组（绿色和黄色）清晰地分离在不同的空间中。

*(注意：您需要将您生成的pca_scores_plot.png图片，上传到您GitHub仓库的figures文件夹下，此图片才能正确显示)*

## 5. 项目结构 (Repository Structure)

```
/
├── README.md                # 本项目说明书
├── requirements.txt         # Python库依赖列表
├── LICENSE                  # MIT 开源许可证
│
├── src/                     # 核心Python源代码
│   ├── main_analysis.py     # 主执行脚本
│   ├── feature_extractor.py # 特征提取模块
│   └── ...
│
├── data/                    # 数据文件
│   ├── raw_data_sample.csv  # 原始数据样本
│   └── analyzed_data.csv    # 分析后数据
│
├── prompts/                 # 实验材料
│   └── ...
│
└── results/                 # 生成的结果
    ├── figures/             # 所有图表图片
    └── tables/              # 所有统计表格
```

## 6. 安装与运行 (Installation & Usage)

1. **克隆本仓库**
  
  Bash
  
  ```
  git clone [您的仓库URL]
  cd [您的仓库名称]
  ```
  
2. **创建并激活虚拟环境**
  
  Bash
  
  ```
  python -m venv venv
  source venv/bin/activate  # 在 Windows上: venv\Scripts\activate
  ```
  
3. **安装依赖**
  
  Bash
  
  ```
  pip install -r requirements.txt
  python -m spacy download zh_core_web_sm
  ```
  
4. **运行分析**
  
  - 将您的原始数据放入 `data/raw/` 文件夹。
    
  - 运行主脚本：
    

Bash

```
python src/main_analysis.py
```

- 所有分析结果将自动保存在 `results/` 文件夹中。

## 7. 如何引用 (Citation)

如果您在您的研究中使用了本项目的代码、数据或思想，请引用本GitHub仓库的链接。

## 8. 许可证 (License)

## 本项目的代码部分采用 [MIT License](https://www.google.com/search?q=LICENSE) 授权。 本项目的文字、数据和研究成果采用 [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 授权。

## 9.联系 (Contact us)
如有问题，请联系[traveler-elaina](wy807110695@gmail.com)或在GitHub上打开问题。
