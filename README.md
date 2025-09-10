# 结构化锚定：一项关于上下文对大型语言模型输出影响的计算研究

**Structural Anchoring: A Computational Study on the Effect of Context on Large Language Model Outputs**

> 🚀 **Live Demo: [点击此处体验交互式Demo](您的Streamlit/Gradio应用链接)**
> 
> *通过我们的交互应用，您可以亲自体验并验证“结构化锚定”效应如何实时改变LLM的输出行为。*

## 1. 项目概述 (Project Overview)

本项目旨在通过一项严谨的计算实验，探索和量化一种我们称之为 **“结构化锚定”（Structural Anchoring）** 的新现象。该现象描述了一个复杂的对话上下文，如何系统性地改变大型语言模型（LLM）后续文本输出的语言学特征。

我们开发了一套包含超过10个NLP指标的自动化分析流水线，用以精确测量LLM在不同上下文条件下的“有序性”、“复杂性”和“创造性”。本项目不仅证实了“结构化锚定”效应的普遍存在，还揭示了不同顶尖模型（如Google Gemini, DeepSeek）在该任务下独特的行为模式，为高级Prompt工程和AI认知研究提供了新的视角和量化工具。

**更重要的是，本研究的发现与方法，为构建更自然、更有情感深度的对话式AI（Conversational AI）提供了可量化的评估工具和行之有效的优化路径。通过引入‘无关上下文’作为对照，本研究的计算实验证实：对话上下文的结构性存在（而非其语义相关性），是触发LLM输出模式从‘知识检索’向‘情境推理’跃迁的关键因素，表现为输出信息熵显著降低（更有序）与新颖度显著提升（更创造性）的统一。**

## 2. 核心发现：“结构化锚定”效应

本研究的核心发现是，LLM的生成模式会受到其先前接触过的信息的**结构特性**的强烈影响，**即便这些信息的语义内容与当前任务无关**。

具体而言，当一个LLM预先接触一个复杂的（无论是否相关的）对话上下文后，它会从一个通用的“知识检索”模式，跃迁至一个“情境推理”模式。这个新模式的特征呈现出一种深刻的悖论：

- **更有序 (More Ordered):** 输出文本在主题上更聚焦，信息熵显著降低。
  
- **但更复杂与新颖 (More Complex & Novel):** 与此同时，输出的文本更出人意料（困惑度升高），也更具创造性（新颖度得分和词汇多样性增加）。
  
#### 可视化结果 (Visualization)

主成分分析（PCA）图直观地展示了三种条件下AI输出模式的系统性差异。`Idle`组（蓝色）与两个有上下文的组（绿色和黄色）清晰地分离在不同的空间中。

![可视化结果](results/gemini/figures/ancova_visualization%20(1).png)

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
| **shannon_entropy** (信息熵 ↓) | 6.68 | 6.51 | 6.46 |
| **novelty_score** (新颖度 ↑) | 0.39 | 0.73 | 0.75 |


## 5. 项目结构 (Repository Structure)

```
/
├── README.md                    # 本项目说明书
├── requirements.txt             # Python库依赖列表
├── LICENSE                      # 开源许可证
│
├── src/                         # <-- 存放您所有可复用的核心Python源代码
│   ├── data_generator.py        # 包含 generate_data_gemini() 和 generate_data_deepseek() 两个函数
│   ├── feature_extractor.py     # 所有NLP指标计算函数 (两个实验共用)
│   └── analysis_and_viz.py      # 所有统计和可视化函数 (两个实验共用)
│
├── prompts/                     # <-- 存放所有实验材料
│   ├── idle_prompts.csv
│   └── dialogue_prompts.csv
│
├── data/                        # <-- 存放所有数据文件
│   ├── gemini/                  # 存放Gemini实验的数据
│   │   ├── raw_data.csv
│   │   └── analyzed_data.csv
│   └── deepseek/                # 存放DeepSeek实验的数据
│       ├── raw_data.csv
│       └── analyzed_data.csv
│
└── results/                     # <-- 存放所有由脚本生成的最终结果
    ├── gemini/                  # 存放Gemini实验的结果
    │   ├── figures/             # (所有图表图片)
    │   └── tables/              # (所有统计表格)
    ├── deepseek/                # 存放DeepSeek实验的结果
    │   ├── figures/
    │   └── tables/
    └── comparative/             # (可选，但强烈推荐) 存放对比两个模型的结果
        └── gemini_vs_deepseek_plot.png
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

## 7. 未来工作 (Future Work)

将实验扩展到更多模型家族（如GPT-4, Claude, Llama）。

探究不同上下文结构（如长度、逻辑复杂度）对“锚定”效应强度的影响。

探索此效应在高级Prompt工程和可控文本生成中的应用。

## 8. 如何引用 (Citation)

如果您在您的研究中使用了本项目的代码、数据或思想，请引用本GitHub仓库的链接。

## 9. 许可证 (License)

本项目的代码部分采用 [MIT License](https://www.google.com/search?q=LICENSE) 授权。 本项目的文字、数据和研究成果采用 [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 授权。

## 10. 联系 (Contact us)
如有问题，请联系[traveler-elaina](wy807110695@gmail.com)或在GitHub上打开问题。
