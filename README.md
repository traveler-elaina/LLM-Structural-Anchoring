# 结构化锚定：语义无关上下文的认知负荷调制研究

**Structural Anchoring: A Cognitive Load Study on Semantic-Irrelevant Contexts in LLMs**

> 🚀 **Live Demo: [点击此处体验交互式Demo](您的Streamlit/Gradio应用链接)**
> 
> *通过交互应用，探索语义无关上下文如何通过认知负荷调制LLM输出，实时验证聚焦脚手架、认知过载和创造性融合效应。*
>
> *注意：Demo由Streamlit Community Cloud托管，永久有效。本地运行请见下方说明。*


## 1. 项目概述 (Project Overview)

本项目通过计算实验（31任务，N=270样本）探索并量化 **“结构化锚定”** 现象：语义无关的结构化上下文如何通过元指令（“对话” vs. “背景”）调制大型语言模型（LLM）的输出，影响其有序性（熵）和创新性（新颖度）。我们提出 **“认知负荷”理论**，将结构化锚定定义为额外“后台任务”，其效果取决于任务固有负荷：

**低负荷任务（如概括、框架设计）：** 上下文作为“聚焦脚手架”，降低熵（如任务4：Δ熵=-0.4446）。

****高负荷任务（如多维度分析）：** 上下文引发“认知过载”，增加熵（如任务15：Δ熵=+0.2632）。

**创意任务（如诗歌、描写）：** 上下文在连贯调制下引入“一致性成本”（如任务6：Δ新颖度=-0.1537），在融合调制下促进创造性（如任务12：Δ新颖度=+0.0417）。

我们开发了包含10+ NLP指标（熵、新颖度、词汇多样性等）的自动化分析流水线，测试了Google Gemini和DeepSeek模型，揭示其独特行为模式。 **此研究为提示工程提供可控设计原则，优化LLM在逻辑推理和创造性生成中的表现，适用于人机交互（HCI）和NLP应用（如DeepSeek的R-1模型优化）。**


## 2. 核心发现：“结构化锚定”效应与认知负荷调制

**“结构化锚定”** 的核心是语义无关上下文通过元指令施加“认知负荷”，触发两种调制模式：

**连贯调制模式（Coherence Modulation Mode）：** 元指令为“继续对话”，LLM作为“对话继承者”，维持上下文一致性。

**低负荷任务：** 聚焦脚手架，增强有序性（任务4：Δ熵=-0.4446，Δ新颖度=-0.0071）。

**创意任务：** 一致性成本，抑制创新（任务6：Δ新颖度=-0.1537）。

**融合调制模式（Fusion Modulation Mode）：** 元指令为“基于背景”，LLM作为“文本分析师”，融合不相关元素。

**创意任务：** 促进创造性，维持新颖度（任务12：Δ新颖度=+0.0417）。

**高负荷任务：** 可能缓解过载（任务30：Δ熵=-0.1134）。

**高负荷任务：** 无论模式，易触发“认知过载”，输出混乱（任务15：Δ熵=+0.2632，Δ新颖度=-0.0320）。

**关键洞察：** 语义无关上下文的“结构存在性”而非语义内容驱动调制，效果由任务负荷和元指令决定。47任务（N=270）验证了这一规律，填补了提示工程中语义无关上下文研究的空白，优于Chain-of-Thought（CoT）等语义相关方法。

  
#### 可视化结果 (Visualization)

主成分分析（PCA）显示三种条件（Idle、Relevant、Unrelated）的输出模式分离：

`Idle`组（蓝色）：高熵、低新颖度。

`Unrelated`组（绿色和黄色）：低负荷任务更有序，高负荷任务更混乱，创意任务新颖度分化。

![可视化结果](results/gemini/figures/ancova_visualization%20(1).png)

我们认为，这种效应的主要驱动力是上下文的 **“结构存在性”** （我们称之为“结构化锚定”），而非其 **“语义相关性”**。

## 3. 实验设计 (Experimental Design)

我们设计了三组对照实验，分离语语义无关上下文的调制效应：

**Idle（无上下文）：** 基线，仅提供任务指令（如“概括林黛玉性格”）。

**Dialogue-Relevant（相关上下文）：** 提供语义相关的对话历史（因提示词操控，参考价值有限）。

**Dialogue-Unrelated（无关上下文）：** 提供语义无关但结构相似的对话历史，作为“结构化锚定”测试。

**任务分类（31任务）：**

**低负荷**（单一逻辑，如任务4：概括林黛玉）：预期聚焦脚手架。

**高负荷**（多维度，如任务15：城市绿化分析）：预期认知过载。

**创意任务**（开放式，如任务6：比喻解释量子纠缠）：预期连贯/融合模式分化。

**指标：** 信息熵（有序性）、新颖度（创新性）、词汇多样性等，自动化流水线分析。
  

4. 主要结果 (Key Results)

270样本（31任务×3条件×2模型）的统计分析（ANOVA, p<0.001）显示：

低负荷任务：Unrelated组熵显著下降（平均Δ熵=-0.30，如任务4），新颖度稳定。

高负荷任务：Unrelated组熵上升（平均Δ熵=+0.12，如任务15），新颖度下降（平均Δ新颖度=-0.03）。

创意任务：连贯调制下新颖度下降（任务6：Δ新颖度=-0.1537），融合调制下新颖度维持/上升（任务12：Δ新颖度=+0.0417）。

**指标** | **Idle均值** | **Unrelated均值** | **示例任务**
**熵** | 6.68 | 6.46 | 任务4：5.4919→5.0473
**新颖度** | 0.39 | 0.75 | 任务12：0.7371→0.7788


## 5. 应用与未来工作 (Application and Future Work)

**提示工程：** 为DeepSeek等LLM优化提供可控元指令（连贯/融合），提升逻辑/创意输出。

**人机交互：** 优化AI交互（如教育、文化分析），CHI LBW投稿中。

**未来验证：**

元指令隔离：测试继承 vs. 合成（任务1、2、10）。

人文学科：验证反模式（任务17、25）。

Baseline：对比CoT（任务4、12）。

将实验扩展到更多模型家族（如GPT-4, Claude, Llama）。

探究不同上下文结构（如长度、逻辑复杂度）对“锚定”效应强度的影响。

探索此效应在高级Prompt工程和可控文本生成中的应用。

## 6. 项目结构 (Repository Structure)

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

## 7. 安装与运行 (Installation & Usage)

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


## 8. 如何引用 (Citation)

如果您在您的研究中使用了本项目的代码、数据或思想，请引用本GitHub仓库的链接。

## 9. 许可证 (License)

本项目的代码部分采用 [MIT License](https://www.google.com/search?q=LICENSE) 授权。 本项目的文字、数据和研究成果采用 [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 授权。

## 10. 联系 (Contact us)
如有问题，请联系[traveler-elaina](wy807110695@gmail.com)或在GitHub上打开问题。
