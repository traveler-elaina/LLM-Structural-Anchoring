# LLM-Structural-Anchoring
# 中文文本分析管道

## 项目概述
此存储库包含一个模块化的Python管道，用于分析中文文本数据，重点关注特征提取、统计分析和可视化。该管道处理文本响应数据集，提取语言和语义特征，执行统计测试（ANOVA和Tukey's HSD），并生成可视化和结构化报告。它设计用于研究级分析，支持使用强大的NLP工具处理中文文本。

## 功能
- **数据加载**：加载带有实验条件和文本响应的CSV数据。
- **特征提取**：计算指标，如香农熵、困惑度、新颖度、类型-标记比率、语义复杂性等。
- **统计分析**：执行单因素ANOVA和Tukey's HSD，以比较不同条件的指标，并计算效应量。
- **可视化**：生成带有置信区间的柱状图和ANCOVA可视化，以探索与prompt长度的关系。
- **报告**：生成总结结果的结构化Markdown报告，适合GitHub作品集。
- **中文支持**：使用适当的分词和字体渲染处理中文文本。

## 安装
1. 克隆存储库：
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows上: venv\Scripts\activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 安装中文字体（非Colab环境，如果需要）：
   ```bash
   sudo apt-get update
   sudo apt-get install -y fonts-wqy-zenhei
   ```
5. 安装spaCy中文模型：
   ```bash
   python -m spacy download zh_core_web_sm
   ```
6. 将输入数据（`experiment_A_raw_data_CHINESE_v3 (1).csv`）放置在`data/`目录中。

## 依赖
在`requirements.txt`中列出：
```
pandas
numpy
scipy
matplotlib
seaborn
statsmodels
scikit-learn
sentence-transformers
keybert
spacy
transformers
networkx
tqdm
torch
```

## 项目结构
```
your-repo-name/
│
├── data/                           # 输入数据（例如experiment_A_raw_data_CHINESE_v3 (1).csv）
├── results/                        # 输出目录，用于图表和报告
├── config.py                       # 配置设置
├── feature_extractor.py            # 特征提取函数
├── statistics_and_visualization.py # 统计分析和绘图
├── main_analysis.py                # 主执行脚本
├── requirements.txt                # 依赖列表
└── README.md                       # 项目文档
```

## 使用
1. 根据需要更新`config.py`中的路径和参数。
2. 运行主脚本：
   ```bash
   python main_analysis.py
   ```
3. 检查`results/`目录中的图表和报告（`analysis_report_v5.1.md`）。

## 配置
编辑`config.py`自定义：
- **RAW_DATA_FILE**：输入CSV路径。
- **ANALYZED_DATA_FILE**：分析数据输出路径。
- **REPORT_FILE**：Markdown报告路径。
- **OUTPUT_DIR**：图表和报告目录。
- **NLP模型**：指定spaCy、SentenceTransformer、KeyBERT等的模型。
- **特征参数**：调整阈值、关键词计数等。
- **可视化设置**：配置图表格式和尺寸。

## 示例
给定具有`condition`、`prompt`、`response`等列的数据集：
1. 管道加载数据。
2. 提取特征，如`shannon_entropy`、`novelty_score`和`analogy_rate`。
3. 执行ANOVA测试不同条件（例如Idle、Dialogue-Relevant、Dialogue-Unrelated）的差异。
4. 对于显著结果运行Tukey's HSD。
5. 生成柱状图和ANCOVA可视化，保存在`results/`中。
6. 生成总结发现的Markdown报告。

## 贡献
1. 分叉存储库。
2. 创建功能分支（`git checkout -b feature/your-feature`）。
3. 提交更改（`git commit -m "添加您的功能"`）。
4. 推送到分支（`git push origin feature/your-feature`）。
5. 打开拉取请求。

## 许可证
MIT许可证。详见[LICENSE](LICENSE)。
本项目的文字与数据内容采用[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)许可协议。

## 联系
如有问题，请联系[traveler-elaina](wy807110695@gmail.com)或在GitHub上打开问题。
