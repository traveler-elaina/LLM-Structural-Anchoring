```python
# config.py
# Configuration settings for the text analysis pipeline

# File paths
RAW_DATA_FILE = "experiment_A_raw_data_CHINESE_v3 (1).csv"
ANALYZED_DATA_FILE = "experiment_A_analyzed_data_FINAL.csv"
REPORT_FILE = "analysis_report_v5.1.md"
OUTPUT_DIR = "results/"  # Directory for saving plots and reports

# NLP model settings
SPACY_MODEL = "zh_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
KEYBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
PROXY_MODEL = "bert-base-chinese"
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Feature extraction parameters
SIMILARITY_THRESHOLD = 0.65  # Threshold for analogy rate
TOP_N_KEYWORDS = 10  # Number of keywords for semantic complexity
NGRAM_N = 2  # N-gram size for distinct_n calculation
MAX_TOKEN_LENGTH = 512  # Max length for tokenizer

# Statistical analysis parameters
SIGNIFICANCE_LEVEL = 0.05  # Alpha for ANOVA and Tukey's HSD

# Visualization settings
PLOT_FORMATS = ["png"]  # File formats for saving plots
FIGURE_SIZE = (8, 6)  # Default figure size for bar plots
ANCOVA_FIGURE_SIZE = (10, 7)  # Figure size for ANCOVA plots
FONT = "WenQuanYi Zen Hei"  # Font for Chinese text rendering

# Prototype analogy sentences
PROTOTYPE_SENTENCES = [
    "这个概念就像一个东西", "它的原理类似于一个过程", "我们可以把它看作是一种事物",
    "这好比是一个例子", "这种情况仿佛是另一回事", "其本质上是一种", "打个比方来说"
]