```python
# feature_extractor.py
# 文本数据特征提取模块

import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
import torch
from config import (
    SPACY_MODEL, SENTENCE_TRANSFORMER_MODEL, KEYBERT_MODEL, PROXY_MODEL,
    SENTIMENT_MODEL, SIMILARITY_THRESHOLD, TOP_N_KEYWORDS, NGRAM_N, MAX_TOKEN_LENGTH,
    PROTOTYPE_SENTENCES
)
from datetime import datetime  # 用于打印日志

# 初始化NLP模型
print(f"[{datetime.now()}] --- 正在加载NLP模型... ---")
nlp = spacy.load(SPACY_MODEL)  # 加载spaCy模型
sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)  # 加载句子转换模型
kw_model = KeyBERT(model=KEYBERT_MODEL)  # 加载KeyBERT模型
proxy_model = pipeline('fill-mask', model=PROXY_MODEL)  # 加载代理模型
proxy_tokenizer = proxy_model.tokenizer  # 获取分词器
sentiment_analyzer = pipeline('sentiment-analysis', model=SENTIMENT_MODEL)  # 加载情感分析模型
prototype_embeddings = sentence_model.encode(PROTOTYPE_SENTENCES)  # 生成原型句子嵌入
print(f"[{datetime.now()}] --- NLP模型加载完成。 ---")

def tokenize_chinese(text: str) -> list:
    """使用spaCy进行中文分词。
    参数:
    text: 输入文本字符串
    返回:
    分词列表，排除标点和空格
    """
    if not text:
        return []  # 如果文本为空，返回空列表
    doc = nlp(text)  # 使用spaCy处理文本
    return [token.text for token in doc if not token.is_punct and not token.is_space]  # 过滤标点和空格

def calculate_shannon_entropy(text: str) -> float:
    """计算文本的香农熵，衡量信息多样性。
    参数:
    text: 输入文本
    返回:
    香农熵值
    """
    tokens = tokenize_chinese(text)  # 分词
    if not tokens:
        return 0.0  # 无token返回0
    freq = Counter(tokens)  # 计算频率
    probs = [f / len(tokens) for f in freq.values()]  # 计算概率
    return entropy(probs, base=2) if probs else 0.0  # 计算熵

def calculate_perplexity_proxy(text: str) -> float:
    """使用语言模型计算困惑度代理。
    参数:
    text: 输入文本
    返回:
    困惑度值或NaN
    """
    if not text:
        return np.nan  # 空文本返回NaN
    try:
        with torch.no_grad():  # 无梯度计算
            inputs = proxy_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH)  # 分词并截断
            outputs = proxy_model(**inputs, labels=inputs["input_ids"])  # 模型前向传播
            return torch.exp(outputs.loss).item()  # 计算困惑度
    except Exception as e:
        print(f"[{datetime.now()}] 困惑度计算错误: {e}")  # 打印错误日志
        return np.nan

def calculate_novelty(response: str, context: str) -> float:
    """计算响应相对于上下文的新颖度。
    参数:
    response: 响应文本
    context: 上下文文本
    返回:
    新颖度分数 (0-1)
    """
    if not response or not context:
        return 0.0  # 任意为空返回0
    try:
        embs = sentence_model.encode([response, context])  # 生成嵌入
        return 1 - cosine_similarity([embs[0]], [embs[1]])[0][0]  # 计算1 - 相似度
    except Exception as e:
        print(f"[{datetime.now()}] 新颖度计算错误: {e}")
        return 0.0

def calculate_semantic_complexity_keybert(text: str, top_n: int = TOP_N_KEYWORDS) -> dict:
    """使用KeyBERT计算语义复杂性。
    参数:
    text: 输入文本
    top_n: 关键词数量
    返回:
    字典包含graph_density和avg_clustering
    """
    if not text or len(text.split()) < 15:
        return {"graph_density": 0, "avg_clustering": 0}  # 文本太短返回0
    try:
        keywords = [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=top_n)]  # 提取关键词
        G = nx.Graph()  # 创建图
        doc = nlp(text)  # 处理文本
        if len(keywords) < 2:
            return {"graph_density": 0, "avg_clustering": 0}  # 关键词不足返回0
        for sentence in doc.sents:  # 遍历句子
            sent_kws = [kw for kw in keywords if kw in sentence.text]  # 句子中关键词
            for i in range(len(sent_kws)):
                for j in range(i + 1, len(sent_kws)):
                    G.add_edge(sent_kws[i], sent_kws[j])  # 添加边
        return {
            "graph_density": nx.density(G) if G.number_of_nodes() > 1 else 0,  # 图密度
            "avg_clustering": nx.average_clustering(G) if G.number_of_nodes() > 1 else 0  # 平均聚类系数
        }
    except Exception as e:
        print(f"[{datetime.now()}] 语义复杂性计算错误: {e}")
        return {"graph_density": 0, "avg_clustering": 0}

def calculate_ttr(text: str) -> float:
    """计算类型-标记比率，衡量词汇多样性。
    参数:
    text: 输入文本
    返回:
    TTR值
    """
    tokens = tokenize_chinese(text)  # 分词
    return len(set(tokens)) / len(tokens) if tokens else 0.0  # 计算比率

def calculate_psi_proxy_chinese(text: str, surprisal: float) -> float:
    """计算PSI代理，结合情感和困惑度。
    参数:
    text: 输入文本
    surprisal: 惊奇度（困惑度）
    返回:
    PSI值或NaN
    """
    if not text or pd.isna(surprisal):
        return np.nan  # 无效输入返回NaN
    try:
        result = sentiment_analyzer(text)[0]  # 情感分析
        valence = result['score'] if result['label'] == 'positive' else -result['score']  # 计算效价
        arousal = abs(valence)  # 计算唤醒度
        return 1.0 * abs(valence) + 1.0 * arousal + 0.1 * surprisal  # 计算PSI
    except Exception as e:
        print(f"[{datetime.now()}] PSI计算错误: {e}")
        return np.nan

def calculate_analogy_rate_semantic(text: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> float:
    """基于语义相似度计算类比句式比率。
    参数:
    text: 输入文本
    similarity_threshold: 相似度阈值
    返回:
    类比比率
    """
    if not text:
        return 0.0  # 空文本返回0
    doc = nlp(text)  # 处理文本
    sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 5]  # 提取有效句子
    if not sentences:
        return 0.0  # 无句子返回0
    try:
        sentence_embeddings = sentence_model.encode(sentences)  # 生成句子嵌入
        similarity_matrix = cosine_similarity(sentence_embeddings, prototype_embeddings)  # 计算相似度矩阵
        max_similarities = similarity_matrix.max(axis=1)  # 最大相似度
        analogy_count = np.sum(max_similarities > similarity_threshold)  # 计数超过阈值的
        return analogy_count / len(sentences)  # 计算比率
    except Exception as e:
        print(f"[{datetime.now()}] 类比率计算错误: {e}")
        return 0.0

def calculate_internal_coherence(text: str) -> float:
    """计算文本内部语义连贯性。
    参数:
    text: 输入文本
    返回:
    连贯性值
    """
    sentences = [sent.text for sent in nlp(text).sents if len(sent.text.strip()) > 5]  # 提取有效句子
    if len(sentences) < 2:
        return 1.0  # 少于2句返回1
    try:
        embeddings = sentence_model.encode(sentences)  # 生成嵌入
        sim_matrix = cosine_similarity(embeddings)  # 相似度矩阵
        indices = np.triu_indices_from(sim_matrix, k=1)  # 上三角索引
        return np.mean(sim_matrix[indices]) if len(indices[0]) > 0 else 1.0  # 计算平均相似度
    except Exception as e:
        print(f"[{datetime.now()}] 连贯性计算错误: {e}")
        return 1.0

def calculate_distinct_n(text: str, n: int = NGRAM_N) -> float:
    """计算n-gram多样性。
    参数:
    text: 输入文本
    n: n-gram大小
    返回:
    多样性值
    """
    tokens = tokenize_chinese(text)  # 分词
    if len(tokens) < n:
        return 0.0  # token不足返回0
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]  # 生成n-gram
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0  # 计算独特比率

def calculate_tokens_per_second(row: pd.Series) -> float:
    """计算每秒token数。
    参数:
    row: 数据行
    返回:
    TPS值
    """
    return row['token_count'] / row['latency'] if row['latency'] > 0 else 0.0  # 计算速度
```