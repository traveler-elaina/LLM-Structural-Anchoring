```python
# main_analysis.py
# 主执行脚本，协调文本分析管道

import pandas as pd
from datetime import datetime
from config import RAW_DATA_FILE, ANALYZED_DATA_FILE, REPORT_FILE
from feature_extractor import (
    calculate_tokens_per_second, calculate_shannon_entropy, calculate_perplexity_proxy,
    calculate_novelty, calculate_semantic_complexity_keybert, calculate_ttr,
    calculate_psi_proxy_chinese, calculate_analogy_rate_semantic, calculate_internal_coherence,
    calculate_distinct_n
)
from statistics_and_visualization import run_statistical_analysis, create_visualizations
from tqdm import tqdm
import time
import numpy as np  # 用于NaN

def analyze_data(df: pd.DataFrame, output_filename: str) -> pd.DataFrame:
    """对数据集进行特征提取。
    参数:
    df: 输入数据框
    output_filename: 输出文件路径
    返回:
    分析后的数据框
    """
    start_time = time.time()  # 记录开始时间
    print(f"[{datetime.now()}] --- 开始特征提取 ---")
    df_analyzed = df.copy()  # 复制数据框
    metrics_to_generate = [
        'tokens_per_second', 'shannon_entropy', 'perplexity_proxy', 'novelty_score', 'ttr',
        'graph_density', 'avg_clustering', 'psi_proxy', 'analogy_rate', 'internal_coherence', 'distinct_2'
    ]  # 要生成的指标
    
    # 预提取文本和上下文
    all_texts = []  # 存储响应文本
    all_contexts = []  # 存储上下文
    total_rows = len(df)  # 总行数
    print(f"[{datetime.now()}] --- 准备文本数据 ({total_rows} 行) ---")
    for index, row in tqdm(df_analyzed.iterrows(), total=total_rows, desc="提取文本进度"):  # 进度条
        response_text = row['response']  # 获取响应
        if pd.isna(response_text) or isinstance(response_text, float) or not response_text.strip() or "ERROR" in response_text:
            all_texts.append("")  # 无效添加空
            all_contexts.append("")  # 无效添加空
            continue
        all_texts.append(response_text)  # 添加响应
        context = ""  # 初始化上下文
        if row['condition'] != 'Idle':
            try:
                context = row['prompt'].split("[CONTEXT]\n")[1].split("\n\n[INSTRUCTION]")[0]  # 提取上下文
            except IndexError:
                context = row['prompt']  # 异常使用prompt
        else:
            context = row['prompt'].replace("[INSTRUCTION]\n", "")  # Idle情况
        all_contexts.append(context)  # 添加上下文
    
    # 特征计算
    print(f"[{datetime.now()}] --- 计算特征 ---")
    progress_step = max(1, total_rows // 10)  # 进度步长
    for index, row in tqdm(df_analyzed.iterrows(), total=total_rows, desc="计算特征进度"):
        if index % progress_step == 0:
            print(f"[{datetime.now()}] 已处理 {index}/{total_rows} 行 ({index/total_rows*100:.1f}%)")  # 打印进度
        response_text = row['response']  # 获取响应
        context = all_contexts[index]  # 获取上下文
        
        if pd.isna(response_text) or isinstance(response_text, float) or not response_text.strip() or "ERROR" in response_text:
            print(f"[{datetime.now()}] 跳过行 {index + 1}: 无效或空响应")  # 日志
            for metric in metrics_to_generate:
                df_analyzed.loc[index, metric] = 0.0 if metric not in ['psi_proxy', 'perplexity_proxy'] else np.nan  # 设置默认值
            continue
        
        # 计算特征
        df_analyzed.loc[index, 'shannon_entropy'] = calculate_shannon_entropy(response_text)
        df_analyzed.loc[index, 'perplexity_proxy'] = calculate_perplexity_proxy(response_text)
        df_analyzed.loc[index, 'ttr'] = calculate_ttr(response_text)
        df_analyzed.loc[index, 'analogy_rate'] = calculate_analogy_rate_semantic(response_text)
        df_analyzed.loc[index, 'distinct_2'] = calculate_distinct_n(response_text)
        df_analyzed.loc[index, 'tokens_per_second'] = calculate_tokens_per_second(row)
        df_analyzed.loc[index, 'novelty_score'] = calculate_novelty(response_text, context)
        complexity_metrics = calculate_semantic_complexity_keybert(response_text)
        df_analyzed.loc[index, 'graph_density'] = complexity_metrics['graph_density']
        df_analyzed.loc[index, 'avg_clustering'] = complexity_metrics['avg_clustering']
        df_analyzed.loc[index, 'internal_coherence'] = calculate_internal_coherence(response_text)
        df_analyzed.loc[index, 'psi_proxy'] = calculate_psi_proxy_chinese(response_text, df_analyzed.loc[index, 'perplexity_proxy'])
    
    df_analyzed['prompt_length'] = df_analyzed['prompt'].str.len()  # 计算prompt长度
    df_analyzed.to_csv(output_filename, index=False, encoding='utf-8-sig')  # 保存CSV
    print(f"[{datetime.now()}] --- 分析完成。保存到 '{output_filename}'，耗时 {time.time() - start_time:.2f} 秒 ---")
    return df_analyzed

def generate_structured_report(df: pd.DataFrame, stats_results: list, plot_files: list, output_filename: str):
    """生成结构化Markdown报告。
    参数:
    df: 数据框
    stats_results: 统计结果
    plot_files: 图表文件
    output_filename: 输出路径
    """
    print(f"[{datetime.now()}] --- 生成结构化报告 ---")
    report_content = f"""
# 文本分析报告 (V5.1)
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 概述
本报告总结了对中文文本数据集的全面分析，包括特征提取、统计分析和可视化结果。
- **输入文件**: {RAW_DATA_FILE}
- **输出数据文件**: {ANALYZED_DATA_FILE}
- **总行数**: {len(df)}
- **实验条件**: {', '.join(df['condition'].unique())}
## 分析指标
| 指标 | 描述 |
|------|------|
| tokens_per_second | 每秒生成的token数，衡量生成速度 |
| shannon_entropy | 文本的香农熵，衡量信息多样性 |
| perplexity_proxy | 文本困惑度，衡量语言模型预测难度 |
| novelty_score | 文本相对于上下文的新颖度 |
| ttr | 类型-标记比率，衡量词汇多样性 |
| graph_density | 语义网络的密度，基于关键词共现 |
| avg_clustering | 语义网络的平均聚类系数 |
| psi_proxy | 心理信息量代理，结合情感和困惑度 |
| analogy_rate | 类比句式的比例 |
| internal_coherence | 文本内部语义连贯性 |
| distinct_2 | 二元n-gram的多样性 |
## 统计分析结果
| 指标 | F值 | P值 | 效应量 (η²) |
|------|------|------|------|
"""
    for result in stats_results:  # 添加统计行
        f_value = f"{result['f_value']:.4f}" if not pd.isna(result['f_value']) else "N/A"
        p_value = f"{result['p_value']:.4f}" if not pd.isna(result['p_value']) else "N/A"
        eta_squared = f"{result['eta_squared']:.4f}" if not pd.isna(result['eta_squared']) else "N/A"
        report_content += f"| {result['metric']} | {f_value} | {p_value} | {eta_squared} |\n"
    report_content += """
## 可视化结果
以下为生成的可视化图表，展示不同条件下的指标表现：
"""
    for plot_file in plot_files:  # 添加图表
        report_content += f"![{plot_file}]({plot_file})\n\n"
    report_content += """
## 结论
- 分析成功完成了特征提取、统计测试和可视化。
- 关键发现：{根据实际结果手动补充}
- 建议：{根据实际结果手动补充后续研究或改进}
## 使用说明
- 本报告由 `main_analysis.py` 自动生成。
- 所有图表和数据文件均保存在当前工作目录。
- 建议将此报告及相关文件上传至GitHub，展示分析流程和结果。
"""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)  # 写入文件
    print(f"[{datetime.now()}] --- 报告保存到 '{output_filename}' ---")

def main():
    """主执行函数。"""
    start_time = time.time()  # 开始时间
    print(f"[{datetime.now()}] --- 开始管道执行 ---")
    
    # 步骤1: 加载数据
    try:
        print(f"[{datetime.now()}] --- 从 '{RAW_DATA_FILE}' 加载数据 ---")
        df_raw = pd.read_csv(RAW_DATA_FILE)  # 读取CSV
        print(f"[{datetime.now()}] --- 加载 {len(df_raw)} 行 ---")
    except FileNotFoundError:
        print(f"[{datetime.now()}] 错误: '{RAW_DATA_FILE}' 未找到。")
        return
    except Exception as e:
        print(f"[{datetime.now()}] 数据加载错误: {e}")
        return
    
    # 步骤2: 特征提取
    df_analyzed = analyze_data(df_raw, ANALYZED_DATA_FILE)  # 分析数据
    
    # 步骤3: 统计分析
    if not df_analyzed.empty:
        stats_results = run_statistical_analysis(df_analyzed)  # 运行统计
        
        # 步骤4: 可视化
        plot_files = create_visualizations(df_analyzed)  # 创建图表
        
        # 步骤5: 生成报告
        generate_structured_report(df_analyzed, stats_results, plot_files, REPORT_FILE)  # 生成报告
    else:
        print(f"[{datetime.now()}] 分析数据为空。跳过后续步骤。")
    
    print(f"[{datetime.now()}] --- 管道完成，耗时 {time.time() - start_time:.2f} 秒 ---")

if __name__ == "__main__":
    main()  # 执行主函数
```