import pandas as pd
import json
import os
import argparse
import sys
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_args():
    """
    配置命令行参数
    """
    parser = argparse.ArgumentParser(description="将 cMedQA2 CSV 数据转换为 ChatML 格式的 JSONL 用于微调")
    parser.add_argument('--question_path', type=str, default=r"data/raw/question.csv", help="问题 CSV 文件路径 (例如: data/raw/question.csv)")
    parser.add_argument('--answer_path', type=str, default=r"data/raw/answer.csv", help="答案 CSV 文件路径 (例如: data/raw/answer.csv)")
    parser.add_argument('--output_path', type=str, default=r"data/processed/train_peft.jsonl", help="输出 JSONL 文件路径 (例如: data/processed/train_peft.jsonl)")
    parser.add_argument('--limit', type=int, default=256, help="仅处理前 N 条问题 ")
    return parser.parse_args()

def load_csv_safe(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    安全加载 CSV 文件，尝试不同的编码格式
    """
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        sys.exit(1)

    try:
        logger.info(f"正在读取 {path} (utf-8)...")
        return pd.read_csv(path, nrows=nrows, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 解码失败，尝试 GBK...")
        try:
            return pd.read_csv(path, nrows=nrows, encoding='gbk')
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            sys.exit(1)

def process_cmedqa2(question_path: str, answer_path: str, output_path: str, limit: Optional[int] = None):
    """
    处理 cMedQA2 数据集的核心逻辑
    """
    # 1. 加载数据
    df_q = load_csv_safe(question_path, nrows=limit)
    logger.info(f"已加载问题数据: {len(df_q)} 条")
    
    # 答案数据量可能很大，建议全部读取后在内存中 merge，或者分块处理
    # 对于百万级数据，Pandas 通常可以处理
    df_a = load_csv_safe(answer_path)
    logger.info(f"已加载答案数据: {len(df_a)} 条")

    # 2. 数据清洗与合并
    # 确保列名存在
    if 'question_id' not in df_q.columns or 'content' not in df_q.columns:
        logger.error("question.csv 缺少必要的列 (question_id, content)")
    
    if 'question_id' not in df_a.columns or 'content' not in df_a.columns:
        logger.error("answer.csv 缺少必要的列 (question_id, content)")


    # 如果有限制，先过滤 df_a 以加速 merge
    if limit:
        target_qids = df_q['question_id'].unique()
        df_a = df_a[df_a['question_id'].isin(target_qids)]
    
    logger.info("正在合并问题与答案...")
    # 使用 inner join，只保留有答案的问题
    merged = pd.merge(df_q, df_a, on='question_id', how='inner', suffixes=('_q', '_a'))
    logger.info(f"合并完成，共 {len(merged)} 条 QA 对")

    # 3. 转换为 ChatML 格式
    logger.info("正在转换为 ChatML 格式并写入文件...")
    
    system_prompt = "你是一个专业的医疗助手，能够提供准确的医疗咨询。"
    
    success_count = 0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in merged.iterrows():
            q_text = str(row['content_q']).strip()
            a_text = str(row['content_a']).strip()

            # 跳过空数据
            if not q_text or not a_text or q_text.lower() == 'nan' or a_text.lower() == 'nan':
                continue

            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q_text},
                    {"role": "assistant", "content": a_text}
                ]
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            success_count += 1
            
            if success_count % 10000 == 0:
                logger.info(f"已处理 {success_count} 条数据...")

    logger.info(f"处理完成！最终生成 {success_count} 条有效训练数据。")
    logger.info(f"输出文件: {output_path}")

def main():
    args = setup_args()
    process_cmedqa2(args.question_path, args.answer_path, args.output_path, args.limit)

if __name__ == "__main__":
    main()