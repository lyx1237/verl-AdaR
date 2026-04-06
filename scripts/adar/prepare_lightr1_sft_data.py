"""
准备Light-R1-SFTData训练数据.
将qihoo360/Light-R1-SFTData的conversations格式转换为verl self-play兼容的parquet格式.

数据来源: https://huggingface.co/datasets/qihoo360/Light-R1-SFTData
原始格式: conversations: [{from: "user", value: ...}, {from: "assistant", value: ...}]
assistant回答中包含<think>推理过程和\\boxed{}最终答案.
"""

import json
import re
import sys
import os
import pandas as pd


def extract_answer_from_boxed(text):
    """从assistant回答中提取\\boxed{}中的最终答案."""
    # 找最后一个\boxed{}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # 退而求其次: 找<answer>标签
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


def extract_query_and_response(conversations):
    """从conversations中提取user query和assistant response."""
    query = None
    response = None
    for msg in conversations:
        if msg["from"] == "user":
            query = msg["value"]
        elif msg["from"] == "assistant":
            response = msg["value"]
    return query, response


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/Light-R1-SFTData"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../data/selfplay/train_lightr1_sft.parquet"
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据 - 支持parquet和json格式
    print(f"---DATA--- 加载数据: {input_dir}")
    all_items = []

    # 尝试加载parquet文件
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if parquet_files:
        for pf in sorted(parquet_files):
            filepath = os.path.join(input_dir, pf)
            print(f"---DATA--- 读取parquet: {filepath}")
            df = pd.read_parquet(filepath)
            for _, row in df.iterrows():
                all_items.append(row.to_dict())
    elif json_files:
        for jf in sorted(json_files):
            filepath = os.path.join(input_dir, jf)
            print(f"---DATA--- 读取json: {filepath}")
            with open(filepath) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_items.extend(data)
                else:
                    all_items.append(data)
    else:
        print(f"---ERROR--- 在 {input_dir} 中未找到parquet或json文件")
        sys.exit(1)

    print(f"---DATA--- 总样本数: {len(all_items)}")

    if max_samples and max_samples < len(all_items):
        import random
        random.seed(42)
        all_items = random.sample(all_items, max_samples)
        print(f"---DATA--- 采样后样本数: {len(all_items)}")

    records = []
    skipped_no_query = 0
    skipped_no_answer = 0

    for idx, item in enumerate(all_items):
        conversations = item.get("conversations", [])
        if not conversations:
            skipped_no_query += 1
            continue

        query, response = extract_query_and_response(conversations)
        if not query or not response:
            skipped_no_query += 1
            continue

        answer = extract_answer_from_boxed(response)
        if not answer:
            skipped_no_answer += 1
            continue

        # 剥离<think>...</think>内容, 只保留最终解答 (Stage1需要简洁的CoT作为输入)
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        record = {
            "data_source": f"adar-selfplay-{idx}",
            "prompt": [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": query},
            ],
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer),
            },
            "extra_info": {
                "id": idx,
                "query": query,
                "chosen": cleaned_response,
                "answer": str(answer),
            },
        }
        records.append(record)

    print(f"---DATA--- 有效样本数: {len(records)}")
    print(f"---DATA--- 跳过(无query/response): {skipped_no_query}")
    print(f"---DATA--- 跳过(无boxed答案): {skipped_no_answer}")

    df = pd.DataFrame(records)
    df.to_parquet(output_path)
    print(f"---DATA--- 输出: {output_path}, 样本数: {len(df)}")
    print(f"---DATA--- 列: {list(df.columns)}")


if __name__ == "__main__":
    main()
