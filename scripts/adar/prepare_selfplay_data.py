"""
准备Self-Play训练数据.
将原始JSON数据转换为verl兼容的parquet格式,
在extra_info中包含query, chosen, answer, 供Self-Play trainer使用.
"""

import json
import sys
import pandas as pd

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/orca_200.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../data/selfplay/train_selfplay.parquet"

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    records = []
    for idx, item in enumerate(data):
        record = {
            "data_source": f"adar-selfplay-{idx}",
            "prompt": [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": item["query"]},
            ],
            "reward_model": {
                "style": "rule",
                "ground_truth": str(item["answer"]),
            },
            "extra_info": {
                "id": idx,
                "query": item["query"],
                "chosen": item["chosen"],
                "answer": str(item["answer"]),
            },
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_parquet(output_path)
    print(f"---DATA--- 输出: {output_path}, 样本数: {len(df)}")
    print(f"---DATA--- 列: {list(df.columns)}")

if __name__ == "__main__":
    main()
