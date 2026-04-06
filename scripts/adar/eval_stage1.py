"""
评估模型的Stage1 (模板+代码生成) 能力.
用vLLM部署模型, 对测试数据运行Stage1 prompt, 检查VA/EC通过率.

用法:
    python eval_stage1.py --model_path <模型路径> --test_data <parquet路径> [--n_samples 4] [--gpu 0]
"""

import argparse
import json
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from recipe.adar_selfplay.auto_pipeline import parse_and_verify, SafeExecutor

# 复制 prompt 模板
T1_INSTRUCTION = r"""Task Description:
You are given a natural language query and its chain-of-thought response. Your task is to:
Generate a Query Template by abstracting specific values into variables.
Generate Python Code that executes the logic described in the COT response using the abstracted variables.

Input Format:
Query: Original query with specific values
Response: Chain-of-thought reasoning that leads to the answer

Output Requirements:
Query Template:
Replace only concrete values in the query with angle-bracketed placeholders like <variable_name>.
Do not replace names or general nouns (e.g., do not change "Jungkook" to <person_name>).
Preserve the original wording and structure of the query as much as possible.
Python Code:
Begin by defining variables that correspond to the placeholders in the template.
Translate the logic in the response into executable Python code.
The code should end with a print() statement that prints only the final result.
Do not include comments with explanations or reasoning.
Use the same variable names as in the template for consistency.

=== START EXAMPLE ===
### Query:
Find A that satisfies 32×A×A×A=42592

### Response:
To find the value of A that satisfies the equation 32×A×A×A=42592, we can rewrite the equation as:
\(32A^3 = 42592\)
Now, we need to isolate A by dividing both sides of the equation by 32:
\(A^3 = \frac{42592}{32}\)
\(A^3 = 1331\)
Now, we take the cube root of both sides to solve for A:
\(A = \sqrt[3]{1331}\)
\(A = 11\)

### Template:
Find A that satisfies <coefficient>×A×A×A=<result>

### Python Code:
```python
# Variable definitions
coefficient = 32
result = 42592

# Calculation
A_cubed = result / coefficient
A = A_cubed ** (1/3)

# Output
print(A)
```
=== END EXAMPLE ===
"""

T1_PROMPT_TEMPLATE = """
Instruction:
### Query:
{query}

### Response:
{response}

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--test_data", required=True, help="测试数据parquet路径")
    parser.add_argument("--n_samples", type=int, default=4, help="每题采样次数")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--max_tokens", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--output", type=str, default=None, help="输出结果JSON路径")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import pandas as pd
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 加载数据
    df = pd.read_parquet(args.test_data)
    print(f"---EVAL--- 加载测试数据: {len(df)} 条")

    # 构建prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = []
    queries = []
    answers = []
    for _, row in df.iterrows():
        extra = row["extra_info"]
        query = extra["query"]
        chosen = extra["chosen"]
        answer = extra["answer"]
        queries.append(query)
        answers.append(answer)

        content = T1_INSTRUCTION + T1_PROMPT_TEMPLATE.format(query=query, response=chosen)
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"---EVAL--- 构建了 {len(prompts)} 个prompt")

    # 加载模型
    print(f"---EVAL--- 加载模型: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        n=args.n_samples,
    )

    print(f"---EVAL--- 开始生成, n_samples={args.n_samples}...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"---EVAL--- 生成完成")

    # 评估
    executor = SafeExecutor()
    total_rollouts = 0
    va_passed_count = 0
    ec_passed_count = 0
    prompt_passed_count = 0  # 至少有一个rollout通过的prompt数

    results = []
    for i, output in enumerate(outputs):
        query = queries[i]
        answer = answers[i]
        prompt_has_pass = False

        for j, completion in enumerate(output.outputs):
            total_rollouts += 1
            text = completion.text
            detail = parse_and_verify(text, query, answer, executor, code_timeout=3.0, return_detail=True)
            va = detail["va_passed"]
            ec = detail["ec_passed"]
            if va:
                va_passed_count += 1
            if ec:
                ec_passed_count += 1
                prompt_has_pass = True

        if prompt_has_pass:
            prompt_passed_count += 1

        results.append({
            "query": query[:100],
            "answer": answer,
            "n_rollouts": args.n_samples,
            "va_passed": va_passed_count,
            "ec_passed": ec_passed_count,
        })

    # 汇总
    print(f"\n{'='*60}")
    print(f"---EVAL--- 模型: {args.model_path}")
    print(f"---EVAL--- 测试数据: {args.test_data}")
    print(f"---EVAL--- 总prompt数: {len(prompts)}")
    print(f"---EVAL--- 每prompt采样: {args.n_samples}")
    print(f"---EVAL--- 总rollout数: {total_rollouts}")
    print(f"---EVAL--- VA通过: {va_passed_count}/{total_rollouts} ({va_passed_count/total_rollouts*100:.1f}%)")
    print(f"---EVAL--- EC通过: {ec_passed_count}/{total_rollouts} ({ec_passed_count/total_rollouts*100:.1f}%)")
    print(f"---EVAL--- Prompt通过率: {prompt_passed_count}/{len(prompts)} ({prompt_passed_count/len(prompts)*100:.1f}%)")
    print(f"{'='*60}")

    # 保存结果
    if args.output:
        summary = {
            "model_path": args.model_path,
            "test_data": args.test_data,
            "n_prompts": len(prompts),
            "n_samples": args.n_samples,
            "total_rollouts": total_rollouts,
            "va_passed": va_passed_count,
            "ec_passed": ec_passed_count,
            "prompt_passed": prompt_passed_count,
            "va_rate": va_passed_count / total_rollouts,
            "ec_rate": ec_passed_count / total_rollouts,
            "prompt_pass_rate": prompt_passed_count / len(prompts),
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"---EVAL--- 结果保存到: {args.output}")


if __name__ == "__main__":
    main()
