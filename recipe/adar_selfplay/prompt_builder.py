"""
PromptBuilder (prompt_builder.py)

负责为AdaR Self-Play的4个阶段构造不同格式的prompt,
并将文本prompt编码为verl的DataProto格式, 供generate_sequences()使用.

4个阶段的prompt格式:
- T1: 原始数学题 + "请提取模板和写解题代码" 的指令
- T2: 扰动后的数学题 + "请解答" 的指令
- T3: 通过EVS的题目 + "请改写这道题" 的指令
- T4: paraphrase后的题目 + "请解答" 的指令
"""

import uuid
import logging

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

logger = logging.getLogger(__name__)


# ============================================================
# Prompt模板定义
# ============================================================

# T1: 模板+代码生成指令
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

# T2/T4: 解答数学题的指令
SOLVE_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

SOLVE_WITH_CODE_PROMPT = r"""
Your task is to provide a clear chain-of-thought (COT) explanation that answers the user's question. A Python script may be provided as part of the input, but it is not mandatory to follow it closely. If the provided code doesn't align with the real-world scenario or if the values and logic in the code are incorrect or irrelevant to the problem, feel free to disregard the script. Instead, focus on reasoning through the problem using your own judgment and logic.
Interpret the question clearly and begin by understanding the problem. If the Python script can offer guidance, you may refer to it, but it's not a requirement. If the provided code does not match the context or contains errors, you are free to work through the solution from scratch without referring to it.
Explicitly state the final answer after completing your reasoning, enclosed in \boxed{}.
"""

SOLVE_WITH_CODE_INSTRUCTION = """
### Query:
{query}

### Python Code:
{code}

### Response:
"""

# T3: paraphrase指令
PARAPHRASE_PROMPT = """You are an AI assistant to help me rephrase questions. Follow the given examples.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Rephrase the above question: What is the amount of money that Olivia has left after purchasing five bagels
for $3 each, if she initially had $23?

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How
many golf balls did he have at the end of wednesday?
Rephrase the above question: After losing 23 golf balls on Tuesday and an additional 2 on Wednesday, how
many golf balls does Michael have left if he initially had 58 golf balls?

Question: Angelo and Melanie want to plan how many hours over the next week they should study together
for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize.
They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each
worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study
total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each
day, and 30 minutes for lunch each day?
Rephrase the above question: Angelo and Melanie need to study 2 chapters in their textbook and 4
worksheets for their upcoming test. They have planned to dedicate 3 hours for each chapter and 1.5 hours for
each worksheet. They can study for a maximum of 4 hours each day, taking into account 10-minute breaks
every hour, 3 10-minute snack breaks per day, and 30 minutes for lunch. How many days do they need to
study in total over the next week to complete their study plan?

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in
total?
Rephrase the above question: If Leah had 32 chocolates and her sister had 42, and they both consumed 35
chocolates, what is the total number of chocolates that they have left?

Question: There were nine computers in the server room. Five more computers were installed each day,
from monday to thursday. How many computers are now in the server room?
Rephrase the above question: If there were initially nine computers in the server room and five more
computers were added each day from Monday to Thursday, what is the current total number of computers in
the server room?

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many
lollipops did Jason give to Denny?
Rephrase the above question: If Jason initially had 20 lollipops and now has 12 after giving some to Denny,
how many lollipops did he give to Denny?

Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged
five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the
rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in
dollars?
Rephrase the above question: Sam purchased 12 boxes, each containing 30 highlighter pens, at $10 per
box. He repackaged five of these boxes into sets of six highlighters and sold them for $3 per set. He sold
the remaining highlighters individually at a rate of three pens for $2. What is the total profit he made in dollars?
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are
done, there will be 21 trees. How many trees did the grove workers plant today?
Rephrase the above question: If there were initially 15 trees in the grove and the grove workers are planning
to plant more trees today, resulting in a total of 21 trees, how many trees did the workers plant today?

Question: {question}
Rephrase the above question: """


class PromptBuilder:
    """
    将文本prompt编码为verl的DataProto格式.

    对于每个阶段, 接收文本列表, 输出包含input_ids/attention_mask/position_ids的DataProto,
    可直接传给actor_rollout_wg.generate_sequences().
    """

    def __init__(self, tokenizer, max_prompt_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _apply_chat_template(self, messages_list: list[list[dict]]) -> list[str]:
        """将messages列表通过chat template转为文本"""
        prompts = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        return prompts

    def _encode_prompts(self, prompts: list[str], max_length: int = None) -> DataProto:
        """
        将文本prompt列表编码为DataProto.
        返回包含 input_ids, attention_mask, position_ids 的DataProto.
        """
        if max_length is None:
            max_length = self.max_prompt_length

        all_input_ids = []
        all_attention_masks = []
        all_raw_prompt_ids = []

        for prompt in prompts:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"][0]  # (seq_len,)
            attention_mask = encoded["attention_mask"][0]

            # 截断 (左截断, 保留最后max_length个token)
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_raw_prompt_ids.append(input_ids.tolist())

        # 左padding到统一长度
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_position_ids = []

        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ])
                attention_mask = torch.cat([
                    torch.zeros(pad_len, dtype=attention_mask.dtype),
                    attention_mask
                ])

            # 计算position_ids
            position_ids = torch.zeros_like(input_ids)
            valid_mask = attention_mask.bool()
            position_ids[valid_mask] = torch.arange(valid_mask.sum().item())

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
            padded_position_ids.append(position_ids)

        # 构造DataProto
        batch_input_ids = torch.stack(padded_input_ids, dim=0)  # (batch, seq_len)
        batch_attention_mask = torch.stack(padded_attention_masks, dim=0)
        batch_position_ids = torch.stack(padded_position_ids, dim=0)

        # uid: 每个prompt一个唯一标识
        uids = np.array([str(uuid.uuid4()) for _ in range(len(prompts))], dtype=object)

        # raw_prompt_ids: numpy array of lists
        raw_prompt_ids_np = np.array(all_raw_prompt_ids, dtype=object)

        data_proto = DataProto.from_dict(
            tensors={
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "position_ids": batch_position_ids,
            },
            non_tensors={
                "raw_prompt_ids": raw_prompt_ids_np,
                "uid": uids,
            },
        )

        return data_proto

    # ============================================================
    # 各阶段prompt构造
    # ============================================================

    def build_t1_prompts(
        self,
        queries: list[str],
        responses: list[str],
        max_length: int = None,
    ) -> DataProto:
        """
        T1阶段: 构造模板+代码生成的prompt.

        Args:
            queries: 原始数学题列表
            responses: 原始CoT回答列表
            max_length: prompt最大长度

        Returns:
            DataProto, 可传给generate_sequences()
        """
        messages_list = []
        for query, response in zip(queries, responses):
            content = T1_INSTRUCTION + T1_PROMPT_TEMPLATE.format(query=query, response=response)
            messages = [{"role": "user", "content": content}]
            messages_list.append(messages)

        prompts = self._apply_chat_template(messages_list)
        result = self._encode_prompts(prompts, max_length=max_length)
        logger.info(f"---PROMPT_BUILDER--- T1: 构造了 {len(queries)} 个模板+代码生成prompt")
        return result

    def build_t2_prompts(
        self,
        queries: list[str],
        codes: list[str],
        max_length: int = None,
    ) -> DataProto:
        """
        T2阶段: 构造解答扰动问题的prompt (带代码提示).

        Args:
            queries: 扰动后的数学题列表
            codes: 对应的代码列表
            max_length: prompt最大长度

        Returns:
            DataProto
        """
        messages_list = []
        for query, code in zip(queries, codes):
            content = SOLVE_WITH_CODE_PROMPT + SOLVE_WITH_CODE_INSTRUCTION.format(query=query, code=code)
            messages = [
                {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            messages_list.append(messages)

        prompts = self._apply_chat_template(messages_list)
        result = self._encode_prompts(prompts, max_length=max_length)
        logger.info(f"---PROMPT_BUILDER--- T2: 构造了 {len(queries)} 个解答扰动问题prompt")
        return result

    def build_t3_prompts(
        self,
        questions: list[str],
        max_length: int = None,
    ) -> DataProto:
        """
        T3阶段: 构造paraphrase的prompt.

        Args:
            questions: 要改写的题目列表
            max_length: prompt最大长度

        Returns:
            DataProto
        """
        messages_list = []
        for question in questions:
            content = PARAPHRASE_PROMPT.format(question=question)
            messages = [{"role": "user", "content": content}]
            messages_list.append(messages)

        prompts = self._apply_chat_template(messages_list)
        result = self._encode_prompts(prompts, max_length=max_length)
        logger.info(f"---PROMPT_BUILDER--- T3: 构造了 {len(questions)} 个paraphrase prompt")
        return result

    def build_t4_prompts(
        self,
        questions: list[str],
        max_length: int = None,
    ) -> DataProto:
        """
        T4阶段: 构造解答paraphrased问题的prompt (纯解答, 无代码提示).

        Args:
            questions: paraphrased的数学题列表
            max_length: prompt最大长度

        Returns:
            DataProto
        """
        messages_list = []
        for question in questions:
            messages = [
                {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            messages_list.append(messages)

        prompts = self._apply_chat_template(messages_list)
        result = self._encode_prompts(prompts, max_length=max_length)
        logger.info(f"---PROMPT_BUILDER--- T4: 构造了 {len(questions)} 个解答prompt")
        return result

    def build_standard_solve_prompts(
        self,
        questions: list[str],
        max_length: int = None,
    ) -> DataProto:
        """
        标准解题prompt (用于T4-only模式, 等同于标准GRPO训练).
        与build_t4_prompts相同.
        """
        return self.build_t4_prompts(questions, max_length=max_length)
