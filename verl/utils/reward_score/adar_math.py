"""
Reward function for AdaR orca-math data sources.
Ground truth answers are numerical values; model responses may use \\boxed{} or plain numbers.
"""

import re


def extract_last_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in the text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left = 0
    right_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left += 1
        if text[i] == "}":
            num_left -= 1
            if num_left == 0:
                right_idx = i
                break
        i += 1
    if right_idx is None:
        return None
    start = text.index("{", idx) + 1
    return text[start:right_idx]


def extract_last_number(text: str) -> float | None:
    """Extract the last number from text, handling commas in numbers."""
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Try boxed first
    boxed = extract_last_boxed(text)
    if boxed is not None:
        nums = re.findall(r"(-?\d+\.?\d*)", boxed)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
    # Fallback: find last number in text
    nums = re.findall(r"(-?\d+\.?\d*)", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def compute_score(solution_str: str, ground_truth, **kwargs) -> float:
    """
    Compute reward score for AdaR data.
    Returns 1.0 if the model's answer matches ground truth within tolerance.
    """
    try:
        if isinstance(ground_truth, (list, tuple)):
            gt_value = float(ground_truth[0])
        else:
            gt_value = float(ground_truth)
    except (ValueError, TypeError):
        print(f"---REWARD--- Failed to parse ground_truth: {ground_truth}")
        return 0.0

    predicted = extract_last_number(solution_str)
    if predicted is None:
        return 0.0

    # Use relative tolerance for large numbers, absolute for small
    if abs(gt_value) > 1e-6:
        if abs(predicted - gt_value) / max(abs(gt_value), 1e-10) < 1e-3:
            return 1.0
    else:
        if abs(predicted - gt_value) < 1e-3:
            return 1.0

    return 0.0
