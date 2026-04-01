"""
Custom reward function for AdaR RLVR training.
Registers reward computation for orca-math data sources.
The ground truth answers are numerical values from code execution.

NOTE: This file was originally in AdaR/scripts/reward_func.py and has been
copied here to make verl/recipe/adar_selfplay self-contained.
"""

import re
import importlib


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
    # Extract content between \boxed{ and }
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


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Compute reward score for AdaR data.
    Returns 1.0 if the model's answer matches ground truth within tolerance.
    """
    try:
        gt_value = float(ground_truth)
    except (ValueError, TypeError):
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


def register_adar_reward():
    """
    Monkey-patch verl's default_compute_score to handle AdaR data sources.
    Call this before training starts.
    """
    import verl.utils.reward_score as reward_module

    original_fn = reward_module.default_compute_score

    def patched_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
        # AdaR data sources start with "orca-80K" or similar
        if "orca" in data_source.lower() or "adar" in data_source.lower():
            return compute_score(solution_str, ground_truth)
        return original_fn(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

    reward_module.default_compute_score = patched_compute_score
    print("---REWARD--- AdaR reward function registered successfully")


if __name__ == "__main__":
    # Test
    assert compute_score("The answer is \\boxed{42}", "42.0") == 1.0
    assert compute_score("The answer is \\boxed{42}", "43.0") == 0.0
    assert compute_score("I got 3.14159", "3.14159") == 1.0
    print("All tests passed!")
