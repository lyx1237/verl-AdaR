"""
API Rollout Client (api_rollout.py)

当某个阶段关闭self-play时, 使用外部API (OpenAI-compatible) 进行异步rollout,
替代本地模型的generate_sequences.

使用aiohttp实现异步并发请求, 支持重试和并发控制.
"""

import asyncio
import json
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


async def _single_api_call(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> str:
    """
    单个API调用, 带重试.

    Args:
        session: aiohttp session
        endpoint: API endpoint URL (e.g., "http://localhost:8000/v1/chat/completions")
        model: 模型名称
        prompt: 输入prompt文本
        max_tokens: 最大生成token数
        temperature: 采样温度
        timeout: 单次请求超时(秒)
        max_retries: 最大重试次数

    Returns:
        生成的response文本, 失败返回空字符串
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.warning(f"---API_ROLLOUT--- HTTP {resp.status}: {error_text[:200]} (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue

                output = await resp.text()
                data = json.loads(output)
                return data["choices"][0]["message"]["content"]

        except asyncio.TimeoutError:
            logger.warning(f"---API_ROLLOUT--- Timeout (attempt {attempt+1}/{max_retries})")
        except (aiohttp.ClientError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"---API_ROLLOUT--- Error: {type(e).__name__}: {e} (attempt {attempt+1})")

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    logger.error(f"---API_ROLLOUT--- All {max_retries} attempts failed for prompt[:50]={prompt[:50]}...")
    return ""


async def _api_generate_batch_async(
    prompts: list[str],
    endpoint: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_concurrent: int = 64,
) -> list[str]:
    """
    异步批量API调用.

    Args:
        prompts: prompt文本列表
        endpoint: API endpoint URL
        model: 模型名称
        max_tokens: 最大生成token数
        temperature: 采样温度
        timeout: 单次请求超时(秒)
        max_retries: 最大重试次数
        max_concurrent: 最大并发请求数

    Returns:
        response文本列表, 与prompts等长
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def _bounded_call(prompt: str) -> str:
            async with semaphore:
                return await _single_api_call(
                    session=session,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=max_retries,
                )

        tasks = [_bounded_call(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


def api_generate_batch(
    prompts: list[str],
    endpoint: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_concurrent: int = 64,
) -> list[str]:
    """
    同步接口: 批量调用外部API生成response.

    Args:
        prompts: prompt文本列表
        endpoint: API endpoint URL (e.g., "http://localhost:8000/v1/chat/completions")
        model: 模型名称
        max_tokens: 最大生成token数
        temperature: 采样温度
        timeout: 单次请求超时(秒)
        max_retries: 最大重试次数
        max_concurrent: 最大并发请求数

    Returns:
        response文本列表, 与prompts等长
    """
    start = time.time()
    logger.info(f"---API_ROLLOUT--- 开始API batch生成: {len(prompts)}个请求, "
                f"endpoint={endpoint}, model={model}, max_concurrent={max_concurrent}")
    print(f"---API_ROLLOUT--- 开始API batch生成: {len(prompts)}个请求")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # 已经在async context中, 使用nest_asyncio或新线程
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(
                asyncio.run,
                _api_generate_batch_async(
                    prompts=prompts,
                    endpoint=endpoint,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=max_retries,
                    max_concurrent=max_concurrent,
                )
            ).result()
    else:
        results = asyncio.run(
            _api_generate_batch_async(
                prompts=prompts,
                endpoint=endpoint,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                max_concurrent=max_concurrent,
            )
        )

    elapsed = time.time() - start
    non_empty = sum(1 for r in results if r.strip())
    logger.info(f"---API_ROLLOUT--- API batch完成: {non_empty}/{len(prompts)}个非空response, "
                f"耗时: {elapsed:.1f}s")
    print(f"---API_ROLLOUT--- API batch完成: {non_empty}/{len(prompts)}个非空, 耗时: {elapsed:.1f}s")

    return results
