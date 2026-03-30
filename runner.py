import asyncio
import json
import re
from pathlib import Path

import os

import openai
import anthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

_openai_client = None
_anthropic_client = None
_gemini_client = None
_openrouter_client = None

OPENROUTER_MODELS = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "mistral-small": "mistralai/mistral-small-24b-instruct-2501",
}


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI()
    return _openai_client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = openai.AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _gemini_client


def get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        _openrouter_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return _openrouter_client

LETTER_MAP = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
RESULTS_DIR = Path("results")


def parse_response(text, question_framing):
    if text is None:
        return None
    text = text.strip().lower()

    if question_framing == "forced_choice":
        for char in text:
            if char in LETTER_MAP:
                return LETTER_MAP[char]

    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))

    return None


async def call_openai(prompt, semaphore, max_retries=8):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(get_openai_client().chat.completions.create(
                    model=prompt["model"],
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=prompt["temperature"],
                    max_completion_tokens=50,
                ), timeout=30)
                return response.choices[0].message.content
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return "ERROR: timeout"
                await asyncio.sleep(2)
            except openai.RateLimitError:
                await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(1)
        return "ERROR: max retries exceeded"


async def call_anthropic(prompt, semaphore, max_retries=8):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(get_anthropic_client().messages.create(
                    model=prompt["model"],
                    system=prompt["system"],
                    messages=[
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=prompt["temperature"],
                    max_tokens=10,
                ), timeout=30)
                return response.content[0].text
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return "ERROR: timeout"
                await asyncio.sleep(2)
            except anthropic.RateLimitError:
                await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(1)
        return "ERROR: max retries exceeded"


async def call_gemini(prompt, semaphore, max_retries=8):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(get_gemini_client().chat.completions.create(
                    model=prompt["model"],
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=prompt["temperature"],
                    max_completion_tokens=50,
                ), timeout=30)
                content = response.choices[0].message.content
                if content is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return "ERROR: empty response"
                return content
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return "ERROR: timeout"
                await asyncio.sleep(2)
            except openai.RateLimitError:
                await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(1)
        return "ERROR: max retries exceeded"


async def call_openrouter(prompt, semaphore, max_retries=8):
    async with semaphore:
        api_model = OPENROUTER_MODELS.get(prompt["model"], prompt["model"])
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(get_openrouter_client().chat.completions.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=prompt["temperature"],
                    max_tokens=10,
                ), timeout=30)
                content = response.choices[0].message.content
                if content is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return "ERROR: empty response"
                return content
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return "ERROR: timeout"
                await asyncio.sleep(2)
            except openai.RateLimitError:
                await asyncio.sleep(min(2 ** attempt, 10))
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(1)
        return "ERROR: max retries exceeded"


MODEL_ROUTES = {
    "gpt-5.4-nano": "openai",
    "gpt-5.4": "openai",
    "claude-sonnet-4-6": "anthropic",
    "gemini-2.5-flash": "gemini",
    "llama-3.3-70b": "openrouter",
    "mistral-small": "openrouter",
}

DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
    "openrouter": call_openrouter,
}


PROVIDER_CONCURRENCY = {
    "openai": 30,
    "anthropic": 5,
    "gemini": 5,
    "openrouter": 20,
}


def _get_route(model):
    route = MODEL_ROUTES.get(model)
    if route is None:
        for prefix, r in [("gpt", "openai"), ("claude", "anthropic"), ("gemini", "gemini"), ("llama", "openrouter"), ("mistral", "openrouter")]:
            if model.startswith(prefix):
                route = r
                break
    return route


async def call_model(prompt, semaphore):
    route = _get_route(prompt["model"])
    if route is None:
        return "ERROR: unknown model"
    return await DISPATCH[route](prompt, semaphore)


async def run_batch(tasks, max_concurrent=20):
    provider_semaphores = {
        provider: asyncio.Semaphore(limit)
        for provider, limit in PROVIDER_CONCURRENCY.items()
    }
    fallback_semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(task):
        prompt = task["prompt"]
        route = _get_route(prompt["model"])
        semaphore = provider_semaphores.get(route, fallback_semaphore)
        raw = await call_model(prompt, semaphore)
        score = parse_response(raw, task["question_framing"])
        return {
            "spec_id": task["spec_id"],
            "profile_id": task["profile_id"],
            "party": task["party"],
            "item": task["item"],
            "repeat": task["repeat"],
            "model": prompt["model"],
            "persona_format": task["persona_format"],
            "question_framing": task["question_framing"],
            "system_prompt": task["system_prompt"],
            "temperature": prompt["temperature"],
            "few_shot": task["few_shot"],
            "raw_response": raw,
            "score": score,
        }

    results = await tqdm_asyncio.gather(
        *[execute(t) for t in tasks],
        desc="Running specifications",
    )
    return results


def save_results(results, filename="pilot_results.json"):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
