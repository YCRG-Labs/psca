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
                response = await get_openai_client().chat.completions.create(
                    model=prompt["model"],
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=prompt["temperature"],
                    max_completion_tokens=10,
                )
                return response.choices[0].message.content
            except openai.RateLimitError:
                await asyncio.sleep(2 ** attempt + 1)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(2)
        return "ERROR: max retries exceeded"


async def call_anthropic(prompt, semaphore, max_retries=8):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await get_anthropic_client().messages.create(
                    model=prompt["model"],
                    system=prompt["system"],
                    messages=[
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=prompt["temperature"],
                    max_tokens=10,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt + 1)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"ERROR: {e}"
                await asyncio.sleep(2)
        return "ERROR: max retries exceeded"


async def call_gemini(prompt, semaphore, max_retries=8):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await get_gemini_client().chat.completions.create(
                    model=prompt["model"],
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=prompt["temperature"],
                    max_completion_tokens=50,
                )
                content = response.choices[0].message.content
                if content is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return "ERROR: empty response"
                return content
            except openai.RateLimitError:
                await asyncio.sleep(2 ** attempt)
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
}

DISPATCH = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}


async def call_model(prompt, semaphore):
    route = MODEL_ROUTES.get(prompt["model"])
    if route is None:
        for prefix, r in [("gpt", "openai"), ("claude", "anthropic"), ("gemini", "gemini")]:
            if prompt["model"].startswith(prefix):
                route = r
                break
    if route is None:
        return "ERROR: unknown model"
    return await DISPATCH[route](prompt, semaphore)


async def run_batch(tasks, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(task):
        prompt = task["prompt"]
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
