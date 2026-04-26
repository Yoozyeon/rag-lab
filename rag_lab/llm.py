
import json
import hashlib
from pathlib import Path
from openai import OpenAI
from .config import OPENAI_API_KEY, LLM_MODEL, CACHE_DIR

SOGANG_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
SOGANG_MODEL = "gpt-5.4-mini"

def _get_client() -> OpenAI:
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=SOGANG_BASE_URL
    )

def llm_call(
    prompt: str,
    system: str = "",
    model: str = SOGANG_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    if use_cache:
        cache_key = hashlib.md5(
            f"{model}:{system}:{prompt}".encode()
        ).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["response"]

    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = resp.choices[0].message.content

    if use_cache:
        cache_file.write_text(json.dumps({"response": result}))

    return result

def llm_call_json(
    prompt: str,
    system: str = "",
    model: str = SOGANG_MODEL,
    max_tokens: int = 2000,
) -> dict | list:
    raw = llm_call(prompt, system, model, max_tokens)
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return json.loads(text)
