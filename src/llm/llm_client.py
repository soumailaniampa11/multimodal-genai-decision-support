import os

import requests
from dotenv import load_dotenv


class GeminiClient:
    """
    Text completion through the Google Gemini API.
    """

    def __init__(self, model_name: str):
        from google import genai

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def complete(self, prompt: str, json_output: bool = False) -> str:

        response = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text


class OllamaClient:
    """
    Text completion with a local open-weight model served by Ollama
    (e.g. Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct from Hugging Face).

    Runs offline, without API quota, and with deterministic decoding
    for reproducible experiments.
    """

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        context_window: int = 8192,
        seed: int = 42,
        timeout: int = 600,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.options = {
            "temperature": temperature,
            "num_ctx": context_window,
            "seed": seed,
        }
        self.timeout = timeout

    def complete(self, prompt: str, json_output: bool = False) -> str:

        body = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
        }

        if json_output:
            body["format"] = "json"

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=body,
                timeout=self.timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. "
                "Start it with: ollama serve"
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{self.model_name}' is not available in Ollama. "
                f"Download it with: ollama pull {self.model_name}"
            )

        response.raise_for_status()

        return response.json()["response"]


class HuggingFaceClient:
    """
    Text completion with an open-weight Hugging Face model through
    Hugging Face Inference Providers (OpenAI-compatible API).

    No local download is needed. Requires HF_TOKEN in .env.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://router.huggingface.co/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        timeout: int = 120,
    ):
        load_dotenv()

        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN is not configured. Create a token at "
                "https://huggingface.co/settings/tokens and add it to .env."
            )

        self.model_name = model_name
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        self.timeout = timeout

    def complete(self, prompt: str, json_output: bool = False) -> str:

        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.parameters,
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=body,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Hugging Face API error {response.status_code} "
                f"for model '{self.model_name}': {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]


def create_llm_client(
    provider: str,
    model_name: str,
    ollama: dict | None = None,
    huggingface: dict | None = None,
):
    """
    Build the LLM client selected in the configuration.

    Args:
        provider: "gemini", "ollama" or "huggingface".
        model_name: model identifier for the chosen provider.
        ollama: optional Ollama settings (host, temperature, ...).
        huggingface: optional Hugging Face settings (base_url, ...).
    """

    if provider == "gemini":
        return GeminiClient(model_name)

    if provider == "ollama":
        return OllamaClient(model_name, **(ollama or {}))

    if provider == "huggingface":
        return HuggingFaceClient(model_name, **(huggingface or {}))

    raise ValueError(
        f"Unknown LLM provider: {provider}. "
        "Use 'gemini', 'ollama' or 'huggingface'."
    )
