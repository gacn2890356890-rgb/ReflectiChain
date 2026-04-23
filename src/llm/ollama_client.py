"""
Ollama LLM Client: OpenAI-compatible wrapper for local Ollama server.
Compatible with the existing llm.chat.completions.create() interface used throughout the codebase.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)


class OllamaChatCompletion:
    """
    OpenAI-compatible Chat Completions interface backed by Ollama.

    Usage:
        from src.llm.ollama_client import OllamaChatCompletion
        client = OllamaChatCompletion(base_url="http://localhost:11434", model="qwen2.5")
        client.chat.completions.create(model="qwen2.5", messages=[...], temperature=0.7)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5",
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    models = [m.get("name", "") for m in data.get("models", [])]
                    logger.info(f"Ollama available. Models: {models}")
                    return True
        except Exception as e:
            logger.warning(f"Ollama server not available at {self.base_url}: {e}")
        return False

    @property
    def chat(self):
        """Chat completions interface — returns self so .chat.create() works."""
        return self

    @property
    def completions(self):
        """Bridge to support client.chat.completions.create() interface."""
        return _CompletionsBridge(self)

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "ChatCompletionResponse":
        """
        Create a chat completion. Mimics OpenAI.ChatCompletion.create() signature.

        Args:
            model: Model name (e.g. "qwen2.5", "llama3", "mistral")
            messages: List of {"role": "user"|"system"|"assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            stop: Stop sequences

        Returns:
            ChatCompletionResponse (compatible with OpenAI response format)
        """
        if not self._available:
            raise RuntimeError(
                f"Ollama server not available at {self.base_url}. "
                "Please ensure Ollama is running: `ollama serve`"
            )

        effective_model = model or self.model

        # Build Ollama request payload
        ollama_payload = {
            "model": effective_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            },
            "stream": False,
        }
        if stop:
            ollama_payload["options"]["stop"] = stop

        # Make HTTP request to Ollama
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(ollama_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Ollama returned status {resp.status}")
                    raw = json.loads(resp.read())

                    return ChatCompletionResponse.from_ollama(raw, model=effective_model)

            except urllib.error.URLError as e:
                last_error = e
                logger.warning(f"Ollama request failed (attempt {attempt + 1}): {e}")
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Ollama response parse failed (attempt {attempt + 1}): {e}")

        raise RuntimeError(
            f"Ollama request failed after {self.max_retries} attempts: {last_error}"
        )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs,
    ) -> str:
        """Simple text generation (used by AdversarialPolicyLLM)."""
        response = self.create(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content


class _CompletionsBridge:
    """Bridge to expose .chat.completions.create() interface on OllamaChatCompletion."""

    def __init__(self, client: "OllamaChatCompletion"):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> "ChatCompletionResponse":
        """Delegate to OllamaChatCompletion.create()."""
        return self._client.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            stop=stop,
            **kwargs,
        )


class ChatMessage:
    """A single chat message, compatible with OpenAI format."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatCompletionChoice:
    """A single completion choice, compatible with OpenAI format."""

    def __init__(self, message: ChatMessage, index: int = 0, finish_reason: str = "stop"):
        self.message = message
        self.index = index
        self.finish_reason = finish_reason


class ChatCompletionUsage:
    """Token usage statistics, compatible with OpenAI format."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class ChatCompletionResponse:
    """
    Full chat completion response, compatible with OpenAI format.
    Used as: response.choices[0].message.content
    """

    def __init__(
        self,
        model: str,
        choices: List[ChatCompletionChoice],
        usage: ChatCompletionUsage,
    ):
        self.model = model
        self.choices = choices
        self.usage = usage
        self.object = "chat.completion"
        self.id = f"ollama-{id(self)}"
        self.created = 0

    @classmethod
    def from_ollama(cls, raw: Dict[str, Any], model: str) -> "ChatCompletionResponse":
        """Parse Ollama's /api/chat response into OpenAI-compatible format."""
        raw_message = raw.get("message", {})
        content = raw_message.get("content", "")

        message = ChatMessage(
            role=raw_message.get("role", "assistant"),
            content=content,
        )

        finish_reason = "stop"
        if raw.get("done_reason"):
            finish_reason = raw["done_reason"]

        choices = [ChatCompletionChoice(message, index=0, finish_reason=finish_reason)]

        # Estimate tokens (Ollama doesn't always return this)
        prompt_tokens = raw.get("prompt_eval_count", 0)
        completion_tokens = raw.get("eval_count", len(content.split()))

        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return cls(model=model, choices=choices, usage=usage)


def create_ollama_client(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[OllamaChatCompletion]:
    """
    Factory function to create an Ollama client.
    Returns None if Ollama is not available, so the codebase gracefully falls back.
    """
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5")

    try:
        client = OllamaChatCompletion(base_url=base_url, model=model)
        if client._available:
            logger.info(f"Ollama client ready: {base_url} / {model}")
            return client
        else:
            logger.warning(
                f"Ollama not available at {base_url}. "
                "Install models with: `ollama pull qwen2.5`"
            )
            return None
    except Exception as e:
        logger.warning(f"Failed to create Ollama client: {e}")
        return None


def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List all models available on the Ollama server."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []
