"""Transport adapter for the official DeepSeek Chat Completions endpoint."""
from contextvars import ContextVar
import os

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com"


def install(llm_utils, client=None):
    from litellm import ModelResponse
    from openai import OpenAI

    if client is None:
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=ENDPOINT, max_retries=2)
    original_format = llm_utils.to_litellm_messages
    original_completion = llm_utils.completion
    history = ContextVar("deepseek_history")

    def format_messages(messages):
        formatted = original_format(messages)
        assert len(formatted) == len(messages)
        history.set((formatted, messages))
        return formatted

    def completion(*args, model, messages, **kwargs):
        if model != MODEL:
            return original_completion(*args, model=model, messages=messages, **kwargs)
        formatted, source = history.get()
        assert formatted is messages
        request = []
        for message, saved in zip(messages, source):
            item = dict(message)
            if item["role"] == "assistant":
                response = getattr(saved, "raw_data", None) or {}
                previous = (response.get("choices") or [{}])[0].get("message") or {}
                if "reasoning_content" in previous:
                    item["reasoning_content"] = previous["reasoning_content"]
                if item.get("tool_calls"):
                    item["tool_calls"] = [
                        {key: call[key] for key in ("id", "type", "function")} for call in item["tool_calls"]
                    ]
                else:
                    item.pop("tool_calls", None)
            request.append(item)
        # LiteLLM's retry count is a client option, never part of the API payload.
        kwargs.pop("num_retries", None)
        answer = client.chat.completions.with_raw_response.create(model=model, messages=request, **kwargs)
        return ModelResponse(**answer.http_response.json())

    llm_utils.to_litellm_messages = format_messages
    llm_utils.completion = completion
