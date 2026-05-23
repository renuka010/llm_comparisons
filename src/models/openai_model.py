import json
import time
from src.config import cfg
from src.models.base import ModelClient, ModelResponse

# Models that only accept the default temperature (1) and reject custom values.
_FIXED_TEMP_PREFIXES = ("o1", "o2", "o3", "o4", "gpt-4.1", "gpt-4o-search", "gpt-5")

# Models that support reasoning_effort to control (or disable) chain-of-thought.
_REASONING_PREFIXES = ("o1", "o2", "o3", "o4", "gpt-5")

_MAX_TOOL_ROUNDS = 3


def _supports_temperature(model: str) -> bool:
    return not any(model.startswith(p) for p in _FIXED_TEMP_PREFIXES)


def _supports_reasoning_effort(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _base_kwargs(model: str, messages: list[dict]) -> dict:
    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_completion_tokens=cfg.FRONTIER_MAX_TOKENS,
    )
    if _supports_temperature(model):
        kwargs["temperature"] = cfg.FRONTIER_TEMPERATURE
    if _supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = "low"
    return kwargs


class OpenAIModelClient(ModelClient):
    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        self._model = cfg.OPENAI_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[dict]) -> ModelResponse:
        try:
            kwargs = _base_kwargs(self._model, messages)

            start = time.time()
            response = self._client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            choice = response.choices[0]
            message = choice.message

            if getattr(message, "refusal", None):
                return ModelResponse(
                    text="",
                    latency_ms=latency_ms,
                    error=f"Model refused: {message.refusal}",
                )

            text = message.content or ""
            finish_reason = getattr(choice, "finish_reason", "unknown")

            if not text:
                if finish_reason == "length":
                    return ModelResponse(
                        text="",
                        latency_ms=latency_ms,
                        error=(
                            f"Token budget exhausted before output was produced "
                            f"(max_completion_tokens={cfg.FRONTIER_MAX_TOKENS}). "
                            f"Increase FRONTIER_MAX_TOKENS in your .env file."
                        ),
                    )
                return ModelResponse(
                    text="",
                    latency_ms=latency_ms,
                    error=f"Empty response from model (finish_reason={finish_reason}). Check model name and API key.",
                )

            token_estimate = response.usage.total_tokens if response.usage else None
            return ModelResponse(text=text, latency_ms=latency_ms, token_estimate=token_estimate)

        except Exception as exc:
            return ModelResponse(text="", latency_ms=0, error=str(exc))

    def generate_with_tools(
        self,
        messages: list[dict],
        tool_map: dict,
    ) -> ModelResponse:
        """
        Agentic tool-call loop for the OpenAI API.
        Runs up to _MAX_TOOL_ROUNDS rounds of tool invocation before returning
        the final text response. tool_map is {function_name: callable(**kwargs)}.
        """
        from src.prompts import OPENAI_SEARCH_MEMORY_TOOL

        tools = [OPENAI_SEARCH_MEMORY_TOOL]
        working_messages = list(messages)
        total_latency = 0
        total_tokens = 0
        tool_calls_log: list[dict] = []

        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                kwargs = _base_kwargs(self._model, working_messages)
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

                start = time.time()
                response = self._client.chat.completions.create(**kwargs)
                total_latency += int((time.time() - start) * 1000)

                if response.usage:
                    total_tokens += response.usage.total_tokens

                choice = response.choices[0]
                message = choice.message

                if getattr(message, "refusal", None):
                    return ModelResponse(
                        text="",
                        latency_ms=total_latency,
                        error=f"Model refused: {message.refusal}",
                    )

                if choice.finish_reason == "tool_calls":
                    # Append the assistant turn that contains the tool_calls
                    working_messages.append(message.model_dump(exclude_none=True))

                    for tool_call in message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)

                        if fn_name in tool_map:
                            result = tool_map[fn_name](**fn_args)
                            if isinstance(result, list):
                                result_str = (
                                    "\n".join(f"- {r}" for r in result)
                                    if result
                                    else "No relevant memories found."
                                )
                            else:
                                result_str = str(result)
                        else:
                            result_str = f"Tool '{fn_name}' not available."

                        tool_calls_log.append(
                            {
                                "tool": fn_name,
                                "query": fn_args.get("query", ""),
                                "result_preview": result_str[:300],
                            }
                        )

                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_str,
                            }
                        )
                    continue

                text = message.content or ""
                return ModelResponse(
                    text=text,
                    latency_ms=total_latency,
                    token_estimate=total_tokens or None,
                    tool_calls_log=tool_calls_log,
                )

            return ModelResponse(
                text="",
                latency_ms=total_latency,
                error="Max tool-call rounds exceeded.",
                tool_calls_log=tool_calls_log,
            )

        except Exception as exc:
            return ModelResponse(text="", latency_ms=0, error=str(exc))
