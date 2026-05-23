import time
from src.config import cfg
from src.models.base import ModelClient, ModelResponse

_MAX_TOOL_ROUNDS = 3


class AnthropicModelClient(ModelClient):
    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self._model = cfg.ANTHROPIC_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def _split_system(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Separate the system message from chat messages (Anthropic API requirement)."""
        system_content = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)
        return system_content, chat_messages or messages

    def generate(self, messages: list[dict]) -> ModelResponse:
        try:
            system_content, chat_messages = self._split_system(messages)

            kwargs: dict = dict(
                model=self._model,
                max_tokens=cfg.FRONTIER_MAX_TOKENS,
                messages=chat_messages,
            )
            if system_content:
                kwargs["system"] = system_content

            start = time.time()
            response = self._client.messages.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            text = response.content[0].text if response.content else ""
            token_estimate = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else None
            )

            return ModelResponse(text=text, latency_ms=latency_ms, token_estimate=token_estimate)

        except Exception as exc:
            return ModelResponse(text="", latency_ms=0, error=str(exc))

    def generate_with_tools(
        self,
        messages: list[dict],
        tool_map: dict,
    ) -> ModelResponse:
        """
        Agentic tool-call loop for the Anthropic API.
        Runs up to _MAX_TOOL_ROUNDS rounds of tool invocation before returning
        the final text response. tool_map is {function_name: callable(**kwargs)}.
        """
        from src.prompts import ANTHROPIC_SEARCH_MEMORY_TOOL

        tools = [ANTHROPIC_SEARCH_MEMORY_TOOL]
        system_content, chat_messages = self._split_system(messages)
        total_latency = 0
        total_tokens = 0
        tool_calls_log: list[dict] = []

        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                kwargs: dict = dict(
                    model=self._model,
                    max_tokens=cfg.FRONTIER_MAX_TOKENS,
                    messages=chat_messages,
                    tools=tools,
                )
                if system_content:
                    kwargs["system"] = system_content

                start = time.time()
                response = self._client.messages.create(**kwargs)
                total_latency += int((time.time() - start) * 1000)

                if response.usage:
                    total_tokens += (
                        response.usage.input_tokens + response.usage.output_tokens
                    )

                if response.stop_reason == "tool_use":
                    # Append the full assistant turn (may mix text + tool_use blocks)
                    chat_messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            fn_name = block.name
                            fn_args = block.input

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
                                    "query": block.input.get("query", ""),
                                    "result_preview": result_str[:300],
                                }
                            )

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_str,
                                }
                            )

                    chat_messages.append({"role": "user", "content": tool_results})
                    continue

                text = "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )
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
