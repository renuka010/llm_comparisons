from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    text: str
    latency_ms: int
    token_estimate: int | None = None
    error: str | None = None
    # Each entry: {"tool": str, "query": str, "result_preview": str}
    tool_calls_log: list[dict] = field(default_factory=list)


class ModelClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict]) -> ModelResponse:
        ...

    def generate_with_tools(
        self,
        messages: list[dict],
        tool_map: dict,
    ) -> ModelResponse:
        """
        Default: ignore tools and fall back to plain generate.
        Frontier subclasses override this to run a proper tool-call loop.
        """
        return self.generate(messages)

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
