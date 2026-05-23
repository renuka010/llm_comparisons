import time
import streamlit as st
from src.config import cfg
from src.models.base import ModelClient, ModelResponse


@st.cache_resource(show_spinner="Loading OSS model…")
def _load_model(model_id: str, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    return tokenizer, model


class OSSModelClient(ModelClient):
    def __init__(self) -> None:
        self._model_id = cfg.OSS_MODEL_ID
        self._device = cfg.OSS_DEVICE
        self._max_new_tokens = cfg.OSS_MAX_NEW_TOKENS
        self._temperature = cfg.OSS_TEMPERATURE

    @property
    def model_name(self) -> str:
        return self._model_id

    def generate(self, messages: list[dict]) -> ModelResponse:
        import torch

        try:
            tokenizer, model = _load_model(self._model_id, self._device)

            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    do_sample=self._temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency_ms = int((time.time() - start) * 1000)

            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            token_estimate = len(new_tokens)

            return ModelResponse(text=text, latency_ms=latency_ms, token_estimate=token_estimate)

        except Exception as exc:
            return ModelResponse(text="", latency_ms=0, error=str(exc))
