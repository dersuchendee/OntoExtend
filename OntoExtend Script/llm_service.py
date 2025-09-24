from __future__ import annotations
import os
import yaml
import openai
import json

from dotenv import load_dotenv
from copy import deepcopy
from typing import Any, Dict, Optional
from abc import abstractmethod
from rich import print
from tiktoken import get_encoding

load_dotenv()


LLM_COSTS_FILE = os.getenv("LLM_COSTS_FILE", "llm_costs.yaml")
ENC = get_encoding("cl100k_base")


class LLMServiceTypes:
    """Enumeration for different types of embedding models."""
    LLM_SERVICE_TYPE_OPENAI = "openai"
    LLM_SERVICE_TYPE_CORPORATE = "corporate"


def load_llm_costs(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        print(f"Warning: LLM costs file not found at {file_path}")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

PRICES_PER_1M: Dict[str, Dict[str, Optional[float]]] = load_llm_costs(LLM_COSTS_FILE)
# print(json.dumps(PRICES_PER_1M, indent=2))


# ==========================
# Cost Tracking & Logging
# ==========================

class TokenCostTracker:
    def __init__(self) -> None:
        self.totals: Dict[str, Dict[str, int]] = {
            # model -> {prompt, completion, embedding}
        }

    def _ensure(self, model: str) -> None:
        if model not in self.totals:
            self.totals[model] = {"prompt": 0, "completion": 0, "embedding": 0}

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return deepcopy(self.totals)

    def diff(self, before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for m in set(before.keys()) | set(after.keys()):
            out[m] = {
                "prompt": after.get(m, {}).get("prompt", 0) - before.get(m, {}).get("prompt", 0),
                "completion": after.get(m, {}).get("completion", 0) - before.get(m, {}).get("completion", 0),
                "embedding": after.get(m, {}).get("embedding", 0) - before.get(m, {}).get("embedding", 0),
            }
        return out

    def note_chat(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._ensure(model)
        self.totals[model]["prompt"] += prompt_tokens
        self.totals[model]["completion"] += completion_tokens

    def note_embed(self, model: str, usage: Any, fallback_text: Optional[str] = None) -> None:
        self._ensure(model)
        if usage and getattr(usage, "total_tokens", None) is not None:
            self.totals[model]["embedding"] += int(usage.total_tokens)
        elif fallback_text is not None:
            self.totals[model]["embedding"] += len(ENC.encode(fallback_text))

    def model_cost_usd(self, model: str, counts: Dict[str, int]) -> float:
        p = PRICES_PER_1M.get(model, {})
        cin = (counts.get("prompt", 0) + counts.get("embedding", 0)) / 1000000.0 * (p.get("input") or 0.0)
        cout = counts.get("completion", 0) / 1000000.0 * (p.get("output") or 0.0)
        return cin + cout

    def totals_cost_usd(self) -> float:
        return sum(self.model_cost_usd(m, c) for m, c in self.totals.items())
    

class LLMService:
    def __init__(self, model: str, temperature: float, tracker: TokenCostTracker) -> None:
        self.model = model
        self.temperature = temperature
        self.tracker = tracker

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str) -> str:
        pass
        
        
class OpenAILLMService(LLMService):
    def __init__(self, model: str, temperature: float, tracker: TokenCostTracker) -> None:
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", None),
        )
        super().__init__(model=model, temperature=temperature, tracker=tracker)

    async def complete(self, prompt: str, system_prompt: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            )
            token_usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
            self.tracker.note_chat(self.model, prompt_tokens, completion_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[red]LLM Error: {e}[/red]")
            raise
        
                
try: 

    from helpers.azure_openai import init_chat_model
            
    class CorporateLLMService(LLMService):
        def __init__(self, model: str, temperature: float, tracker: TokenCostTracker) -> None:
            self.llm_model = init_chat_model(model, temperature=temperature)
            super().__init__(model=model, temperature=temperature, tracker=tracker)

        async def complete(self, prompt: str, system_prompt: str) -> str:
            try:
                resp = await self.llm_model.ainvoke(
                    input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                )
                token_usage = getattr(resp, "usage_metadata", None)
                prompt_tokens = int(token_usage['input_tokens']) if 'input_tokens' in token_usage else 0
                completion_tokens = int(token_usage['output_tokens']) if 'output_tokens' in token_usage else 0
                self.tracker.note_chat(self.model, prompt_tokens, completion_tokens)            
                return resp.content or ""
            except Exception as e:
                print(f"[red]LLM Error: {e}[/red]")
                raise

    corporate_llm_service_available = True

except Exception as e:
    corporate_llm_service_available = False


def get_llm_service(llm_service: str, model: str, temperature: float, tracker: TokenCostTracker) -> LLMService:
    if llm_service == LLMServiceTypes.LLM_SERVICE_TYPE_OPENAI:
        return OpenAILLMService(model, temperature, tracker)
    elif corporate_llm_service_available and llm_service == LLMServiceTypes.LLM_SERVICE_TYPE_CORPORATE:
        return CorporateLLMService(model, temperature, tracker)
    else:
        raise ValueError(f"Unsupported LLM service: {llm_service}. Supported services are 'openai' and 'corporate'.")