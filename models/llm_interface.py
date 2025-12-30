# LLM Interface - Abstract wrapper for any LLM backend

import asyncio
import os
import json
from abc import ABC, abstractmethod
from typing import Optional, List


class LLMInterface(ABC):
    """Abstract interface for LLM calls."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate a response from the LLM."""
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing without actual API calls."""
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        # Simple mock response based on prompt
        if "expert" in (system_prompt or "").lower():
            return f"[Mock Expert Response] Analyzing: {prompt[:50]}..."
        if "critic" in (system_prompt or "").lower():
            return "[Mock Critic] No major inconsistencies detected."
        return f"[Mock Response] I understand your query about: {prompt[:50]}..."


class LLMRouter(LLMInterface):
    """
    Router that uses different LLMs based on task type.
    
    - FastLLM: for fast_path simple responses
    - MainLLM: for experts, critic, reflection (quality-critical)
    
    Homeostasis measures quality from MainLLM only.
    All responses are filtered to remove LLM self-identification.
    """
    
    def __init__(
        self,
        fast_llm: LLMInterface,
        main_llm: LLMInterface,
        filter_identity: bool = True
    ):
        self.fast_llm = fast_llm
        self.main_llm = main_llm
        self._current_mode = "main"
        self.filter_identity = filter_identity
        
        # Identity filter (lazy import to avoid circular deps)
        self._identity_filter = None
        
        # Stats for monitoring
        self.fast_calls = 0
        self.main_calls = 0
    
    def _get_filter(self):
        """Lazy load identity filter."""
        if self._identity_filter is None and self.filter_identity:
            try:
                from core.identity_filter import IdentityFilter
                self._identity_filter = IdentityFilter()
            except ImportError:
                self._identity_filter = False  # Mark as unavailable
        return self._identity_filter
    
    def _filter_response(self, response: str) -> str:
        """Filter identity mentions from response."""
        filt = self._get_filter()
        if filt and filt is not False:
            return filt.filter_response(response)
        return response
    
    def set_mode(self, mode: str):
        """Set current mode: 'fast' or 'main'."""
        self._current_mode = mode
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        if self._current_mode == "fast":
            self.fast_calls += 1
            response = await self.fast_llm.generate(prompt, system_prompt, temperature, max_tokens, stop)
        else:
            self.main_calls += 1
            response = await self.main_llm.generate(prompt, system_prompt, temperature, max_tokens, stop)
        return self._filter_response(response)
    
    async def generate_fast(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None
    ) -> str:
        """Explicitly use fast LLM."""
        self.fast_calls += 1
        response = await self.fast_llm.generate(prompt, system_prompt, temperature, max_tokens, stop)
        return self._filter_response(response)
    
    async def generate_main(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        """Explicitly use main LLM (for quality-critical tasks)."""
        self.main_calls += 1
        response = await self.main_llm.generate(prompt, system_prompt, temperature, max_tokens, stop)
        return self._filter_response(response)
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        total = self.fast_calls + self.main_calls
        return {
            "fast_calls": self.fast_calls,
            "main_calls": self.main_calls,
            "fast_ratio": self.fast_calls / total if total > 0 else 0
        }


class OllamaLLM(LLMInterface):
    """Ollama backend (local LLM)."""
    
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        import aiohttp
        
        # Try /api/chat first (newer format)
        try:
            return await self._chat_api(prompt, system_prompt, temperature, max_tokens, stop)
        except Exception as e:
            if "400" in str(e) or "404" in str(e):
                # Fallback to /api/generate (older format)
                return await self._generate_api(prompt, system_prompt, temperature, max_tokens, stop)
            raise
    
    async def _chat_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> str:
        import aiohttp
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        if stop:
            options["stop"] = stop
            
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "")
                else:
                    error_text = await resp.text()
                    raise Exception(f"Ollama chat error {resp.status}: {error_text[:200]}")
    
    async def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]]
    ) -> str:
        import aiohttp
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        if stop:
            options["stop"] = stop
            
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": options
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "")
                else:
                    error_text = await resp.text()
                    raise Exception(f"Ollama generate error {resp.status}: {error_text[:200]}")


class OpenAILLM(LLMInterface):
    """OpenAI API backend."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        import aiohttp
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if stop:
            payload["stop"] = stop
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenAI error: {resp.status}")


class FunctionGemmaLLM(LLMInterface):
    """
    ToolCaller - specialized tool-calling model.
    Previously FunctionGemma, now using Qwen2.5:7b for better reasoning.
    """
    
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._tools = []
    
    def set_tools(self, tools: list[dict]):
        """Set available tools in Ollama format."""
        self._tools = tools
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> str:
        import aiohttp
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens
        }
        if stop:
            options["stop"] = stop
            
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "")
                else:
                    error_text = await resp.text()
                    raise Exception(f"Ollama chat error {resp.status}: {error_text[:200]}")

    async def call_tool(self, prompt: str, tools: list[dict] = None) -> Optional[dict]:
        """
        Force the model to return a tool call in JSON format.
        """
        actual_tools = tools if tools is not None else self._tools
        
        system_msg = """You are a function caller. 
Given a user request and a list of tools, you MUST return ONLY a JSON object matching the ToolCall schema.
Do NOT explain. Do NOT say anything else.

Schema:
{
  "tool": "name_of_tool",
  "arguments": { ... }
}
"""
        tools_str = json.dumps(actual_tools, indent=2, ensure_ascii=False)
        full_prompt = f"Available Tools:\n{tools_str}\n\nTask: {prompt}"
        
        raw_json = await self.generate(full_prompt, system_msg, temperature=0.0)
        
        try:
            # Clean up potential markdown formatting
            clean_json = raw_json.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            
            return json.loads(clean_json.strip())
        except:
            print(f"[ToolCaller] Failed to parse JSON: {raw_json}")
            return None
