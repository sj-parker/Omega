# LLM Interface - Abstract wrapper for any LLM backend

from abc import ABC, abstractmethod
from typing import Optional
import os


class LLMInterface(ABC):
    """Abstract interface for LLM calls."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
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
        max_tokens: int = 1024
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
        max_tokens: int = 1024
    ) -> str:
        if self._current_mode == "fast":
            self.fast_calls += 1
            response = await self.fast_llm.generate(prompt, system_prompt, temperature, max_tokens)
        else:
            self.main_calls += 1
            response = await self.main_llm.generate(prompt, system_prompt, temperature, max_tokens)
        return self._filter_response(response)
    
    async def generate_fast(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Explicitly use fast LLM."""
        self.fast_calls += 1
        response = await self.fast_llm.generate(prompt, system_prompt, temperature, max_tokens)
        return self._filter_response(response)
    
    async def generate_main(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Explicitly use main LLM (for quality-critical tasks)."""
        self.main_calls += 1
        response = await self.main_llm.generate(prompt, system_prompt, temperature, max_tokens)
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
        max_tokens: int = 1024
    ) -> str:
        import aiohttp
        
        # Try /api/chat first (newer format)
        try:
            return await self._chat_api(prompt, system_prompt, temperature, max_tokens)
        except Exception as e:
            if "400" in str(e) or "404" in str(e):
                # Fallback to /api/generate (older format)
                return await self._generate_api(prompt, system_prompt, temperature, max_tokens)
            raise
    
    async def _chat_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        import aiohttp
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
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
        max_tokens: int
    ) -> str:
        import aiohttp
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
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
        max_tokens: int = 1024
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
        temperature: float = 0.0,  # Zero temp for precise tool calls
        max_tokens: int = 512
    ) -> str:
        """Generate a tool call (or text response)."""
        import aiohttp
        
        # Qwen prefers explicit system prompts
        if not system_prompt:
            system_prompt = "You are a helpful assistant that outputs only JSON."
            
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
                "stop": ["<|endoftext|>", "<|im_end|>"]
            },
            "format": "json"  # Native JSON mode for Qwen
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Use /api/generate instead of /api/chat for better raw control
                async with session.post(f"{self.base_url}/api/generate", json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response", "")
                    else:
                        error_text = await resp.text()
                        print(f"Error calling {self.model}: {error_text}")
                        return "{}"
            except Exception as e:
                print(f"Connection error to {self.model}: {e}")
                return "{}"
        

    
    async def call_tool(
        self,
        task_description: str,
        tools: list[dict],
        context: str = ""
    ) -> Optional[dict]:
        """
        High-level method: given a task, return the appropriate tool call.
        Returns dict with 'tool' and 'arguments', or None if no tool needed.
        """
        # ═══════════════════════════════════════════════════════════════
        # FIX 1: Filter invalid "None" queries from experts
        # ═══════════════════════════════════════════════════════════════
        if not task_description or task_description.lower().strip() in ["none", "none.", "n/a", ""]:
            print(f"[FunctionGemma] BLOCKED: Invalid task description '{task_description}'")
            return None
        
        # ═══════════════════════════════════════════════════════════════
        # FIX 2: Direct math detection - bypass LLM entirely for simple math
        # ═══════════════════════════════════════════════════════════════
        import re
        math_match = re.search(r"(\d+(?:\.\d+)?)\s*[\*\×x]\s*(\d+(?:\.\d+)?)", task_description)
        if math_match:
            # Simple multiplication - return calculate result directly
            a, b = float(math_match.group(1)), float(math_match.group(2))
            result = a * b
            print(f"[FunctionGemma] MATH BYPASS: {a} × {b} = {result}")
            # Return a pseudo-observation so experts can use it
            return {"tool": "direct_calculation", "arguments": {"result": result, "expression": f"{a} × {b}"}}
        
        # ═══════════════════════════════════════════════════════════════
        # FIX 3: Block impossible calculation requests
        # ═══════════════════════════════════════════════════════════════
        IMPOSSIBLE_CALC_PATTERNS = [
            r"probability", r"вероятност", r"ймовірност",
            r"correlation", r"корреляц", r"кореляц",
            r"analyze", r"аналіз", r"анализ",
            r"assess", r"оценить", r"оцінити"
        ]
        task_lower = task_description.lower()
        if any(re.search(p, task_lower) for p in IMPOSSIBLE_CALC_PATTERNS):
            if "calculate" in task_lower or "compute" in task_lower or "вычисл" in task_lower:
                print(f"[FunctionGemma] BLOCKED: Impossible calculation request (probability/correlation/analyze)")
                return None
        
        # Cleanup: Experts improperly use snake_case
        task_description = task_description.replace("_", " ")
        
        # FIX: Strip "search" prefix that experts often add
        import re
        task_description = re.sub(r"^search\s+", "", task_description, flags=re.IGNORECASE)
        task_description = re.sub(r"^find\s+", "", task_description, flags=re.IGNORECASE)
        
        # STRICT TOOL FILTERING (Code Surgery)
        # We explicitly restricting the tools available to the LLM based on keywords.
        # This prevents "search for calculation" errors.
        
        filtered_tools = tools # Default: all tools
        forced_instruction = "" # Initialize variable
        
        key_task = task_description.lower()
        
        
        # 1. Calculation Intent (Check FIRST to avoid "what" masking it)
        if any(w in key_task for w in ["calculate", "compute", "drain", "charge", "linear", "rate"]):
             # Keep calculate AND search tools (Fallback mechanism for complex math/puzzles)
             filtered_tools = [t for t in tools if "calculate" in t['function']['name'] or "search" in t['function']['name']]
             forced_instruction = '\nCONSTRAINT: Use "calculate_*" tools ONLY for linear rates. For puzzles, probability, or complex math, use "search_and_extract".'
             print(f"[QwenTool] Strict Filter: Intent=CALCULATION. Tools: {[t['function']['name'] for t in filtered_tools]}")

        # 2. Search Intent (Only if NOT calculation)
        elif any(w in key_task for w in ["search", "find", "limit", "price", "weather", "verify", "check", "news", "current"]):
             search_tools = [t for t in tools if t.get('function', {}).get('name') in ['search_and_extract', 'verify_fact']]
             filtered_tools = search_tools
             forced_instruction = "CONSTRAINT: You MUST use 'search_and_extract' or 'verify_fact'. Do NOT calculate."
             print(f"[QwenTool] Strict Filter: Intent=SEARCH. Tools: {[t.get('function', {}).get('name') for t in filtered_tools]}")
             

        self.set_tools(filtered_tools)
        
        # Enhanced prompt for Qwen: Provide tools definition and ask for JSON
        tools_def = [t.get('function') for t in filtered_tools]
        import json
        
        prompt = f"""You are a tool calling assistant. Given a task, select the best tool and providing arguments in JSON format.

Task: {task_description}

Available Tools:
{json.dumps(tools_def, indent=2)}

{f"Context: {context}" if context else ""}

IMPORTANT RULES:
1. Return ONLY the JSON object.
2. For current data (weather, prices) use "search_and_extract".
3. For calculations use "calculate_linear_change" ONLY if applicable.
4. If no tool fits, return {{"tool": "none", "arguments": {{}}}}

{forced_instruction}

Respond with JSON ONLY:"""

        response = await self.generate(prompt, temperature=0.0)

        
        try:
            import json
            data = json.loads(response)
            
            # HALLUCINATION DETECTION: Check if query is relevant to task
            if data.get("tool") == "search_and_extract":
                query = data.get("arguments", {}).get("query", "").lower()
                task_lower = task_description.lower()
                
                # Known hallucination patterns (expanded)
                HALLUCINATION_PHRASES = [
                    "weather london", "weather berlin", "weather", "current weather", 
                    "check weather", "real-time weather", "погода",
                    "search_and_extract", "search for real-time", "search me for",
                    "apple stock", "stock price"
                ]
                
                # Check for explicit hallucination phrases
                is_known_hallucination = any(phrase in query for phrase in HALLUCINATION_PHRASES)
                
                # Check if task mentions these topics legitimately
                WEATHER_KEYWORDS = ["weather", "погода", "london", "лондон", "berlin", "берлін", "temperature", "температура"]
                STOCK_KEYWORDS = ["stock", "акці", "apple", "price", "ціна", "курс"]
                
                task_is_about_weather = any(w in task_lower for w in WEATHER_KEYWORDS)
                task_is_about_stocks = any(w in task_lower for w in STOCK_KEYWORDS)
                
                # IMPROVED Semantic relevance check
                # Extract meaningful words (>3 chars, not common stopwords)
                STOPWORDS = {"the", "and", "for", "that", "this", "with", "from", "search", "find", "about"}
                task_words = set(w for w in task_lower.split() if len(w) > 3 and w not in STOPWORDS)
                query_words = set(w for w in query.split() if len(w) > 3 and w not in STOPWORDS)
                shared_words = task_words & query_words
                
                # Must have NO shared words AND task must have enough context to compare
                is_semantically_irrelevant = len(shared_words) == 0 and len(task_words) > 3
                
                # FIX: Don't flag as hallucination if query is clearly derived from task
                query_in_task = query in task_lower or task_lower in query
                
                if (is_known_hallucination and not task_is_about_weather and not task_is_about_stocks) and not query_in_task:
                    print(f"[FunctionGemma] HALLUCINATION DETECTED: '{query[:50]}' unrelated to '{task_description[:50]}'")
                    # Fallback: use task description as query
                    return {"tool": "search_and_extract", "arguments": {"query": task_description}}
            
            # HEALING: If model returns raw args {"start":...} without "tool" wrapper
            if isinstance(data, dict) and "tool" not in data:
                 if len(filtered_tools) == 1:
                      guessed_tool = filtered_tools[0]['function']['name']
                      print(f"[FunctionGemma] JSON HEALING: Wrapping raw args for {guessed_tool}")
                      return {"tool": guessed_tool, "arguments": data}
            
            return data
        except:
            return None

