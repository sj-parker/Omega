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
    FunctionGemma - specialized tool-calling model.
    Uses native Ollama tools=[] format.
    """
    
    def __init__(self, model: str = "functiongemma", base_url: str = "http://localhost:11434"):
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
        temperature: float = 0.3,  # Lower temp for precise tool calls
        max_tokens: int = 512
    ) -> str:
        """Generate a tool call (or text response)."""
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
        
        # Add tools if defined
        if self._tools:
            payload["tools"] = self._tools
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    msg = data.get("message", {})
                    
                    # Check for tool calls
                    if msg.get("tool_calls"):
                        tool_call = msg["tool_calls"][0]
                        func = tool_call.get("function", {})
                        # Return as JSON string
                        import json
                        return json.dumps({
                            "tool": func.get("name"),
                            "arguments": func.get("arguments", {})
                        })
                    
                    return msg.get("content", "")
                else:
                    error_text = await resp.text()
                    raise Exception(f"FunctionGemma error {resp.status}: {error_text}")
    
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
        # Cleanup: Experts improperly use snake_case
        task_description = task_description.replace("_", " ")
        
        # STRICT TOOL FILTERING (Code Surgery)
        # We explicitly restricting the tools available to the LLM based on keywords.
        # This prevents "search for calculation" errors.
        
        filtered_tools = tools # Default: all tools
        forced_instruction = "" # Initialize variable
        
        key_task = task_description.lower()
        
        
        # 1. Search Intent (Prioritize over calc if ambiguous)
        if any(w in key_task for w in ["search", "find", "limit", "price", "weather", "who", "what", "where", "verify", "check"]):
             search_tools = [t for t in tools if t.get('function', {}).get('name') in ['search_and_extract', 'verify_fact']]
             filtered_tools = search_tools
             forced_instruction = "CONSTRAINT: You MUST use 'search_and_extract' or 'verify_fact'. Do NOT calculate."
             print(f"[FunctionGemma] Strict Filter: Locked to {[t.get('function', {}).get('name') for t in filtered_tools]}")

        # 2. Calculation Intent (Only if NOT search)
        elif any(w in key_task for w in ["calculate", "compute", "drain", "charge", "linear"]):
             # Keep calculate AND search tools (Fallback mechanism for complex math/puzzles)
             filtered_tools = [t for t in tools if "calculate" in t['function']['name'] or "search" in t['function']['name']]
             forced_instruction = '\nCONSTRAINT: Use "calculate_*" tools ONLY for linear rates. For puzzles, probability, or complex math, use "search_and_extract".'
             print(f"[FunctionGemma] Strict Filter: Locked to {[t['function']['name'] for t in filtered_tools]}")
             

        self.set_tools(filtered_tools)
        
        # Enhanced prompt with clear tool selection guidance
        prompt = f"""You are a tool dispatcher. Given a task, select the correct tool and provide arguments.

CRITICAL RULES:
1. For CURRENT/REAL-TIME data (prices, weather, news, exchange rates) -> ALWAYS use "search_and_extract"
2. For mathematical calculations (linear change, projections) -> use "calculate_linear_change"
3. For resource allocation -> use "calculate_resource_allocation"
4. For fact verification -> use "verify_fact"

Task: {task_description}

Available tools: {[t.get('function', {}).get('name') for t in filtered_tools]}

{f"Context: {context}" if context else ""}

IMPORTANT RULES FOR search_and_extract:
- The "query" should be a CLEAN, SIMPLE search phrase matching the task.
- "target" is optional (use if specific data needed).
- DO NOT use search_and_extract for calculations!

IMPORTANT RULES FOR calculate_linear_change:
- Use this tool if the task involves "calculate", "compute", "drain", "charge", "rate".
- Do NOT search for how to calculate. Use the tool directly.
- PRESERVE PRECISION: Do NOT round numbers. Use 83.4, not 83.
- PERCENTAGES: "2% per minute" means rate = -2 (minus 2 units), NOT -0.02.

EXAMPLES:
Task: "Check weather in London" -> {{"tool": "search_and_extract", "arguments": {{"query": "current weather London", "target": "weather"}}}}
Task: "Apple stock price" -> {{"tool": "search_and_extract", "arguments": {{"query": "current Apple stock price", "target": "price"}}}}
Task: "Calculate battery drain from 83.4% with -1.4 rate" -> {{"tool": "calculate_linear_change", "arguments": {{"start": 83.4, "rate": -1.4, "time": 10, "variable_name": "final_charge"}}}}
Task: "calculate_linear_change for battery..." -> {{"tool": "calculate_linear_change", ...}}

{forced_instruction}

Output ONLY valid JSON: {{"tool": "tool_name", "arguments": {{...}}}}"""

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
                
                # Semantic relevance check: query should share at least one meaningful word with task
                task_words = set(w for w in task_lower.split() if len(w) > 3)
                query_words = set(w for w in query.split() if len(w) > 3)
                shared_words = task_words & query_words
                
                is_semantically_irrelevant = len(shared_words) == 0 and len(task_words) > 2
                
                if (is_known_hallucination and not task_is_about_weather and not task_is_about_stocks) or is_semantically_irrelevant:
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

