# Orchestrator
# Central dispatcher for the modular Omega system

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, List, Callable, Awaitable, Protocol, TYPE_CHECKING
from enum import Enum
import time

if TYPE_CHECKING:
    from core.task_queue import Task, TaskResult, Priority


class ModuleCapability(Enum):
    """Capabilities that modules can provide."""
    SEARCH = "search"
    CALCULATE = "calculate"
    REASON = "reason"
    REMEMBER = "remember"
    VALIDATE = "validate"
    GENERATE = "generate"
    CLASSIFY = "classify"


@dataclass
class ModuleInterface:
    """
    Contract that defines what a module can do.
    Used for module registration and hot-swapping.
    """
    name: str
    input_types: list[str]      # Task types this module accepts
    output_types: list[str]     # Result types this module produces
    capabilities: list[ModuleCapability]
    fallback_module: Optional[str] = None  # Module to use if this one fails
    priority: int = 50          # Higher = preferred when multiple modules match
    is_async: bool = True       # Whether the module's process is async
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "capabilities": [c.value for c in self.capabilities],
            "fallback_module": self.fallback_module,
            "priority": self.priority
        }


class ModuleProtocol(Protocol):
    """Protocol that all Omega modules should implement."""
    
    async def process(self, *args, **kwargs) -> Any:
        """Process a request."""
        ...
    
    def get_interface(self) -> ModuleInterface:
        """Return module's interface definition."""
        ...


@dataclass
class ModuleHealth:
    """Health status of a module."""
    name: str
    is_healthy: bool
    last_call_time_ms: float = 0
    total_calls: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    avg_response_time_ms: float = 0
    
    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.error_count / self.total_calls


@dataclass
class DispatchResult:
    """Result of dispatching a task to a module."""
    module_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    used_fallback: bool = False
    
    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "success": self.success,
            "data": str(self.data)[:200] if self.data else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "used_fallback": self.used_fallback
        }


class Orchestrator:
    """
    Orchestrator - Central dispatcher for the modular Omega system.
    
    Features:
    - Module registration with defined interfaces
    - Hot-swap module replacement without restart
    - Automatic fallback on module failure
    - Health monitoring
    - Capability-based routing
    
    This enables true modularity: modules can be replaced without breaking the system.
    """
    
    def __init__(self, default_timeout: float = 30.0):
        # Module registry: name -> (module_instance, interface)
        self._modules: Dict[str, tuple[Any, ModuleInterface]] = {}
        
        # Capability index: capability -> [module_names]
        self._capability_index: Dict[ModuleCapability, List[str]] = {}
        
        # Input type index: input_type -> [module_names]
        self._input_type_index: Dict[str, List[str]] = {}
        
        # Health tracking
        self._health: Dict[str, ModuleHealth] = {}
        
        # Configuration
        self.default_timeout = default_timeout
        
        # Stats
        self._stats = {
            "total_dispatches": 0,
            "successful_dispatches": 0,
            "fallback_used": 0,
            "by_module": {}
        }
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def register_module(
        self,
        name: str,
        module: Any,
        interface: ModuleInterface
    ) -> bool:
        """
        Register a module with its interface.
        
        Args:
            name: Unique module name
            module: The module instance
            interface: Module's interface contract
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            # Ensure interface name matches
            if interface.name != name:
                interface.name = name
            
            self._modules[name] = (module, interface)
            
            # Update capability index
            for cap in interface.capabilities:
                if cap not in self._capability_index:
                    self._capability_index[cap] = []
                if name not in self._capability_index[cap]:
                    self._capability_index[cap].append(name)
            
            # Update input type index
            for input_type in interface.input_types:
                if input_type not in self._input_type_index:
                    self._input_type_index[input_type] = []
                if name not in self._input_type_index[input_type]:
                    self._input_type_index[input_type].append(name)
            
            # Initialize health tracking
            self._health[name] = ModuleHealth(name=name, is_healthy=True)
            
            print(f"[Orchestrator] Registered module '{name}' with capabilities: {[c.value for c in interface.capabilities]}")
            return True
    
    async def replace_module(
        self,
        name: str,
        new_module: Any,
        new_interface: Optional[ModuleInterface] = None
    ) -> bool:
        """
        Hot-swap a module with a new implementation.
        
        Args:
            name: Name of the module to replace
            new_module: New module instance
            new_interface: Optional new interface (uses old one if not provided)
            
        Returns:
            True if replacement successful
        """
        async with self._lock:
            if name not in self._modules:
                print(f"[Orchestrator] Cannot replace '{name}': not registered")
                return False
            
            _, old_interface = self._modules[name]
            interface = new_interface or old_interface
            
            # Preserve health stats
            old_health = self._health.get(name)
            
            # Replace
            self._modules[name] = (new_module, interface)
            
            # Reset health but keep historical stats
            if old_health:
                self._health[name] = ModuleHealth(
                    name=name,
                    is_healthy=True,
                    total_calls=old_health.total_calls
                )
            
            print(f"[Orchestrator] Hot-swapped module '{name}'")
            return True
    
    async def unregister_module(self, name: str) -> bool:
        """Remove a module from the registry."""
        async with self._lock:
            if name not in self._modules:
                return False
            
            _, interface = self._modules[name]
            
            # Remove from capability index
            for cap in interface.capabilities:
                if cap in self._capability_index and name in self._capability_index[cap]:
                    self._capability_index[cap].remove(name)
            
            # Remove from input type index
            for input_type in interface.input_types:
                if input_type in self._input_type_index and name in self._input_type_index[input_type]:
                    self._input_type_index[input_type].remove(name)
            
            del self._modules[name]
            del self._health[name]
            
            print(f"[Orchestrator] Unregistered module '{name}'")
            return True
    
    async def dispatch(
        self,
        task_type: str,
        payload: Any,
        preferred_module: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> DispatchResult:
        """
        Dispatch a task to the appropriate module.
        
        Args:
            task_type: Type of task (e.g., "search", "calculate", "classify")
            payload: Task payload
            preferred_module: Optional specific module to use
            timeout: Optional timeout override
            
        Returns:
            DispatchResult with the module's response
        """
        self._stats["total_dispatches"] += 1
        timeout = timeout or self.default_timeout
        
        # Find appropriate module
        module_name = preferred_module
        if not module_name or module_name not in self._modules:
            module_name = self._find_module_for_task(task_type)
        
        if not module_name:
            return DispatchResult(
                module_name="none",
                success=False,
                error=f"No module registered for task type '{task_type}'"
            )
        
        # Dispatch to module
        result = await self._dispatch_to_module(module_name, payload, timeout)
        
        # If failed, try fallback
        if not result.success:
            _, interface = self._modules.get(module_name, (None, None))
            if interface and interface.fallback_module:
                fallback_name = interface.fallback_module
                if fallback_name in self._modules:
                    print(f"[Orchestrator] Using fallback '{fallback_name}' for failed '{module_name}'")
                    result = await self._dispatch_to_module(fallback_name, payload, timeout)
                    result.used_fallback = True
                    self._stats["fallback_used"] += 1
        
        if result.success:
            self._stats["successful_dispatches"] += 1
        
        # Update per-module stats
        if module_name not in self._stats["by_module"]:
            self._stats["by_module"][module_name] = {"calls": 0, "errors": 0}
        self._stats["by_module"][module_name]["calls"] += 1
        if not result.success:
            self._stats["by_module"][module_name]["errors"] += 1
        
        return result
    
    async def dispatch_by_capability(
        self,
        capability: ModuleCapability,
        payload: Any,
        timeout: Optional[float] = None
    ) -> DispatchResult:
        """
        Dispatch a task to a module with a specific capability.
        
        Selects the best module based on priority and health.
        """
        module_name = self._find_module_by_capability(capability)
        if not module_name:
            return DispatchResult(
                module_name="none",
                success=False,
                error=f"No module with capability '{capability.value}'"
            )
        
        return await self._dispatch_to_module(module_name, payload, timeout or self.default_timeout)
    
    def _find_module_for_task(self, task_type: str) -> Optional[str]:
        """Find the best module for a task type."""
        candidates = self._input_type_index.get(task_type, [])
        if not candidates:
            return None
        
        # Sort by priority (higher first) and health
        def score(name: str) -> tuple:
            _, interface = self._modules[name]
            health = self._health.get(name)
            is_healthy = health.is_healthy if health else True
            return (-1 if not is_healthy else 0, -interface.priority)
        
        candidates.sort(key=score)
        return candidates[0]
    
    def _find_module_by_capability(self, capability: ModuleCapability) -> Optional[str]:
        """Find the best module with a specific capability."""
        candidates = self._capability_index.get(capability, [])
        if not candidates:
            return None
        
        # Sort by priority and health
        def score(name: str) -> tuple:
            _, interface = self._modules[name]
            health = self._health.get(name)
            is_healthy = health.is_healthy if health else True
            return (-1 if not is_healthy else 0, -interface.priority)
        
        candidates.sort(key=score)
        return candidates[0]
    
    async def _dispatch_to_module(
        self,
        module_name: str,
        payload: Any,
        timeout: float
    ) -> DispatchResult:
        """Actually dispatch to a module."""
        if module_name not in self._modules:
            return DispatchResult(
                module_name=module_name,
                success=False,
                error=f"Module '{module_name}' not found"
            )
        
        module, interface = self._modules[module_name]
        health = self._health[module_name]
        
        start_time = time.time()
        
        try:
            # Call the module's process method
            if interface.is_async:
                result = await asyncio.wait_for(
                    module.process(payload) if hasattr(module, 'process') 
                    else module(payload),
                    timeout=timeout
                )
            else:
                # Run sync module in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: module.process(payload) if hasattr(module, 'process') else module(payload)
                    ),
                    timeout=timeout
                )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Update health
            health.total_calls += 1
            health.last_call_time_ms = elapsed_ms
            health.avg_response_time_ms = (
                health.avg_response_time_ms * 0.9 + elapsed_ms * 0.1
            )
            health.is_healthy = True
            
            return DispatchResult(
                module_name=module_name,
                success=True,
                data=result,
                execution_time_ms=elapsed_ms
            )
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            health.total_calls += 1
            health.error_count += 1
            health.last_error = "Timeout"
            health.is_healthy = health.error_rate < 0.5
            
            return DispatchResult(
                module_name=module_name,
                success=False,
                error=f"Timeout after {timeout}s",
                execution_time_ms=elapsed_ms
            )
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            health.total_calls += 1
            health.error_count += 1
            health.last_error = str(e)
            health.is_healthy = health.error_rate < 0.5
            
            return DispatchResult(
                module_name=module_name,
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms
            )
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get a module instance by name."""
        if name in self._modules:
            module, _ = self._modules[name]
            return module
        return None
    
    def get_module_interface(self, name: str) -> Optional[ModuleInterface]:
        """Get a module's interface by name."""
        if name in self._modules:
            _, interface = self._modules[name]
            return interface
        return None
    
    def get_module_health(self, name: Optional[str] = None) -> Dict[str, ModuleHealth]:
        """Get health status of one or all modules."""
        if name:
            return {name: self._health.get(name)} if name in self._health else {}
        return dict(self._health)
    
    def get_all_modules(self) -> List[str]:
        """Get list of all registered module names."""
        return list(self._modules.keys())
    
    def get_modules_by_capability(self, capability: ModuleCapability) -> List[str]:
        """Get modules that have a specific capability."""
        return self._capability_index.get(capability, [])
    
    def get_stats(self) -> dict:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "registered_modules": len(self._modules),
            "capabilities_registered": len(self._capability_index),
            "healthy_modules": sum(1 for h in self._health.values() if h.is_healthy),
            "success_rate": (
                self._stats["successful_dispatches"] / max(1, self._stats["total_dispatches"])
            )
        }
    
    def health_report(self) -> dict:
        """Generate a comprehensive health report."""
        modules_report = {}
        for name, health in self._health.items():
            _, interface = self._modules[name]
            modules_report[name] = {
                "is_healthy": health.is_healthy,
                "total_calls": health.total_calls,
                "error_rate": f"{health.error_rate:.1%}",
                "avg_response_ms": f"{health.avg_response_time_ms:.0f}",
                "last_error": health.last_error,
                "capabilities": [c.value for c in interface.capabilities]
            }
        
        return {
            "status": "healthy" if all(h.is_healthy for h in self._health.values()) else "degraded",
            "total_modules": len(self._modules),
            "healthy_count": sum(1 for h in self._health.values() if h.is_healthy),
            "modules": modules_report
        }
