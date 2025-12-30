# Task Queue with Priority System
# Central task management for the Omega system

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, Any, Callable, Awaitable, TYPE_CHECKING
from collections import defaultdict
import heapq

from models.schemas import ContextScope

if TYPE_CHECKING:
    from core.context_manager import ContextManager


class Priority(IntEnum):
    """Task priority levels. Lower number = higher priority."""
    CRITICAL = 0    # Security issues, system errors
    HIGH = 1        # User-facing responses
    MEDIUM = 2      # Background enrichment, tool calls
    LOW = 3         # Reflection, learning
    BACKGROUND = 4  # Analytics, cleanup, compaction


@dataclass
class Task:
    """A unit of work in the system."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""  # "user_query", "tool_call", "search", "expert_consult", etc.
    payload: Any = None
    priority: Priority = Priority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    
    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    
    # Source tracking
    source_module: str = ""
    target_module: str = ""
    
    # Execution state
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    heartbeat: Optional[datetime] = None
    
    # Retry logic
    max_retries: int = 3
    retry_count: int = 0
    
    # Context Slicing (NEW)
    context_scope: ContextScope = ContextScope.NONE
    context_filter: Optional[str] = None  # Semantic filter for RELEVANT scope
    injected_context: Optional[dict] = None  # Filled before execution
    
    def __lt__(self, other: 'Task') -> bool:
        """For heapq comparison - lower priority number wins, then older task."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority.name,
            "status": self.status,
            "source": self.source_module,
            "target": self.target_module,
            "context_scope": self.context_scope.value,
            "has_context": self.injected_context is not None,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    source_module: str = ""
    steps: list[dict] = field(default_factory=list)  # Captured trace steps
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "data": str(self.data)[:200] if self.data else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "steps": self.steps
        }


class TaskQueue:
    """
    Priority-based task queue with dependency tracking.
    
    Features:
    - Priority scheduling (CRITICAL > HIGH > MEDIUM > LOW > BACKGROUND)
    - Task dependencies (task B waits for task A)
    - Async processing
    - Task cancellation
    - Retry logic
    """
    
    def __init__(self, max_concurrent: int = 5):
        self._queue: list[Task] = []  # heapq
        self._tasks: dict[str, Task] = {}  # task_id -> Task
        self._dependents: dict[str, list[str]] = defaultdict(list)  # task_id -> dependent task_ids
        self._completed: dict[str, TaskResult] = {}  # task_id -> result
        self._handlers: dict[str, Callable[[Task], Awaitable[TaskResult]]] = {}
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._status_callbacks: list[Callable[[Task], None]] = []
        
    def register_handler(self, task_type: str, handler: Callable[[Task], Awaitable[TaskResult]]):
        """Register a handler for a specific task type."""
        self._handlers[task_type] = handler
        print(f"[TaskQueue] Registered handler for '{task_type}'")
    
    def inject_context(self, task: Task, context_manager: 'ContextManager'):
        """
        Inject appropriate context into task before execution.
        
        This is the core of Context Slicing - each task gets only
        the context it needs based on its context_scope.
        """
        task.injected_context = context_manager.get_scoped_context(
            scope=task.context_scope,
            semantic_filter=task.context_filter
        )
        if task.injected_context:
            print(f"[TaskQueue] Injected {task.context_scope.value} context into {task.task_id[:8]}")
        
    async def enqueue(self, task: Task) -> str:
        """Add a task to the queue. Returns task_id."""
        async with self._lock:
            self._tasks[task.task_id] = task
            
            # Check if dependencies are met
            if self._can_run(task):
                heapq.heappush(self._queue, task)
            
            # Track dependents
            for dep_id in task.depends_on:
                self._dependents[dep_id].append(task.task_id)
                
        async with self._condition:
            self._condition.notify()
            
        print(f"[TaskQueue] Enqueued {task.task_type} (priority={task.priority.name}, id={task.task_id[:8]})")
        return task.task_id
    
    def _can_run(self, task: Task) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in task.depends_on:
            if dep_id not in self._completed:
                return False
        return True
    
    async def dequeue(self, stop_event: Optional[asyncio.Event] = None) -> Optional[Task]:
        """Get the next task to process (blocks if queue is empty)."""
        async with self._condition:
            while not self._queue and self._running_count < self._max_concurrent:
                if stop_event and stop_event.is_set():
                    return None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    if stop_event and stop_event.is_set():
                        return None
                    continue
            
            if self._queue:
                task = heapq.heappop(self._queue)
                task.status = "running"
                self._running_count += 1
                return task
            return None
    
    async def complete(self, task_id: str, result: TaskResult):
        """Mark a task as completed and wake up dependents."""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = "completed" if result.success else "failed"
                self._tasks[task_id].result = result.data
                if result.error:
                    self._tasks[task_id].error = result.error
            
            self._completed[task_id] = result
            self._running_count -= 1
            
            # Wake up dependent tasks
            for dependent_id in self._dependents.get(task_id, []):
                dep_task = self._tasks.get(dependent_id)
                if dep_task and self._can_run(dep_task):
                    heapq.heappush(self._queue, dep_task)
                    
        async with self._condition:
            self._condition.notify_all()
            
    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == "pending":
                    task.status = "cancelled"
                    # Remove from queue
                    self._queue = [t for t in self._queue if t.task_id != task_id]
                    heapq.heapify(self._queue)
                    return True
        return False
    
    async def wait_for(self, task_id: str, timeout: float = 30.0) -> Optional[TaskResult]:
        """Wait for a specific task to complete."""
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            async with self._lock:
                if task_id in self._completed:
                    return self._completed[task_id]
            await asyncio.sleep(0.1)
            
        return None
    
    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return len(self._queue)
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        status_counts = defaultdict(int)
        for task in self._tasks.values():
            status_counts[task.status] += 1
            
        priority_counts = defaultdict(int)
        for task in self._queue:
            priority_counts[task.priority.name] += 1
            
        return {
            "total_tasks": len(self._tasks),
            "pending": len(self._queue),
            "running": self._running_count,
            "completed": len(self._completed),
            "by_status": dict(status_counts),
            "pending_by_priority": dict(priority_counts)
        }
    
    async def process_one(self, stop_event: Optional[asyncio.Event] = None) -> Optional[TaskResult]:
        """Process a single task from the queue."""
        task = await self.dequeue(stop_event)
        if not task:
            return None
            
        task.status = "running"
        task.started_at = datetime.now()
        task.heartbeat = datetime.now()
        self._notify_status_change(task)
        handler = self._handlers.get(task.task_type)
        from core.tracer import tracer
        
        # Capture steps even if the handler doesn't have its own capture_steps
        with tracer.capture_steps() as steps:
            if not handler:
                result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"No handler registered for task type '{task.task_type}'"
                )
            else:
                import time
                start = time.time()
                
                # Automatically inject results from dependencies into payload
                if task.depends_on:
                    dep_results = []
                    for dep_id in task.depends_on:
                        if dep_id in self._completed:
                            res = self._completed[dep_id]
                            if res.success:
                                dep_results.append(res.data)
                    
                    if isinstance(task.payload, dict):
                        task.payload["results"] = dep_results
                
                try:
                    result = await handler(task)
                    result.execution_time_ms = int((time.time() - start) * 1000)
                except Exception as e:
                    result = TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=str(e),
                        execution_time_ms=int((time.time() - start) * 1000),
                        steps=[s.to_dict() for s in steps] # Capture whatever we got before crash
                    )
                    
                    # Retry logic
                    task.retry_count += 1
                    if task.retry_count < task.max_retries:
                        task.status = "pending"
                        async with self._lock:
                            heapq.heappush(self._queue, task)
                            self._running_count -= 1
                        print(f"[TaskQueue] Retrying {task.task_id[:8]} ({task.retry_count}/{task.max_retries})")
                        return result
            
            # If capture_steps was used inside handler, it might have its own steps.
            # If not, we use the ones captured here.
            if not result.steps and steps:
                result.steps = [s.to_dict() for s in steps]
        
        await self.complete(task.task_id, result)
        return result
    
    async def run_worker(self, stop_event: asyncio.Event):
        """Background worker that processes tasks continuously."""
        while not stop_event.is_set():
            try:
                # Blocking call but responsive to stop_event
                result = await self.process_one(stop_event)
                if result:
                    status = "✓" if result.success else "✗"
                    print(f"[TaskQueue] {status} Task {result.task_id[:8]} completed in {result.execution_time_ms}ms")
            except Exception as e:
                if not stop_event.is_set():
                    print(f"[TaskQueue] Worker error: {e}")
                    await asyncio.sleep(0.5)

    async def run_watchdog(self, stop_event: asyncio.Event, timeout_seconds: float = 60.0):
        """Monitor for stuck tasks and handle timeouts."""
        print(f"[TaskWatchdog] Started (timeout={timeout_seconds}s)")
        while not stop_event.is_set():
            try:
                await asyncio.sleep(5.0) # Check every 5 seconds
                now = datetime.now()
                stuck_tasks = []
                
                async with self._lock:
                    for task_id, task in self._tasks.items():
                        if task.status == "running" and task.started_at:
                            elapsed = (now - task.started_at).total_seconds()
                            if elapsed > timeout_seconds:
                                stuck_tasks.append(task)
                
                for task in stuck_tasks:
                    print(f"[TaskWatchdog] Detected stuck task: {task.task_id[:8]} ({task.task_type})")
                    result = TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=f"Task timed out after {timeout_seconds}s",
                        source_module="watchdog"
                    )
                    await self.complete(task.task_id, result)
                    
            except Exception as e:
                print(f"[TaskWatchdog] Error: {e}")

    def on_status_change(self, callback: Callable[[Task], None]):
        """Register a callback for task status updates."""
        self._status_callbacks.append(callback)

    def _notify_status_change(self, task: Task):
        """Notify all callbacks about status change."""
        for cb in self._status_callbacks:
            try:
                cb(task)
            except:
                pass
