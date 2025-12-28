# Task Queue with Priority System
# Central task management for the Omega system

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional, Any, Callable, Awaitable
from collections import defaultdict
import heapq


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
    
    # Retry logic
    max_retries: int = 3
    retry_count: int = 0
    
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
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "data": str(self.data)[:200] if self.data else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
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
        
    def register_handler(self, task_type: str, handler: Callable[[Task], Awaitable[TaskResult]]):
        """Register a handler for a specific task type."""
        self._handlers[task_type] = handler
        print(f"[TaskQueue] Registered handler for '{task_type}'")
        
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
    
    async def dequeue(self) -> Optional[Task]:
        """Get the next task to process (blocks if queue is empty)."""
        async with self._condition:
            while not self._queue and self._running_count < self._max_concurrent:
                await self._condition.wait()
            
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
    
    async def process_one(self) -> Optional[TaskResult]:
        """Process a single task from the queue."""
        task = await self.dequeue()
        if not task:
            return None
            
        handler = self._handlers.get(task.task_type)
        if not handler:
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=f"No handler registered for task type '{task.task_type}'"
            )
        else:
            import time
            start = time.time()
            try:
                result = await handler(task)
                result.execution_time_ms = int((time.time() - start) * 1000)
            except Exception as e:
                result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    execution_time_ms=int((time.time() - start) * 1000)
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
        
        await self.complete(task.task_id, result)
        return result
    
    async def run_worker(self, stop_event: asyncio.Event):
        """Background worker that processes tasks continuously."""
        while not stop_event.is_set():
            try:
                result = await asyncio.wait_for(self.process_one(), timeout=1.0)
                if result:
                    status = "✓" if result.success else "✗"
                    print(f"[TaskQueue] {status} Task {result.task_id[:8]} completed in {result.execution_time_ms}ms")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[TaskQueue] Worker error: {e}")
