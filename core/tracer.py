# Global Tracer for Cognitive System
# Captures steps of request processing for visualization

import contextvars
import contextlib
from datetime import datetime
from typing import Optional, Any
import uuid

from models.schemas import TraceStep

# Context-safe storage for the current active trace
_active_steps = contextvars.ContextVar("active_steps", default=None)
_active_episode_id = contextvars.ContextVar("active_episode_id", default=None)

class Tracer:
    """
    Thread-safe tracer using context variables.
    
    Usage:
    async with tracer.session():
        tracer.add_step("Module", "Action", "Description")
        ...
    """
    
    @staticmethod
    def start_session(episode_id: Optional[str] = None):
        """Start a new tracing session for the current context."""
        _active_steps.set([])
        _active_episode_id.set(episode_id or str(uuid.uuid4()))
        return _active_episode_id.get()

    @staticmethod
    def add_step(module: str, name: str, description: str, data_in: Any = None, data_out: Any = None, timestamp: Optional[datetime] = None):
        """Add a step to the current context's trace."""
        steps = _active_steps.get()
        if steps is not None:
            step = TraceStep(
                module=module,
                name=name,
                description=description,
                data_in=data_in,
                data_out=data_out,
                timestamp=timestamp or datetime.now()
            )
            steps.append(step)
            # print(f"[Tracer] Step added: {module}.{name}")
        else:
             # If no session, do nothing or log a warning
             pass

    @staticmethod
    def get_steps() -> list[TraceStep]:
        """Get all steps for the current context."""
        return _active_steps.get() or []

    @staticmethod
    def get_episode_id() -> Optional[str]:
        return _active_episode_id.get()

    @staticmethod
    def end_session():
        """Clear the current context's trace."""
        steps = _active_steps.get()
        _active_steps.set(None)
        _active_episode_id.set(None)
        return steps

    @staticmethod
    @contextlib.contextmanager
    def capture_steps():
        """
        Context manager to capture steps in a local scope.
        Yields the list of steps captured during the block.
        """
        old_steps = _active_steps.get()
        local_steps = []
        _active_steps.set(local_steps)
        try:
            yield local_steps
        finally:
            _active_steps.set(old_steps)

# Global singleton-like access
tracer = Tracer()
