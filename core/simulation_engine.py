# Simulation Engine
# Deterministic computation module - code does the math, not LLM
# Supports: FSM (state machines), Math (linear/percent), Physics

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import re


# ============================================================================
# BASE CLASSES
# ============================================================================

class SimulationType(Enum):
    """Type of simulation."""
    FSM = "fsm"           # Finite State Machine (robots, resources, queues)
    MATH = "math"         # Linear equations, percentages
    PHYSICS = "physics"   # Physical simulations
    LOGIC = "logic"       # Logical reasoning


@dataclass
class SimulationStep:
    """A single step in a simulation."""
    time: int  # Minutes from start
    state: str
    values: Dict[str, Any]
    event: str = ""
    
    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "state": self.state,
            "values": self.values,
            "event": self.event
        }


@dataclass
class SimulationResult:
    """Result of a simulation."""
    success: bool
    final_values: Dict[str, Any]
    steps: List[SimulationStep] = field(default_factory=list)
    answer_text: str = ""
    error: str = ""
    
    def get_trace(self, max_steps: int = 10) -> str:
        """Get human-readable trace of key steps."""
        if not self.steps:
            return "No steps recorded."
        
        # Show first few and last few steps
        lines = ["**Simulation Trace:**"]
        steps_to_show = self.steps[:3] + self.steps[-3:] if len(self.steps) > 6 else self.steps
        
        for step in steps_to_show:
            time_str = f"t={step.time}min"
            values_str = ", ".join(f"{k}={v}" for k, v in step.values.items())
            lines.append(f"  {time_str}: [{step.state}] {values_str} {step.event}")
        
        if len(self.steps) > 6:
            lines.insert(4, f"  ... ({len(self.steps) - 6} steps omitted) ...")
        
        return "\n".join(lines)


# ============================================================================
# FSM SIMULATOR - For state-based problems (robots, resources, queues)
# ============================================================================

class EntityState(Enum):
    """Common states for entities."""
    IDLE = "idle"
    CHARGING = "charging"
    WORKING = "working"
    WAITING = "waiting"
    LOW_BATTERY = "low_battery"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Entity:
    """An entity in the simulation (robot, vehicle, etc)."""
    name: str
    charge: float  # Current charge %
    state: EntityState = EntityState.IDLE
    priority: int = 0  # Higher = more priority
    
    # Rates
    charge_rate: float = 1.0   # % per minute when charging
    work_rate: float = 1.0     # % per minute when working
    
    # Constraints
    min_charge_for_work: float = 0.0  # Minimum charge to start working
    shutdown_threshold: float = 5.0   # Shutdown if charge drops below
    
    # Task tracking
    current_task: Optional[str] = None
    work_remaining: int = 0  # Minutes of work left
    
    def can_work(self) -> bool:
        """Check if entity can start working."""
        return self.charge >= self.min_charge_for_work and self.charge > self.shutdown_threshold
    
    def needs_charge_for_task(self, task_min_charge: float) -> bool:
        """Check if entity needs to charge before task."""
        return self.charge < task_min_charge


@dataclass
class Task:
    """A task to be completed."""
    name: str
    duration: int  # Minutes
    min_charge: float  # Minimum charge required to start
    priority: str = "normal"  # critical, urgent, normal
    deadline: Optional[int] = None  # Minutes from start (e.g., 60 = 1 hour)
    
    def priority_value(self) -> int:
        return {"critical": 3, "urgent": 2, "normal": 1}.get(self.priority.lower(), 0)


@dataclass
class Resource:
    """A shared resource (charging port, etc)."""
    name: str
    capacity: int  # How many entities can use at once
    current_users: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        return len(self.current_users) < self.capacity
    
    def allocate(self, entity_name: str) -> bool:
        if self.is_available():
            self.current_users.append(entity_name)
            return True
        return False
    
    def release(self, entity_name: str):
        if entity_name in self.current_users:
            self.current_users.remove(entity_name)


class FSMSimulator:
    """
    Finite State Machine Simulator.
    
    Simulates state-based problems minute by minute:
    - Robot charging/working scenarios
    - Resource allocation (charging ports, queues)
    - Priority-based scheduling
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.tasks: List[Task] = []
        self.resources: Dict[str, Resource] = {}
        self.time: int = 0  # Current time in minutes
        self.start_time: datetime = datetime.now().replace(second=0, microsecond=0)
        self.steps: List[SimulationStep] = []
        self.max_simulation_time: int = 1440  # Max 24 hours
    
    def add_entity(self, entity: Entity):
        """Add an entity to the simulation."""
        self.entities[entity.name] = entity
    
    def add_task(self, task: Task):
        """Add a task to the simulation."""
        self.tasks.append(task)
        # Sort by priority
        self.tasks.sort(key=lambda t: -t.priority_value())
    
    def add_resource(self, resource: Resource):
        """Add a resource to the simulation."""
        self.resources[resource.name] = resource
    
    def set_start_time(self, hour: int, minute: int = 0):
        """Set simulation start time."""
        self.start_time = self.start_time.replace(hour=hour, minute=minute)
    
    def get_current_time_str(self) -> str:
        """Get current simulation time as HH:MM."""
        current = self.start_time + timedelta(minutes=self.time)
        return current.strftime("%H:%M")
    
    def _record_step(self, event: str = ""):
        """Record current state as a step."""
        values = {}
        for name, entity in self.entities.items():
            values[f"{name}_charge"] = round(entity.charge, 1)
            values[f"{name}_state"] = entity.state.value
        
        self.steps.append(SimulationStep(
            time=self.time,
            state=self.get_current_time_str(),
            values=values,
            event=event
        ))
    
    def simulate_single_entity_task(
        self,
        entity_name: str,
        task: Task,
        charging_port: Optional[str] = None
    ) -> SimulationResult:
        """
        Simulate a single entity completing a task.
        
        Returns when task is complete or entity fails.
        """
        if entity_name not in self.entities:
            return SimulationResult(success=False, final_values={}, error=f"Entity {entity_name} not found")
        
        entity = self.entities[entity_name]
        port = self.resources.get(charging_port) if charging_port else None
        
        self._record_step(f"Start: {entity.name} needs to do '{task.name}'")
        
        # Phase 1: Charge if needed
        if entity.needs_charge_for_task(task.min_charge):
            if port and not port.is_available():
                return SimulationResult(
                    success=False,
                    final_values={"charge": entity.charge},
                    steps=self.steps,
                    error=f"Charging port {charging_port} not available"
                )
            
            if port:
                port.allocate(entity.name)
            
            entity.state = EntityState.CHARGING
            charge_needed = task.min_charge - entity.charge
            charge_time = int(charge_needed / entity.charge_rate)
            
            self._record_step(f"Charging: need {charge_needed}% ({charge_time} min)")
            
            for _ in range(charge_time):
                entity.charge += entity.charge_rate
                self.time += 1
                if self.time > self.max_simulation_time:
                    return SimulationResult(success=False, final_values={}, error="Simulation timeout")
            
            entity.charge = min(100.0, entity.charge)
            
            if port:
                port.release(entity.name)
            
            self._record_step(f"Charged to {entity.charge}%")
        
        # Phase 2: Work
        entity.state = EntityState.WORKING
        entity.current_task = task.name
        entity.work_remaining = task.duration
        
        self._record_step(f"Working: '{task.name}' for {task.duration} min")
        
        for _ in range(task.duration):
            entity.charge -= entity.work_rate
            entity.work_remaining -= 1
            self.time += 1
            
            # Check shutdown threshold
            if entity.charge <= entity.shutdown_threshold:
                entity.state = EntityState.LOW_BATTERY
                self._record_step(f"LOW BATTERY SHUTDOWN at {entity.charge}%!")
                return SimulationResult(
                    success=False,
                    final_values={
                        "charge": round(entity.charge, 1),
                        "time": self.get_current_time_str(),
                        "work_remaining": entity.work_remaining
                    },
                    steps=self.steps,
                    error=f"Low battery shutdown at {round(entity.charge, 1)}%",
                    answer_text=f"FAILED: Robot shut down at {self.get_current_time_str()} with {round(entity.charge, 1)}% charge. {entity.work_remaining} min of work remaining."
                )
            
            if self.time > self.max_simulation_time:
                return SimulationResult(success=False, final_values={}, error="Simulation timeout")
        
        # Task complete
        entity.state = EntityState.COMPLETED
        entity.current_task = None
        
        end_time = self.get_current_time_str()
        final_charge = round(entity.charge, 1)
        
        self._record_step(f"COMPLETED at {end_time} with {final_charge}% charge")
        
        # Check deadline
        deadline_status = ""
        if task.deadline:
            if self.time > task.deadline:
                late_by = self.time - task.deadline
                deadline_status = f"LATE by {late_by} minutes!"
            else:
                early_by = task.deadline - self.time
                deadline_status = f"On time ({early_by} min early)"
        
        return SimulationResult(
            success=True,
            final_values={
                "charge": final_charge,
                "time": end_time,
                "total_minutes": self.time,
                "deadline_status": deadline_status
            },
            steps=self.steps,
            answer_text=f"Task '{task.name}' completed at {end_time}. Remaining charge: {final_charge}%. {deadline_status}"
        )
    
    def calculate_completion_time(
        self,
        entity_name: str,
        task: Task
    ) -> SimulationResult:
        """
        Calculate when an entity will complete a task.
        
        Formula:
        - Charge time = (min_charge - current_charge) / charge_rate
        - Total time = charge_time + task_duration
        - End time = start_time + total_time
        """
        if entity_name not in self.entities:
            return SimulationResult(success=False, final_values={}, error=f"Entity {entity_name} not found")
        
        entity = self.entities[entity_name]
        
        # Calculate charge time
        if entity.charge < task.min_charge:
            charge_needed = task.min_charge - entity.charge
            charge_time = int(charge_needed / entity.charge_rate)
        else:
            charge_time = 0
        
        # Total time
        total_time = charge_time + task.duration
        
        # End time
        end_datetime = self.start_time + timedelta(minutes=total_time)
        end_time_str = end_datetime.strftime("%H:%M")
        
        # Final charge (after working)
        charge_after_charging = min(100.0, entity.charge + charge_time * entity.charge_rate)
        final_charge = charge_after_charging - task.duration * entity.work_rate
        
        # Deadline check
        deadline_status = ""
        late_by = 0
        if task.deadline:
            if total_time > task.deadline:
                late_by = total_time - task.deadline
                deadline_status = f"ОПОЗДАНИЕ на {late_by} минут"
            else:
                deadline_status = f"Успеваем (запас {task.deadline - total_time} мин)"
        
        return SimulationResult(
            success=True,
            final_values={
                "charge_time": charge_time,
                "work_time": task.duration,
                "total_time": total_time,
                "end_time": end_time_str,
                "final_charge": round(final_charge, 1),
                "late_by": late_by
            },
            steps=[],
            answer_text=f"""**Расчёт для {entity_name}:**
- Текущий заряд: {entity.charge}%
- Нужен заряд: {task.min_charge}%
- Время зарядки: {charge_time} мин (скорость +{entity.charge_rate}%/мин)
- Время работы: {task.duration} мин
- **Общее время: {total_time} мин**
- **Завершение: {end_time_str}**
- Заряд после работы: {round(final_charge, 1)}%
{f"- {deadline_status}" if deadline_status else ""}"""
        )


# ============================================================================
# MATH SOLVER - For linear/percentage calculations
# ============================================================================

class MathSolver:
    """
    Deterministic math solver.
    
    Handles:
    - Linear changes (start + rate * time)
    - Percentages
    - Proportions
    """
    
    @staticmethod
    def linear_change(start: float, rate: float, time: float) -> float:
        """Calculate linear change: result = start + rate * time"""
        return start + rate * time
    
    @staticmethod
    def time_to_reach(start: float, target: float, rate: float) -> float:
        """Calculate time to reach target: time = (target - start) / rate"""
        if rate == 0:
            return float('inf') if start != target else 0
        return (target - start) / rate
    
    @staticmethod
    def percentage_of(value: float, percent: float) -> float:
        """Calculate percentage of a value."""
        return value * (percent / 100)
    
    @staticmethod
    def apply_discount(price: float, discount_percent: float) -> float:
        """Apply discount to price."""
        return price * (1 - discount_percent / 100)
    
    @staticmethod
    def apply_multiplier(price: float, multiplier: float) -> float:
        """Apply price multiplier (e.g., peak hours)."""
        return price * multiplier


# ============================================================================
# SIMULATION ENGINE - Main entry point
# ============================================================================

class SimulationEngine:
    """
    Main Simulation Engine.
    
    Analyzes problems and routes to appropriate simulator.
    Code does the math - not LLM.
    """
    
    def __init__(self):
        self.fsm = FSMSimulator()
        self.math = MathSolver()
        self._stats = {
            "simulations_run": 0,
            "by_type": {}
        }
    
    def reset_fsm(self):
        """Reset FSM simulator for new problem."""
        self.fsm = FSMSimulator()
    
    def detect_simulation_type(self, query: str) -> SimulationType:
        """Detect what type of simulation is needed."""
        query_lower = query.lower()
        
        # FSM patterns (robots, charging, queues)
        fsm_patterns = [
            r"робот|robot",
            r"заряд.*%|charge.*%",
            r"idle|charging|working",
            r"порт|port|slot",
            r"очередь|queue|priority",
            r"задач.*длительность|task.*duration",
        ]
        
        for pattern in fsm_patterns:
            if re.search(pattern, query_lower):
                return SimulationType.FSM
        
        # Math patterns
        math_patterns = [
            r"\d+\s*[+\-*/×÷]\s*\d+",
            r"процент|percent|%.*от|of",
            r"скидк|discount",
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, query_lower):
                return SimulationType.MATH
        
        # Physics patterns
        physics_patterns = [
            r"вакуум|vacuum",
            r"давлен|pressure",
            r"гравитац|gravity|падени|fall",
            r"температур|temperature",
        ]
        
        for pattern in physics_patterns:
            if re.search(pattern, query_lower):
                return SimulationType.PHYSICS
        
        return SimulationType.LOGIC
    
    def parse_robot_scenario(self, query: str) -> Optional[Dict]:
        """
        Parse robot scenario from natural language.
        
        Extracts:
        - Robots (name, charge %)
        - Tasks (name, duration, min_charge)
        - Resources (ports)
        - Time constraints
        """
        result = {
            "entities": [],
            "tasks": [],
            "resources": [],
            "start_time": None,
            "question": ""
        }
        
        # Extract start time
        time_match = re.search(r"(?:старт|start|сейчас|now)[:\s]*(\d{1,2}):(\d{2})", query, re.I)
        if time_match:
            result["start_time"] = (int(time_match.group(1)), int(time_match.group(2)))
        else:
            # Try other formats
            time_match = re.search(r"(\d{1,2}):(\d{2})", query)
            if time_match:
                result["start_time"] = (int(time_match.group(1)), int(time_match.group(2)))
        
        # Extract robots
        robot_patterns = [
            r"(?:робот|robot)[\s\-]*([А-Яа-яA-Za-z])[:.\s]*(\d+)%?\s*(?:заряд|charge)?",
            r"(?:робот|robot)[\s\-]*([А-Яа-яA-Za-z]).*?(\d+)\s*%",
        ]
        
        for pattern in robot_patterns:
            for match in re.finditer(pattern, query, re.I):
                name = f"Robot-{match.group(1).upper()}"
                charge = float(match.group(2))
                if not any(e["name"] == name for e in result["entities"]):
                    result["entities"].append({"name": name, "charge": charge})
        
        # Extract tasks
        task_pattern = r"(?:task|задача)\s*[\"']?([^\"'(]+)[\"']?\s*\(([^)]+)\)"
        for match in re.finditer(task_pattern, query, re.I):
            task_name = match.group(1).strip()
            details = match.group(2)
            
            # Parse duration
            duration_match = re.search(r"(\d+)\s*(?:мин|min)", details, re.I)
            duration = int(duration_match.group(1)) if duration_match else 0
            
            # Parse min charge
            charge_match = re.search(r"(?:мин|min).*?(\d+)%|(\d+)%.*(?:заряд|charge)", details, re.I)
            min_charge = float(charge_match.group(1) or charge_match.group(2)) if charge_match else 0
            
            # Parse priority
            priority = "normal"
            if "critical" in details.lower():
                priority = "critical"
            elif "urgent" in details.lower():
                priority = "urgent"
            
            result["tasks"].append({
                "name": task_name,
                "duration": duration,
                "min_charge": min_charge,
                "priority": priority
            })
        
        # Extract ports
        port_match = re.search(r"(\d+)\s*(?:свободн|free|available)?.*?(?:порт|port)", query, re.I)
        if port_match:
            result["resources"].append({
                "name": "charging_port",
                "capacity": int(port_match.group(1))
            })
        
        # Extract rates
        charge_rate_match = re.search(r"(?:заряд|charge).*?([+-]?\d+(?:\.\d+)?)\s*%\s*/?\s*(?:мин|min)", query, re.I)
        work_rate_match = re.search(r"(?:работ|work|расход).*?([+-]?\d+(?:\.\d+)?)\s*%\s*/?\s*(?:мин|min)", query, re.I)
        
        result["charge_rate"] = float(charge_rate_match.group(1)) if charge_rate_match else 1.0
        result["work_rate"] = abs(float(work_rate_match.group(1))) if work_rate_match else 1.0
        
        # Extract deadline
        deadline_match = re.search(r"(?:до|before|deadline)\s*(\d{1,2}):(\d{2})", query, re.I)
        if deadline_match and result["start_time"]:
            deadline_hour = int(deadline_match.group(1))
            deadline_min = int(deadline_match.group(2))
            start_h, start_m = result["start_time"]
            result["deadline_minutes"] = (deadline_hour - start_h) * 60 + (deadline_min - start_m)
        
        # Extract shutdown threshold
        shutdown_match = re.search(r"(?:отключ|shutdown|выключ).*?(\d+)\s*%", query, re.I)
        result["shutdown_threshold"] = float(shutdown_match.group(1)) if shutdown_match else 5.0
        
        return result
    
    def run_robot_simulation(self, scenario: Dict) -> SimulationResult:
        """Run simulation based on parsed scenario."""
        self.reset_fsm()
        
        # Set start time
        if scenario.get("start_time"):
            self.fsm.set_start_time(*scenario["start_time"])
        
        # Add entities
        for e in scenario.get("entities", []):
            entity = Entity(
                name=e["name"],
                charge=e["charge"],
                charge_rate=scenario.get("charge_rate", 1.0),
                work_rate=scenario.get("work_rate", 1.0),
                shutdown_threshold=scenario.get("shutdown_threshold", 5.0)
            )
            self.fsm.add_entity(entity)
        
        # Add resources
        for r in scenario.get("resources", []):
            self.fsm.add_resource(Resource(name=r["name"], capacity=r["capacity"]))
        
        # Add tasks
        for t in scenario.get("tasks", []):
            task = Task(
                name=t["name"],
                duration=t["duration"],
                min_charge=t["min_charge"],
                priority=t.get("priority", "normal"),
                deadline=scenario.get("deadline_minutes")
            )
            self.fsm.add_task(task)
        
        # Run simulation for first entity and first task
        if scenario["entities"] and scenario["tasks"]:
            entity_name = scenario["entities"][0]["name"]
            task = self.fsm.tasks[0]
            
            return self.fsm.calculate_completion_time(entity_name, task)
        
        return SimulationResult(success=False, final_values={}, error="No entities or tasks to simulate")
    
    def get_stats(self) -> dict:
        return self._stats
