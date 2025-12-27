# Homeostasis Controller
# Maintains system stability through policy calibration

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import PolicySpace, PolicyUpdate, HomeostasisMetrics


class HomeostasisController:
    """
    Homeostasis Controller.
    
    Purpose: Maintain system stability.
    
    Controlled parameters:
    - Average response confidence
    - Expert disagreement
    - Decision cost
    - Repeat question rate
    - Risk level
    
    Principle:
    - Parameters have acceptable ranges (wide initially)
    - On deviation, generate policy corrections
    
    This is not learning, this is calibration.
    Homeostasis is about stability, not precision.
    """
    
    def __init__(self, policy: PolicySpace, storage_path: Optional[Path] = None):
        self.policy = policy
        self.update_history: list[PolicyUpdate] = []
        self.storage_path = storage_path or Path("./learning_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Try to load saved policy on startup
        self.load_policy()
        
        # Target ranges (safe, not "correct")
        self.targets = {
            "confidence": {"min": 0.5, "max": 0.85, "current": 0.7},
            "expert_disagreement": {"min": 0.0, "max": 0.4, "current": 0.2},
            "cost_ms": {"min": 100, "max": 2000, "current": 500},
            "success_rate": {"min": 0.6, "max": 1.0, "current": 0.8}
        }
    
    def analyze(self, metrics: dict) -> Optional[PolicyUpdate]:
        """
        Analyze current metrics and generate policy update if needed.
        
        Returns None if system is within acceptable ranges.
        """
        updates = {}
        reasons = []
        
        # Check confidence
        avg_conf = metrics.get("avg_confidence", 0.7)
        if avg_conf < self.targets["confidence"]["min"]:
            # Confidence too low → be more cautious, use more experts
            updates["expert_call_threshold"] = 0.05  # Lower threshold
            reasons.append(f"Low confidence ({avg_conf:.2f})")
        elif avg_conf > self.targets["confidence"]["max"]:
            # Confidence too high → might be overconfident
            updates["expert_call_threshold"] = -0.05  # Raise threshold
            reasons.append(f"High confidence ({avg_conf:.2f})")
        
        # Check cost
        avg_cost = metrics.get("avg_cost_ms", 500)
        if avg_cost > self.targets["cost_ms"]["max"]:
            # Too expensive → prefer faster paths
            updates["fast_path_bias"] = 0.05
            reasons.append(f"High cost ({avg_cost:.0f}ms)")
        elif avg_cost < self.targets["cost_ms"]["min"]:
            # Very fast → might be too shallow
            updates["fast_path_bias"] = -0.03
            reasons.append(f"Low cost ({avg_cost:.0f}ms)")
        
        # Check success rate
        success_rate = metrics.get("success_rate", 0.8)
        if success_rate < self.targets["success_rate"]["min"]:
            # Low success → be more careful
            updates["expert_call_threshold"] = updates.get("expert_call_threshold", 0) + 0.05
            updates["memory_write_threshold"] = -0.05  # Write more to memory
            reasons.append(f"Low success rate ({success_rate:.2f})")
        
        # Check expert usage
        expert_rate = metrics.get("expert_usage_rate", 0.3)
        if expert_rate > 0.7:
            # Using experts too much
            updates["expert_call_threshold"] = updates.get("expert_call_threshold", 0) - 0.05
            reasons.append(f"High expert usage ({expert_rate:.2f})")
        
        if not updates:
            return None
        
        # Clamp updates to prevent extreme changes
        for key, delta in updates.items():
            updates[key] = max(-0.1, min(0.1, delta))
        
        policy_update = PolicyUpdate(
            updates=updates,
            reason="; ".join(reasons)
        )
        
        self.update_history.append(policy_update)
        return policy_update
    
    def apply_update(self, update: PolicyUpdate):
        """Apply a policy update and auto-save."""
        self.policy.apply_update(update.updates)
        self.save_policy()  # Auto-save after each update
    
    def get_current_metrics(self) -> HomeostasisMetrics:
        """Get current target metrics (for monitoring)."""
        return HomeostasisMetrics(
            avg_confidence=self.targets["confidence"]["current"],
            avg_expert_disagreement=self.targets["expert_disagreement"]["current"],
            avg_decision_cost_ms=self.targets["cost_ms"]["current"],
            repeat_question_rate=0.1,
            risk_level=0.1
        )
    
    def update_targets(self, key: str, value: float):
        """Update the current observed value for a target."""
        if key in self.targets:
            self.targets[key]["current"] = value
    
    def get_health_report(self) -> dict:
        """Get a health report of the system."""
        report = {
            "status": "healthy",
            "deviations": [],
            "recent_updates": len(self.update_history)
        }
        
        for key, target in self.targets.items():
            current = target["current"]
            if current < target["min"]:
                report["deviations"].append(f"{key} too low: {current:.2f} < {target['min']}")
                report["status"] = "warning"
            elif current > target["max"]:
                report["deviations"].append(f"{key} too high: {current:.2f} > {target['max']}")
                report["status"] = "warning"
        
        return report
    
    def save_policy(self):
        """
        Save current policy to disk for persistence across restarts.
        """
        policy_file = self.storage_path / "policy.json"
        try:
            data = {
                "fast_path_bias": self.policy.fast_path_bias,
                "expert_call_threshold": self.policy.expert_call_threshold,
                "creative_range": list(self.policy.creative_range),
                "memory_write_threshold": self.policy.memory_write_threshold,
                "semantic_rules": self.policy.semantic_rules,
                "saved_at": datetime.now().isoformat()
            }
            with open(policy_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Homeostasis] Error saving policy: {e}")
    
    def load_policy(self):
        """
        Load saved policy from disk.
        """
        policy_file = self.storage_path / "policy.json"
        if not policy_file.exists():
            return
        
        try:
            with open(policy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Apply saved values to policy
            if "fast_path_bias" in data:
                self.policy.fast_path_bias = data["fast_path_bias"]
            if "expert_call_threshold" in data:
                self.policy.expert_call_threshold = data["expert_call_threshold"]
            if "memory_write_threshold" in data:
                self.policy.memory_write_threshold = data["memory_write_threshold"]
            if "semantic_rules" in data:
                self.policy.semantic_rules = data["semantic_rules"]
            if "creative_range" in data and len(data["creative_range"]) == 2:
                self.policy.creative_range = tuple(data["creative_range"])
            
            print(f"[Homeostasis] Loaded policy from {policy_file}")
        except Exception as e:
            print(f"[Homeostasis] Error loading policy: {e}")
