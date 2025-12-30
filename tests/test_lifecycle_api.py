
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
import web # Import the module, not just the app

# Manually mock the system instance
mock_system = MagicMock()
mock_system.learning_decoder = MagicMock()
mock_system.learning_decoder.raw_traces = []
web.system = mock_system

client = TestClient(web.app)

def test_lifecycle_empty():
    """Test 404 when no traces exist."""
    response = client.get("/api/lifecycle/latest")
    assert response.status_code == 404
    assert response.json() == {"error": "No traces yet"}

def test_lifecycle_with_data():
    """Test full timeline generation."""
    # Inject a mock trace
    mock_trace = MagicMock()
    mock_trace.episode_id = "test-123"
    mock_trace.timestamp = datetime.now()
    mock_trace.user_input = "Calculate 2+2"
    mock_trace.context_snapshot = {"user_identity": {"trust": 1.0}}
    mock_trace.decision = {
        "action": "respond",
        "depth_used": "deep",
        "confidence": 0.9,
        "cost": {"time_ms": 100}
    }
    mock_trace.expert_outputs = [
        {"expert_type": "math", "confidence": 1.0}
    ]
    mock_trace.final_response = "The answer is 4."
    
    web.system.learning_decoder.raw_traces = [mock_trace]
    
    response = client.get("/api/lifecycle/latest")
    assert response.status_code == 200
    data = response.json()
    
    assert data["trace_id"] == "test-123"
    timeline = data["timeline"]
    assert len(timeline) == 5
    assert timeline[0]["step"] == "Input"
    assert timeline[2]["step"] == "Operational Module"
    assert timeline[2]["status"] == "warning"  # Deep depth = warning color
    assert timeline[3]["step"] == 'Expert (math)' 

if __name__ == "__main__":
    try:
        test_lifecycle_empty()
        test_lifecycle_with_data()
        print("OK: text_lifecycle_api passed!")
    except AssertionError as e:
        print(f"FAIL: Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: Error: {e}")
        sys.exit(1)
