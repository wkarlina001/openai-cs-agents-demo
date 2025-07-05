# metrics_logger.py
import json
import os
import datetime

METRICS_FILE = "log/metrics.log"
metrics_dir = os.path.dirname(METRICS_FILE)

# Check if the directory exists
if not os.path.exists(metrics_dir):
    # If it doesn't exist, create it
    os.makedirs(metrics_dir)
    print(f"Created directory: {metrics_dir}")
else:
    print(f"Log Directory already exists: {metrics_dir}")

def log_metric(event_type: str, data: dict):
    """
    Logs a metric event to a file.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "data": data
    }
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged metric: {event_type} - {data}")

# Example usage (for testing)
if __name__ == "__main__":
    log_metric("user_query", {"message": "What is my account balance?"})
    log_metric("tool_usage", {"tool_name": "faq_retrieval", "agent": "FAQ Agent"})
    log_metric("retrieval_analytics", {"type": "hit", "query": "internet speed", "docs_retrieved": 3})
    log_metric("retrieval_analytics", {"type": "miss", "query": "pizza delivery", "reason": "out of scope"})