"""
Simple Alert Manager (Phase 2)
File: monitoring/alerting.py

Purpose:
- Log important pipeline / monitoring alerts to JSON
- Keep alerting simple and observable
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json

from src.utils import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Lightweight alert logger for monitoring & pipelines."""

    def __init__(self, log_path: str = "logs/alerts.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "INFO",
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Log an alert entry.

        Args:
            alert_type: e.g. data_drift, model_performance, pipeline_failure
            message: Human-readable message
            severity: INFO | WARNING | CRITICAL
            metadata: Optional extra context
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
        }

        # Load existing alerts
        alerts = []
        if self.log_path.exists():
            try:
                alerts = json.loads(self.log_path.read_text())
            except json.JSONDecodeError:
                alerts = []

        alerts.append(alert)

        # Save alerts
        self.log_path.write_text(json.dumps(alerts, indent=2))

        # Console log
        if severity == "CRITICAL":
            logger.error(message)
        elif severity == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

