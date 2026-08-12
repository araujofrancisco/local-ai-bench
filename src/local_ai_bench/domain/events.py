"""Structured events emitted by the runner to power progress and JSONL logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    kind: str
    host: str | None = None
    model: str | None = None
    plugin: str | None = None
    case_id: str | None = None
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        for key in ("host", "model", "plugin", "case_id", "message"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out["data"] = self.data
        return out


class Events:
    """Event kind constants (mirrors the event stream in PLAN.md §19.6)."""

    RUN_STARTED = "RunStarted"
    HOST_CHECKED = "HostChecked"
    MODEL_DISCOVERED = "ModelDiscovered"
    PLUGIN_STARTED = "PluginStarted"
    CASE_STARTED = "CaseStarted"
    CASE_COMPLETED = "CaseCompleted"
    CASE_FAILED = "CaseFailed"
    PLUGIN_COMPLETED = "PluginCompleted"
    CONTEXT_PROBE_STARTED = "ContextProbeStarted"
    CONTEXT_PROBE_COMPLETED = "ContextProbeCompleted"
    RUN_COMPLETED = "RunCompleted"
    REPORT_GENERATED = "ReportGenerated"