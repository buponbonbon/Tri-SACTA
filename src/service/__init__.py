"""Tri-SACTA Service Module (lazy exports to avoid eager Pydantic/FastAPI-schema coupling)."""

__all__ = ["BRMORouter", "OrchestratorConfig"]


def __getattr__(name: str):
    if name in __all__:
        from src.service.router import BRMORouter, OrchestratorConfig
        return {"BRMORouter": BRMORouter, "OrchestratorConfig": OrchestratorConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
