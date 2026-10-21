"""Hazard / exposure / vulnerability kept separate; composite only when all three exist."""
from typing import Optional

def risk_components(hazard: Optional[float], exposure: Optional[float], vulnerability: Optional[float]) -> dict:
    """Each input is a 0..1 relative score (normalise upstream and document the method)."""
    comp = {"hazard": hazard, "exposure": exposure, "vulnerability": vulnerability}
    for k, v in comp.items():
        if v is not None and not 0 <= v <= 1:
            raise ValueError(f"{k} must be within 0..1")
    missing = [k for k, v in comp.items() if v is None]
    if missing:
        return {"components": comp, "priority_index": None,
                "note": f"Incomplete ({', '.join(missing)} missing): no composite shown. Not low risk."}
    return {"components": comp, "priority_index": hazard * exposure * vulnerability ** 0.5,
            "note": "Relative prioritisation index (hazard x exposure x sqrt(vulnerability)). "
                    "Not an official disaster-risk metric; weights are unvalidated. Inspect components."}
