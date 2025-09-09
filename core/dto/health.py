from typing import Optional, TypedDict


class HealthResult(TypedDict):
    ok: bool
    error: Optional[str]
