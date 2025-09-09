import redis

from core.dto.health import HealthResult


def ping_redis(url: str) -> HealthResult:
    try:
        r = redis.Redis.from_url(
            url,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
        )
        pong = r.ping()
        return HealthResult(ok=bool(pong))
    except Exception as e:
        return HealthResult(ok=False, error=str(e))
