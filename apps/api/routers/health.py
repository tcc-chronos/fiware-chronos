from datetime import datetime, timezone

from fastapi import APIRouter

from adapters.messaging.rabbitmq_client import ping_rabbitmq
from adapters.messaging.redis_client import ping_redis
from adapters.mongo.mongo_client import ping_mongo
from config.settings import settings
from core.dto.health import HealthDependencies, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    mongo = ping_mongo(settings.MONGO_URI)
    rabbit = ping_rabbitmq(settings.BROKER_URL)
    redis_status = ping_redis(settings.RESULT_BACKEND)

    now = datetime.now(timezone.utc)
    uptime_s = None
    try:
        from apps.api.main import app

        started = getattr(app.state, "started_at", None)
        if started:
            uptime_s = (now - started).total_seconds()
    except Exception:
        pass

    all_ok = all([mongo.ok, rabbit.ok, redis_status.ok])

    response = HealthResponse(
        status="ok" if all_ok else "degraded",
        uptime_s=uptime_s,
        deps=HealthDependencies(mongo=mongo, broker=rabbit, redis=redis_status),
        version=settings.APP_VERSION,
        timestamp=now.isoformat(),
    )
    return response
