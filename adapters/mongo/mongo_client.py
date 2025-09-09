from pymongo import MongoClient
from pymongo.errors import PyMongoError

from core.dto.health import HealthResult


def ping_mongo(uri: str) -> HealthResult:
    try:
        client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return {"ok": True, "error": None}
    except PyMongoError as e:
        return {"ok": False, "error": str(e)}
