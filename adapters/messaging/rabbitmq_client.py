import pika

from core.dto.health import HealthResult


def ping_rabbitmq(amqp_url: str) -> HealthResult:
    try:
        params = pika.URLParameters(amqp_url)
        params.socket_timeout = 2
        connection = pika.BlockingConnection(params)
        try:
            channel = connection.channel()
            channel.close()
        finally:
            connection.close()
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}
