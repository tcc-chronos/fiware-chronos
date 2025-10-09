# Logging Configuration

Chronos uses `structlog` to produce structured application logs. Output format and verbosity are controlled via environment variables.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). | `INFO` |
| `LOG_FORMAT` | Human-readable log format used in development. | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` |
| `LOG_FILE_PATH` | Optional file path for log output. | Console only |

## Recommended Formats

- **Development** – plain text for readability.
- **Production** – JSON output for log aggregation backends (Loki, ELK, etc.). Set `LOG_FORMAT=json`.

## Usage Example

```python
from src.shared import get_logger

logger = get_logger(__name__)

logger.info("operation.started")
logger.debug("operation.parameters", window=48, horizon=6)

logger.info("request.received", method="GET", path="/api/models")
logger.error("forecast.publish.failed", error_code=500, entity_id="urn:ngsi-ld:Sensor:001")

user_logger = logger.bind(user_id="12345", session_id="abc-123")
user_logger.info("user.login.success")
```

## Sample Output

### Development

```
2025-01-15 10:15:32,123 - chronos.training - INFO - operation.started
2025-01-15 10:15:32,125 - chronos.training - INFO - request.received method=GET path=/api/models
```

### Production (JSON)

```json
{"timestamp": "2025-01-15T10:15:32.123Z", "level": "INFO", "logger": "chronos.training", "event": "operation.started", "app": "fiware-chronos"}
{"timestamp": "2025-01-15T10:15:32.125Z", "level": "INFO", "logger": "chronos.training", "event": "request.received", "method": "GET", "path": "/api/models"}
```
