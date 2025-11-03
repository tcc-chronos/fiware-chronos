# FIWARE Interoperability (NGSI)

This guide provides reproducible NGSI-v2 examples showing how Chronos publishes predictions to Orion Context Broker and how other applications can query those predictions directly from Orion and STH‑Comet. The goal is to make interoperability verifiable and repeatable for researchers.

## Prerequisites

- Orion base URL (e.g., `http://localhost:1026`)
- STH‑Comet base URL (e.g., `http://localhost:8666`)
- FIWARE headers:
  - `Fiware-Service: smart`
  - `Fiware-ServicePath: /`
- Target entity identifiers:
  - `entityId` (e.g., `urn:ngsi-ld:Sensor:001`)
  - `entityType` (e.g., `Sensor`)
- Attribute used to publish forecasts: `forecastSeries` (configurable via `PREDICTION_ATTRIBUTE`).

## NGSI Payloads

### Entity (minimal)

```json
{
  "id": "urn:ngsi-ld:Sensor:001",
  "type": "Sensor"
}
```

### Forecast attribute update (Chronos publication)

```json
{
  "forecastSeries": {
    "type": "StructuredValue",
    "value": [
      { "step": 1, "targetTimestamp": "2025-01-01T00:00:00Z", "value": 12.34 },
      { "step": 2, "targetTimestamp": "2025-01-01T00:10:00Z", "value": 12.10 }
    ],
    "metadata": {
      "modelId": { "type": "Text", "value": "<model-uuid>" },
      "trainingId": { "type": "Text", "value": "<training-uuid>" },
      "feature": { "type": "Text", "value": "humidity" },
      "generatedAt": { "type": "DateTime", "value": "2025-01-01T00:00:00Z" }
    }
  }
}
```

### Orion subscription to STH‑Comet (store forecast history)

```json
{
  "description": "Store forecastSeries history",
  "subject": {
    "entities": [{ "id": "urn:ngsi-ld:Sensor:001", "type": "Sensor" }],
    "condition": { "attrs": ["forecastSeries"] }
  },
  "notification": {
    "http": { "url": "http://sth-comet:8666/notify" },
    "attrsFormat": "legacy"
  },
  "throttling": 1
}
```

Chronos creates this subscription automatically when you enable recurring predictions, but the example above allows independent verification.

## cURL Examples

Set environment variables first:

```bash
export ORION=http://localhost:1026
export STH=http://localhost:8666
export FIWARE_SERVICE=smart
export FIWARE_SERVICE_PATH=/
export ENTITY_ID=urn:ngsi-ld:Sensor:001
export ENTITY_TYPE=Sensor
export ATTR=forecastSeries
```

- Ensure entity (idempotent create):

```bash
curl -i -X POST "$ORION/v2/entities" \
  -H 'Content-Type: application/json' \
  -H "Fiware-Service: $FIWARE_SERVICE" \
  -H "Fiware-ServicePath: $FIWARE_SERVICE_PATH" \
  -d '{"id":"'"$ENTITY_ID"'","type":"'"$ENTITY_TYPE"'"}'
```

- Publish forecast to Orion:

```bash
cat > payload.json <<'JSON'
{
  "forecastSeries": {
    "type": "StructuredValue",
    "value": [
      { "step": 1, "targetTimestamp": "2025-01-01T00:00:00Z", "value": 12.34 },
      { "step": 2, "targetTimestamp": "2025-01-01T00:10:00Z", "value": 12.10 }
    ],
    "metadata": {
      "modelId": { "type": "Text", "value": "123e4567-e89b-12d3-a456-426614174000" },
      "trainingId": { "type": "Text", "value": "223e4567-e89b-12d3-a456-426614174001" },
      "feature": { "type": "Text", "value": "humidity" },
      "generatedAt": { "type": "DateTime", "value": "2025-01-01T00:00:00Z" }
    }
  }
}
JSON

curl -i -X POST "$ORION/v2/entities/$ENTITY_ID/attrs" \
  -H 'Content-Type: application/json' \
  -H "Fiware-Service: $FIWARE_SERVICE" \
  -H "Fiware-ServicePath: $FIWARE_SERVICE_PATH" \
  --data-binary @payload.json
```

- Create STH‑Comet subscription (if not handled by Chronos):

```bash
cat > sub.json <<'JSON'
{
  "description": "Store forecastSeries history",
  "subject": {
    "entities": [{ "id": "'"$ENTITY_ID"'", "type": "'"$ENTITY_TYPE"'" }],
    "condition": { "attrs": ["forecastSeries"] }
  },
  "notification": {
    "http": { "url": "http://sth-comet:8666/notify" },
    "attrsFormat": "legacy"
  },
  "throttling": 1
}
JSON

curl -i -X POST "$ORION/v2/subscriptions" \
  -H 'Content-Type: application/json' \
  -H "Fiware-Service: $FIWARE_SERVICE" \
  -H "Fiware-ServicePath: $FIWARE_SERVICE_PATH" \
  --data-binary @sub.json
```

- Query Orion (keyValues):

```bash
curl -s "$ORION/v2/entities/$ENTITY_ID?type=$ENTITY_TYPE&options=keyValues" \
  -H "Fiware-Service: $FIWARE_SERVICE" \
  -H "Fiware-ServicePath: $FIWARE_SERVICE_PATH" | jq
```

- Query STH‑Comet history of `forecastSeries`:

```bash
curl -s "$STH/STH/v1/contextEntities/type/$ENTITY_TYPE/id/$ENTITY_ID/attributes/$ATTR?lastN=5" \
  -H "Fiware-Service: $FIWARE_SERVICE" \
  -H "Fiware-ServicePath: $FIWARE_SERVICE_PATH" | jq
```

## Python Scripts (reproducible)

For convenience, this repository provides ready-to-run scripts under `examples/ngsi/`:

- `publish_forecast.py` — Ensures the entity exists and publishes a forecast attribute to Orion.
- `read_entity.py` — Reads an entity from Orion in keyValues mode.
- `query_sth.py` — Queries STH‑Comet for recent values of `forecastSeries`.
- `create_subscription.py` — Creates a subscription in Orion pointing to STH‑Comet (if needed).

Set environment variables (same as in cURL examples) before running the scripts.

