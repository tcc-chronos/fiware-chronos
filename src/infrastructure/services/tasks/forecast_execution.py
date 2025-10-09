"""Celery task that executes scheduled forecasts and publishes them to Orion."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import CallbackTask, logger
from src.infrastructure.settings import get_settings


@celery_app.task(bind=True, base=CallbackTask, name="execute_forecast")
def execute_forecast(
    self,
    *,
    training_job_id: str,
    model_id: Optional[str],
) -> Dict[str, Any]:
    """Run a prediction for the specified training job and publish it to Orion."""

    try:
        from src.application.use_cases.model_prediction_use_case import (
            ModelPredictionUseCase,
        )
        from src.domain.entities.prediction import ForecastSeriesPoint, PredictionRecord
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.gateways.iot_agent_gateway import IoTAgentGateway
        from src.infrastructure.gateways.orion_gateway import OrionGateway
        from src.infrastructure.gateways.sth_comet_gateway import STHCometGateway
        from src.infrastructure.repositories.gridfs_model_artifacts_repository import (
            GridFSModelArtifactsRepository,
        )
        from src.infrastructure.repositories.model_repository import ModelRepository
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )

        settings = get_settings()

        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        model_repo = ModelRepository(database)
        sth_gateway = STHCometGateway(settings.fiware.sth_url)
        iot_gateway = IoTAgentGateway(settings.fiware.iot_agent_url)
        orion_gateway = OrionGateway(settings.fiware.orion_url)
        artifacts_repo = GridFSModelArtifactsRepository(
            mongo_client=database.client,
            database_name=settings.database.database_name,
        )

        if model_id is None:
            training_job = asyncio.run(
                training_job_repo.get_by_id(UUID(training_job_id))
            )
            if not training_job or not training_job.model_id:
                raise ValueError(
                    f"Training job {training_job_id} does not reference a model"
                )
            model_id = str(training_job.model_id)

        model_uuid = UUID(model_id)
        training_uuid = UUID(training_job_id)

        training_job = asyncio.run(training_job_repo.get_by_id(training_uuid))
        if not training_job:
            raise ValueError(f"Training job {training_job_id} not found")
        if not training_job.prediction_config.enabled:
            logger.info(
                "forecast.execution.skip_disabled",
                training_job_id=training_job_id,
            )
            return {"status": "skipped", "reason": "disabled"}
        if not training_job.prediction_config.entity_id:
            raise ValueError("Prediction entity ID is not configured")

        prediction_use_case = ModelPredictionUseCase(
            model_repository=model_repo,
            training_job_repository=training_job_repo,
            artifacts_repository=artifacts_repo,
            sth_gateway=sth_gateway,
            iot_agent_gateway=iot_gateway,
            fiware_service=settings.fiware.service,
            fiware_service_path=settings.fiware.service_path,
        )

        prediction_response = asyncio.run(
            prediction_use_case.execute(model_uuid, training_uuid)
        )

        model = asyncio.run(model_repo.find_by_id(model_uuid))
        if not model:
            raise ValueError(f"Model {model_id} not found")

        series: List[ForecastSeriesPoint] = []
        fallback_interval = training_job.sampling_interval_seconds or 0
        for item in prediction_response.predictions:
            timestamp = item.timestamp
            if timestamp is None and fallback_interval > 0:
                timestamp = prediction_response.generated_at + timedelta(
                    seconds=fallback_interval * item.step
                )
            if timestamp is None:
                timestamp = prediction_response.generated_at
            series.append(
                ForecastSeriesPoint(
                    step=item.step,
                    value=item.value,
                    target_timestamp=timestamp,
                )
            )

        record = PredictionRecord(
            entity_id=training_job.prediction_config.entity_id,
            entity_type=training_job.prediction_config.entity_type,
            source_entity=model.entity_id,
            model_id=str(model.id),
            training_id=str(training_uuid),
            horizon=prediction_response.forecast_horizon,
            feature=model.feature,
            generated_at=prediction_response.generated_at,
            series=series,
            metadata=training_job.prediction_config.metadata,
        )

        asyncio.run(
            orion_gateway.upsert_prediction(
                record,
                service=settings.fiware.service,
                service_path=settings.fiware.service_path,
            )
        )

        return {
            "status": "completed",
            "training_job_id": training_job_id,
            "model_id": model_id,
            "points": len(series),
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "forecast.execution.failed",
            training_job_id=training_job_id,
            model_id=model_id,
            error=str(exc),
            exc_info=exc,
        )
        raise
