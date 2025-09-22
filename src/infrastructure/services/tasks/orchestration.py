"""Celery tasks that orchestrate full training runs."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from celery import chord, group

from src.domain.entities.training_job import DataCollectionJob, TrainingStatus
from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import CallbackTask, logger
from src.infrastructure.services.tasks.data_collection import collect_data_chunk
from src.infrastructure.services.tasks.processing import process_collected_data


@celery_app.task(bind=True, base=CallbackTask, name="orchestrate_training")
def orchestrate_training(
    self,
    training_job_id: str,
    model_id: str,
    last_n: int,
) -> Dict[str, Any]:
    """Orchestrate data collection, preprocessing and training for a job."""

    try:
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.gateways.sth_comet_gateway import STHCometGateway
        from src.infrastructure.repositories.model_repository import ModelRepository
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        settings = get_settings()

        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        model_repo = ModelRepository(database)

        sth_gateway = STHCometGateway(settings.fiware.sth_url)

        model = asyncio.run(model_repo.find_by_id(UUID(model_id)))
        if not model:
            raise ValueError(f"Model {model_id} not found")

        window_size = model.lookback_window

        logger.info(
            "Starting training orchestration",
            training_job_id=training_job_id,
            model_id=model_id,
            window_size=window_size,
            last_n=last_n,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            feature=model.feature,
        )

        min_required_data = window_size + 20
        if last_n < min_required_data:
            raise ValueError(
                f"Insufficient training data requested: {last_n} points. "
                f"Need at least {min_required_data} points for "
                f"window_size={window_size}."
            )

        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.COLLECTING_DATA,
                data_collection_start=datetime.now(timezone.utc),
            )
        )

        if not model.entity_type or not model.entity_id:
            raise ValueError(f"Model {model_id} missing entity_type or entity_id")

        total_count = asyncio.run(
            sth_gateway.get_total_count_from_header(
                entity_type=model.entity_type,
                entity_id=model.entity_id,
                attribute=model.feature,
                fiware_service="smart",
                fiware_servicepath="/",
            )
        )

        logger.info(
            "Got total available data count",
            total_count=total_count,
            requested_last_n=last_n,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            attribute=model.feature,
        )

        if total_count < last_n:
            logger.warning(
                "Requested more data than available",
                requested=last_n,
                available=total_count,
            )
            last_n = total_count

        if total_count == 0:
            raise ValueError(f"No data available for entity {model.entity_id}")

        max_per_request = settings.fiware.max_per_request
        collection_jobs: List[DataCollectionJob] = []
        remaining = last_n
        current_h_offset = max(0, total_count - last_n)

        while remaining > 0:
            chunk_size = min(remaining, max_per_request)
            job = DataCollectionJob(h_offset=current_h_offset, last_n=chunk_size)
            collection_jobs.append(job)

            remaining -= chunk_size
            current_h_offset += chunk_size

        for job in collection_jobs:
            asyncio.run(
                training_job_repo.add_data_collection_job(UUID(training_job_id), job)
            )

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs={"orchestration_task_id": self.request.id},
            )
        )

        logger.info(
            "Created data collection strategy",
            training_job_id=training_job_id,
            total_jobs=len(collection_jobs),
            total_requested=last_n,
            total_available=total_count,
            chunk_size=max_per_request,
            starting_offset=max(0, total_count - last_n),
            estimated_parallel_requests=len(collection_jobs),
        )

        data_collection_tasks = group(
            [
                collect_data_chunk.s(
                    job_id=str(job.id),
                    training_job_id=training_job_id,
                    entity_type=model.entity_type,
                    entity_id=model.entity_id,
                    attribute=model.feature,
                    h_limit=job.last_n,
                    h_offset=job.h_offset,
                ).set(queue="data_collection", task_id=str(job.id))
                for job in collection_jobs
            ]
        )

        process_task_id = f"{training_job_id}:process"

        chord_result = chord(
            data_collection_tasks,
            process_collected_data.s(
                training_job_id=training_job_id,
                model_config={
                    "id": str(model.id),
                    "name": model.name,
                    "model_type": model.model_type.value,
                    "rnn_units": model.rnn_units,
                    "dense_units": model.dense_units,
                    "rnn_dropout": model.rnn_dropout,
                    "dense_dropout": model.dense_dropout,
                    "learning_rate": model.learning_rate,
                    "batch_size": model.batch_size,
                    "epochs": model.epochs,
                    "early_stopping_patience": model.early_stopping_patience,
                    "feature": model.feature,
                    "entity_type": model.entity_type,
                    "entity_id": model.entity_id,
                },
                window_size=window_size,
                last_n=last_n,
            ),
        ).apply_async(queue="orchestration", task_id=process_task_id)

        chord_group_id = getattr(chord_result.parent, "id", None)
        task_ref_payload = {
            "processing_task_id": process_task_id,
            "chord_callback_id": chord_result.id,
        }
        if chord_group_id:
            task_ref_payload["chord_group_id"] = chord_group_id

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs=task_ref_payload,
            )
        )

        logger.info(
            "celery.orchestration.started",
            training_job_id=training_job_id,
            collection_tasks=len(collection_jobs),
            chord_id=chord_result.id,
        )

        return {
            "training_job_id": training_job_id,
            "status": "data_collection_started",
            "collection_tasks": len(collection_jobs),
            "chord_id": chord_result.id,
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "celery.orchestration.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

        try:
            from src.infrastructure.database.mongo_database import MongoDatabase
            from src.infrastructure.repositories.training_job_repository import (
                TrainingJobRepository,
            )
            from src.main.config import get_settings

            settings = get_settings()
            database = MongoDatabase(
                mongo_uri=settings.database.mongo_uri,
                db_name=settings.database.database_name,
            )
            training_job_repo = TrainingJobRepository(database)

            asyncio.run(
                training_job_repo.fail_training_job(
                    UUID(training_job_id),
                    error=str(exc),
                    error_details={
                        "orchestration_error": True,
                        "task_id": self.request.id,
                        "model_id": model_id,
                        "requested_data_points": last_n,
                    },
                )
            )
        except Exception as update_error:  # pragma: no cover
            logger.error(
                "celery.orchestration.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        logger.error(
            "celery.orchestration.not_retrying",
            training_job_id=training_job_id,
            model_id=model_id,
            last_n=last_n,
        )
        raise exc
