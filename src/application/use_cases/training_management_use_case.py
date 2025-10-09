"""
Application Use Cases - Training Management

This module contains use cases for managing training jobs.
"""

import json
from datetime import datetime, timezone
from math import ceil
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from src.application.dtos.training_dto import (
    DataCollectionJobDTO,
    StartTrainingResponseDTO,
    TrainingJobDTO,
    TrainingJobSummaryDTO,
    TrainingMetricsDTO,
    TrainingRequestDTO,
)
from src.domain.entities.errors import ModelValidationError
from src.domain.entities.model import Model, ModelStatus
from src.domain.entities.training_job import (
    DataCollectionStatus,
    TrainingJob,
    TrainingStatus,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.orion_gateway import IOrionGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.ports.training_orchestrator import ITrainingOrchestrator
from src.domain.repositories.model_artifacts_repository import IModelArtifactsRepository
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository
from src.domain.services import validate_model_configuration

logger = structlog.get_logger(__name__)


class TrainingManagementError(Exception):
    """Exception raised when training management operations fail."""

    pass


class TrainingManagementUseCase:
    """Use case for managing training jobs."""

    def __init__(
        self,
        training_job_repository: ITrainingJobRepository,
        model_repository: IModelRepository,
        artifacts_repository: IModelArtifactsRepository,
        sth_gateway: ISTHCometGateway,
        training_orchestrator: ITrainingOrchestrator,
        iot_agent_gateway: IIoTAgentGateway,
        orion_gateway: IOrionGateway,
        fiware_service: str = "smart",
        fiware_service_path: str = "/",
    ):
        """
        Initialize the training management use case.

        Args:
            training_job_repository: Repository for training jobs
            model_repository: Repository for models
            artifacts_repository: Repository for stored model artifacts
        """
        self.training_job_repository = training_job_repository
        self.model_repository = model_repository
        self.artifacts_repository = artifacts_repository
        self.sth_gateway = sth_gateway
        self.training_orchestrator = training_orchestrator
        self.iot_agent_gateway = iot_agent_gateway
        self.orion_gateway = orion_gateway
        self.fiware_service = fiware_service
        self.fiware_service_path = fiware_service_path

    async def start_training(
        self, model_id: UUID, request: TrainingRequestDTO
    ) -> StartTrainingResponseDTO:
        """
        Start a new training job for a model.

        Args:
            model_id: ID of the model to train
            request: Training request parameters

        Returns:
            Response with training job ID

        Raises:
            TrainingManagementError: When training cannot be started
        """
        previous_status: Optional[ModelStatus] = None
        model: Optional[Model] = None

        try:
            # Validate model exists
            model = await self.model_repository.find_by_id(model_id)
            if not model:
                raise TrainingManagementError(f"Model {model_id} not found")

            previous_status = model.status

            logger.info(
                "Starting training job",
                model_id=str(model_id),
                lookback_window=model.lookback_window,
                last_n=request.last_n,
            )

            # Validate model configuration
            if not model.entity_type or not model.entity_id or not model.feature:
                raise TrainingManagementError(
                    "Model must have entity_type, entity_id, and feature configured"
                )

            try:
                validate_model_configuration(model)
            except ModelValidationError as exc:
                raise TrainingManagementError(
                    f"Model configuration invalid: {exc.message}"
                ) from exc

            train_ratio = 1.0 - model.validation_ratio - model.test_ratio
            if train_ratio <= 0.0:
                raise TrainingManagementError(
                    "Validation and test ratios leave no data for training."
                )

            base_min = model.lookback_window + model.forecast_horizon
            dynamic_min = int(
                ceil((model.lookback_window + model.forecast_horizon + 2) / train_ratio)
            )
            min_required_points = max(base_min, dynamic_min)

            if request.last_n < min_required_points:
                message = (
                    "Requested data window is too small for the configured "
                    "lookback window and data splits. Provide at least "
                    f"{min_required_points} points for lookback_window="
                    f"{model.lookback_window}, forecast_horizon="
                    f"{model.forecast_horizon}, validation_ratio="
                    f"{model.validation_ratio}, and test_ratio={model.test_ratio}."
                )
                raise TrainingManagementError(message)

            try:
                total_available = await self.sth_gateway.get_total_count_from_header(
                    entity_type=model.entity_type,
                    entity_id=model.entity_id,
                    attribute=model.feature,
                    fiware_service=self.fiware_service,
                    fiware_servicepath=self.fiware_service_path,
                )
            except Exception as exc:
                raise TrainingManagementError(
                    f"Failed to validate STH-Comet availability: {exc}"
                ) from exc

            if total_available <= 0:
                raise TrainingManagementError(
                    "No data available in STH-Comet for the configured entity/feature."
                )

            if total_available < min_required_points:
                raise TrainingManagementError(
                    "Insufficient historical data for training. "
                    f"Required at least {min_required_points} points but only "
                    f"{total_available} are available."
                )

            if request.last_n > total_available:
                raise TrainingManagementError(
                    f"Requested {request.last_n} data points but only "
                    f"{total_available} are available in STH-Comet."
                )

            # Check for existing running training jobs
            existing_jobs = await self.training_job_repository.get_by_model_id(model_id)
            running_jobs = [
                job
                for job in existing_jobs
                if job.status
                in [
                    TrainingStatus.PENDING,
                    TrainingStatus.CANCEL_REQUESTED,
                    TrainingStatus.COLLECTING_DATA,
                    TrainingStatus.PREPROCESSING,
                    TrainingStatus.TRAINING,
                ]
            ]

            if running_jobs:
                raise TrainingManagementError(
                    f"Model {model_id} already has a running training job: "
                    f"{running_jobs[0].id}"
                )

            model.status = ModelStatus.TRAINING
            model.update_timestamp()
            await self.model_repository.update(model)

            # Create new training job
            training_job = TrainingJob(
                model_id=model_id,
                last_n=request.last_n,
                total_data_points_requested=request.last_n,
                start_time=datetime.now(timezone.utc),
            )

            # Save training job
            created_job = await self.training_job_repository.create(training_job)

            task_id = await self.training_orchestrator.dispatch_training_job(
                training_job_id=created_job.id,
                model_id=model_id,
                last_n=request.last_n,
            )

            if task_id:
                await self.training_job_repository.update_task_refs(
                    created_job.id,
                    task_refs={"orchestration_task_id": task_id},
                )

            logger.info(
                "Orchestration task dispatched",
                training_job_id=str(created_job.id),
                task_id=task_id,
                status="sent",
            )

            logger.info(
                "Training job started successfully",
                training_job_id=str(created_job.id),
                model_id=str(model_id),
            )

            return StartTrainingResponseDTO(
                training_job_id=created_job.id, status=created_job.status
            )

        except Exception as e:
            if "model" in locals() and model:
                try:
                    revert_status = previous_status or ModelStatus.DRAFT
                    if (
                        previous_status == ModelStatus.TRAINED
                        and model.has_trained_artifacts()
                    ):
                        revert_status = ModelStatus.TRAINED

                    model.status = revert_status
                    model.update_timestamp()
                    await self.model_repository.update(model)
                except Exception as status_error:  # pragma: no cover - best effort
                    logger.warning(
                        "Failed to revert model status after training start error",
                        model_id=str(model_id),
                        error=str(status_error),
                    )
            logger.error(
                "Failed to start training", model_id=str(model_id), error=str(e)
            )
            if isinstance(e, TrainingManagementError):
                raise e
            raise TrainingManagementError(f"Failed to start training: {str(e)}") from e

    async def get_training_job(
        self, model_id: UUID, training_job_id: UUID
    ) -> Optional[TrainingJobDTO]:
        """
        Get detailed information about a training job.

        Args:
            model_id: ID of the model the training job should belong to
            training_job_id: ID of the training job

        Returns:
            Training job details or None if not found
        """
        try:
            training_job = await self.training_job_repository.get_by_id(training_job_id)

            if not training_job or training_job.model_id != model_id:
                return None

            metadata = await self._extract_training_metadata(training_job)
            return self._to_training_job_dto(training_job, metadata)

        except Exception as e:
            logger.error(
                "Failed to get training job",
                model_id=str(model_id),
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise TrainingManagementError(
                f"Failed to get training job: {str(e)}"
            ) from e

    async def list_training_jobs_by_model(
        self, model_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[TrainingJobSummaryDTO]:
        """List training jobs for a specific model."""
        try:
            training_jobs = await self.training_job_repository.get_by_model_id(model_id)

            # Apply pagination manually since repository returns full list
            training_jobs = training_jobs[skip : skip + limit]

            results: List[TrainingJobSummaryDTO] = []
            for job in training_jobs:
                metadata = await self._extract_training_metadata(job)
                results.append(self._to_training_job_summary_dto(job, metadata))

            return results

        except Exception as e:
            logger.error(
                "Failed to list training jobs for model",
                model_id=str(model_id),
                error=str(e),
            )
            raise TrainingManagementError(
                f"Failed to list training jobs: {str(e)}"
            ) from e

    async def cancel_training_job(self, model_id: UUID, training_job_id: UUID) -> bool:
        """
        Cancel a running training job.

        Args:
            model_id: ID of the model the training job should belong to
            training_job_id: ID of the training job to cancel

        Returns:
            True if cancelled successfully, False if the job is not found for the model

        Raises:
            TrainingManagementError: When job cannot be cancelled
        """
        try:
            training_job = await self.training_job_repository.get_by_id(training_job_id)

            if not training_job or training_job.model_id != model_id:
                logger.warning(
                    "Training job not found for cancellation",
                    training_job_id=str(training_job_id),
                    model_id=str(model_id),
                )
                return False

            if training_job.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            ]:
                raise TrainingManagementError(
                    f"Training job {training_job_id} is already "
                    f"{training_job.status.value}"
                )

            if training_job.status != TrainingStatus.CANCEL_REQUESTED:
                await self.training_job_repository.update_training_job_status(
                    training_job_id, TrainingStatus.CANCEL_REQUESTED
                )
                training_job.status = TrainingStatus.CANCEL_REQUESTED

            task_refs = training_job.task_refs or {}
            revoke_ids = {
                str(task_id)
                for task_id in (
                    task_refs.get("orchestration_task_id"),
                    task_refs.get("chord_callback_id"),
                    task_refs.get("chord_group_id"),
                    task_refs.get("processing_task_id"),
                    task_refs.get("training_task_id"),
                )
                if task_id
            }

            data_collection_ids = task_refs.get("data_collection_task_ids") or []
            revoke_ids.update(
                str(task_id) for task_id in data_collection_ids if task_id
            )

            if revoke_ids:
                logger.info(
                    "Revoking orchestration tasks",
                    training_job_id=str(training_job_id),
                    task_count=len(revoke_ids),
                )
                await self.training_orchestrator.revoke_tasks(list(revoke_ids))

            now = datetime.now(timezone.utc)

            # Update outstanding collection jobs to cancelled
            for job in training_job.data_collection_jobs:
                if job.status in (
                    DataCollectionStatus.PENDING,
                    DataCollectionStatus.IN_PROGRESS,
                ):
                    await self.training_job_repository.update_data_collection_job_status(  # noqa: E501
                        training_job_id,
                        job.id,
                        DataCollectionStatus.CANCELLED,
                        end_time=now,
                        error="Cancelled by user request",
                    )

            data_collection_end = training_job.data_collection_end
            if not data_collection_end and training_job.data_collection_start:
                data_collection_end = now

            preprocessing_end = training_job.preprocessing_end
            if not preprocessing_end and training_job.preprocessing_start:
                preprocessing_end = now

            training_end = training_job.training_end
            if not training_end and training_job.training_start:
                training_end = now

            await self.training_job_repository.update_training_job_status(
                training_job_id,
                TrainingStatus.CANCELLED,
                data_collection_end=data_collection_end,
                preprocessing_end=preprocessing_end,
                training_end=training_end,
                total_data_points_collected=training_job.total_data_points_collected,
                end_time=now,
            )

            model = await self.model_repository.find_by_id(model_id)
            if model:
                model.status = (
                    ModelStatus.TRAINED
                    if model.has_trained_artifacts()
                    else ModelStatus.DRAFT
                )
                model.update_timestamp()
                await self.model_repository.update(model)

            await self.training_orchestrator.schedule_cleanup(training_job_id)

            logger.info("Training job cancelled", training_job_id=str(training_job_id))

            return True

        except Exception as e:
            logger.error(
                "Failed to cancel training job",
                model_id=str(model_id),
                training_job_id=str(training_job_id),
                error=str(e),
            )
            if isinstance(e, TrainingManagementError):
                raise e
            raise TrainingManagementError(
                f"Failed to cancel training job: {str(e)}"
            ) from e

    async def delete_training_job(self, model_id: UUID, training_job_id: UUID) -> bool:
        """
        Delete a training job and its generated artifacts.

        Args:
            model_id: ID of the model the training job should belong to
            training_job_id: ID of the training job to delete

        Returns:
            True when the training job is deleted successfully, False when the
            training job is not found for the requested model.

        Raises:
            TrainingManagementError: When the job cannot be deleted
        """
        try:
            training_job = await self.training_job_repository.get_by_id(training_job_id)

            if not training_job or training_job.model_id != model_id:
                logger.warning(
                    "Training job not found for deletion",
                    training_job_id=str(training_job_id),
                    model_id=str(model_id),
                )
                return False

            if training_job.status in [
                TrainingStatus.PENDING,
                TrainingStatus.CANCEL_REQUESTED,
                TrainingStatus.COLLECTING_DATA,
                TrainingStatus.PREPROCESSING,
                TrainingStatus.TRAINING,
            ]:
                raise TrainingManagementError(
                    f"Training job {training_job_id} for model {model_id} "
                    "is still running. Cancel it before deleting."
                )

            await self._cleanup_prediction_resources(training_job)

            artifact_ids = [
                training_job.model_artifact_id,
                training_job.x_scaler_artifact_id,
                training_job.y_scaler_artifact_id,
                training_job.metadata_artifact_id,
            ]

            for artifact_id in artifact_ids:
                if not artifact_id:
                    continue

                try:
                    deleted = await self.artifacts_repository.delete_artifact(
                        artifact_id
                    )

                    if deleted:
                        logger.info(
                            "Training artifact deleted",
                            training_job_id=str(training_job_id),
                            artifact_id=artifact_id,
                        )
                    else:
                        logger.warning(
                            "Training artifact not found for deletion",
                            training_job_id=str(training_job_id),
                            artifact_id=artifact_id,
                        )

                except Exception as e:
                    logger.error(
                        "Failed to delete training artifact",
                        training_job_id=str(training_job_id),
                        artifact_id=artifact_id,
                        error=str(e),
                    )
                    raise TrainingManagementError(
                        "Failed to delete training artifacts"
                    ) from e

            deleted = await self.training_job_repository.delete(training_job_id)

            if deleted:
                logger.info(
                    "Training job deleted successfully",
                    training_job_id=str(training_job_id),
                )

                remaining_jobs = await self.training_job_repository.get_by_model_id(
                    model_id
                )

                should_revert_model = False
                if not remaining_jobs:
                    should_revert_model = True
                else:
                    non_successful_statuses = {
                        TrainingStatus.CANCELLED,
                        TrainingStatus.FAILED,
                    }
                    should_revert_model = all(
                        job.status in non_successful_statuses for job in remaining_jobs
                    )

                if should_revert_model:
                    model_record = await self.model_repository.find_by_id(model_id)
                    if model_record:
                        model_record.clear_artifacts()
                        model_record.status = ModelStatus.DRAFT
                        model_record.update_timestamp()
                        await self.model_repository.update(model_record)
                        logger.info(
                            "Model reverted to draft after deleting trainings",
                            model_id=str(model_id),
                        )

            return deleted

        except TrainingManagementError:
            raise

        except Exception as e:
            logger.error(
                "Failed to delete training job",
                model_id=str(model_id),
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise TrainingManagementError(
                f"Failed to delete training job: {str(e)}"
            ) from e

    async def _cleanup_prediction_resources(self, training_job: TrainingJob) -> None:
        """Remove prediction-related FIWARE resources for the training job."""

        entity_id = (training_job.prediction_config.entity_id or "").strip()
        if entity_id:
            try:
                await self.orion_gateway.delete_entity(
                    entity_id=entity_id,
                    service=self.fiware_service,
                    service_path=self.fiware_service_path,
                )
            except Exception as exc:
                raise TrainingManagementError(
                    f"Failed to delete Orion entity {entity_id}: {exc}"
                ) from exc

        device_id = self._extract_prediction_device_id(training_job)
        if device_id:
            try:
                await self.iot_agent_gateway.delete_device(
                    device_id,
                    service=self.fiware_service,
                    service_path=self.fiware_service_path,
                )
            except Exception as exc:
                raise TrainingManagementError(
                    f"Failed to delete IoT Agent device {device_id}: {exc}"
                ) from exc

    def _extract_prediction_device_id(self, training_job: TrainingJob) -> Optional[str]:
        metadata = training_job.prediction_config.metadata or {}
        candidate_keys = (
            "device_id",
            "deviceId",
            "iot_device_id",
            "iotDeviceId",
        )
        for key in candidate_keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    async def _extract_training_metadata(
        self, training_job: TrainingJob
    ) -> Optional[Dict[str, Any]]:
        metadata_id = training_job.metadata_artifact_id
        if not metadata_id:
            return None

        try:
            artifact = await self.artifacts_repository.get_artifact_by_id(metadata_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "training.metadata.fetch_failed",
                training_job_id=str(training_job.id),
                metadata_artifact_id=metadata_id,
                error=str(exc),
            )
            return None

        if not artifact:
            logger.info(
                "training.metadata.not_found",
                training_job_id=str(training_job.id),
                metadata_artifact_id=metadata_id,
            )
            return None

        try:
            return json.loads(artifact.content.decode("utf-8"))
        except Exception as exc:
            logger.warning(
                "training.metadata.parse_failed",
                training_job_id=str(training_job.id),
                metadata_artifact_id=metadata_id,
                error=str(exc),
            )
            return None

    def _to_training_job_dto(
        self,
        training_job: TrainingJob,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainingJobDTO:
        """Convert TrainingJob entity to DTO."""

        # Convert data collection jobs
        data_collection_jobs = [
            DataCollectionJobDTO(
                id=job.id,
                h_offset=job.h_offset,
                last_n=job.last_n,
                status=job.status,
                start_time=job.start_time,
                end_time=job.end_time,
                error=job.error,
                data_points_collected=job.data_points_collected,
            )
            for job in training_job.data_collection_jobs
        ]

        # Convert metrics
        metrics = None
        if training_job.metrics:
            metrics = TrainingMetricsDTO(
                mse=training_job.metrics.mse,
                mae=training_job.metrics.mae,
                rmse=training_job.metrics.rmse,
                theil_u=training_job.metrics.theil_u,
                mape=training_job.metrics.mape,
                r2=training_job.metrics.r2,
                mae_pct=training_job.metrics.mae_pct,
                rmse_pct=training_job.metrics.rmse_pct,
                best_train_loss=training_job.metrics.best_train_loss,
                best_val_loss=training_job.metrics.best_val_loss,
                best_epoch=training_job.metrics.best_epoch,
            )

        training_history = (
            metadata.get("training_history") if isinstance(metadata, dict) else None
        )

        return TrainingJobDTO(
            id=training_job.id,
            model_id=training_job.model_id,
            status=training_job.status,
            last_n=training_job.last_n,
            data_collection_jobs=data_collection_jobs,
            total_data_points_requested=training_job.total_data_points_requested,
            total_data_points_collected=training_job.total_data_points_collected,
            data_collection_progress=training_job.get_data_collection_progress(),
            start_time=training_job.start_time,
            end_time=training_job.end_time,
            data_collection_start=training_job.data_collection_start,
            data_collection_end=training_job.data_collection_end,
            preprocessing_start=training_job.preprocessing_start,
            preprocessing_end=training_job.preprocessing_end,
            training_start=training_job.training_start,
            training_end=training_job.training_end,
            metrics=metrics,
            model_artifact_id=training_job.model_artifact_id,
            metadata_artifact_id=training_job.metadata_artifact_id,
            error=training_job.error,
            error_details=training_job.error_details,
            prediction_enabled=training_job.prediction_config.enabled,
            prediction_entity_id=training_job.prediction_config.entity_id,
            sampling_interval_seconds=training_job.sampling_interval_seconds,
            next_prediction_at=training_job.next_prediction_at,
            created_at=training_job.created_at,
            updated_at=training_job.updated_at,
            total_duration_seconds=training_job.get_total_duration(),
            training_duration_seconds=training_job.get_training_duration(),
            training_history=training_history,
        )

    def _to_training_job_summary_dto(
        self,
        training_job: TrainingJob,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainingJobSummaryDTO:
        """Convert TrainingJob entity to summary DTO."""

        training_history = (
            metadata.get("training_history") if isinstance(metadata, dict) else None
        )

        return TrainingJobSummaryDTO(
            id=training_job.id,
            model_id=training_job.model_id,
            status=training_job.status,
            data_collection_progress=training_job.get_data_collection_progress(),
            start_time=training_job.start_time,
            end_time=training_job.end_time,
            error=training_job.error,
            created_at=training_job.created_at,
            metadata_artifact_id=training_job.metadata_artifact_id,
            training_history=training_history,
            prediction_enabled=training_job.prediction_config.enabled,
            next_prediction_at=training_job.next_prediction_at,
        )
