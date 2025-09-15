"""
Infrastructure Repository - Training Job MongoDB Implementation

This module implements the training job repository using MongoDB.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

import structlog
from pymongo.errors import PyMongoError

from src.domain.entities.training_job import (
    DataCollectionJob,
    DataCollectionStatus,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from src.domain.repositories.training_job_repository import ITrainingJobRepository
from src.infrastructure.database.mongo_database import MongoDatabase

logger = structlog.get_logger(__name__)


class TrainingJobRepository(ITrainingJobRepository):
    """MongoDB implementation of training job repository."""

    def __init__(self, database: MongoDatabase):
        """Initialize repository with database connection."""
        self.database = database
        self.collection_name = "training_jobs"

    async def create(self, training_job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Convert to document
            document = self._to_document(training_job)

            # Insert document
            result = collection.insert_one(document)

            if result.inserted_id:
                logger.info(
                    "Training job created",
                    training_job_id=str(training_job.id),
                    model_id=(
                        str(training_job.model_id) if training_job.model_id else None
                    ),
                )
                return training_job
            else:
                raise Exception("Failed to insert training job")

        except PyMongoError as e:
            logger.error(
                "Failed to create training job",
                training_job_id=str(training_job.id),
                error=str(e),
            )
            raise e

    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        """Get training job by ID."""
        try:
            collection = self.database.get_collection(self.collection_name)

            document = collection.find_one({"id": str(training_job_id)})

            if document:
                return self._from_document(document)
            return None

        except PyMongoError as e:
            logger.error(
                "Failed to get training job",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise e

    async def get_by_model_id(self, model_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific model."""
        try:
            collection = self.database.get_collection(self.collection_name)

            cursor = collection.find({"model_id": str(model_id)}).sort("created_at", -1)
            documents = list(cursor)

            return [self._from_document(doc) for doc in documents]

        except PyMongoError as e:
            logger.error(
                "Failed to get training jobs by model ID",
                model_id=str(model_id),
                error=str(e),
            )
            raise e

    async def update(self, training_job: TrainingJob) -> TrainingJob:
        """Update a training job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Update timestamp
            training_job.update_timestamp()

            # Convert to document
            document = self._to_document(training_job)

            # Update document
            result = collection.replace_one({"id": str(training_job.id)}, document)

            if result.modified_count > 0:
                logger.info(
                    "Training job updated", training_job_id=str(training_job.id)
                )
                return training_job
            else:
                raise Exception("Training job not found or not modified")

        except PyMongoError as e:
            logger.error(
                "Failed to update training job",
                training_job_id=str(training_job.id),
                error=str(e),
            )
            raise e

    async def delete(self, training_job_id: UUID) -> bool:
        """Delete a training job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            result = collection.delete_one({"id": str(training_job_id)})

            if result.deleted_count > 0:
                logger.info(
                    "Training job deleted", training_job_id=str(training_job_id)
                )
                return True
            return False

        except PyMongoError as e:
            logger.error(
                "Failed to delete training job",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise e

    async def list_all(self, skip: int = 0, limit: int = 100) -> List[TrainingJob]:
        """List all training jobs with pagination."""
        try:
            collection = self.database.get_collection(self.collection_name)

            cursor = collection.find().sort("created_at", -1).skip(skip).limit(limit)
            documents = list(cursor)

            return [self._from_document(doc) for doc in documents]

        except PyMongoError as e:
            logger.error(
                "Failed to list training jobs", skip=skip, limit=limit, error=str(e)
            )
            raise e

    async def add_data_collection_job(
        self, training_job_id: UUID, job: DataCollectionJob
    ) -> bool:
        """Add a data collection job to a training job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Convert job to document
            job_doc = {
                "id": str(job.id),
                "h_offset": job.h_offset,
                "last_n": job.last_n,
                "status": job.status.value,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "error": job.error,
                "data_points_collected": job.data_points_collected,
            }

            # Add to data_collection_jobs array
            result = collection.update_one(
                {"id": str(training_job_id)},
                {
                    "$push": {"data_collection_jobs": job_doc},
                    "$set": {"updated_at": datetime.utcnow().isoformat()},
                },
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(
                "Failed to add data collection job",
                training_job_id=str(training_job_id),
                job_id=str(job.id),
                error=str(e),
            )
            raise e

    async def update_data_collection_job_status(
        self,
        training_job_id: UUID,
        job_id: UUID,
        status: DataCollectionStatus,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error: Optional[str] = None,
        data_points_collected: Optional[int] = None,
    ) -> bool:
        """Update the status of a data collection job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Build update document
            update_fields = {
                "data_collection_jobs.$.status": status.value,
                "updated_at": datetime.utcnow().isoformat(),
            }

            if start_time:
                update_fields["data_collection_jobs.$.start_time"] = (
                    start_time.isoformat()
                )
            if end_time:
                update_fields["data_collection_jobs.$.end_time"] = end_time.isoformat()
            if error:
                update_fields["data_collection_jobs.$.error"] = error
            if data_points_collected is not None:
                update_fields["data_collection_jobs.$.data_points_collected"] = str(
                    data_points_collected
                )

            # Update specific job in array
            result = collection.update_one(
                {"id": str(training_job_id), "data_collection_jobs.id": str(job_id)},
                {"$set": update_fields},
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(
                "Failed to update data collection job status",
                training_job_id=str(training_job_id),
                job_id=str(job_id),
                error=str(e),
            )
            raise e

    async def update_training_job_status(
        self,
        training_job_id: UUID,
        status: TrainingStatus,
        data_collection_start: Optional[datetime] = None,
        data_collection_end: Optional[datetime] = None,
        preprocessing_start: Optional[datetime] = None,
        preprocessing_end: Optional[datetime] = None,
        training_start: Optional[datetime] = None,
        training_end: Optional[datetime] = None,
        total_data_points_collected: Optional[int] = None,
    ) -> bool:
        """Update the status and timestamps of a training job."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Build update document
            update_fields = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat(),
            }

            if data_collection_start:
                update_fields["data_collection_start"] = (
                    data_collection_start.isoformat()
                )
            if data_collection_end:
                update_fields["data_collection_end"] = data_collection_end.isoformat()
            if preprocessing_start:
                update_fields["preprocessing_start"] = preprocessing_start.isoformat()
            if preprocessing_end:
                update_fields["preprocessing_end"] = preprocessing_end.isoformat()
            if training_start:
                update_fields["training_start"] = training_start.isoformat()
            if training_end:
                update_fields["training_end"] = training_end.isoformat()
            if total_data_points_collected is not None:
                update_fields["total_data_points_collected"] = str(
                    total_data_points_collected
                )

            # Update document
            result = collection.update_one(
                {"id": str(training_job_id)}, {"$set": update_fields}
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(
                "Failed to update training job status",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise e

    async def complete_training_job(
        self,
        training_job_id: UUID,
        metrics: TrainingMetrics,
        model_artifact_path: str,
        x_scaler_path: str,
        y_scaler_path: str,
        metadata_path: str,
    ) -> bool:
        """Complete a training job with results."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Convert metrics to dict
            metrics_dict = {
                "mse": metrics.mse,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "r2": metrics.r2,
                "mae_pct": metrics.mae_pct,
                "rmse_pct": metrics.rmse_pct,
                "best_train_loss": metrics.best_train_loss,
                "best_val_loss": metrics.best_val_loss,
                "best_epoch": metrics.best_epoch,
            }

            # Update document
            result = collection.update_one(
                {"id": str(training_job_id)},
                {
                    "$set": {
                        "status": TrainingStatus.COMPLETED.value,
                        "end_time": datetime.utcnow().isoformat(),
                        "metrics": metrics_dict,
                        "model_artifact_path": model_artifact_path,
                        "x_scaler_path": x_scaler_path,
                        "y_scaler_path": y_scaler_path,
                        "metadata_path": metadata_path,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(
                "Failed to complete training job",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise e

    async def fail_training_job(
        self, training_job_id: UUID, error: str, error_details: Optional[dict] = None
    ) -> bool:
        """Mark a training job as failed."""
        try:
            collection = self.database.get_collection(self.collection_name)

            # Update document
            result = collection.update_one(
                {"id": str(training_job_id)},
                {
                    "$set": {
                        "status": TrainingStatus.FAILED.value,
                        "end_time": datetime.utcnow().isoformat(),
                        "error": error,
                        "error_details": error_details,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(
                "Failed to mark training job as failed",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise e

    def _to_document(self, training_job: TrainingJob) -> dict:
        """Convert TrainingJob entity to MongoDB document."""

        # Convert data collection jobs
        data_collection_jobs = []
        for job in training_job.data_collection_jobs:
            job_doc = {
                "id": str(job.id),
                "h_offset": job.h_offset,
                "last_n": job.last_n,
                "status": job.status.value,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "error": job.error,
                "data_points_collected": job.data_points_collected,
            }
            data_collection_jobs.append(job_doc)

        # Convert metrics
        metrics_doc = None
        if training_job.metrics:
            metrics_doc = {
                "mse": training_job.metrics.mse,
                "mae": training_job.metrics.mae,
                "rmse": training_job.metrics.rmse,
                "mape": training_job.metrics.mape,
                "r2": training_job.metrics.r2,
                "mae_pct": training_job.metrics.mae_pct,
                "rmse_pct": training_job.metrics.rmse_pct,
                "best_train_loss": training_job.metrics.best_train_loss,
                "best_val_loss": training_job.metrics.best_val_loss,
                "best_epoch": training_job.metrics.best_epoch,
            }

        return {
            "id": str(training_job.id),
            "model_id": str(training_job.model_id) if training_job.model_id else None,
            "status": training_job.status.value,
            "last_n": training_job.last_n,
            "data_collection_jobs": data_collection_jobs,
            "total_data_points_requested": training_job.total_data_points_requested,
            "total_data_points_collected": training_job.total_data_points_collected,
            "start_time": (
                training_job.start_time.isoformat() if training_job.start_time else None
            ),
            "end_time": (
                training_job.end_time.isoformat() if training_job.end_time else None
            ),
            "data_collection_start": (
                training_job.data_collection_start.isoformat()
                if training_job.data_collection_start
                else None
            ),
            "data_collection_end": (
                training_job.data_collection_end.isoformat()
                if training_job.data_collection_end
                else None
            ),
            "preprocessing_start": (
                training_job.preprocessing_start.isoformat()
                if training_job.preprocessing_start
                else None
            ),
            "preprocessing_end": (
                training_job.preprocessing_end.isoformat()
                if training_job.preprocessing_end
                else None
            ),
            "training_start": (
                training_job.training_start.isoformat()
                if training_job.training_start
                else None
            ),
            "training_end": (
                training_job.training_end.isoformat()
                if training_job.training_end
                else None
            ),
            "metrics": metrics_doc,
            "model_artifact_path": training_job.model_artifact_path,
            "x_scaler_path": training_job.x_scaler_path,
            "y_scaler_path": training_job.y_scaler_path,
            "metadata_path": training_job.metadata_path,
            "error": training_job.error,
            "error_details": training_job.error_details,
            "created_at": training_job.created_at.isoformat(),
            "updated_at": training_job.updated_at.isoformat(),
        }

    def _from_document(self, document: dict) -> TrainingJob:
        """Convert MongoDB document to TrainingJob entity."""

        # Convert data collection jobs
        data_collection_jobs = []
        for job_doc in document.get("data_collection_jobs", []):
            job = DataCollectionJob(
                id=UUID(job_doc["id"]),
                h_offset=job_doc["h_offset"],
                last_n=job_doc["last_n"],
                status=DataCollectionStatus(job_doc["status"]),
                start_time=(
                    datetime.fromisoformat(job_doc["start_time"])
                    if job_doc.get("start_time")
                    else None
                ),
                end_time=(
                    datetime.fromisoformat(job_doc["end_time"])
                    if job_doc.get("end_time")
                    else None
                ),
                error=job_doc.get("error"),
                data_points_collected=job_doc.get("data_points_collected", 0),
            )
            data_collection_jobs.append(job)

        # Convert metrics
        metrics = None
        if document.get("metrics"):
            metrics_doc = document["metrics"]
            metrics = TrainingMetrics(
                mse=metrics_doc.get("mse"),
                mae=metrics_doc.get("mae"),
                rmse=metrics_doc.get("rmse"),
                mape=metrics_doc.get("mape"),
                r2=metrics_doc.get("r2"),
                mae_pct=metrics_doc.get("mae_pct"),
                rmse_pct=metrics_doc.get("rmse_pct"),
                best_train_loss=metrics_doc.get("best_train_loss"),
                best_val_loss=metrics_doc.get("best_val_loss"),
                best_epoch=metrics_doc.get("best_epoch"),
            )

        return TrainingJob(
            id=UUID(document["id"]),
            model_id=UUID(document["model_id"]) if document.get("model_id") else None,
            status=TrainingStatus(document["status"]),
            last_n=document["last_n"],
            data_collection_jobs=data_collection_jobs,
            total_data_points_requested=document.get("total_data_points_requested", 0),
            total_data_points_collected=document.get("total_data_points_collected", 0),
            start_time=(
                datetime.fromisoformat(document["start_time"])
                if document.get("start_time")
                else None
            ),
            end_time=(
                datetime.fromisoformat(document["end_time"])
                if document.get("end_time")
                else None
            ),
            data_collection_start=(
                datetime.fromisoformat(document["data_collection_start"])
                if document.get("data_collection_start")
                else None
            ),
            data_collection_end=(
                datetime.fromisoformat(document["data_collection_end"])
                if document.get("data_collection_end")
                else None
            ),
            preprocessing_start=(
                datetime.fromisoformat(document["preprocessing_start"])
                if document.get("preprocessing_start")
                else None
            ),
            preprocessing_end=(
                datetime.fromisoformat(document["preprocessing_end"])
                if document.get("preprocessing_end")
                else None
            ),
            training_start=(
                datetime.fromisoformat(document["training_start"])
                if document.get("training_start")
                else None
            ),
            training_end=(
                datetime.fromisoformat(document["training_end"])
                if document.get("training_end")
                else None
            ),
            metrics=metrics,
            model_artifact_path=document.get("model_artifact_path"),
            x_scaler_path=document.get("x_scaler_path"),
            y_scaler_path=document.get("y_scaler_path"),
            metadata_path=document.get("metadata_path"),
            error=document.get("error"),
            error_details=document.get("error_details"),
            created_at=datetime.fromisoformat(document["created_at"]),
            updated_at=datetime.fromisoformat(document["updated_at"]),
        )
