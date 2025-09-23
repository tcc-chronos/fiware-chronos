"""Domain service helpers for validating model configurations."""

from typing import List

from src.domain.entities.errors import ModelValidationError
from src.domain.entities.model import DenseLayerConfig, Model, RNNLayerConfig


def _validate_rnn_layers(layers: List[RNNLayerConfig], errors: List[str]) -> None:
    if not layers:
        errors.append("Model must have at least one RNN layer configured.")
        return

    for idx, layer in enumerate(layers, start=1):
        prefix = f"RNN layer #{idx}"
        if layer.units <= 0:
            errors.append(f"{prefix} must have more than 0 units.")
        if not 0.0 <= layer.dropout < 1.0:
            errors.append(
                f"{prefix} dropout must be between 0 (inclusive) and 1 (exclusive)."
            )
        if not 0.0 <= layer.recurrent_dropout < 1.0:
            errors.append(
                f"{prefix} recurrent_dropout must be between 0 "
                f"(inclusive) and 1 (exclusive)."
            )


def _validate_dense_layers(layers: List[DenseLayerConfig], errors: List[str]) -> None:
    for idx, layer in enumerate(layers, start=1):
        prefix = f"Dense layer #{idx}"
        if layer.units <= 0:
            errors.append(f"{prefix} must have more than 0 units.")
        if not 0.0 <= layer.dropout < 1.0:
            errors.append(
                f"{prefix} dropout must be between 0 (inclusive) and 1 (exclusive)."
            )
        if not layer.activation:
            errors.append(f"{prefix} must define a non-empty activation function.")


def validate_model_configuration(model: Model) -> None:
    """Validate a model's hyperparameters and configuration.

    Raises:
        ModelValidationError: If one or more validation rules fail.
    """

    errors: List[str] = []

    _validate_rnn_layers(model.rnn_layers, errors)
    _validate_dense_layers(model.dense_layers, errors)

    if model.batch_size <= 0:
        errors.append("Batch size must be greater than 0.")
    if model.epochs <= 0:
        errors.append("Number of epochs must be greater than 0.")
    if not 0.0 < model.learning_rate <= 1.0:
        errors.append("Learning rate must be greater than 0 and at most 1.")
    if not 0.0 <= model.validation_split < 1.0:
        errors.append(
            "Validation split must be between 0 (inclusive) and 1 (exclusive)."
        )
    if model.lookback_window <= 0:
        errors.append("Lookback window must be greater than 0.")
    if model.forecast_horizon <= 0:
        errors.append("Forecast horizon must be greater than 0.")

    if model.early_stopping_patience is not None:
        if model.early_stopping_patience <= 0:
            errors.append(
                "Early stopping patience must be greater than 0 when provided."
            )
        if (
            model.early_stopping_patience
            and model.early_stopping_patience > model.epochs
        ):
            errors.append(
                "Early stopping patience cannot be greater "
                "than the total number of epochs."
            )

    if not model.entity_type or not model.entity_type.strip():
        errors.append("FIWARE entity type must be provided.")
    if not model.entity_id or not model.entity_id.strip():
        errors.append("FIWARE entity ID must be provided.")
    if not model.feature or not model.feature.strip():
        errors.append("Feature attribute must be provided.")

    if errors:
        raise ModelValidationError(
            "Model configuration is invalid.", details={"errors": errors}
        )
