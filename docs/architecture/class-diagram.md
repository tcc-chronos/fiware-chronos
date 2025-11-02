# Class Diagram (Logical)

This diagram presents the main domain entities, core interfaces (ports), and selected infrastructure implementations.

```mermaid
classDiagram
  direction LR

  class Model {
    +UUID id
    +str name
    +ModelType model_type
    +ModelStatus status
    +int lookback_window
    +int forecast_horizon
    +str entity_type
    +str entity_id
    +bool has_successful_training()
  }

  class TrainingJob {
    +UUID id
    +UUID model_id
    +TrainingStatus status
    +TrainingMetrics metrics
    +datetime next_prediction_at
    +set_sampling_interval(seconds)
  }

  class PredictionRecord {
    +str entity_id
    +str entity_type
    +int horizon
    +ForecastSeriesPoint[] series
  }

  class IModelRepository {
    <<interface>>
    +get_by_id(id)
    +list(...)
    +create(model)
    +update(model)
    +delete(id)
  }

  class ITrainingJobRepository {
    <<interface>>
    +create(job)
    +get_by_id(id)
    +update(...)
    +update_prediction_schedule(...)
    +claim_prediction_schedule(...)
  }

  class IModelArtifactsRepository {
    <<interface>>
    +save_artifact(...)
    +get_artifact(...)
    +delete_model_artifacts(...)
  }

  class ModelRepository
  class TrainingJobRepository
  class GridFSModelArtifactsRepository

  Model --> TrainingJob : trains
  TrainingJob --> PredictionRecord : produces

  IModelRepository <|.. ModelRepository
  ITrainingJobRepository <|.. TrainingJobRepository
  IModelArtifactsRepository <|.. GridFSModelArtifactsRepository
```

Note: This logical diagram focuses on relationships and interfaces rather than exhaustive attributes.
