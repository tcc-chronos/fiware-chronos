# Class Diagram (Logical)

This diagram presents the main domain entities, core interfaces (ports), and selected infrastructure implementations.

```mermaid
classDiagram
  direction TB

  class Model {
    +UUID id
    +str name
    +ModelType model_type
    +ModelStatus status
    +int lookback_window
    +int forecast_horizon
    +str feature
    +str entity_type
    +str entity_id
    +int batch_size
    +int epochs
    +float learning_rate
    +RNNLayerConfig[] rnn_layers
    +DenseLayerConfig[] dense_layers
    +bool has_successful_training()
  }

  class RNNLayerConfig {
    +int units
    +float dropout
    +float recurrent_dropout
  }

  class DenseLayerConfig {
    +int units
    +float dropout
    +str activation
  }

  class TrainingJob {
    +UUID id
    +UUID model_id
    +TrainingStatus status
    +int last_n
    +DataCollectionJob[] data_collection_jobs
    +int total_data_points_requested
    +int total_data_points_collected
    +datetime start_time
    +datetime end_time
    +TrainingMetrics metrics
    +int? sampling_interval_seconds
    +datetime? next_prediction_at
    +TrainingPredictionConfig prediction_config
    +set_sampling_interval(seconds)
  }

  class DataCollectionJob {
    +UUID id
    +int h_offset
    +int last_n
    +DataCollectionStatus status
    +datetime start_time
    +datetime end_time
    +int data_points_collected
  }

  class TrainingPredictionConfig {
    +bool enabled
    +str? service_group
    +str? entity_id
    +str entity_type
    +str? subscription_id
  }

  class TrainingMetrics {
    +float? mse
    +float? mae
    +float? rmse
    +float? mape
    +float? r2
    +float? best_train_loss
    +float? best_val_loss
    +int? best_epoch
  }

  class PredictionRecord {
    +str entity_id
    +str entity_type
    +int horizon
    +str feature
    +ForecastSeriesPoint[] series
  }

  class ForecastSeriesPoint {
    +int step
    +float value
    +datetime target_timestamp
  }

  class HistoricDataPoint {
    +datetime timestamp
    +float value
  }

  Model --> "*" TrainingJob : has trainings
  Model o-- RNNLayerConfig : uses
  Model o-- DenseLayerConfig : uses
  TrainingJob *-- "*" DataCollectionJob : collects
  TrainingJob o-- TrainingMetrics : results
  TrainingJob o-- TrainingPredictionConfig : scheduling
  DataCollectionJob --> "*" HistoricDataPoint : produces
  TrainingJob --> PredictionRecord : produces
  PredictionRecord --> "*" ForecastSeriesPoint
```

Note: This logical diagram focuses on relationships and interfaces rather than exhaustive attributes.
