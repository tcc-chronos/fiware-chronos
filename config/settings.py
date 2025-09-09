from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = Field(default="chronos-ge")
    APP_VERSION: str = Field(default="0.1.0")
    API_PORT: int = Field(default=8000)

    MONGO_URI: str = Field(default="mongodb://localhost:27017/chronos")
    BROKER_URL: str = Field(default="amqp://chronos:chronos@rabbitmq:5672/chronos")
    RESULT_BACKEND: str = Field(default="redis://localhost:6379/0")

    GIT_COMMIT: str = Field(default="unknown")
    BUILD_TIME: str = Field(default="unknown")

    class Config:
        env_file = ".env"


settings = Settings()
