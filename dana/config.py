from enum import StrEnum
import os

from pydantic import BaseSettings, ConfigDict


# Storage configuration
class StorageType(StrEnum):
    """
    Use StrEnum to avoid issues with string comparison.
    """

    FILE = "file"
    # TODO: Implement other storage types
    # S3 = "s3"
    # GCS = "gcs"
    # AZURE = "azure"
    # LOCAL = "local"


class StorageConfig(BaseSettings):
    type: StorageType
    model_config = ConfigDict(use_enum_values=True)


class FileStorageConfig(StorageConfig):
    type: StorageType = StorageType.FILE
    workspace_folder: str | None


# Model configuration
class ModelTargetConfig(BaseSettings):
    provider: str
    model: str
    api_key: str | None = None
    endpoint: str | None = None
    extra: dict[str, str] | None = None


# MAIN CONFIG
class Config(BaseSettings):
    storage_cfg: StorageConfig
    models: list[ModelTargetConfig] = []


storage_mode = os.getenv("DANA_STORAGE_MODE", "file")
if storage_mode == StorageType.FILE:
    storage_cfg = FileStorageConfig()
else:
    raise ValueError(f"Invalid storage mode: {storage_mode}")

config = Config()
