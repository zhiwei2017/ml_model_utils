from enum import Enum


class MlflowModelStage(Enum):
    """Enumeration for mlflow stages: Staging, Archived, Production and None."""
    STAGING = "Staging"
    ARCHIVED = "Archived"
    PRODUCTION = "Production"
    NONE = "None"
