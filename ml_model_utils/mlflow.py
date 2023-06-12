import mlflow  # type: ignore
from mlflow.tracking import MlflowClient  # type: ignore
from typing import Dict, Any
from .constants import MlflowModelStage


def mlflow_config(uri: str, experiment_name: str) -> None:
    """Configure mlflow with tracking url and experiment.

    Args:
        uri (str): mlflow tracking uri.
        experiment_name (str): mlflow experiment name for storing the model.

    Examples:
        >>> mlflow_config("https://mlflow.dummy.com", "dummy-model")
    """
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log metrics for the model to upload later.

    Args:
        metrics (:obj:`dict` of (str, any): metrics of the model.

    Examples:
        >>> log_metrics(dict(precision=0.91, recall=0.90, f1_score=0.905,
        ...                  support=300))
    """
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kv, vv in v.items():
                mlflow.log_metric(f"{k}_{kv}", vv)
        else:
            mlflow.log_metric(k, v)


def change_model_stage(
        model_name: str,
        model_version: str,
        current_version_stage: MlflowModelStage = MlflowModelStage.STAGING,
        previous_version_stage: MlflowModelStage = MlflowModelStage.ARCHIVED) -> None:
    """Change the stage of the newly uploaded model, and change the stage of the last version
    of the model.

    Args:
        model_name (str): name of the newly uploaded model.
        model_version (str): version of the newly uploaded model.
        current_version_stage (:obj:`MlflowModelStage`): stage of the current version of the model
          to use. Default is MlflowModelStage.STAGING.
        previous_version_stage (:obj:`MlflowModelStage`): stage of the previous version of the
          model to use. MlflowModelStage.ARCHIVED.

    Examples:
        >>> from ml_model_utils.constants import MlflowModelStage
        >>> change_model_stage("dummy_model", "3",
        ...                    MlflowModelStage.STAGING, MlflowModelStage.ARCHIVED)
    """
    client = MlflowClient()
    if int(model_version) > 1:
        client.transition_model_version_stage(
            name=model_name,
            version=str(int(model_version) - 1),
            stage=previous_version_stage.value
        )
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=current_version_stage.value
    )


def upload_model(uri: str,
                 experiment_name: str,
                 model_name: str,
                 artifact_path: str,
                 model: Any,
                 metrics: Dict[str, Any],
                 current_version_stage: MlflowModelStage = MlflowModelStage.STAGING,
                 previous_version_stage: MlflowModelStage = MlflowModelStage.ARCHIVED) -> None:
    """Upload model to mlflow.

    Args:
        uri (str): mlflow tracking uri.
        experiment_name (str): mlflow experiment name for storing the model.
        model_name (str): mlflow model name.
        artifact_path (str): path for storing the ml model.
        model (any): trained ml model instance.
        metrics (:obj:`dict` of (str, any)): metrics of the model.
        current_version_stage (:obj:`MlflowModelStage`): stage of the current version of the model
          to use. Default is MlflowModelStage.STAGING.
        previous_version_stage (:obj:`MlflowModelStage`): stage of the previous version of the
          model to use. MlflowModelStage.ARCHIVED.

    Examples:
        >>> from ml_model_utils.constants import MlflowModelStage
        >>> upload_model("https://mlflow.dummy.com", "dummy-model", "dummy_model",
        ...              "classifier", model,
        ...              dict(precision=0.91, recall=0.90, f1_score=0.905, support=300),
        ...              MlflowModelStage.STAGING, MlflowModelStage.ARCHIVED)
    """
    mlflow_config(uri, experiment_name)
    # upload model
    with mlflow.start_run() as run:
        log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path)
    # register model
    model_uri = "runs:/{}/{}".format(run.info.run_id, artifact_path)
    mv = mlflow.register_model(model_uri, model_name)
    # adapt model stage
    change_model_stage(model_name=model_name,
                       model_version=mv.version,
                       current_version_stage=current_version_stage,
                       previous_version_stage=previous_version_stage)
