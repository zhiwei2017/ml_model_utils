import pytest
from unittest import mock
from ml_model_utils.mlflow import (
    mlflow_config, log_metrics, change_model_stage, MlflowModelStage,
    upload_model
)


@mock.patch("ml_model_utils.mlflow.mlflow")
def test_mlflow_config(mocked_mlflow):
    mlflow_config("https://dummy_mlflow_url.com", "dummy_model")
    mocked_mlflow.set_tracking_uri.assert_called_once_with("https://dummy_mlflow_url.com")
    mocked_mlflow.set_experiment.assert_called_once_with("dummy_model")


@mock.patch("ml_model_utils.mlflow.mlflow")
def test_log_metrics(mocked_mlflow):
    metrics = dict(precision=0.91, recall=0.90, f1_score=0.905, support=300)
    log_metrics(metrics)
    calls = [mock.call("precision", 0.91),
             mock.call("recall", 0.90),
             mock.call("f1_score", 0.905),
             mock.call("support", 300)]
    mocked_mlflow.log_metric.assert_has_calls(calls, any_order=True)

    metrics = dict(dummy_model_2=dict(precision=0.91, recall=0.90, f1_score=0.905, support=300),
                   dummy_model_1=dict(precision=0.51, recall=0.50, f1_score=0.505, support=300))
    log_metrics(metrics)
    calls = [mock.call("dummy_model_2_precision", 0.91),
             mock.call("dummy_model_2_recall", 0.90),
             mock.call("dummy_model_2_f1_score", 0.905),
             mock.call("dummy_model_2_support", 300),
             mock.call("dummy_model_1_precision", 0.51),
             mock.call("dummy_model_1_recall", 0.50),
             mock.call("dummy_model_1_f1_score", 0.505),
             mock.call("dummy_model_1_support", 300)]
    mocked_mlflow.log_metric.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize("model_version", ['1', '2'])
@mock.patch("ml_model_utils.mlflow.MlflowClient")
def test_change_mlflow_model_stage(mocked_mlflow_client, model_version):
    change_model_stage("dummy_model", model_version, MlflowModelStage.PRODUCTION,
                       MlflowModelStage.ARCHIVED)
    calls = [mock.call(name="dummy_model",
                       version=str(int(model_version) - 1),
                       stage=MlflowModelStage.ARCHIVED.value),
             mock.call(name="dummy_model",
                       version=model_version,
                       stage=MlflowModelStage.PRODUCTION.value)]
    if int(model_version) > 1:
        mocked_mlflow_client.return_value.transition_model_version_stage.assert_has_calls(calls)
    else:
        mocked_mlflow_client.return_value.transition_model_version_stage.assert_has_calls(calls[1:])


@pytest.mark.parametrize("model_version", ['1', '2'])
@mock.patch("ml_model_utils.mlflow.MlflowClient")
@mock.patch("ml_model_utils.mlflow.mlflow")
def test_upload_model_to_mlflow(mocked_mlflow, mocked_mlflow_client, model_version):
    run_id = "h432kj5ry8ilw7fh43rt89"
    enter = mock.MagicMock(return_value=mock.MagicMock(info=mock.Mock(run_id=run_id)))
    mocked_mlflow.start_run.return_value = mock.MagicMock(__enter__=enter)
    mocked_mlflow.register_model.return_value = mock.MagicMock(version=model_version)

    model_name = "dummy_model"
    model = object()
    artifact_path = "dummy_path"
    upload_model(uri="https://dummy_mlflow_url.com",
                 experiment_name=model_name,
                 model_name=model_name,
                 artifact_path=artifact_path,
                 model=model,
                 metrics=dict(precision=0.91, recall=0.90, f1_score=0.905, support=300),
                 current_version_stage=MlflowModelStage.PRODUCTION,
                 previous_version_stage=MlflowModelStage.ARCHIVED)

    mocked_mlflow.set_tracking_uri.assert_called_once_with("https://dummy_mlflow_url.com")
    mocked_mlflow.set_experiment.assert_called_once_with("dummy_model")

    mocked_mlflow.start_run.assert_called_once()
    mocked_mlflow.sklearn.log_model.assert_called_once_with(model, artifact_path)
    mocked_mlflow.register_model.assert_called_once_with("runs:/{}/{}".format(run_id, artifact_path), model_name)


    calls = [mock.call("precision", 0.91),
             mock.call("recall", 0.90),
             mock.call("f1_score", 0.905),
             mock.call("support", 300)]
    mocked_mlflow.log_metric.assert_has_calls(calls, any_order=True)

    calls = [mock.call(name="dummy_model",
                       version=str(int(model_version) - 1),
                       stage=MlflowModelStage.ARCHIVED.value),
             mock.call(name="dummy_model",
                       version=model_version,
                       stage=MlflowModelStage.PRODUCTION.value)]
    if int(model_version) > 1:
        mocked_mlflow_client.return_value.transition_model_version_stage.assert_has_calls(calls)
    else:
        mocked_mlflow_client.return_value.transition_model_version_stage.assert_has_calls(calls[1:])


