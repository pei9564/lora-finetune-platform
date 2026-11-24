from types import SimpleNamespace
from app.auth.jwt_utils import create_token

from app.api.routes import task
from app.train import publication


def test_retry_publication_reuses_registered_version(
    monkeypatch, tmp_path, test_client, test_config
):
    adapter = tmp_path / "final_model"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"saved adapter")
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    publication.prepare_publication(
        adapter, "run-1", "experiment", test_config,
        {"eval_accuracy": 0.85, "eval_loss": 0.2}, model,
    )

    version = SimpleNamespace(name="bert-base-uncased", version=1, current_stage="None")
    versions = []

    def register_model(*args):
        versions.append(version)
        return version

    def transition_model_version_stage(**kwargs):
        if version.current_stage == "None":
            version.current_stage = "Staging"
            raise RuntimeError("response lost after stage transition")

    client = SimpleNamespace(
        search_model_versions=lambda query: versions,
        log_artifacts=lambda *args: None,
        transition_model_version_stage=transition_model_version_stage,
    )
    monkeypatch.setattr(publication.mlflow.tracking, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(publication.mlflow, "register_model", register_model)
    monkeypatch.setattr(publication.registry, "save_model_card", lambda card: None)
    assert publication.publish_model(adapter, model)["status"] == "failed"

    result = {"adapter_path": str(adapter), "config": {"user_id": "owner"}}
    backend = SimpleNamespace(get_task_meta=lambda task_id: {"status": "SUCCESS", "result": result})
    monkeypatch.setattr(task, "AsyncResult", lambda task_id: SimpleNamespace(backend=backend))
    endpoint = "/task/training-job/publication/retry"

    denied = test_client.post(endpoint, headers={"Authorization": f"Bearer {create_token('other', 'user')}"})
    assert denied.status_code == 403
    response = test_client.post(endpoint, headers={"Authorization": f"Bearer {create_token('owner', 'user')}"})
    assert response.status_code == 200
    assert response.json() == {"status": "published", "version": 1}
    assert test_client.post(endpoint).json() == response.json()
    assert versions == [version]
    assert weights.read_bytes() == b"saved adapter"
