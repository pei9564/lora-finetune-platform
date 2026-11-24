from types import SimpleNamespace
from app.auth.jwt_utils import create_token

from app.api.routes import task
from app.models.model_registry import ModelRegistry
from app.train import publication


def test_retry_publication_reuses_registered_version(
    monkeypatch, tmp_path, test_client, test_config
):
    verify_publication(monkeypatch, tmp_path, test_client, test_config, fail_once=True)


def test_successful_publication_updates_single_card(
    monkeypatch, tmp_path, test_client, test_config
):
    verify_publication(monkeypatch, tmp_path, test_client, test_config, fail_once=False)


def verify_publication(monkeypatch, tmp_path, test_client, test_config, fail_once):
    adapter = tmp_path / "final_model"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"saved adapter")
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    config = tmp_path / "config.yaml"
    test_config.save_yaml(str(config))
    registry = ModelRegistry(str(tmp_path / "registry"))
    monkeypatch.setattr(publication, "registry", registry)
    publication.prepare_publication(
        adapter, "run-1", test_config,
        {"eval_accuracy": 0.85, "eval_loss": 0.2}, model, config,
    )
    card_path = tmp_path / "registry" / "task_run-1.json"
    assert registry.models["task_run-1"].publication_status == "pending"
    assert card_path.exists()

    version = SimpleNamespace(name="bert-base-uncased", version=1, current_stage="None")
    versions = []
    uploaded_configs = []

    def register_model(*args):
        versions.append(version)
        return version

    def transition_model_version_stage(**kwargs):
        if version.current_stage == "None":
            version.current_stage = "Staging"
            if fail_once:
                raise RuntimeError("response lost after stage transition")

    client = SimpleNamespace(
        search_model_versions=lambda query: versions,
        log_artifacts=lambda *args: None,
        log_artifact=lambda *args: uploaded_configs.append(args),
        transition_model_version_stage=transition_model_version_stage,
    )
    monkeypatch.setattr(publication.mlflow.tracking, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(publication.mlflow, "register_model", register_model)
    expected_status = "failed" if fail_once else "published"
    assert publication.publish_model(adapter, model)["status"] == expected_status
    assert registry.models["task_run-1"].publication_status == expected_status

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
    assert list(card_path.parent.glob("*.json")) == [card_path]
    assert uploaded_configs == [("run-1", str(config), "config")]
    card = ModelRegistry(str(card_path.parent)).models["task_run-1"]
    assert card.publication_status == "published"
    assert card.run_id == "run-1"
    assert card.metrics.accuracy == 0.85
    assert card.version == 1
    assert card.adapter_path == str(adapter)
    assert card.config_path == str(config)
    assert card.config_artifact == "config/config.yaml"
