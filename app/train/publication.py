"""Publish a saved training result independently of the training loop."""

import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import fcntl
import mlflow
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

from app.models.model_registry import ModelCard, registry

logger = logging.getLogger(__name__)


def _write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)


def model_card_id(run_id: str) -> str:
    return f"task_{run_id}"


def _save_card(adapter, state):
    model_name = state["base_model"].split("/")[-1]
    card = ModelCard(
        id=model_card_id(state["run_id"]),
        name=model_name,
        base_model=state["base_model"],
        language="en",
        task="text-classification",
        description=f"LoRA fine-tuned model on {state['dataset']} dataset",
        metrics=state["metrics"],
        version=state.get("version"),
        stage=state.get("stage"),
        run_id=state["run_id"],
        tags=[state["dataset"], "lora", state["base_model"]],
        experiment_name=state.get("experiment_name"),
        adapter_path=str(adapter),
        config_path=state.get("config_path"),
        config_artifact=state.get("config_artifact"),
        publication_status=state["status"],
    )
    if not registry.save_model_card(card):
        raise RuntimeError(f"Could not save model card {card.id}")


def prepare_publication(adapter_path, run_id, config, metrics, model, config_path=None):
    path = Path(adapter_path).parent / "publication.json"
    base_revision = getattr(model.config, "_commit_hash", None)
    state = {
        "status": "pending",
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "config_path": os.fspath(config_path) if config_path is not None else None,
        "base_model": config.model.name,
        "base_revision": base_revision if isinstance(base_revision, str) else None,
        "num_labels": config.model.num_labels,
        "dataset": config.data.dataset_name,
        "metrics": {"accuracy": metrics["eval_accuracy"], "loss": metrics["eval_loss"]},
    }
    _write_state(path, state)
    _save_card(adapter_path, state)


def publication_status(adapter_path):
    path = Path(adapter_path).parent / "publication.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    return {key: state[key] for key in ("status", "version", "error") if key in state}


def publish_model(adapter_path, model=None):
    """Upload and register the saved model, reusing any version for this run."""
    adapter = Path(adapter_path)
    path = adapter.parent / "publication.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["status"] == "published":
            return publication_status(adapter)

        try:
            run_id = state["run_id"]
            model_name = state["base_model"].split("/")[-1]
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            config_path = state.get("config_path")
            if config_path and not state.get("config_artifact"):
                client.log_artifact(run_id, config_path, "config")
                state["config_artifact"] = f"config/{Path(config_path).name}"
            version = next(
                (item for item in client.search_model_versions(f"run_id = '{run_id}'")
                 if item.name == model_name),
                None,
            )
            if version is None:
                package = adapter / "mlflow_model"
                if not package.exists():
                    if model is None:
                        base = AutoModelForSequenceClassification.from_pretrained(
                            state["base_model"],
                            revision=state["base_revision"],
                            num_labels=state["num_labels"],
                        )
                        model = PeftModel.from_pretrained(base, adapter)
                    with TemporaryDirectory(dir=adapter.parent) as temporary:
                        staged = Path(temporary) / "model"
                        mlflow.pytorch.save_model(
                            model, str(staged), serialization_format="pickle"
                        )
                        staged.rename(package)
                client.log_artifacts(run_id, str(package), "final_model")
                version = mlflow.register_model(
                    f"runs:/{run_id}/final_model",
                    model_name,
                )
            if version.current_stage == "None":
                client.transition_model_version_stage(
                    name=model_name, version=version.version, stage="Staging"
                )

            state.update(
                status="published",
                version=int(version.version),
                stage=version.current_stage if version.current_stage != "None" else "Staging",
            )
            state.pop("error", None)
            _save_card(adapter, state)
        except Exception:
            logger.exception("Model publication failed for run %s", state["run_id"])
            state.pop("version", None)
            state.pop("stage", None)
            state.update(status="failed", error="模型發布失敗，可重試發布")
            try:
                _save_card(adapter, state)
            except RuntimeError:
                logger.exception("Could not update model card for run %s", state["run_id"])

        _write_state(path, state)
        return publication_status(adapter)
