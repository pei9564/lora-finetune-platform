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


def prepare_publication(adapter_path, run_id, run_name, config, metrics, model):
    path = Path(adapter_path).parent / "publication.json"
    base_revision = getattr(model.config, "_commit_hash", None)
    _write_state(path, {
        "status": "pending",
        "run_id": run_id,
        "run_name": run_name,
        "base_model": config.model.name,
        "base_revision": base_revision if isinstance(base_revision, str) else None,
        "num_labels": config.model.num_labels,
        "dataset": config.data.dataset_name,
        "metrics": {"accuracy": metrics["eval_accuracy"], "loss": metrics["eval_loss"]},
    })


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

            registry.save_model_card(ModelCard(
                id=f"task_{state['run_name']}",
                name=model_name,
                base_model=model_name,
                language="en",
                task="text-classification",
                description=f"LoRA fine-tuned model on {state['dataset']} dataset",
                metrics=state["metrics"],
                version=version.version,
                stage=version.current_stage if version.current_stage != "None" else "Staging",
                run_id=run_id,
                tags=[state["dataset"], "lora", model_name],
            ))
        except Exception:
            logger.exception("Model publication failed for run %s", state["run_id"])
            state.update(status="failed", error="模型發布失敗，可重試發布")
        else:
            state.update(status="published", version=int(version.version))
            state.pop("error", None)

        _write_state(path, state)
        return publication_status(adapter)
