"""
測試配置和共用 fixtures
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

# Provide deterministic JWT config for tests before importing app modules
os.environ.setdefault("JWT_SECRET", "test-secret-key")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

# Ensure logging directory exists before app modules configure file handlers
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
# Ensure results directory exists before app modules access databases
os.makedirs(os.path.join(os.getcwd(), "results"), exist_ok=True)


# Mock Celery before any app imports
mock_celery_app = MagicMock()
mock_celery_app.task = lambda *args, **kwargs: lambda func: func
sys.modules["app.tasks.celery_app"] = mock_celery_app

from app.core.config import Config  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """設置測試環境"""
    # 確保資料庫目錄存在
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 確保日誌目錄存在，避免 FileHandler 建立失敗
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Mock Celery settings
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")

    # Mock 審計日誌相關操作
    def mock_noop(*args, **kwargs):
        return None

    def mock_get_audit_logs(*args, **kwargs):
        return [
            {
                "id": 1,
                "user_id": "test_user",
                "role": "admin",
                "action": "GET /test",
                "method": "GET",
                "path": "/test",
                "status_code": 200,
                "timestamp": int(time.time()),
            }
        ]

    monkeypatch.setattr("app.monitor.audit_utils.init_audit_table", mock_noop)
    monkeypatch.setattr("app.monitor.audit_utils.save_audit_log", mock_noop)
    monkeypatch.setattr("app.monitor.audit_utils.get_audit_logs", mock_get_audit_logs)

    # Mock 模型儲存相關操作
    monkeypatch.setattr("torch.save", mock_noop)
    monkeypatch.setattr("safetensors.torch.save_file", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_model", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_state", mock_noop)

    # 全局 Mock AsyncResult 以避免 DisabledBackend 问题
    def mock_async_result_global(task_id):
        owner = task_id.split("-task")[0]
        mock_task = MagicMock()
        mock_task.id = task_id
        mock_task.status = "SUCCESS"
        mock_task.result = {
            "status": "success",
            "train": {"global_step": 100},
            "eval": {"accuracy": 0.85},
            "config": {"user_id": owner},
        }
        mock_task.ready.return_value = True
        mock_task.failed.return_value = False

        mock_backend = MagicMock()
        task_meta = {
            "status": "SUCCESS",
            "result": mock_task.result,
            "task_id": task_id,
            "kwargs": {"config": {"user_id": owner}},
        }
        mock_backend.get_task_meta.return_value = task_meta
        mock_backend._get_task_meta_for.return_value = task_meta
        mock_backend.as_tuple.return_value = (
            task_meta["status"],
            task_meta["result"],
            None,
        )
        mock_task.backend = mock_backend

        return mock_task

    # 在所有可能的地方 patch AsyncResult
    monkeypatch.setattr("celery.result.AsyncResult", mock_async_result_global)
    monkeypatch.setattr("app.api.routes.task.AsyncResult", mock_async_result_global)


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch, tmp_path):
    import types
    from pathlib import Path

    class _FakeRun:
        def __init__(self, run_id="run-1"):
            self.info = types.SimpleNamespace(run_id=run_id, run_name=run_id)
            self.data = types.SimpleNamespace(metrics={}, params={}, tags={})

    class _FakeRunContext(_FakeRun):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeMlflowClient:
        def __init__(self, mlflow_module, *args, **kwargs):
            self._mlflow = mlflow_module

        def get_run(self, run_id):
            return self._mlflow.get_run(run_id)

        def search_runs(self, *args, **kwargs):
            return pd.DataFrame([])

        def search_model_versions(self, *args, **kwargs):
            return []

        def set_registered_model_alias(self, *args, **kwargs):
            return None

        def get_registered_model(self, *args, **kwargs):
            return None

        def create_registered_model(self, *args, **kwargs):
            return None

        def delete_run(self, run_id):
            return None

        def transition_model_version_stage(self, *args, **kwargs):
            return None

        def get_experiment_by_name(self, name):
            return self._mlflow.get_experiment_by_name(name)

        def get_experiment(self, experiment_id):
            return self._mlflow.get_experiment(experiment_id)

    class _FakeMlflowModule:
        def __init__(self):
            self._experiments = {}
            self._runs = {}
            self.exceptions = types.SimpleNamespace(MlflowException=Exception)
            self.entities = types.SimpleNamespace(
                ViewType=types.SimpleNamespace(ACTIVE_ONLY=0)
            )
            self.pytorch = types.SimpleNamespace(
                save_model=lambda model, path: Path(path).mkdir(
                    parents=True, exist_ok=True
                )
            )

        def set_tracking_uri(self, uri):
            self.tracking_uri = uri

        def set_tracking_token(self, token):
            self.tracking_token = token

        def get_experiment_by_name(self, name):
            experiment = self._experiments.get(name)
            if not experiment:
                return None
            return types.SimpleNamespace(
                experiment_id=experiment["id"],
                artifact_location=experiment["artifact_location"],
                name=name,
            )

        def create_experiment(self, name, artifact_location=None, tags=None):
            experiment_id = f"exp-{len(self._experiments) + 1}"
            self._experiments[name] = {
                "id": experiment_id,
                "artifact_location": artifact_location or str(tmp_path / "mlruns"),
            }
            return experiment_id

        def get_experiment(self, experiment_id):
            for data in self._experiments.values():
                if data["id"] == experiment_id:
                    return types.SimpleNamespace(
                        experiment_id=experiment_id,
                        artifact_location=data["artifact_location"],
                        name="experiment",
                    )
            return types.SimpleNamespace(
                experiment_id=experiment_id,
                artifact_location=str(tmp_path / "mlruns"),
                name="experiment",
            )

        def set_experiment(self, experiment_id):
            self._active_experiment = experiment_id

        def start_run(self, experiment_id=None, run_name=None):
            run_id = f"run-{len(self._runs) + 1}"
            run_name = run_name or run_id
            run = _FakeRunContext(run_id=run_id)
            run.info.run_name = run_name
            self._runs[run_id] = run
            return run

        def log_params(self, params):
            return None

        def log_metrics(self, metrics, step=None):
            return None

        def log_artifact(self, local_path, artifact_path=None):
            return None

        def log_artifacts(self, local_dir, artifact_path=None):
            return None

        def get_run(self, run_id):
            return self._runs.get(run_id, _FakeRun(run_id))

        def search_runs(self, *args, **kwargs):
            return pd.DataFrame([])

        def register_model(self, model_uri, name):
            return types.SimpleNamespace(
                name=name, version=1, run_id="run-1", current_stage="Production"
            )

    fake_mlflow = _FakeMlflowModule()

    def _client_factory(*args, **kwargs):
        return _FakeMlflowClient(fake_mlflow, *args, **kwargs)

    fake_mlflow.tracking = types.SimpleNamespace(MlflowClient=_client_factory)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    modules_to_patch = [
        "app.core.mlflow_config",
        "app.train.runner",
        "app.train.evaluator",
        "app.models.model_registry",
        "app.api.routes.mlflow",
    ]
    for module_path in modules_to_patch:
        module = __import__(module_path, fromlist=["mlflow"])
        monkeypatch.setattr(module, "mlflow", fake_mlflow)

    monkeypatch.setattr("app.core.mlflow_config.MlflowClient", _client_factory)
    monkeypatch.setattr("app.train.runner.MlflowClient", _client_factory, raising=False)
    monkeypatch.setattr(
        "app.api.routes.mlflow.MlflowClient", _client_factory, raising=False
    )

    def fake_init_mlflow():
        experiment = fake_mlflow.get_experiment_by_name("finetune-platform")
        if experiment is None:
            exp_id = fake_mlflow.create_experiment(
                "finetune-platform", artifact_location=str(tmp_path / "mlruns")
            )
            experiment = fake_mlflow.get_experiment(exp_id)
        config = {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "finetune-platform",
            "experiment_id": experiment.experiment_id,
            "artifact_location": experiment.artifact_location,
        }
        return config

    monkeypatch.setattr("app.core.mlflow_config.init_mlflow", fake_init_mlflow)
    monkeypatch.setattr("app.api.routes.mlflow.init_mlflow", fake_init_mlflow)
    monkeypatch.setattr("app.train.runner.init_mlflow", fake_init_mlflow)


@pytest.fixture(autouse=True)
def mock_transformers_offline(monkeypatch):
    from app.train import evaluator, preprocess, runner

    class _DummyTokenizer:
        pad_token_id = 0

        def encode(self, text):
            return list(range(max(len(str(text)) // 4, 1)))

        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                items = [texts]
            else:
                items = list(texts)
            max_length = kwargs.get("max_length", 128)
            return {
                "input_ids": [[1] * max_length for _ in items],
                "attention_mask": [[1] * max_length for _ in items],
                "length": [min(len(str(text)), max_length) for text in items],
            }

    class _DummyModel:
        def to(self, device):
            return self

        def parameters(self):
            class _Param:
                def __init__(self, requires_grad=True):
                    self.requires_grad = requires_grad

                def numel(self):
                    return 1

            return iter([_Param(True), _Param(False)])

        def print_trainable_parameters(self):
            return None

    dummy_tokenizer = _DummyTokenizer()
    dummy_model = _DummyModel()

    monkeypatch.setattr(
        runner.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=dummy_tokenizer),
    )
    monkeypatch.setattr(
        runner.AutoModelForSequenceClassification,
        "from_pretrained",
        MagicMock(return_value=dummy_model),
    )
    monkeypatch.setattr(
        runner,
        "get_peft_model",
        MagicMock(side_effect=lambda model, config: model),
    )
    metric_mock = MagicMock()
    metric_mock.compute.return_value = {"accuracy": 1.0}
    monkeypatch.setattr(evaluator.evaluate, "load", MagicMock(return_value=metric_mock))

    fake_dataset = {
        "train": Dataset.from_dict({"sentence": ["sample"] * 20, "label": [1, 0] * 10}),
        "validation": Dataset.from_dict(
            {"sentence": ["sample"] * 5, "label": [1, 0, 1, 0, 1]}
        ),
    }
    monkeypatch.setattr(
        preprocess, "load_dataset", MagicMock(return_value=fake_dataset)
    )


@pytest.fixture
def mock_auth():
    """提供測試用管理員使用者與 token"""
    from app.auth.jwt_utils import create_token

    token = create_token("test_user", "admin")
    return {"user_id": "test_user", "role": "admin", "token": token}


@pytest.fixture
def mock_token(mock_auth):
    """提供測試用 JWT token"""
    return mock_auth["token"]


@pytest.fixture
def auth_headers(mock_token):
    """提供認證 headers"""
    return {"Authorization": f"Bearer {mock_token}"}


@pytest.fixture
def test_client(mock_auth, auth_headers):
    """提供 FastAPI 測試客戶端"""
    client = TestClient(app)
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def mock_celery(monkeypatch):
    """Mock Celery 任務"""

    def create_task_result(task_id, status="SUCCESS", result=None, error=None):
        """创建一个任务结果对象"""
        owner = task_id.split("-task")[0]
        task = MagicMock()
        task.id = task_id  # 確保這是字符串
        task.status = status
        if error:
            task.result = error
        else:
            payload = (
                result.copy()
                if isinstance(result, dict)
                else {
                    "status": "success",
                    "train": {"global_step": 100},
                    "eval": {"accuracy": 0.85},
                }
            )
            payload.setdefault("config", {"user_id": owner})
            task.result = payload
        task.ready.return_value = status != "PENDING"
        task.failed.return_value = status == "FAILURE"

        # 设置后端
        task.backend = MagicMock()
        task_meta = {
            "status": status,
            "result": task.result,
            "task_id": task_id,
            "kwargs": {"config": {"user_id": owner}},
        }
        task.backend.get_task_meta.return_value = task_meta
        task.backend._get_task_meta_for.return_value = task_meta
        task.backend.as_tuple.return_value = (status, task.result, None)
        return task

    # 预定义的任务结果
    success_result = create_task_result(
        "test-task-123",
        status="SUCCESS",
        result={
            "status": "success",
            "train": {"global_step": 100},
            "eval": {"accuracy": 0.85},
        },
    )

    error_result = create_task_result(
        "error-task-123", status="FAILURE", error=ValueError("訓練數據集不能為空")
    )

    oom_result = create_task_result(
        "error-task-456",
        status="FAILURE",
        error=RuntimeError("GPU 記憶體不足: 已使用 15.0GB / 總計 16.0GB"),
    )

    pending_result = create_task_result("pending-task", status="PENDING")

    invalid_result = MagicMock()
    invalid_result.id = "invalid-task-id"
    invalid_result.backend = MagicMock()
    invalid_result.backend.get_task_meta.side_effect = Exception("Task not found")
    invalid_result.backend._get_task_meta_for.side_effect = Exception("Task not found")

    # 創建 mock train_lora 任務 - 確保 task.id 是字符串
    mock_train_lora = MagicMock()

    def mock_delay(*args, **kwargs):
        # 根據配置返回不同的任務結果
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            exp_name = kwargs["config"].get("experiment_name", "").lower()
            if "error" in exp_name:
                return error_result
            elif "oom" in exp_name:
                return oom_result
        return success_result

    mock_train_lora.delay = MagicMock(side_effect=mock_delay)
    mock_train_lora.apply_async = MagicMock(side_effect=mock_delay)
    mock_train_lora.__name__ = "train_lora"

    # Mock AsyncResult 类
    def mock_async_result_class(task_id):
        if task_id == "invalid-task-id":
            return invalid_result
        elif task_id == "error-task-123":
            return error_result
        elif task_id == "error-task-456":
            return oom_result
        elif task_id == "pending-task":
            return pending_result
        else:
            return create_task_result(task_id)

    # train.py 現在透過 create_training_job 提交任務，直接 mock 該函式
    def mock_create_training_job(config, request=None, **kwargs):
        return mock_delay(config=config)

    # 設置所有的 mock - 使用 try/except 避免模組導入問題
    monkeypatch.setattr(
        "app.api.routes.train.create_training_job", mock_create_training_job
    )
    monkeypatch.setattr("app.api.routes.task.AsyncResult", mock_async_result_class)
    monkeypatch.setattr("celery.result.AsyncResult", mock_async_result_class)

    # 嘗試 mock training 模組，如果失敗就跳過
    try:
        monkeypatch.setattr("app.tasks.training.train_lora", mock_train_lora)
    except (AttributeError, ImportError):
        pass

    return mock_train_lora


@pytest.fixture
def test_config():
    """提供測試用配置"""
    return Config(
        experiment_name="test_experiment",
        model={"name": "bert-base-uncased", "num_labels": 2},
        data={
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_samples": 20,
            "eval_samples": 5,
            "max_length": 128,
            "validation_rules": {
                "min_text_length": 5,
                "max_text_length": 500,
                "allow_empty": False,
                "remove_html": True,
            },
        },
        training={
            "output_dir": "results/test",
            "eval_strategy": "steps",
            "learning_rate": 1e-3,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 1,
            "logging_steps": 10,
        },
        lora={
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["query", "value"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "SEQ_CLS",
        },
        system={
            "experiment_name": "test_experiment",
            "save_config": True,
        },
    )


@pytest.fixture
def test_dataset():
    """提供測試用小型數據集"""
    data = {
        "sentence": [
            "This is a great movie!",
            "The film was terrible.",
            "I love this book so much!",
            "What a waste of time.",
            "Amazing performance by the actors!",
        ]
        * 4,  # 重複 4 次得到 20 筆數據
        "label": [1, 0, 1, 0, 1] * 4,  # 1=正面, 0=負面
    }
    return Dataset.from_pandas(pd.DataFrame(data))


@pytest.fixture
def empty_dataset():
    """提供空數據集"""
    return Dataset.from_pandas(pd.DataFrame({"sentence": [], "label": []}))


@pytest.fixture
def long_sequence_dataset():
    """提供超長序列數據集"""
    long_text = " ".join(["very"] * 1000) + " long text"  # 產生超長文本
    data = {"sentence": [long_text] * 5, "label": [1] * 5}
    return Dataset.from_pandas(pd.DataFrame(data))
