"""訓練任務定義"""

import os
from copy import deepcopy
from datetime import datetime
from time import perf_counter
from typing import Dict
from uuid import uuid4

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure

from fastapi import HTTPException, Request

from app.auth.jwt_utils import decode_token
from app.core.config import Config
from app.core.logger import setup_system_logger
from app.exceptions import (
    OutOfMemoryError,
    TrainingError,
    TrainingTimeoutError,
)
from app.tasks import celery_app
from app.train import (
    load_and_process_data,
    load_model_and_tokenizer,
    setup_device,
    setup_lora,
    setup_training,
    train_and_evaluate,
)
from app.train.publication import model_card_id, publication_status
from app.monitor.exporter import record_task_failure, record_task_success

# 配置任務錯誤日誌
task_logger = setup_system_logger(
    name="celery.task.error", log_file="logs/task_errors.log", console_output=True
)

DEFAULT_TENANT_NAMESPACE = "tenant-user"


def get_tenant_from_token(request: Request) -> str:
    """從請求的 JWT Token 解析租戶資訊。"""

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return DEFAULT_TENANT_NAMESPACE

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return DEFAULT_TENANT_NAMESPACE

    token = parts[1]
    try:
        payload = decode_token(token)
    except HTTPException:
        return DEFAULT_TENANT_NAMESPACE

    tenant = payload.get("tenant")
    if tenant:
        return tenant

    role = payload.get("role")
    return "tenant-admin" if role == "admin" else DEFAULT_TENANT_NAMESPACE


def create_training_job(config: Dict, request: Request):
    """建立訓練任務並注入租戶命名空間。"""

    tenant = get_tenant_from_token(request)
    job_config = deepcopy(config)
    metadata = job_config.get("metadata") or {}
    metadata["namespace"] = tenant
    job_config["metadata"] = metadata
    return train_lora.delay(config=job_config)


@task_failure.connect
def handle_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """處理任務失敗的信號處理器

    記錄任務失敗的詳細信息，包括：
    - 任務 ID
    - 錯誤類型
    - 錯誤訊息
    - 重試次數（如果有）
    """
    retry_count = kwargs.get("request", {}).get("retries", 0)
    task_name = sender.name if sender else "unknown"

    task_logger.error(
        f"Task {task_name}[{task_id}] failed (retry {retry_count}): "
        f"{exception.__class__.__name__} - {str(exception)}"
    )


@celery_app.task(
    autoretry_for=(OutOfMemoryError, SoftTimeLimitExceeded),
    retry_backoff=True,
    retry_backoff_max=600,  # 最大延遲 10 分鐘
    retry_jitter=True,  # 添加隨機變化以避免同時重試
    max_retries=3,
    soft_time_limit=3600,  # 1 小時超時
)
def train_lora(config: Dict) -> Dict:
    """執行 LoRA 訓練任務

    Args:
        config: 完整的訓練配置字典

    Returns:
        Dict: 包含訓練結果的字典

    Raises:
        OutOfMemoryError: 當訓練過程中發生記憶體不足
        TrainingTimeoutError: 當訓練超過時間限制
        TrainingError: 其他訓練相關錯誤
    """
    start_time = perf_counter()
    metadata = config.get("metadata", {}) or {}
    namespace = metadata.get("namespace", DEFAULT_TENANT_NAMESPACE)

    try:
        config = Config(**config)

        # 設置實驗目錄（使用實驗名稱和時間戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(config.experiment_name)
        exp_dir = os.path.join("results", f"{name}_{timestamp}_{uuid4().hex}")
        os.makedirs(exp_dir, exist_ok=True)

        # 設置輸出目錄
        artifacts_dir = os.path.join(exp_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        config.training.output_dir = artifacts_dir

        # 保存配置文件
        config_path = os.path.join(exp_dir, "config.yaml")
        config.save_yaml(config_path)

        # 設置訓練環境
        device = setup_device(config)
        model, tokenizer = load_model_and_tokenizer(config, device)
        train_dataset, eval_dataset = load_and_process_data(config, tokenizer)
        model = setup_lora(config, model, device)

        trainer = setup_training(config, model, train_dataset, eval_dataset, exp_dir)
        train_result, eval_result, mlflow_run_id = train_and_evaluate(
            config, trainer, config_path=config_path
        )

        duration = perf_counter() - start_time
        record_task_success(duration_seconds=duration)

        adapter_path = os.path.join(config.training.output_dir, "final_model")

        # 返回結果
        return {
            "status": "success",
            "train": {
                "global_step": train_result.global_step,
                "runtime": train_result.metrics["train_runtime"],
            },
            "eval": {"accuracy": eval_result["eval_accuracy"]},
            "model_id": model_card_id(mlflow_run_id),
            "mlflow_run_id": mlflow_run_id,
            "adapter_path": adapter_path,
            "publication": publication_status(adapter_path),
            "config": {
                "experiment_name": config.experiment_name,
                "user_id": config.user_id,
            },
            "metadata": {"namespace": namespace},
        }
    except RuntimeError as e:
        duration = perf_counter() - start_time
        record_task_failure(duration_seconds=duration)
        if "out of memory" in str(e).lower():
            raise OutOfMemoryError(f"訓練過程記憶體不足: {str(e)}")
        raise TrainingError(f"訓練過程發生錯誤: {str(e)}")
    except SoftTimeLimitExceeded:
        duration = perf_counter() - start_time
        record_task_failure(duration_seconds=duration)
        raise TrainingTimeoutError("訓練超過時間限制")
    except Exception as e:
        duration = perf_counter() - start_time
        record_task_failure(duration_seconds=duration)
        raise TrainingError(f"未預期的錯誤: {str(e)}")
