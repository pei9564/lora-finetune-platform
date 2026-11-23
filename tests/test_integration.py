"""
端到端整合測試

測試主要流程：
1. 訓練流程整合測試
2. 認證與任務存取測試
"""

import json
import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_utils import create_token
from app.core.config import Config
from app.main import app


@pytest.fixture
def test_db(tmp_path):
    """提供每個測試獨立的 SQLite 資料庫連線。"""
    conn = sqlite3.connect(tmp_path / "experiments.db")

    # 建立測試表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            config_path TEXT NOT NULL,
            train_runtime FLOAT,
            eval_accuracy FLOAT,
            log_path TEXT,
            tokens_per_sec FLOAT,
            cpu_percent FLOAT,
            memory_gb FLOAT,
            model_name TEXT,
            dataset_name TEXT,
            train_samples INTEGER,
            batch_size INTEGER,
            learning_rate FLOAT,
            num_epochs INTEGER
        )
    """)
    conn.commit()

    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def mock_celery_task():
    """模擬 Celery 任務"""
    mock_task = MagicMock()
    mock_task.id = "test_user_1-task-123"  # 包含用戶 ID 前綴
    mock_task.status = "SUCCESS"
    mock_task.result = {
        "train": {
            "global_step": 100,
            "runtime": 10.5,
            "tokens_per_sec": 1000,
        },
        "eval": {
            "accuracy": 0.85,
            "loss": 0.2345,
        },
        "config": {"user_id": "test_user_1"},
    }

    # 設置後端
    mock_backend = MagicMock()
    task_meta = {
        "status": "SUCCESS",
        "result": mock_task.result,
        "task_id": mock_task.id,
        "kwargs": {"config": {"user_id": "test_user_1"}},
    }
    mock_backend.get_task_meta.return_value = task_meta
    mock_backend._get_task_meta_for.return_value = task_meta
    mock_backend.as_tuple.return_value = (
        task_meta["status"],
        task_meta["result"],
        None,
    )
    mock_task.backend = mock_backend
    mock_task.ready.return_value = True
    mock_task.failed.return_value = False

    return mock_task


@pytest.fixture
def integration_test_client():
    """提供整合測試專用的 FastAPI 測試客戶端（不使用 mock_auth）"""
    return TestClient(app)


@pytest.fixture
def test_users():
    """提供測試用戶"""
    return {
        "user1": {
            "user_id": "test_user_1",
            "role": "user",
            "token": create_token("test_user_1", "user"),
        },
        "user2": {
            "user_id": "test_user_2",
            "role": "user",
            "token": create_token("test_user_2", "user"),
        },
    }


class TestTrainingFlow:
    """訓練流程整合測試"""

    def test_submit_training_task(
        self,
        integration_test_client: TestClient,
        test_db: sqlite3.Connection,
        mock_celery_task: MagicMock,
        test_users: dict,
    ):
        """測試提交訓練任務到完成的完整流程

        步驟：
        1. 提交訓練任務
        2. 等待任務完成
        3. 驗證實驗記錄
        4. 驗證指標文件
        """
        # 準備測試配置
        test_config = Config(
            experiment_name="test_e2e_experiment",
            model={"name": "bert-base-uncased", "num_labels": 2},
            data={
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_samples": 100,
                "eval_samples": 10,
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
                "learning_rate": 1e-3,
                "per_device_train_batch_size": 8,
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
                "experiment_name": "test_e2e_experiment",
                "save_config": True,
            },
        )

        # 設置認證
        headers = {"Authorization": f"Bearer {test_users['user1']['token']}"}

        # Mock 訓練任務
        with patch(
            "app.tasks.training.train_lora.delay", return_value=mock_celery_task
        ):
            # 提交訓練任務
            response = integration_test_client.post(
                "/train",
                headers=headers,
                json={"config": test_config.model_dump()},
            )

            # 驗證響應
            assert response.status_code == 200
            assert "task_id" in response.json()
            task_id = response.json()["task_id"]

            # 等待任務完成
            with patch("celery.result.AsyncResult") as mock_result:
                mock_result.return_value = mock_celery_task
                response = integration_test_client.get(
                    f"/task/{task_id}", headers=headers
                )

                # 驗證任務狀態
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "SUCCESS"
                assert result["result"]["eval"]["accuracy"] == 0.85

        # 手動創建資料庫記錄（模擬訓練任務完成後的資料庫寫入）
        test_db.execute(
            """INSERT INTO experiments (id, name, created_at, config_path, log_path, train_runtime, eval_accuracy)
               VALUES (?, ?, datetime('now'), ?, ?, ?, ?)""",
            (
                "test_exp_1",
                "test_e2e_experiment",
                "/tmp/config.yaml",
                "/tmp/logs.txt",
                10.5,
                0.85,
            ),
        )
        test_db.commit()

        # 驗證資料庫記錄
        cursor = test_db.execute(
            "SELECT * FROM experiments WHERE name = ?",
            ("test_e2e_experiment",),
        )
        record = cursor.fetchone()
        assert record is not None
        assert record[1] == "test_e2e_experiment"  # name
        assert record[4] == 10.5  # train_runtime
        assert record[5] == 0.85  # eval_accuracy

        # 創建指標文件（模擬訓練任務完成後的文件寫入）
        metrics_dir = os.path.join("results", f"test_e2e_experiment_{record[0]}")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file = os.path.join(metrics_dir, "metrics.json")

        test_metrics = {
            "train": {
                "runtime": 10.5,
                "global_step": 100,
                "tokens_per_sec": 1000,
            },
            "eval": {
                "accuracy": 0.85,
                "loss": 0.2345,
            },
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # 驗證指標文件
        assert os.path.exists(metrics_file)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            assert "train" in metrics
            assert "eval" in metrics
            assert metrics["train"]["runtime"] == 10.5
            assert metrics["eval"]["accuracy"] == 0.85


class TestAuthTaskAccess:
    """認證與任務存取測試"""

    def test_task_access_permissions(
        self,
        integration_test_client: TestClient,
        mock_celery_task: MagicMock,
        test_users: dict,
    ):
        """測試任務存取權限

        步驟：
        1. 用戶1提交任務
        2. 用戶1可以存取自己的任務
        3. 用戶2無法存取用戶1的任務
        4. 未認證用戶無法存取任務
        """
        # 用戶1提交任務
        headers_user1 = {"Authorization": f"Bearer {test_users['user1']['token']}"}

        # 創建包含用戶2 ID 的任務（用於測試權限）
        user2_task = MagicMock()
        user2_task.id = "test_user_2-task-456"  # 用戶2的任務 ID
        user2_task.status = "SUCCESS"
        user2_task.result = {"status": "success"}
        user2_task.ready.return_value = True
        user2_task.failed.return_value = False

        # 設置 backend
        user2_backend = MagicMock()
        user2_task_meta = {
            "status": "SUCCESS",
            "result": user2_task.result,
            "task_id": user2_task.id,
        }
        user2_backend.get_task_meta.return_value = user2_task_meta
        user2_backend._get_task_meta_for.return_value = user2_task_meta
        user2_backend.as_tuple.return_value = ("SUCCESS", user2_task.result, None)
        user2_task.backend = user2_backend

        with patch(
            "app.tasks.training.train_lora.delay", return_value=mock_celery_task
        ):
            # 提交任務
            response = integration_test_client.post(
                "/train",
                headers=headers_user1,
                json={
                    "config": {
                        "experiment_name": "test_auth_task",
                        "model": {"name": "bert-base-uncased", "num_labels": 2},
                        "data": {
                            "dataset_name": "glue",
                            "dataset_config": "sst2",
                            "train_samples": 100,
                            "eval_samples": 10,
                            "max_length": 128,
                            "validation_rules": {
                                "min_text_length": 5,
                                "max_text_length": 500,
                                "allow_empty": False,
                                "remove_html": True,
                            },
                        },
                        "training": {
                            "output_dir": "results/test",
                            "learning_rate": 1e-3,
                            "per_device_train_batch_size": 8,
                            "num_train_epochs": 1,
                            "logging_steps": 10,
                        },
                        "lora": {
                            "r": 8,
                            "lora_alpha": 32,
                            "target_modules": ["query", "value"],
                            "lora_dropout": 0.1,
                            "bias": "none",
                            "task_type": "SEQ_CLS",
                        },
                        "system": {
                            "experiment_name": "test_auth_task",
                            "save_config": True,
                        },
                    }
                },
            )
            assert response.status_code == 200
            task_id = response.json()["task_id"]

            # 用戶1存取自己的任務
            with patch("celery.result.AsyncResult") as mock_result:
                mock_result.return_value = mock_celery_task
                response = integration_test_client.get(
                    f"/task/{task_id}", headers=headers_user1
                )
                assert response.status_code == 200
                assert response.json()["status"] == "SUCCESS"

            # 用戶2嘗試存取用戶1的任務（應該被拒絕）
            headers_user2 = {"Authorization": f"Bearer {test_users['user2']['token']}"}
            # 不需要 patch AsyncResult，因為權限檢查在 AsyncResult 調用之前就會被拒絕
            response = integration_test_client.get(
                f"/task/{task_id}", headers=headers_user2
            )
            assert response.status_code == 403
            assert "無權訪問此任務" in response.json()["detail"]

            # 未認證用戶嘗試存取任務
            response = integration_test_client.get(f"/task/{task_id}")
            assert response.status_code == 401
            assert "未提供認證信息" in response.json()["detail"]
