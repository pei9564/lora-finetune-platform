"""
訓練相關路由
"""

from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.jwt_utils import get_current_user
from app.core.config import (
    Config,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    SystemConfig,
    TrainingConfig,
)
from app.tasks.training import create_training_job

router = APIRouter(tags=["Training"])


class SubmittedTrainingConfig(Config):
    """Require the existing API sections while validating their contents."""

    experiment_name: str = Field(min_length=1, pattern=r"\S")
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoRAConfig
    system: SystemConfig


class TrainingRequest(BaseModel):
    """訓練請求模型"""

    config: SubmittedTrainingConfig


@router.post("/train")
async def start_training(
    request: TrainingRequest,
    http_request: Request,
    user: Dict = Depends(get_current_user),
) -> Dict[str, str]:
    """提交訓練任務，添加錯誤處理

    Args:
        request: 包含完整訓練配置的請求

    Returns:
        Dict[str, str]: 包含任務 ID 的字典

    Raises:
        HTTPException: 當任務提交失敗時
    """
    try:
        # 設置用戶 ID 並提交任務
        config_dict = request.config.model_dump()
        config_dict["user_id"] = user["user_id"]
        task = create_training_job(config=config_dict, request=http_request)

        if not task or not hasattr(task, "id"):
            raise HTTPException(status_code=500, detail="任務提交失敗")

        return {"task_id": task.id}

    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"提交訓練任務失敗: {e}")
        raise HTTPException(status_code=500, detail="內部服務器錯誤")
