"""
配置系統
使用 Pydantic 進行參數驗證和管理
"""

import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """模型相關配置"""

    name: str = Field(default="distilbert-base-uncased", description="預訓練模型名稱")
    num_labels: int = Field(default=2, description="分類標籤數量")


class LoRAConfig(BaseModel):
    """LoRA 相關配置"""

    r: int = Field(default=8, gt=0, description="LoRA 秩")
    lora_alpha: int = Field(default=16, description="LoRA alpha 參數")
    target_modules: List[str] = Field(
        default=["q_lin", "v_lin"], description="目標模組列表"
    )
    lora_dropout: float = Field(default=0.1, ge=0, le=1, description="LoRA dropout 率")
    bias: str = Field(default="none", description="偏差類型")
    task_type: str = Field(default="SEQ_CLS", description="任務類型")


class DataConfig(BaseModel):
    """資料相關配置"""

    dataset_name: str = Field(default="glue", description="資料集名稱")
    dataset_config: str = Field(default="sst2", description="資料集配置")
    train_samples: int = Field(default=500, gt=0, description="訓練樣本數")
    eval_samples: int = Field(default=100, gt=0, description="驗證樣本數")
    max_length: int = Field(default=128, gt=0, description="最大序列長度")
    validation_rules: dict = Field(
        default={
            "min_text_length": 5,
            "max_text_length": 500,
            "allow_empty": False,
            "remove_html": True,
        },
        description="資料驗證規則",
    )


class TrainingConfig(BaseModel):
    """訓練相關配置"""

    output_dir: str = Field(default="./results", description="輸出目錄")
    eval_strategy: str = Field(default="epoch", description="評估策略")
    learning_rate: float = Field(default=5e-4, gt=0, allow_inf_nan=False, description="學習率")
    per_device_train_batch_size: int = Field(
        default=2, gt=0, description="每個設備的訓練批次大小"
    )
    num_train_epochs: int = Field(default=1, gt=0, description="訓練輪數")
    logging_steps: int = Field(default=10, description="日誌記錄步數")
    device: Optional[str] = Field(default=None, description="指定訓練設備")


class SystemConfig(BaseModel):
    """系統相關配置"""

    experiment_name: str = Field(default="default_experiment", description="實驗名稱")
    save_config: bool = Field(default=True, description="是否保存配置")


class Config(BaseModel):
    """總配置"""

    experiment_name: str = Field(default="default_experiment", description="實驗名稱")
    user_id: Optional[str] = Field(default=None, description="用戶 ID")
    model: ModelConfig = Field(default_factory=ModelConfig, description="模型配置")
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description="LoRA 配置")
    data: DataConfig = Field(default_factory=DataConfig, description="資料配置")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="訓練配置"
    )
    system: SystemConfig = Field(default_factory=SystemConfig, description="系統配置")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """從 YAML 文件載入配置"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_yaml(self, save_path: str) -> None:
        """保存配置到 YAML 文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict(), f, allow_unicode=True, sort_keys=False)


def load_config(config_path: str = "config/default.yaml") -> Config:
    """載入配置"""
    return Config.from_yaml(config_path)
