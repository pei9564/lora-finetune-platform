# Finetune Platform — 端到端 LoRA 訓練與實驗管理平台

一個由**一人獨立完成、歷時 30 天**設計、實作並交付的全生命週期 MLOps 平台。

涵蓋 LoRA 微調、非同步任務隔離、實驗治理、RBAC 安全控管，以及端到端可觀測性，形成「**資料 → 訓練 → 部署 → 觀測**」的完整閉環。

**🏆 2025 iThome 鐵人賽「生成式 AI 組」佳作**
**📖 系列文章：[打造 AI 微調平台：從系統設計到 AI 協作的 30 天實戰筆記](https://ithelp.ithome.com.tw/users/20151660/ironman/8264)**

---

## 為什麼做這個專案

大多數微調教學在 `trainer.train()` 之後就結束了。這個專案進一步回答了更困難的工程問題：

- 長時間的訓練任務如何不阻塞 API？
- 如何跨實驗追蹤、比較並治理模型版本？
- 如何在生產環境（而非只有開發環境）監控訓練系統？
- 如何讓整個技術棧可重複、可自動化地部署？
- **如何靠一個人，在 30 天內完成這一切？**

最後這個問題的答案，正是這個專案最不一樣的地方。

---

## 開發方法

一人於 30 天內完成，期間運用 AI pair programming（ChatGPT-4o + Cursor）加速原型開發、減少樣板程式碼撰寫。核心架構、模組邊界與測試套件皆由本人親自設計與撰寫 — AI 加快了探索的速度，但承重的工程決策皆出自我手。

| 工具 | 角色定位 |
|------|---------|
| ChatGPT-4o | 設計取捨討論、技術文件草稿 |
| Cursor | 編輯器內 pair programming、重構與快速迭代 |

**實務上長什麼樣子？**

AI 很擅長快速生成看起來合理的程式碼，但生產系統的要求更高。即使生成的程式碼能通過語法檢查，仍需要持續 debug、修正邏輯、進行整合測試，才能真正可用。隨著模組數量增加，清晰的模組邊界與完整的測試覆蓋，就是「持續前進」與「不斷重做」的分野。

真正有效的策略是：**先寫測試，再疊功能。** AI 讓探索更快；嚴謹的測試與 debug，才讓產品真正落地。

> 架構決策、Prompt 策略與工程筆記，均詳細記錄於[鐵人賽系列文章](https://ithelp.ithome.com.tw/users/20151660/ironman/8264)中。

---

## Demo 展示

完整端對端流程：叢集啟動 → 登入 → 提交訓練任務 → MLflow 記錄 → 模型 Registry → 推論 → Grafana 監控。

<video src="docs/demo.mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

---

## 建立了什麼

歷時 30 天，完整實作並串聯以下模組：

| 模組類別 | 已實作能力 |
|---------|----------|
| 任務管理 | Celery + Redis 任務排程與狀態追蹤 |
| 實驗追蹤 | MLflow 自動記錄參數、指標與模型產物 |
| 模型共享 | Model Registry + 語義推薦 API |
| 身分與權限 | JWT + RBAC + Audit Log |
| 部署流程 | Helm 部署 + GitHub Actions CI/CD |
| 可觀測性 | Prometheus Exporter + Grafana Dashboard |
| 穩定性測試 | Locust 壓測 + 效能監控 |
| 租戶隔離 | Namespace + ResourceQuota |

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (Streamlit)                       │
│          提交任務 · 監控進度 · 瀏覽實驗結果                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                     FastAPI (REST API)                      │
│    JWT/RBAC 認證 · 任務派發 · 模型搜尋 · 審計日誌              │
└──────────┬─────────────────────────────────┬────────────────┘
           │ Celery task                      │ Prometheus /metrics
┌──────────▼──────────────┐      ┌───────────▼───────────────┐
│   Redis（消息代理）      │      │   Prometheus + Grafana    │
│   任務佇列 · 結果存儲    │      │   系統與任務指標監控        │
└──────────┬──────────────┘      └───────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                    Celery Worker                            │
│                                                             │
│   載入資料集 → 分詞 → LoRA 注入 → 訓練 → 評估                  │
│         │                               │                   │
│   DataConfig 驗證                       MLflow 記錄          │
│                                         │                   │
│                                    MLflow Registry          │
│                                    ModelCard JSON           │
└─────────────────────────────────────────────────────────────┘
```

---

## 技術棧與設計決策

### 非同步任務執行 — Celery + Redis

訓練任務可能執行數分鐘到數小時。讓 HTTP 請求在此期間持續等待是不可行的。

**設計思路：** API 立即回傳 `task_id`，任務在 Celery worker 進程中執行。客戶端透過輪詢 `/task/{task_id}` 取得狀態更新。這樣的設計將請求處理與計算完全解耦，同時提升系統韌性 — 當 worker 崩潰時，任務可自動重試。

```python
# OOM 或逾時時自動重試，並帶指數退避
@celery_app.task(
    autoretry_for=(OutOfMemoryError, SoftTimeLimitExceeded),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=3,
    soft_time_limit=3600,
)
def train_lora(config_dict: dict) -> dict:
    ...
```

Redis 同時作為 Celery broker 與 result backend，保持最小的依賴規模。

### 參數高效微調 — PEFT / LoRA

對大型 transformer 模型進行全量微調在算力上難以負擔。LoRA 將可訓練的秩分解矩陣注入注意力層，僅訓練約 0.1–1% 的參數，同時保留模型大部分的原始能力。

```python
lora_config = LoraConfig(
    r=config.lora.r,                              # 秩：控制適配器容量
    lora_alpha=config.lora.lora_alpha,            # 縮放因子
    target_modules=config.lora.target_modules,   # 指定要適配的層
    lora_dropout=config.lora.lora_dropout,
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
```

### 實驗追蹤與模型治理 — MLflow

每次訓練執行都會自動將參數、指標與模型產物記錄至 MLflow。平台實作了完整的模型生命週期管理：

```
訓練完成
    │
    ▼
MLflow run 記錄（參數 + 指標 + 模型產物）
    │
    ▼
模型註冊至 MLflow Registry
    │
    ▼
階段：Staging ──► Production ──► Archived
    │
    ▼
ModelCard JSON 儲存（base_model, task, metrics, tags, run_id）
```

任何版本都可透過 API 重現、比較、晉升或降級。

### 配置管理 — Pydantic Models + YAML

所有訓練參數在任務提交前透過 Pydantic 模型驗證，在訓練開始前就攔截錯誤配置，而不是訓練到一半才發現問題。

```python
class DataConfig(BaseModel):
    dataset_name: str = Field(default="glue")
    dataset_config: str = Field(default="sst2")   # HuggingFace 資料集子集
    train_samples: int = Field(default=500)
    eval_samples: int = Field(default=100)
    max_length: int = Field(default=128)
    validation_rules: dict = Field(default={...})
```

配置可序列化為 YAML 並作為 MLflow artifact 記錄，確保每次執行完整可重現。

### 安全機制 — JWT + RBAC + 審計日誌

API 以 JWT 認證與角色存取控制保護。每個 API 呼叫由 middleware 層自動攔截，並寫入 SQLite 審計日誌。

```
請求 → AuditLogMiddleware → 路由處理器
             │
             ▼
  audit_log(user_id, role, method, path, status_code, timestamp)
```

為多使用者或團隊環境提供可追溯的操作紀錄。

### 可觀測性 — Prometheus + Grafana

平台暴露 `/metrics` 端點，由 Prometheus 抓取並在預配置的 Grafana Dashboard 中視覺化。

| 指標 | 類型 | 用途 |
|------|------|------|
| `task_success_total` | Counter | 追蹤任務成功率 |
| `task_failure_total` | Counter | 偵測失敗模式 |
| `task_queue_length` | Gauge | 識別系統壅塞 |
| `task_duration_seconds` | Histogram | 任務延遲分布 |
| `system_cpu_percent` | Gauge | Worker 負載 |
| `system_memory_usage_gigabytes` | Gauge | 資源消耗 |

### 多硬體支援

設備選擇在執行時動態決定，同一份程式碼無需修改即可在開發筆電與 GPU 伺服器上運行。

```python
def setup_device(config: Config) -> str:
    if config.training.device == "auto":
        if torch.cuda.is_available():            return "cuda"
        elif torch.backends.mps.is_available():  return "mps"   # Apple Silicon
        else:                                    return "cpu"
    return config.training.device
```

---

## API 一覽

| Endpoint | 方法 | 認證 | 說明 |
|----------|------|------|------|
| `/auth/login` | POST | — | 取得 JWT Token |
| `/train` | POST | JWT | 提交微調任務 |
| `/task/{id}` | GET | JWT | 輪詢任務狀態 |
| `/experiments` | GET | JWT | 列出所有 MLflow 實驗 |
| `/experiments/mlflow/{run_id}` | GET | JWT | 取得執行詳情 |
| `/models/search` | GET | JWT | 搜尋模型註冊表 |
| `/models/recommend` | POST | JWT | 語義模型推薦 |
| `/models/transition` | POST | Admin | 晉升/降級模型階段 |
| `/audit` | GET | Admin | 查詢審計日誌 |
| `/metrics` | GET | — | Prometheus 指標端點 |

---

## 專案結構

```
app/
├── api/routes/         # FastAPI 路由處理器
│   ├── auth.py         # 登入 / Token 發放
│   ├── train.py        # 任務提交
│   ├── task.py         # 狀態輪詢
│   ├── experiments.py  # MLflow 實驗查詢
│   ├── models.py       # 模型搜尋與推薦
│   └── audit.py        # 審計日誌存取
│
├── auth/               # JWT 工具、RBAC 輔助函式
├── core/               # 配置（Pydantic）、MLflow 初始化、日誌
│
├── tasks/              # Celery 任務定義
│   └── training.py     # train_lora 任務，含重試邏輯
│
├── train/              # 訓練流水線
│   ├── runner.py       # 編排：資料 → LoRA → 訓練 → 評估 → 註冊
│   ├── preprocess.py   # 分詞處理
│   └── evaluator.py    # 指標計算 + 進度 callback
│
├── models/
│   └── model_registry.py   # ModelCard schema + 搜尋/推薦邏輯
│
├── monitor/
│   ├── exporter.py         # Prometheus 指標
│   ├── audit_utils.py      # Middleware + SQLite 審計日誌
│   └── system_metrics.py   # CPU/記憶體輪詢
│
├── data/
│   ├── validation.py       # 輸入資料驗證規則
│   ├── analysis.py         # 資料集統計分析
│   └── versioning.py       # 資料集版本追蹤
│
└── tools/
    ├── checkpoint_manager.py
    ├── artifact_utils.py
    └── analyze_metrics.py
```

---

## 部署

### 本地開發（Docker Compose）

```bash
cp .env.example .env
docker compose up -d
```

| 服務 | Port | 用途 |
|------|------|------|
| FastAPI | 8000 | REST API |
| Streamlit UI | 8501 | 網頁操作介面 |
| MLflow | 5001 | 實驗追蹤 UI |
| Prometheus | 9090 | 指標抓取 |
| Grafana | 3000 | 儀表板 |
| Redis | 6379 | Broker + 結果後端 |

### Kubernetes（Helm）

```bash
# 開發環境
helm install finetune charts/finetune-platform -f charts/finetune-platform/values.yaml

# 正式環境
helm upgrade finetune charts/finetune-platform \
  -f charts/finetune-platform/values.yaml \
  -f charts/finetune-platform/values.prod.yaml
```

Helm Chart 一鍵佈建所有服務、PersistentVolumes、資源配額、Secrets 與命名空間隔離。

### CI/CD（GitHub Actions）

| 觸發條件 | 流水線 |
|---------|--------|
| 任意 PR / push | Lint (flake8) + 單元測試 |
| Push 至 `main` | Lint + 測試 + Helm dry-run |
| Tag `day-*` | Docker 建置 + 推送至 DockerHub |

---

## 30 天系統演進軌跡

| 週期 | 目標主題 | 關鍵成果 |
|------|---------|---------|
| Week 1 | 任務與資料流成形 | FastAPI + Celery 任務架構、資料驗證與序列化 |
| Week 2 | 實驗可重現性 | MLflow 實驗追蹤、Config YAML 管理 |
| Week 3 | 模型重用與治理 | Model Registry、推薦 API、RBAC |
| Week 4 | 部署與監控 | CI/CD、Helm、Prometheus + Grafana |
| Week 5 | 多租戶與收尾 | Namespace + Quota、壓測、Demo Day |

---

## 執行測試

```bash
# 所有測試
pytest

# 含覆蓋率報告
pytest --cov=app tests/

# 指定模組
pytest tests/test_api.py -v
```

測試涵蓋：API 端點、認證機制、錯誤處理、Celery 任務邏輯、監控指標。

---

## 主要工程取捨

**審計日誌使用 SQLite** — 簡單、零運維成本，滿足單節點審計需求。多節點部署時應遷移至 PostgreSQL。

**Docker Compose 中 Celery 使用 `-P solo`** — 簡化本地開發體驗；生產環境 Helm Chart 配置了正式的 worker 並行度。

**MLflow 使用 SQLite 後端** — 適用於開發環境與小型團隊。`values.prod.yaml` 設計為指向外部資料庫。

**Pydantic v2 進行配置驗證** — 嚴格型別檢查在消耗任何 GPU 時間之前就攔截配置錯誤。
