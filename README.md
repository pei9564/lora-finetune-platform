# Finetune Platform — 端到端 LoRA 訓練與實驗管理平台

本專案是一個基於生產環境思維設計的 LoRA (Low-Rank Adaptation) 微調平台，旨在整合從數據校驗、非同步訓練、實驗追蹤到模型治理的完整生命週期。

本專案榮獲 **2025 iThome 鐵人賽「生成式 AI 組」佳作**。

* **專案主題**：打造 AI 微調平台：從系統設計到 AI 協作的 30 天實戰筆記
* **開發模式**：本專案深度實踐了 AI 協作開發範式，全系統約 80% 核心代碼與配置由 **ChatGPT-4o** 與 **Cursor** 協作生成。開發過程中的架構決策、Prompt 指令集以及人機調優筆記，均詳細記錄於鐵人賽系列文章中。

---

## 系統核心架構

本平台採用解耦的雲原生分層架構，確保計算密集型任務（LLM Fine-tuning）與管理平面（Control Plane）的隔離，以適應生產環境的擴展需求：

* **接入與控制層 (Control Plane)**：
  * **API Layer**: 使用 FastAPI 構建，透過 Kubernetes Service 暴露接口。負責處理任務提交、JWT 驗證、RBAC 權限控管與任務狀態分發。

* **邏輯調度層 (Orchestration Layer)**：
  * **Task Orchestration**: 透過 Celery + Redis 實現非同步任務調度。Redis 作為 Message Broker 確保任務可靠分發，並針對長時訓練任務設計了狀態輪詢與錯誤重試邏輯。

* **執行運算層 (Execution Layer)**：
  * **Fine-tuning Engine**: 訓練腳本支援自動硬體檢測。在 Linux 環境優先使用 NVIDIA CUDA，在 macOS (M3/M4) 環境則自動啟用 Apple MPS 加速。
  * **Kubernetes Pod Resources Limit**: 嚴格控管運算資源，確保訓練任務的穩定性。

* **存儲與治理層 (Governance Layer)**：
  * **MLOps Integration**: 整合 MLflow Tracking 即時記錄實驗指標與產物，並透過 Model Registry 實現實驗版本化與階段狀態管理。

---

## 技術細節與功能實現

### 1. 實驗追蹤與模型治理 (MLOps)

系統不再僅僅產出權重文件，而是將每次訓練視為一個完整的實驗實體：

* **自動化 Model Card**: 每次訓練結束後，系統會自動產出符合規範的 JSON 元數據，作為模型治理與語義推薦的基礎。
* **生命週期管理**: 整合 MLflow Registry API，支援模型在 `Staging`（測試中）、`Production`（已部署）、`Archived`（已歸檔）各階段的狀態轉換，確保環境穩定性。

#### Model Card JSON 範例

```json
{
  "id": "chinese-sentiment-v1",
  "name": "Chinese Sentiment Model",
  "base_model": "bert-base-chinese",
  "language": "zh",
  "task": "sentiment",
  "description": "Fine-tuned BERT for Chinese movie reviews",
  "metrics": { "accuracy": 0.89 },
  "tags": ["中文", "情感分析", "bert"],
  "embedding": [0.1, 0.2, -0.05, 0.3]
}
```

### 2. 數據驗證與向量化推薦

* **Schema-Driven Data Validation**: 在進入訓練隊列前，對上傳數據進行格式檢查與分佈分析，避免無效任務佔用資源。
* **語義推薦系統**: 利用 Model Card 中的 `embedding` 向量實現語義相似度推薦。用戶可透過自然語言描述需求，系統將自動推薦最符合的適配器（Adapter）。

### 3. 可觀測性 (Observability)

* **Prometheus Exporter**: 監控 `task_queue_length`（積壓監控）、`task_duration_seconds`（效能分析）及 `task_failure_total`。
* **Grafana Dashboard**: 提供可視化面板，實時監測節點負載與訓練任務趨勢。

---

## 部署與自動化 (CI/CD)

本專案採用 Helm 作為部署標準：

* **Helm Chart 結構**: 分離 `values.yaml` (Dev) 與 `values.prod.yaml` (Production)，並透過 StatefulSet 維護 Redis 數據持久化。
* **GitHub Actions 工作流**:
* **CI**: 自動執行 Pytest 單元測試與 Linting。
* **CD**: 通過 Helm Dry-run 驗證並自動建置帶有版本標籤的 Docker 鏡像。

---

## 快速啟動範例

### 環境配置

於根目錄建立 `.env` 文件：

```env
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
JWT_SECRET=your_secret_key
```

### 使用 Helm 部署至 Kubernetes

```bash
helm install finetune-platform ./charts/finetune-platform -f ./charts/finetune-platform/values.yaml
```

---

## 參考資料

* **實戰筆記**：[打造 AI 微調平台：從系統設計到 AI 協作的 30 天實戰筆記](https://ithelp.ithome.com.tw/users/20151660/ironman/8264)
* **開發工具**：ChatGPT-4o, Cursor, GitHub Actions, Helm
