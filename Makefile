.PHONY: setup-conda run-local logs-local \
        data-analyze data-validate data-versions \
        analyze-metrics analyze-by-model analyze-by-dataset \
        lint lint-conda test test-v test-conda deps \
        start-services stop-services restart-services logs-services logs-service \
        docker-build docker-push \
        helm-dryrun helm-deploy helm-uninstall \
        k8s-setup k8s-build k8s-build-fast k8s-deploy k8s-quick-deploy \
        k8s-status k8s-logs k8s-restart k8s-scale k8s-verify k8s-cleanup k8s-full-cleanup \
        serve predict-health predict-text predict-positive predict-negative load-test \
        help

# ==============================================================================
# 🔧 環境設定與共用變數
# ==============================================================================

ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\1/p' .env)
endif

PYTHON_VERSION := 3.11
PYTHONPATH := $(PWD)
IMAGE ?= finetune-app:latest

HELM_RELEASE ?= finetune-platform
HELM_NAMESPACE ?= lora-system
HELM_CHART ?= charts/finetune-platform
HELM_VALUES ?= $(HELM_CHART)/values.yaml
HELM_PROD_VALUES ?= $(HELM_CHART)/values.prod.yaml
HELM_COMMON_FLAGS := -f $(HELM_VALUES) -f $(HELM_PROD_VALUES) --namespace $(HELM_NAMESPACE) --create-namespace

# 共用函數
define detect_env
	if command -v nvidia-smi &> /dev/null; then \
		ENV_NAME="lora-gpu"; \
	elif uname -m | grep -q "arm64"; then \
		ENV_NAME="lora-m3"; \
	else \
		ENV_NAME="lora-cpu"; \
	fi; \
	echo "📦 Environment: $$ENV_NAME";
endef

define check_conda
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝。請先執行：brew install --cask miniforge"; \
		exit 1; \
	fi
endef

define check_env_exists
	if ! conda env list | grep -q "$$ENV_NAME"; then \
		echo "❌ Conda 環境不存在，請先執行 make setup-conda"; \
		exit 1; \
	fi;
endef

# ==============================================================================
# 🧱 Conda 環境與本地訓練
# ==============================================================================

setup-conda:
	@echo "🔍 檢查 Conda 環境..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		if conda env list | grep -q "$$ENV_NAME"; then \
			echo "✅ 環境 $$ENV_NAME 已存在"; \
		else \
			echo "📦 建立環境 $$ENV_NAME..."; \
			conda create --name $$ENV_NAME python=$(PYTHON_VERSION) -y; \
		fi; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		pip install --upgrade pip && pip install -r requirements.txt; \
		echo "✅ 完成！" \
	'

run-local:
	@echo "🚀 啟動本地 LoRA 訓練..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		PYTHONPATH=$(PWD) python -u app/train_lora_v2.py $(ARGS)'

logs-local:
	@latest_dir=$$(ls -td results/*/ 2>/dev/null | head -n1); \
	if [ -f "$$latest_dir/logs.txt" ]; then \
		echo "📋 最新實驗日誌（20 行）:"; \
		tail -n 20 "$$latest_dir/logs.txt"; \
	else \
		echo "❌ 未找到實驗日誌，請先執行 make run-local"; \
	fi

# ==============================================================================
# 📦 資料與分析工具
# ==============================================================================

data-analyze:
	@$(call run_data_tool,"分析資料","analysis")

data-validate:
	@$(call run_data_tool,"驗證資料","validation")

data-versions:
	@$(call run_data_tool,"管理版本","versioning")

analyze-metrics:
	@echo "📊 分析實驗效能..."
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		PYTHONPATH=$(PWD) python -m app.tools.analyze_metrics $(ARGS)'

analyze-by-model:
	@$(MAKE) analyze-metrics ARGS="--group-by model_name"

analyze-by-dataset:
	@$(MAKE) analyze-metrics ARGS="--group-by dataset_name"

# ==============================================================================
# 🧪 測試與 Lint
# ==============================================================================

lint:
	@if [ -n "$$CI" ]; then echo "🧹 Linting (CI)"; flake8; else $(MAKE) lint-conda; fi

lint-conda:
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && flake8'

test:
	@if [ -n "$$CI" ]; then pytest; else $(MAKE) test-conda; fi

test-conda:
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && pytest tests/ -v'

test-v:
	@$(MAKE) test-conda ARGS="-v -s"

deps:
	@echo "📊 生成依賴圖 (docs/deps.svg)..."
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		pydeps app --only app --rmprefix app. -T svg -o docs/deps.svg'

# ==============================================================================
# 🐳 Docker 本地服務
# ==============================================================================

start-services:
	@echo "🚀 啟動 Docker 服務..."
	docker compose up --build -d
	@echo "✅ API: http://localhost:8000 | Grafana:3000 | MLflow:5001"

stop-services: 
	@docker compose down && echo "🛑 所有服務已停止"

restart-services: stop-services start-services
logs-services: 
	@docker compose logs -f api worker ui
logs-service:
	@docker compose logs -f $(service)

# ==============================================================================
# ☸️ Kubernetes 操作
# ==============================================================================

k8s-setup:
	@echo "☸️ 啟動 Minikube..."
	@minikube start --driver=docker --memory=4096 --cpus=2

k8s-build: 
	@./k8s/k8s.sh build
k8s-build-fast: 
	@./k8s/k8s.sh build-fast
k8s-deploy: 
	@./k8s/k8s.sh deploy
k8s-quick-deploy: k8s-setup k8s-build-fast k8s-deploy
k8s-status: 
	@./k8s/k8s.sh status
k8s-logs: 
	@./k8s/k8s.sh logs $(service)
k8s-restart: 
	@./k8s/k8s.sh restart
k8s-scale: 
	@./k8s/k8s.sh scale $(replicas)
k8s-cleanup: 
	@./k8s/k8s.sh cleanup
k8s-full-cleanup: 
	@./k8s/k8s.sh full-cleanup

# ==============================================================================
# 🚀 CI/CD & 部署
# ==============================================================================

docker-build:
	@echo "🐳 Build Docker image: $(IMAGE)"
	docker build -t $(IMAGE) .

docker-push:
	@echo "🚀 Push Docker image: $(IMAGE)"
	docker push $(IMAGE)

helm-dryrun:
	@helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) $(HELM_COMMON_FLAGS) --dry-run=client --debug

helm-deploy:
	@helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) $(HELM_COMMON_FLAGS)

helm-uninstall:
	@helm uninstall $(HELM_RELEASE) --namespace $(HELM_NAMESPACE)

# ==============================================================================
# 🤖 推論與測試
# ==============================================================================

serve:
	@echo "🚀 啟動推論服務..."
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		PYTHONPATH=$(PWD) python app/tasks/inference.py'

predict-health:
	@curl -s http://localhost:8002/health | python3 -m json.tool
predict-text:
	@curl -s -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d '{"text": "$(text)"}' | python3 -m json.tool
predict-positive:
	@$(MAKE) predict-text text="This movie was fantastic!"
predict-negative:
	@$(MAKE) predict-text text="This movie was terrible."

load-test:
	@echo "🐝 Running Locust load test..."
	@bash -c '$(detect_env); $(check_env_exists); \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && \
		locust -f tests/load_test.py --headless -u 5 -r 5'

# ==============================================================================
# 🧭 使用說明
# ==============================================================================

help:
	@echo ""
	@echo "📘 Finetune Platform Makefile — 常用指令一覽"
	@echo ""
	@echo "🧱 基礎環境"
	@echo "  make setup-conda        建立 Conda 環境並安裝依賴"
	@echo ""
	@echo "🚀 訓練與日誌"
	@echo "  make run-local          啟動 LoRA 訓練"
	@echo "  make logs-local         查看最新訓練日誌"
	@echo ""
	@echo "🧪 測試與 Lint"
	@echo "  make lint               代碼檢查"
	@echo "  make test               單元測試 (pytest)"
	@echo "  make deps               生成依賴圖 (docs/deps.svg)"
	@echo ""
	@echo "🐳 Docker"
	@echo "  make start-services     啟動 API / Worker / Grafana"
	@echo "  make stop-services      停止所有容器"
	@echo ""
	@echo "☸️  Kubernetes"
	@echo "  make k8s-quick-deploy   一鍵建構 + 部署"
	@echo "  make k8s-status         檢查叢集狀態"
	@echo ""
	@echo "🧰 CI/CD"
	@echo "  make docker-build       建構 Docker 映像"
	@echo "  make helm-dryrun        模擬 Helm 部署"
	@echo "  make helm-deploy        正式部署 Helm Chart"
	@echo ""
	@echo "🤖 推論與測試"
	@echo "  make serve              啟動推論服務"
	@echo "  make predict-text text='Hello world!'"
	@echo ""
	@echo "📊 資料與分析"
	@echo "  make data-analyze       分析資料集分布"
	@echo "  make analyze-metrics    分析實驗效能"
	@echo ""
	@echo "🐝 壓測工具"
	@echo "  make load-test          啟動 Locust 壓力測試"
	@echo ""
	@echo "💡 提示："
	@echo "  1️⃣ 先執行 make setup-conda 初始化環境"
	@echo "  2️⃣ 執行 make run-local 進行訓練"
	@echo "  3️⃣ make test / lint / helm-dryrun 驗證系統"
	@echo ""
