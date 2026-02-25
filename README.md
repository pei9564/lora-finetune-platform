# Finetune Platform — Production-Grade LoRA Training & MLOps System

A **full-lifecycle MLOps platform** for LoRA fine-tuning of transformer models — built by one developer in 30 days, using AI as a force multiplier.

The system covers the complete pipeline: **Data → Training → Deployment → Observation**, with production engineering baked in from the start: async task isolation, experiment governance, RBAC security, and end-to-end observability.

**🏆 Honorable Mention — iThome Ironman 2025, Generative AI Track**
**📖 Series Articles: [Building an AI Fine-Tuning Platform: 30 Days of System Design & AI Collaboration](https://ithelp.ithome.com.tw/users/20151660/ironman/8264)**

---

## Why This Project

Most fine-tuning tutorials stop at `trainer.train()`. This project asks what comes after — the harder engineering questions that production systems actually demand:

- How do you run long training jobs without blocking the API?
- How do you track, compare, and govern model versions across experiments?
- How do you monitor a training system in production, not just dev?
- How do you deploy this whole stack repeatably?
- **How do you build all of this — alone, in 30 days?**

The answer to that last question is what makes this project different.

---

## Built with AI Collaboration

This platform was built through deep human-AI collaboration. Roughly **80% of core code and configuration was co-generated with ChatGPT-4o and Cursor** — not as a shortcut, but as a deliberate engineering methodology.

| Tool | Role |
|------|------|
| ChatGPT-4o | Architecture planning, design tradeoff discussions, documentation |
| Cursor | In-editor pair programming, refactoring, rapid iteration |

**The result:** a solo developer was able to architect, implement, and ship a system that would typically require a small backend team — in 30 days.

**What AI collaboration actually looks like in practice:**

AI is fast at generating plausible-looking code, but production systems demand more. Even when generated code passed syntax checks, it still required debugging, logic correction, and integration testing to actually work. As module count grew, early AI-generated code often lacked generality — test coverage and clear module boundaries became critical to keep things from collapsing under their own weight.

The approach that worked: **test first, then stack features.** AI accelerated the pace of exploration and seeing the right direction; real debugging and tests made the product land.

> Architecture decisions, prompt strategies, and human-AI tuning notes are documented in full in the [Ironman series](https://ithelp.ithome.com.tw/users/20151660/ironman/8264).

---

## Demo

End-to-end flow: cluster startup → login → submit training job → MLflow experiment recorded → model Registry → inference → Grafana monitoring.

<video src="docs/demo.mp4" controls width="100%">
  Your browser does not support the video tag.
</video>

---

## What Was Built

Over 30 days, the following modules were fully implemented and connected into a closed-loop pipeline.

| Module | Capabilities Delivered |
|--------|----------------------|
| Task Management | Celery + Redis job scheduling and status tracking |
| Experiment Tracking | MLflow auto-logging of params, metrics, and artifacts |
| Model Sharing | Model Registry + semantic recommendation API |
| Auth & Permissions | JWT + RBAC + Audit Log |
| Deployment | Helm chart + GitHub Actions CI/CD |
| Observability | Prometheus Exporter + Grafana Dashboard |
| Load Testing | Locust stress tests + performance monitoring |
| Tenant Isolation | Namespace + ResourceQuota |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI (Streamlit)                   │
│         Submit jobs · Monitor progress · Browse results     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────┐
│                     FastAPI (REST API)                      │
│   Auth (JWT/RBAC) · Task dispatch · Model search · Audit    │
└──────────┬─────────────────────────────────┬────────────────┘
           │ Celery task                      │ Prometheus /metrics
┌──────────▼──────────────┐      ┌───────────▼───────────────┐
│  Redis (Message Broker) │      │   Prometheus + Grafana    │
│  Task queue · Results   │      │   System & task metrics   │
└──────────┬──────────────┘      └───────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                    Celery Worker                            │
│                                                             │
│   load_dataset → tokenize → LoRA inject → train → eval      │
│          │                                      │           │
│   DataConfig validation              MLflow logging         │
│                                          │                  │
│                                 MLflow Registry             │
│                                 ModelCard JSON              │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack & Design Decisions

### Async Task Execution — Celery + Redis

Training jobs can run for minutes or hours. Blocking an HTTP request for that duration is not viable.

**Design:** The API immediately returns a `task_id`, and the job runs in a Celery worker process. The client polls `/task/{task_id}` for status updates. This decouples request handling from compute, and makes the system resilient — if a worker crashes, the task can be retried.

```python
# Auto-retry on OOM or timeout, with exponential backoff
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

Redis serves as both the Celery broker and the result backend, keeping the dependency footprint small.

### Parameter-Efficient Fine-Tuning — PEFT / LoRA

Full fine-tuning of large transformer models is compute-prohibitive for most use cases. LoRA injects trainable rank-decomposition matrices into attention layers, training only ~0.1–1% of parameters while preserving most of the model's original capability.

```python
lora_config = LoraConfig(
    r=config.lora.r,               # Rank: controls adapter capacity
    lora_alpha=config.lora.lora_alpha,
    target_modules=config.lora.target_modules,
    lora_dropout=config.lora.lora_dropout,
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
```

### Experiment Tracking & Model Governance — MLflow

Every training run automatically logs parameters, metrics, and artifacts to MLflow. The platform implements a full model lifecycle:

```
Training completes
      │
      ▼
MLflow run logged (params + metrics + model artifact)
      │
      ▼
Model registered to MLflow Registry
      │
      ▼
Stage: Staging ──► Production ──► Archived
      │
      ▼
ModelCard JSON saved (base_model, task, metrics, tags, run_id)
```

Any version can be reproduced, compared, or promoted/demoted via API.

### Configuration — Pydantic Models + YAML

All training parameters are validated through Pydantic models before a job starts, catching misconfiguration at submission time rather than mid-training.

```python
class DataConfig(BaseModel):
    dataset_name: str = Field(default="glue")
    dataset_config: str = Field(default="sst2")   # HuggingFace dataset subset
    train_samples: int = Field(default=500)
    eval_samples: int = Field(default=100)
    max_length: int = Field(default=128)
    validation_rules: dict = Field(default={...})
```

### Security — JWT + RBAC + Audit Log

The API is protected with JWT authentication and role-based access control. Every API call is automatically captured by a middleware layer and written to an SQLite audit log.

```
Request → AuditLogMiddleware → Route handler
               │
               ▼
    audit_log(user_id, role, method, path, status_code, timestamp)
```

### Observability — Prometheus + Grafana

| Metric | Type | Purpose |
|--------|------|---------|
| `task_success_total` | Counter | Track job outcomes |
| `task_failure_total` | Counter | Detect failure patterns |
| `task_queue_length` | Gauge | Identify backpressure |
| `task_duration_seconds` | Histogram | Latency distribution |
| `system_cpu_percent` | Gauge | Worker load |
| `system_memory_usage_gigabytes` | Gauge | Resource consumption |

### Multi-Hardware Support

Device selection is resolved at runtime so the same codebase runs on development laptops and GPU servers without code changes.

```python
def setup_device(config: Config) -> str:
    if config.training.device == "auto":
        if torch.cuda.is_available():            return "cuda"
        elif torch.backends.mps.is_available():  return "mps"   # Apple Silicon
        else:                                    return "cpu"
    return config.training.device
```

---

## API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Issue JWT token |
| `/train` | POST | JWT | Submit a fine-tuning job |
| `/task/{id}` | GET | JWT | Poll task status |
| `/experiments` | GET | JWT | List all MLflow experiments |
| `/experiments/mlflow/{run_id}` | GET | JWT | Fetch run details |
| `/models/search` | GET | JWT | Search model registry |
| `/models/recommend` | POST | JWT | Semantic model recommendation |
| `/models/transition` | POST | Admin | Promote/demote model stage |
| `/audit` | GET | Admin | Query audit log |
| `/metrics` | GET | — | Prometheus metrics endpoint |

---

## Project Structure

```
app/
├── api/routes/         # FastAPI route handlers
│   ├── auth.py         # Login / token issuance
│   ├── train.py        # Job submission
│   ├── task.py         # Status polling
│   ├── experiments.py  # MLflow experiment queries
│   ├── models.py       # Model search & recommendation
│   └── audit.py        # Audit log access
│
├── auth/               # JWT utilities, RBAC helpers
├── core/               # Config (Pydantic), MLflow init, logging
│
├── tasks/              # Celery task definitions
│   └── training.py     # train_lora task with retry logic
│
├── train/              # Training pipeline
│   ├── runner.py       # Orchestrates: data → LoRA → train → eval → registry
│   ├── preprocess.py   # Tokenization
│   └── evaluator.py    # Metrics computation + progress callback
│
├── models/
│   └── model_registry.py   # ModelCard schema + search/recommend logic
│
├── monitor/
│   ├── exporter.py         # Prometheus metrics
│   ├── audit_utils.py      # Middleware + SQLite audit log
│   └── system_metrics.py   # CPU/memory polling
│
├── data/
│   ├── validation.py       # Input data validation rules
│   ├── analysis.py         # Dataset statistics
│   └── versioning.py       # Dataset version tracking
│
└── tools/
    ├── checkpoint_manager.py
    ├── artifact_utils.py
    └── analyze_metrics.py
```

---

## Deployment

### Local Development (Docker Compose)

```bash
cp .env.example .env
docker compose up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI | 8000 | REST API |
| Streamlit UI | 8501 | Web interface |
| MLflow | 5001 | Experiment tracking UI |
| Prometheus | 9090 | Metrics scraping |
| Grafana | 3000 | Dashboards |
| Redis | 6379 | Broker + result backend |

### Kubernetes (Helm)

```bash
# Development
helm install finetune charts/finetune-platform -f charts/finetune-platform/values.yaml

# Production
helm upgrade finetune charts/finetune-platform \
  -f charts/finetune-platform/values.yaml \
  -f charts/finetune-platform/values.prod.yaml
```

### CI/CD (GitHub Actions)

| Trigger | Pipeline |
|---------|----------|
| Any PR / push | Lint (flake8) + Unit tests |
| Push to `main` | Lint + Tests + Helm dry-run |
| Tag `day-*` | Docker build + push to DockerHub |

---

## 30-Day Evolution

| Period | Focus | Key Deliverables |
|--------|-------|-----------------|
| Week 1 | Task & data pipeline | FastAPI + Celery architecture, data validation |
| Week 2 | Experiment reproducibility | MLflow tracking, Config YAML management |
| Week 3 | Model reuse & governance | Model Registry, recommendation API, RBAC |
| Week 4 | Deployment & monitoring | CI/CD, Helm, Prometheus + Grafana |
| Week 5 | Multi-tenancy & wrap-up | Namespace + Quota, load testing, Demo Day |

---

## Running Tests

```bash
pytest
pytest --cov=app tests/
pytest tests/test_api.py -v
```

Test coverage: API endpoints, authentication, error handling, Celery task logic, monitoring metrics.

---

## Key Engineering Trade-offs

**SQLite for audit log** — zero-ops, sufficient for single-node requirements. Would migrate to PostgreSQL for multi-node deployments.

**Celery `-P solo` in Docker Compose** — simplifies local development; production Helm chart configures proper worker concurrency.

**MLflow with SQLite backend** — adequate for development and small teams. `values.prod.yaml` is designed to point at an external database.

**Pydantic v2 for config validation** — strict typing catches config errors before any GPU time is spent.
