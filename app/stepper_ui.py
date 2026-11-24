"""
訓練任務管理 UI
使用 Streamlit 實現任務提交和進度追蹤
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st
import yaml

from app.core.settings import API_URL
from app.ui_messages import message


LANGUAGES = {"zh-TW": "繁體中文", "en": "English"}
STATUS_MESSAGES = {
    "PENDING": "pending",
    "STARTED": "running",
    "RUNNING": "running",
    "SUCCESS": "finished",
    "FINISHED": "finished",
    "FAILURE": "failed",
    "FAILED": "failed",
    "SCHEDULED": "scheduled",
    "ALL": "all",
}


def tr(key: str, *values) -> str:
    return message(st.session_state.get("ui_language", "zh-TW"), key, *values)


def status_label(status: str) -> str:
    key = STATUS_MESSAGES.get(status)
    return tr(key) if key else status


def load_default_config() -> Dict:
    """載入預設配置

    Returns:
        Dict: 預設配置
    """
    try:
        with open("config/default.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(tr("config_error", e))
        return {}


def login(username: str, password: str) -> Optional[Dict]:
    """登入並獲取 JWT token 和用戶資訊

    Args:
        username: 使用者名稱
        password: 密碼

    Returns:
        Dict: 包含 token 和用戶資訊的字典，如果登入失敗則返回 None
    """
    try:
        with st.spinner(tr("signing_in")):  # 添加載入指示器
            # 使用 API_URL 環境變數（在 Docker 中設置為 http://api:8000）
            response = requests.post(
                f"{API_URL}/auth/login",
                json={"username": username, "password": password},
                timeout=5,  # 設置 5 秒超時
            )

        if response.status_code == 200:
            data = response.json()
            return {
                "token": data["token"],
                "role": data["role"],
                "user_id": data["user_id"],
            }
        elif response.status_code == 401:
            st.error(tr("invalid_credentials"))
            return None
        else:
            st.error(tr("login_server_error", response.status_code))
            return None

    except requests.exceptions.Timeout:
        st.error(tr("login_timeout"))
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(tr("connection_error", str(e)))
        return None
    except Exception as e:
        st.error(tr("login_error", str(e)))
        return None


def get_task_status(task_id: str) -> Optional[Dict]:
    """查詢任務狀態

    Args:
        task_id: 任務 ID

    Returns:
        Dict: 任務狀態資訊，如果請求失敗則返回 None
    """
    try:
        # 從 session state 獲取 token
        token = st.session_state.get("jwt_token")
        if not token:
            st.error(tr("sign_in_required"))
            return None

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_URL}/task/{task_id}", headers=headers)
        return response.json()
    except Exception as e:
        st.error(tr("query_error", e))
        return None


def render_stepper(status: str):
    """渲染進度指示器

    Args:
        status: 任務狀態
    """
    # 定義所有步驟
    steps = [tr("pending"), tr("running"), tr("complete_or_failed")]
    current_step = 0

    # 根據狀態確定當前步驟
    if status == "PENDING":
        current_step = 0
    elif status == "STARTED":
        current_step = 1
    elif status in ["SUCCESS", "FAILURE"]:
        current_step = 2

    # 創建進度條容器
    cols = st.columns(len(steps))

    # 渲染每個步驟
    for i, (step, col) in enumerate(zip(steps, cols)):
        # 設置步驟樣式
        if i < current_step:  # 已完成
            color = "green"
            symbol = "✅"
        elif i == current_step:  # 當前
            color = "blue"
            symbol = "🔄"
            if status == "FAILURE" and i == 2:
                color = "red"
                symbol = "❌"
            elif status == "SUCCESS" and i == 2:
                color = "green"
                symbol = "✅"
        else:  # 未開始
            color = "gray"
            symbol = "⭕️"

        # 在每個 column 中顯示步驟
        with col:
            st.markdown(
                f'<div style="text-align: center; color: {color};">'
                f"<h4>{symbol} {step}</h4>"
                f"</div>",
                unsafe_allow_html=True,
            )


def check_auth() -> bool:
    """檢查是否已登入

    Returns:
        bool: 是否已登入
    """
    if "jwt_token" not in st.session_state:
        st.warning(tr("sign_in_required"))

        # 使用表單進行登入
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input(tr("username"), key="login_username")
            password = st.text_input(tr("password"), type="password", key="login_password")

            # 使用 columns 來對齊按鈕
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                submitted = st.form_submit_button(tr("sign_in"), use_container_width=True)
            with btn_col2:
                guest_login = st.form_submit_button(
                    tr("guest_sign_in"), use_container_width=True
                )

            if submitted or guest_login:
                # 如果是訪客登入，使用預設帳密
                if guest_login:
                    username = "guest"
                    password = "guest123"
                elif not username or not password:
                    st.error(tr("credentials_required"))
                    return False

                auth_data = login(username, password)
                if auth_data:
                    st.session_state.jwt_token = auth_data["token"]
                    st.session_state.user_role = auth_data["role"]
                    st.session_state.user_id = auth_data["user_id"]
                    st.success(
                        tr("login_success", auth_data['user_id'], auth_data['role'])
                    )
                    st.rerun()
                    return True

        return False
    return True


def submit_training_task(
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
) -> Optional[str]:
    """提交訓練任務

    Args:
        experiment_name: 實驗名稱
        model_name: 預訓練模型名稱
        dataset: 資料集名稱
        train_samples: 訓練樣本數
        eval_samples: 驗證樣本數
        batch_size: 批次大小
        lora_r: LoRA 秩
        lora_alpha: LoRA Alpha
        learning_rate: 學習率
        epochs: 訓練輪數
        device: 訓練設備

    Returns:
        Optional[str]: 任務 ID，如果提交失敗則返回 None
    """
    try:
        # 準備配置
        config = load_default_config()

        # 更新配置
        if experiment_name:
            config["experiment_name"] = experiment_name
        if model_name:
            config["model"]["name"] = model_name
        if dataset:
            # 處理資料集名稱
            if "/" in dataset:
                dataset_name, dataset_config = dataset.split("/")
            else:
                dataset_name = dataset
                dataset_config = None
            config["data"]["dataset_name"] = dataset_name
            if dataset_config:
                config["data"]["dataset_config"] = dataset_config
        if train_samples:
            config["data"]["train_samples"] = train_samples
        if eval_samples:
            config["data"]["eval_samples"] = eval_samples
        if batch_size:
            config["training"]["per_device_train_batch_size"] = batch_size
        if lora_r:
            config["lora"]["r"] = lora_r
        if lora_alpha:
            config["lora"]["lora_alpha"] = lora_alpha
        if learning_rate:
            config["training"]["learning_rate"] = learning_rate
        if epochs:
            config["training"]["num_train_epochs"] = epochs
        if device and device != "auto":
            config["training"]["device"] = device

        # 提交任務
        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        response = requests.post(
            f"{API_URL}/train",
            json={"config": config},
            headers=headers,
        )
        if response.status_code == 422:
            for error in response.json()["detail"]:
                field = ".".join(map(str, error["loc"][1:]))
                st.error(tr("field_error", field, error["msg"]))
            return None
        response.raise_for_status()
        return response.json().get("task_id")
    except Exception as e:
        st.error(tr("submission_error", e))
        return None


def render_task_form():
    """渲染任務提交表單"""
    st.markdown(tr("submit_heading"))

    # 載入預設配置
    default_config = load_default_config()

    # 創建表單
    with st.form("training_form"):
        # 實驗設置
        st.markdown(tr("experiment_heading"))
        experiment_name = st.text_input(
            tr("experiment_name"),
            value=default_config.get("experiment_name", "default_experiment"),
            help=tr("experiment_name_help"),
            key="training_experiment_name",
        )

        # 模型設置
        st.markdown(tr("model_heading"))
        model_name = st.selectbox(
            tr("base_model"),
            options=["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
            index=0,
            help=tr("base_model_help"),
            key="training_model_name",
        )

        # 資料設置
        st.markdown(tr("data_heading"))
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox(
                tr("dataset"),
                options=["glue/sst2", "glue/cola", "imdb"],
                index=0,
                help=tr("dataset_help"),
                key="training_dataset",
            )
            train_samples = st.number_input(
                tr("training_samples"),
                value=int(default_config.get("data", {}).get("train_samples", 500)),
                min_value=100,
                max_value=10000,
                step=100,
                help=tr("training_samples_help"),
                key="training_train_samples",
            )
        with col2:
            eval_samples = st.number_input(
                tr("evaluation_samples"),
                value=int(default_config.get("data", {}).get("eval_samples", 100)),
                min_value=50,
                max_value=2000,
                step=50,
                help=tr("evaluation_samples_help"),
                key="training_eval_samples",
            )
            batch_size = st.number_input(
                tr("batch_size"),
                value=int(
                    default_config.get("training", {}).get(
                        "per_device_train_batch_size", 2
                    )
                ),
                min_value=1,
                max_value=32,
                help=tr("batch_size_help"),
                key="training_batch_size",
            )

        # LoRA 設置
        st.markdown(tr("lora_heading"))
        col3, col4 = st.columns(2)
        with col3:
            lora_r = st.number_input(
                tr("lora_rank"),
                value=int(default_config.get("lora", {}).get("r", 8)),
                min_value=1,
                max_value=64,
                help=tr("lora_rank_help"),
                key="training_lora_r",
            )
            learning_rate = st.number_input(
                tr("learning_rate"),
                value=float(
                    default_config.get("training", {}).get("learning_rate", 5e-4)
                ),
                format="%.0e",
                help=tr("learning_rate_help"),
                key="training_learning_rate",
            )
        with col4:
            lora_alpha = st.number_input(
                "LoRA Alpha",
                value=int(default_config.get("lora", {}).get("lora_alpha", 16)),
                min_value=1,
                max_value=128,
                help=tr("lora_alpha_help"),
                key="training_lora_alpha",
            )
            epochs = st.number_input(
                tr("epochs"),
                value=int(
                    default_config.get("training", {}).get("num_train_epochs", 1)
                ),
                min_value=1,
                max_value=10,
                help=tr("epochs_help"),
                key="training_epochs",
            )

        # 系統設置
        st.markdown(tr("system_heading"))
        device = st.selectbox(
            tr("training_device"),
            options=["auto", "cuda", "mps", "cpu"],
            index=0,
            help=tr("training_device_help"),
            key="training_device",
        )

        # 提交按鈕
        submitted = st.form_submit_button(tr("submit_training"))
        if submitted:
            # 提交任務
            task_id = submit_training_task(
                experiment_name=experiment_name,
                model_name=model_name,
                dataset=dataset,
                train_samples=train_samples,
                eval_samples=eval_samples,
                batch_size=batch_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                epochs=epochs,
                device=None if device == "auto" else device,
            )

            if task_id:
                st.success(tr("submission_success", task_id))
                # 將任務 ID 存入 session state
                st.session_state.progress_task_id = task_id


def render_experiment_list():
    """渲染實驗列表"""
    # 標題列與重新整理按鈕並排
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown(tr("experiments_heading"))
    with col2:
        if st.button("🔄", help=tr("refresh")):
            st.rerun()

    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning(tr("sign_in_experiments"))
        return

    # 篩選條件
    with st.expander(tr("filters"), expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name_filter = st.text_input(tr("experiment_name"), help=tr("name_filter_help"), key="history_name_filter")
            min_accuracy = st.slider(
                tr("minimum_accuracy"),
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help=tr("minimum_accuracy_help"),
                key="history_min_accuracy",
            )
        with col2:
            max_runtime = st.number_input(
                tr("maximum_runtime"),
                min_value=0,
                value=0,
                help=tr("maximum_runtime_help"),
                key="history_max_runtime",
            )
            sort_by = st.selectbox(
                tr("sort_by"),
                options=["created_at", "name", "train_runtime", "eval_accuracy"],
                format_func=lambda x: {
                    "created_at": tr("created_at"),
                    "name": tr("experiment_name"),
                    "train_runtime": tr("training_time"),
                    "eval_accuracy": tr("evaluation_accuracy"),
                }[x],
                key="history_sort_by",
            )
        desc = st.checkbox(tr("descending"), value=True, key="history_desc")

    # 發送請求
    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}

    # 先獲取統計資訊
    try:
        stats_response = requests.get(
            f"{API_URL}/experiments/stats", headers=headers, timeout=5
        )
        stats = stats_response.json()

        # 顯示統計資訊
        if stats.get("total_experiments", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(tr("total_experiments"), stats["total_experiments"])
            with col2:
                st.metric(tr("average_accuracy"), f"{stats.get('avg_accuracy', 0):.2%}")
            with col3:
                st.metric(tr("best_accuracy"), f"{stats.get('best_accuracy', 0):.2%}")
            with col4:
                st.metric(tr("shortest_runtime"), f"{stats.get('min_runtime', 0):.1f}s")
        else:
            st.info(tr("no_experiments"))
            st.markdown("---")
            st.markdown(tr("getting_started"))
            st.markdown(tr("open_submit"))
            st.markdown(tr("configure_training"))
            st.markdown(tr("start_training"))
            return

    except Exception:
        st.warning(tr("stats_unavailable"))

        # 獲取實驗列表
        params = {
            "sort_by": sort_by,
            "desc": desc,
            "limit": 100,
        }
        if name_filter:
            params["name"] = name_filter
        if min_accuracy > 0:
            params["min_accuracy"] = min_accuracy
        if max_runtime > 0:
            params["max_runtime"] = max_runtime

        response = requests.get(
            f"{API_URL}/experiments", params=params, headers=headers, timeout=5
        )
        experiments = response.json()

        # 顯示實驗列表
        if experiments:
            st.markdown(tr("experiments_list"))
            # 創建實驗表格
            data = []
            for exp in experiments:
                data.append(
                    {
                        tr("experiment_name"): exp["name"],
                        tr("created_at"): datetime.fromisoformat(exp["created_at"]).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        tr("training_time"): f"{exp['train_runtime']:.1f}s",
                        tr("accuracy"): f"{exp['eval_accuracy']:.2%}",
                        "ID": exp["id"],
                    }
                )

            # 顯示表格
            st.dataframe(
                data,
                column_config={
                    "ID": st.column_config.TextColumn(
                        "ID",
                        help=tr("experiment_id_help"),
                        width="medium",
                    ),
                },
                hide_index=True,
                width="stretch",
            )
        else:
            if name_filter or min_accuracy > 0 or max_runtime > 0:
                st.info(tr("no_matching_experiments"))
            else:
                st.info(tr("no_experiments"))
                st.markdown("---")
                st.markdown(tr("getting_started"))
                st.markdown(tr("open_submit"))
                st.markdown(tr("configure_training"))
                st.markdown(tr("start_training"))

    except requests.exceptions.Timeout:
        st.error(tr("experiments_timeout"))
    except requests.exceptions.ConnectionError:
        st.error(tr("server_unreachable"))
    except Exception as e:
        st.error(tr("experiments_error", str(e)))
        st.markdown("---")
        st.markdown(tr("troubleshooting"))
        st.markdown(tr("refresh_page"))
        st.markdown(tr("check_connection"))
        st.markdown(tr("check_service"))


def render_task_progress():
    """渲染任務進度"""
    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning(tr("sign_in_progress"))
        return

    task_id = st.text_input(tr("task_id"), key="progress_task_id")
    check_status = st.button(tr("check_status"))

    # 如果有輸入 ID 且點擊查詢
    if task_id and check_status:
        status_placeholder = st.empty()

        try:
            while True:
                # 查詢狀態
                result = get_task_status(task_id)
                if not result:
                    st.error(tr("task_unavailable"))
                    break

                # 清空佔位元件並顯示新狀態
                with status_placeholder.container():
                    st.markdown("---")

                    # 安全地獲取狀態
                    status = result.get("status")
                    if not status:
                        st.error(tr("missing_task_status"))
                        break

                    st.markdown(tr("task_status", status_label(status)))
                    render_stepper(status)

                    # 如果有結果，顯示
                    if "result" in result:
                        st.markdown("---")
                        st.markdown(tr("training_result"))
                        st.json(result["result"])
                    # 如果有錯誤，顯示
                    elif "error" in result:
                        st.markdown("---")
                        st.error(tr("task_error", result.get('error', tr("unknown_error"))))

                    # 根據狀態顯示不同的提示
                    if status == "PENDING":
                        st.info(tr("task_pending"))
                    elif status == "STARTED":
                        st.info(tr("task_running"))
                    elif status == "SUCCESS":
                        publication = result.get("result", {}).get("publication", {})
                        if publication.get("status") in ("failed", "pending"):
                            st.warning(tr("publication_pending"))
                            st.session_state["retry_publication_task"] = task_id
                        else:
                            st.success(tr("task_complete"))
                            st.session_state.pop("retry_publication_task", None)
                    elif status == "FAILURE":
                        st.error(tr("task_failed"))
                    else:
                        st.warning(tr("unknown_status", status))

                # 如果任務完成或失敗，停止輪詢
                if status in ["SUCCESS", "FAILURE"]:
                    break

                # 等待 2 秒後再次查詢
                time.sleep(2)

        except Exception as e:
            st.error(tr("task_query_error", str(e)))
            st.markdown("---")
            st.markdown(tr("troubleshooting"))
            st.markdown(tr("check_task_id"))
            st.markdown(tr("check_connection"))
            st.markdown(tr("retry_later"))

    if task_id and st.session_state.get("retry_publication_task") == task_id:
        if st.button(tr("retry_publication")):
            try:
                response = requests.post(
                    f"{API_URL}/task/{task_id}/publication/retry",
                    headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                    timeout=120,
                )
                response.raise_for_status()
                publication = response.json()
                if publication["status"] == "published":
                    st.success(tr("publication_success", publication['version']))
                    st.session_state.pop("retry_publication_task", None)
                else:
                    st.error(tr("publication_error", publication["error"]))
            except requests.RequestException as exc:
                st.error(tr("publication_error", exc))


def main():
    """主函數"""
    st.sidebar.selectbox(
        "Language / 介面語言",
        options=list(LANGUAGES),
        format_func=LANGUAGES.__getitem__,
        key="ui_language",
    )
    st.title(tr("app_title"))

    # 檢查登入狀態
    if not check_auth():
        return

    # 添加頁籤
    tab1, tab2, tab3 = st.tabs([tr("submit_tab"), tr("progress_tab"), tr("experiments_tab")])

    # 提交任務頁籤
    with tab1:
        render_task_form()

    # 追蹤進度頁籤
    with tab2:
        render_task_progress()

    # 實驗記錄頁籤
    with tab3:
        render_mlflow_experiments()


def render_mlflow_experiments():
    """渲染實驗記錄頁面"""
    st.title(tr("mlflow_heading"))

    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning(tr("sign_in_mlflow"))
        return

    # API endpoints
    API_BASE = API_URL  # 使用環境變數中的 API URL
    # 使用環境變數或預設值，並確保使用正確的端口
    MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")
    # 如果是容器內部的 URL，轉換為外部訪問的 URL
    if MLFLOW_UI == "http://mlflow:5000":
        MLFLOW_UI = "http://localhost:5001"

    def format_time(timestamp):
        """Format timestamp to readable datetime."""
        if not timestamp:
            return "-"
        try:
            # 嘗試解析 ISO 格式時間字符串
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            try:
                # 如果不是 ISO 格式，嘗試作為數字時間戳處理
                ts = float(timestamp)
                return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return "-"

    def format_duration(start_time, end_time):
        """Calculate and format duration between timestamps."""
        if not (start_time and end_time):
            return "-"
        try:
            # 嘗試解析 ISO 格式時間字符串
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration = (end_dt - start_dt).total_seconds()
        except (ValueError, AttributeError):
            try:
                # 如果不是 ISO 格式，嘗試作為數字時間戳處理
                start_ts = float(start_time)
                end_ts = float(end_time)
                duration = (end_ts - start_ts) / 1000
            except (ValueError, TypeError):
                return "-"

        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration / 60:.1f}m"
        else:
            return f"{duration / 3600:.1f}h"

    # Sidebar filters
    st.sidebar.title(tr("sidebar_filters"))
    run_statuses = ["FINISHED", "RUNNING", "FAILED", "SCHEDULED", "ALL"]
    status_labels = {status: status_label(status) for status in run_statuses}
    status_filter = st.sidebar.selectbox(
        tr("run_status"),
        run_statuses,
        format_func=status_labels.__getitem__,
        key="mlflow_status_filter",
    )
    limit = st.sidebar.number_input(tr("run_limit"), min_value=1, max_value=100, value=10, key="mlflow_limit")

    # Fetch experiment runs
    try:
        params = {"limit": limit}
        if status_filter != "ALL":
            params["status"] = status_filter

        # 從 session state 獲取 token
        token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        response = requests.get(
            f"{API_BASE}/mlflow",
            params=params,
            headers=headers,
            timeout=15,  # 增加超時時間到 15 秒
        )

        if response.status_code == 200:
            data = response.json()

            # 移除調試信息顯示

            if not data.get("runs") or len(data["runs"]) == 0:
                st.info(tr("no_mlflow_runs"))
                st.markdown("---")
                st.markdown(tr("tip"))
                st.markdown(tr("create_runs"))
                st.markdown(tr("check_mlflow"))
                st.markdown(tr("open_mlflow", MLFLOW_UI))
                return
        else:
            st.warning(tr("mlflow_http_error", response.status_code))
            st.info(tr("mlflow_unavailable"))
            st.markdown("---")
            st.markdown(tr("possible_causes"))
            st.markdown(tr("mlflow_starting"))
            st.markdown(tr("api_restart_needed"))
            st.markdown(tr("check_mlflow_ui", MLFLOW_UI))
            return
    except requests.exceptions.ConnectionError:
        st.error(tr("api_unreachable"))
        st.info(tr("mlflow_unavailable"))
        st.markdown("---")
        st.markdown(tr("solutions"))
        st.markdown(tr("check_api"))
        st.markdown(tr("restart_api"))
        st.markdown(tr("check_mlflow_ui", MLFLOW_UI))
        return
    except requests.exceptions.Timeout:
        st.warning(tr("api_timeout"))
        st.info(tr("mlflow_unavailable"))
        st.markdown("---")
        st.markdown(tr("retry_or_check"))
        return
    except Exception as e:
        st.error(tr("unexpected_error", str(e)))
        st.info(tr("mlflow_unavailable"))
        st.markdown("---")
        st.markdown(tr("check_logs"))
        return

    # 如果成功獲取資料，顯示實驗列表
    if data.get("runs") and len(data["runs"]) > 0:
        # Create DataFrame for better display
        runs_data = []
        for run in data["runs"]:
            runs_data.append(
                {
                    "Run ID": f"[{run.get('run_id', '-')}]({MLFLOW_UI}/#/experiments/{data.get('experiment_id', '1')}/runs/{run.get('run_id', '-')})",
                    tr("status"): run.get("status", "-"),
                    tr("started_at"): format_time(run.get("start_time")),
                    tr("duration"): format_duration(
                        run.get("start_time"), run.get("end_time")
                    ),
                    tr("model"): run.get("params", {}).get("model_name", "-"),
                    tr("accuracy"): f"{float(run.get('metrics', {}).get('eval_accuracy') or 0):.4f}",
                    tr("training_time"): f"{float(run.get('metrics', {}).get('training_time') or 0):.1f}s",
                    tr("batch_size"): run.get("params", {}).get("batch_size", "-"),
                    tr("learning_rate"): run.get("params", {}).get("learning_rate", "-"),
                }
            )

        df = pd.DataFrame(runs_data)

        # Display metrics summary
        st.subheader(tr("recent_runs"))
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_acc = pd.to_numeric(df[tr("accuracy")], errors="coerce").mean()
            st.metric(tr("average_accuracy"), f"{avg_acc:.4f}")

        with col2:
            success_rate = (df[tr("status")] == "FINISHED").mean() * 100
            st.metric(tr("success_rate"), f"{success_rate:.1f}%")

        with col3:
            total_runs = len(df)
            st.metric(tr("total_runs"), total_runs)

        # Display runs table
        st.dataframe(
            df.assign(**{tr("status"): df[tr("status")].map(status_label)}),
            column_config={
                "Run ID": st.column_config.Column(
                    "Run ID",
                    help=tr("run_link_help"),
                    width="small",
                ),
                tr("status"): st.column_config.Column(
                    tr("status"),
                    help=tr("run_status"),
                    width="small",
                ),
                tr("accuracy"): st.column_config.NumberColumn(
                    tr("accuracy"),
                    help=tr("accuracy_help"),
                    format="%.4f",
                    width="small",
                ),
            },
            hide_index=True,
        )

        # Add MLflow UI link
        st.markdown("---")
        st.markdown(
            tr("run_details", MLFLOW_UI)
        )


if __name__ == "__main__":
    main()
