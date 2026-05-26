from __future__ import annotations

import copy
import json
import math
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

TARGET_COLS = [f"vib_{idx:02d}" for idx in range(1, 15)]
DRIFT_TIME_FEATURE_COL = "time_since_start_days"
TIME_FEATURE_COLS: list[str] = [DRIFT_TIME_FEATURE_COL]
POWER_MODE_CODE_COL = "power_mode_code"
HORIZON_STEP_FEATURE_COL = "horizon_step_norm"
ANOMALY_COLS = ["stage1_anomaly", "stage2_anomaly", "stage_any_anomaly"]
CORE_COLS = ["timestamp", "mode", "power_mode", "is_target_mode", *TARGET_COLS, *ANOMALY_COLS]
SUPPORTED_MODEL_KEYS = ("sarima", "var", "lightgbm", "tcn", "transformer")


@dataclass
class Part2Config:
    prepared_path: Path = Path("data") / "prepared_year_after_preprocessing.csv"
    use_only_target_mode: bool = False
    keep_anomalies: bool = True
    use_only_complete_months: bool = True
    model_rule: str | None = "10s"
    power_mode_filter: str | None = None
    split_segments_by_power_mode: bool = False

    lookback_steps: int = 240
    horizon_steps: int = 7 * 24 * 120
    window_step: int = 120

    valid_months: int = 1
    test_months: int = 1
    benchmark_target_start: str | None = None
    enabled_models: tuple[str, ...] = SUPPORTED_MODEL_KEYS

    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    primary_quantile: float = 0.50
    conformal_alpha: float = 0.10

    active_targets: list[str] = field(default_factory=lambda: TARGET_COLS.copy())
    lag_steps: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 120, 240)
    rolling_windows: tuple[int, ...] = (4, 16, 64, 240)

    stat_history_limit: int = 720
    max_search_windows_stat: int = 4
    max_benchmark_test_windows: int = 1

    max_train_windows_ml: int | None = 20_000
    max_valid_windows_ml: int | None = 1

    max_train_windows_tcn: int | None = 5_000
    max_valid_windows_tcn: int | None = 4
    max_train_windows_transformer: int | None = 4_000
    max_valid_windows_transformer: int | None = 4

    sarima_orders: tuple[tuple[int, int, int], ...] = (
        (1, 0, 0),
        (2, 0, 0),
        (1, 0, 1),
        (2, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 0),
    )
    sarima_seasonal_orders: tuple[tuple[int, int, int, int], ...] = ((0, 0, 0, 0),)
    var_maxlags_grid: tuple[int, ...] = (4, 8, 12, 16, 24)

    lgbm_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 60,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ]
    )

    tcn_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "channels": (64, 64, 96, 96, 128),
                "kernel_size": 5,
                "dropout": 0.12,
                "lr": 5e-4,
                "batch_size": 96,
                "epochs": 22,
                "weight_decay": 1e-4,
            },
            {
                "channels": (64, 96, 96, 128, 128),
                "kernel_size": 5,
                "dropout": 0.15,
                "lr": 4e-4,
                "batch_size": 64,
                "epochs": 24,
                "weight_decay": 2e-4,
            },
            {
                "channels": (64, 64, 64, 96, 96, 128),
                "kernel_size": 3,
                "dropout": 0.15,
                "lr": 5e-4,
                "batch_size": 96,
                "epochs": 24,
                "weight_decay": 2e-4,
            },
            {
                "channels": (96, 96, 128, 128, 160),
                "kernel_size": 7,
                "dropout": 0.18,
                "lr": 3e-4,
                "batch_size": 64,
                "epochs": 26,
                "weight_decay": 3e-4,
            },
        ]
    )
    tcn_rollout_steps: int = 24
    tcn_patience: int = 5
    tcn_scheduler_patience: int = 2
    tcn_scheduler_factor: float = 0.5
    tcn_grad_clip_norm: float = 1.0

    transformer_grid: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.10,
                "lr": 8e-4,
                "batch_size": 128,
                "epochs": 18,
                "weight_decay": 1e-4,
            },
            {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.12,
                "lr": 6e-4,
                "batch_size": 96,
                "epochs": 20,
                "weight_decay": 1e-4,
            },
            {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.15,
                "lr": 4e-4,
                "batch_size": 64,
                "epochs": 22,
                "weight_decay": 2e-4,
            },
            {
                "d_model": 192,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 384,
                "dropout": 0.15,
                "lr": 3e-4,
                "batch_size": 48,
                "epochs": 24,
                "weight_decay": 2e-4,
            },
        ]
    )
    transformer_rollout_steps: int = 24
    transformer_patience: int = 5
    transformer_scheduler_patience: int = 2
    transformer_scheduler_factor: float = 0.5
    transformer_grad_clip_norm: float = 1.0

    interval_scale_grid: tuple[float, ...] = (0.60, 0.75, 0.90, 1.00, 1.10)

    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    show_progress: bool = True
    progress_leave: bool = False


@dataclass
class ExperimentBundle:
    frame: pd.DataFrame
    windows: pd.DataFrame
    train_windows: pd.DataFrame
    valid_windows: pd.DataFrame
    test_windows: pd.DataFrame
    benchmark_test_windows: pd.DataFrame
    complete_months: list[str]
    train_months: list[str]
    valid_months_list: list[str]
    test_months_list: list[str]
    power_mode_values: list[str]
    model_step_seconds: int


def resolve_enabled_model_keys(enabled_models: Any) -> list[str]:
    if enabled_models is None:
        return list(SUPPORTED_MODEL_KEYS)
    if isinstance(enabled_models, str):
        enabled_models = [enabled_models]
    resolved: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for item in enabled_models:
        key = str(item).strip().lower()
        if key not in SUPPORTED_MODEL_KEYS:
            unknown.append(str(item))
            continue
        if key in seen:
            continue
        resolved.append(key)
        seen.add(key)
    if unknown:
        supported = ", ".join(SUPPORTED_MODEL_KEYS)
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unsupported models in enabled_models: {unknown_text}. Supported models: {supported}.")
    if not resolved:
        supported = ", ".join(SUPPORTED_MODEL_KEYS)
        raise ValueError(f"enabled_models must contain at least one supported model. Supported models: {supported}.")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_progress_value(value: Any) -> str:
    if isinstance(value, float):
        if value >= 100 or value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_progress_value(item) for item in value) + "]"
    return str(value)


def format_params_for_progress(params: dict[str, Any], max_length: int = 96) -> str:
    parts = [f"{key}={format_progress_value(value)}" for key, value in params.items()]
    text = ", ".join(parts)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def create_progress(
    cfg: Part2Config,
    iterable: Any = None,
    *,
    total: int | None = None,
    desc: str = "",
    leave: bool | None = None,
) -> tqdm:
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        leave=cfg.progress_leave if leave is None else leave,
        disable=not cfg.show_progress,
        dynamic_ncols=True,
    )


def infer_step_seconds_from_series(timestamps: pd.Series) -> int:
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 1
    mode = diffs.mode()
    if not mode.empty:
        return int(mode.iat[0])
    return int(diffs.median())


def infer_rule_seconds(rule: str) -> int:
    return int(pd.to_timedelta(rule).total_seconds())


def duration_to_steps(duration: str | pd.Timedelta, rule: str) -> int:
    duration_delta = pd.to_timedelta(duration)
    rule_delta = pd.to_timedelta(rule)
    if rule_delta <= pd.Timedelta(0):
        raise ValueError("rule must be a positive timedelta.")
    return int(duration_delta / rule_delta)


def add_drift_time_feature_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    start_ts = frame["timestamp"].min()
    frame[DRIFT_TIME_FEATURE_COL] = (
        (frame["timestamp"] - start_ts).dt.total_seconds() / 86_400.0
    ).astype(np.float32)
    return frame


def power_mode_sort_key(value: str) -> tuple[int, int, int, str]:
    match = re.fullmatch(r"(\d+)-(\d+)", str(value))
    if match:
        return 0, int(match.group(1)), int(match.group(2)), str(value)
    return 1, 0, 0, str(value)


def resolve_power_mode_values(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["power_mode_band"].dropna().astype(str).unique().tolist(), key=power_mode_sort_key)


def build_power_mode_code_map(frame: pd.DataFrame) -> dict[str, int]:
    return {value: idx + 1 for idx, value in enumerate(resolve_power_mode_values(frame))}


def add_power_mode_code_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    code_map = build_power_mode_code_map(frame)
    frame[POWER_MODE_CODE_COL] = (
        frame["power_mode_band"].astype(str).map(code_map).fillna(0).astype(np.float32)
    )
    return frame


def resolve_power_mode_feature_cols(frame: pd.DataFrame) -> list[str]:
    if POWER_MODE_CODE_COL in frame.columns:
        return [POWER_MODE_CODE_COL]
    return []


def resolve_known_covariate_cols(frame: pd.DataFrame) -> list[str]:
    return [*TIME_FEATURE_COLS, *resolve_power_mode_feature_cols(frame)]


def resolve_sequence_input_cols(frame: pd.DataFrame, cfg: Part2Config, target_cols: list[str] | None = None) -> list[str]:
    target_cols = target_cols or cfg.active_targets
    known_covariates = resolve_known_covariate_cols(frame)
    return [*target_cols, *known_covariates]


def resolve_target_input_indices(input_cols: list[str], target_cols: list[str]) -> list[int]:
    return [input_cols.index(col) for col in target_cols if col in input_cols]


def month_end_inclusive(period: pd.Period, step_seconds: int) -> pd.Timestamp:
    return period.end_time.floor("s") - pd.Timedelta(seconds=max(step_seconds - 1, 0))


def find_complete_months(frame: pd.DataFrame, step_seconds: int) -> list[pd.Period]:
    months: list[pd.Period] = []
    for period, block in frame.groupby(frame["timestamp"].dt.to_period("M"), sort=True):
        month_start = period.start_time
        month_end = month_end_inclusive(period, step_seconds)
        actual_start = block["timestamp"].min()
        actual_end = block["timestamp"].max()
        starts_in_month = actual_start <= month_start + pd.Timedelta(seconds=step_seconds)
        ends_in_month = actual_end >= month_end
        if starts_in_month and ends_in_month:
            months.append(period)
    return months

def bool_max(series: pd.Series) -> bool:
    return bool(series.fillna(False).astype(bool).max())


def normalize_power_mode_value(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return pd.NA
    match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if match:
        return f"{int(match.group(1))}-{int(match.group(2))}"
    return text


def load_prepared_frame(prepared_path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = prepared_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(
            prepared_path,
            usecols=columns,
            encoding="utf-8",
            low_memory=False,
        )
    elif suffix == ".parquet":
        frame = pd.read_parquet(prepared_path, columns=columns)
    else:
        raise ValueError(f"Неподдерживаемый формат prepared dataset: {prepared_path.suffix}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame[TARGET_COLS] = frame[TARGET_COLS].apply(pd.to_numeric, errors="coerce")
    for col in ["is_target_mode", *ANOMALY_COLS]:
        frame[col] = frame[col].astype("boolean")
    frame["mode"] = frame["mode"].astype("string").str.strip()
    frame["power_mode"] = frame["power_mode"].astype("string").str.strip()
    frame["power_mode_band"] = frame["power_mode"].map(normalize_power_mode_value).astype("string")
    return frame


def build_segment_ids(
    frame: pd.DataFrame,
    expected_step_seconds: int,
    split_on_power_mode: bool,
) -> pd.Series:
    diffs = frame["timestamp"].diff().dt.total_seconds()
    segment_breaks = diffs.ne(expected_step_seconds).fillna(True).astype(bool)
    if split_on_power_mode and "power_mode_band" in frame.columns:
        power_mode = frame["power_mode_band"].astype("string").fillna("__MISSING__")
        power_mode_breaks = power_mode.ne(power_mode.shift()).fillna(True).astype(bool)
        segment_breaks = segment_breaks | power_mode_breaks
    if len(segment_breaks):
        segment_breaks.iloc[0] = True
    segment_ids = segment_breaks.to_numpy(dtype=np.int32).cumsum()
    return pd.Series(segment_ids, index=frame.index, dtype="int64")


def downsample_within_segments(frame: pd.DataFrame, rule: str, target_cols: list[str]) -> pd.DataFrame:
    agg_map: dict[str, Any] = {col: "mean" for col in target_cols}
    agg_map.update(
        {
            "mode": "last",
            "power_mode": "last",
            "power_mode_band": "last",
            "is_target_mode": "last",
            "stage1_anomaly": bool_max,
            "stage2_anomaly": bool_max,
            "stage_any_anomaly": bool_max,
        }
    )
    working = frame.copy()
    working["bucket_ts"] = working["timestamp"].dt.floor(rule)
    grouped = (
        working.groupby(["segment_id", "bucket_ts"], sort=True)
        .agg(agg_map)
        .reset_index()
        .rename(columns={"bucket_ts": "timestamp"})
    )
    grouped["segment_id"] = grouped.groupby("segment_id", sort=False).ngroup().astype(int)
    grouped = grouped.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    return grouped


def prepare_modeling_frame(cfg: Part2Config) -> tuple[pd.DataFrame, list[str], int]:
    frame = load_prepared_frame(cfg.prepared_path, CORE_COLS).sort_values("timestamp").reset_index(drop=True)
    source_step = infer_step_seconds_from_series(frame["timestamp"])

    complete_months = (
        find_complete_months(frame, source_step)
        if cfg.use_only_complete_months
        else sorted(frame["timestamp"].dt.to_period("M").unique().tolist())
    )
    if cfg.use_only_complete_months:
        frame = frame[frame["timestamp"].dt.to_period("M").isin(complete_months)].copy()

    if cfg.use_only_target_mode:
        frame = frame[frame["is_target_mode"].fillna(False)].copy()

    if not cfg.keep_anomalies:
        frame = frame[~frame["stage_any_anomaly"].fillna(False)].copy()

    frame = frame.dropna(subset=cfg.active_targets).copy()
    frame = frame.dropna(subset=["power_mode_band"]).copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["segment_id"] = build_segment_ids(
        frame,
        expected_step_seconds=source_step,
        split_on_power_mode=cfg.split_segments_by_power_mode,
    )

    model_step = source_step
    if cfg.model_rule and cfg.model_rule != f"{source_step}s":
        frame = downsample_within_segments(frame, cfg.model_rule, cfg.active_targets)
        model_step = infer_rule_seconds(cfg.model_rule)

    frame = add_drift_time_feature_column(frame)
    frame = add_power_mode_code_column(frame)
    frame = frame.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    frame["row_pos"] = np.arange(len(frame), dtype=np.int64)
    return frame, [str(item) for item in complete_months], model_step


def build_window_index(frame: pd.DataFrame, cfg: Part2Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment_id, block in frame.groupby("segment_id", sort=True):
        positions = block["row_pos"].to_numpy(dtype=np.int64)
        if len(positions) < cfg.lookback_steps + cfg.horizon_steps:
            continue
        limit = len(positions) - cfg.lookback_steps - cfg.horizon_steps + 1
        segment_start = int(positions[0])
        segment_end = int(positions[-1])
        for offset in range(0, limit, cfg.window_step):
            lookback_start = int(positions[offset])
            origin_pos = int(positions[offset + cfg.lookback_steps - 1])
            target_start = int(positions[offset + cfg.lookback_steps])
            target_end = int(positions[offset + cfg.lookback_steps + cfg.horizon_steps - 1])
            target_end_ts = frame.at[target_end, "timestamp"]
            rows.append(
                {
                    "segment_id": int(segment_id),
                    "row_start": lookback_start,
                    "origin_pos": origin_pos,
                    "target_start_pos": target_start,
                    "target_end_pos": target_end,
                    "segment_start_pos": segment_start,
                    "segment_end_pos": segment_end,
                    "origin_ts": frame.at[origin_pos, "timestamp"],
                    "target_start_ts": frame.at[target_start, "timestamp"],
                    "target_end_ts": target_end_ts,
                    "target_month": target_end_ts.to_period("M"),
                    "power_mode_band": frame.at[target_end, "power_mode_band"],
                }
            )
    return pd.DataFrame(rows)


def assign_window_splits(
    windows: pd.DataFrame,
    complete_months: list[str],
    cfg: Part2Config,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if windows.empty:
        raise ValueError("Не удалось сформировать ни одного окна. Проверь lookback/horizon и фильтр режимов.")

    complete_periods = [pd.Period(month, freq="M") for month in complete_months]
    available = sorted(period for period in complete_periods if period in set(windows["target_month"]))
    need = cfg.valid_months + cfg.test_months + 1
    if len(available) < need:
        raise ValueError(f"Недостаточно полных месяцев для split: требуется минимум {need}, доступно {len(available)}.")

    test_months = available[-cfg.test_months :]
    valid_months = available[-(cfg.test_months + cfg.valid_months) : -cfg.test_months]
    train_months = available[: -(cfg.test_months + cfg.valid_months)]

    split_series = pd.Series("drop", index=windows.index, dtype="object")
    split_series.loc[windows["target_month"].isin(train_months)] = "train"
    split_series.loc[windows["target_month"].isin(valid_months)] = "valid"
    split_series.loc[windows["target_month"].isin(test_months)] = "test"

    windows = windows.copy()
    windows["split"] = split_series
    windows = windows[windows["split"] != "drop"].reset_index(drop=True)
    windows["target_month_str"] = windows["target_month"].astype(str)
    return (
        windows,
        [str(item) for item in train_months],
        [str(item) for item in valid_months],
        [str(item) for item in test_months],
    )


def create_experiment_bundle(cfg: Part2Config) -> ExperimentBundle:
    frame, complete_months, model_step = prepare_modeling_frame(cfg)
    windows = build_window_index(frame, cfg)
    windows, train_months, valid_months, test_months = assign_window_splits(windows, complete_months, cfg)
    train_windows = windows[windows["split"] == "train"].reset_index(drop=True)
    valid_windows = windows[windows["split"] == "valid"].reset_index(drop=True)
    test_windows = windows[windows["split"] == "test"].reset_index(drop=True)
    benchmark_test_windows = build_benchmark_test_windows(test_windows, cfg)
    power_mode_values = resolve_power_mode_values(frame)
    return ExperimentBundle(
        frame=frame,
        windows=windows,
        train_windows=train_windows,
        valid_windows=valid_windows,
        test_windows=test_windows,
        benchmark_test_windows=benchmark_test_windows,
        complete_months=complete_months,
        train_months=train_months,
        valid_months_list=valid_months,
        test_months_list=test_months,
        power_mode_values=power_mode_values,
        model_step_seconds=model_step,
    )


def summarize_bundle(bundle: ExperimentBundle) -> pd.DataFrame:
    frame = bundle.frame
    rows = [
        {
            "scope": "frame",
            "rows": int(len(frame)),
            "segments": int(frame["segment_id"].nunique()),
            "timestamp_min": frame["timestamp"].min(),
            "timestamp_max": frame["timestamp"].max(),
            "months": ", ".join(bundle.complete_months),
            "power_modes": ", ".join(bundle.power_mode_values),
        },
        {
            "scope": "train_windows",
            "rows": int(len(bundle.train_windows)),
            "segments": int(bundle.train_windows["segment_id"].nunique()),
            "timestamp_min": bundle.train_windows["origin_ts"].min(),
            "timestamp_max": bundle.train_windows["target_end_ts"].max(),
            "months": ", ".join(bundle.train_months),
            "power_modes": ", ".join(sorted(bundle.train_windows["power_mode_band"].dropna().astype(str).unique().tolist())),
        },
        {
            "scope": "valid_windows",
            "rows": int(len(bundle.valid_windows)),
            "segments": int(bundle.valid_windows["segment_id"].nunique()),
            "timestamp_min": bundle.valid_windows["origin_ts"].min(),
            "timestamp_max": bundle.valid_windows["target_end_ts"].max(),
            "months": ", ".join(bundle.valid_months_list),
            "power_modes": ", ".join(sorted(bundle.valid_windows["power_mode_band"].dropna().astype(str).unique().tolist())),
        },
        {
            "scope": "test_windows",
            "rows": int(len(bundle.test_windows)),
            "segments": int(bundle.test_windows["segment_id"].nunique()),
            "timestamp_min": bundle.test_windows["origin_ts"].min(),
            "timestamp_max": bundle.test_windows["target_end_ts"].max(),
            "months": ", ".join(bundle.test_months_list),
            "power_modes": ", ".join(sorted(bundle.test_windows["power_mode_band"].dropna().astype(str).unique().tolist())),
        },
        {
            "scope": "benchmark_test_windows",
            "rows": int(len(bundle.benchmark_test_windows)),
            "segments": int(bundle.benchmark_test_windows["segment_id"].nunique()),
            "timestamp_min": bundle.benchmark_test_windows["origin_ts"].min(),
            "timestamp_max": bundle.benchmark_test_windows["target_end_ts"].max(),
            "months": ", ".join(bundle.test_months_list),
            "power_modes": ", ".join(sorted(bundle.benchmark_test_windows["power_mode_band"].dropna().astype(str).unique().tolist())),
        },
    ]
    return pd.DataFrame(rows)


def sample_windows_evenly(windows: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None or len(windows) <= max_windows:
        return windows.reset_index(drop=True).copy()
    positions = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    positions = np.unique(positions)
    return windows.iloc[positions].reset_index(drop=True).copy()


def build_benchmark_test_windows(test_windows: pd.DataFrame, cfg: Part2Config) -> pd.DataFrame:
    target_windows = test_windows.reset_index(drop=True).copy()
    if cfg.benchmark_target_start is not None:
        target_start = pd.to_datetime(cfg.benchmark_target_start)
        ranked = target_windows.assign(
            _target_start_delta=(target_windows["target_start_ts"] - target_start).abs()
        ).sort_values(["_target_start_delta", "target_start_ts"])
        return ranked.drop(columns="_target_start_delta").head(max(1, cfg.max_benchmark_test_windows)).reset_index(drop=True)
    target_windows = filter_windows_for_target_power_mode(target_windows, cfg.power_mode_filter)
    return sample_windows_evenly(target_windows, cfg.max_benchmark_test_windows)


def build_validation_folds(bundle: ExperimentBundle, cfg: Part2Config) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    fold_train = bundle.train_windows.reset_index(drop=True).copy()
    fold_valid = bundle.valid_windows.reset_index(drop=True).copy()
    target_power_mode = cfg.power_mode_filter
    if target_power_mode is not None:
        fold_valid = fold_valid[fold_valid["power_mode_band"].eq(target_power_mode)].reset_index(drop=True)
    if fold_train.empty:
        raise ValueError("Не удалось собрать holdout fold: train_windows пустой.")
    if fold_valid.empty:
        raise ValueError("Не удалось собрать holdout fold: valid_windows пустой.")
    valid_months = sorted(fold_valid["target_month"].astype(str).unique().tolist())
    fold_name = "valid:" + ",".join(valid_months) if valid_months else "valid"
    return [(fold_train, fold_valid, fold_name)]


def build_final_fit_stop_windows(
    bundle: ExperimentBundle,
    cfg: Part2Config,
    max_train_windows: int | None,
    max_valid_windows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_windows, holdout_windows, _ = build_validation_folds(bundle, cfg)[0]
    return (
        sample_windows_evenly(fit_windows, max_train_windows),
        sample_windows_evenly(holdout_windows, max_valid_windows),
    )


def filter_windows_for_target_power_mode(windows: pd.DataFrame, target_power_mode: str | None) -> pd.DataFrame:
    if target_power_mode is None:
        return windows.reset_index(drop=True).copy()
    filtered = windows[windows["power_mode_band"].eq(target_power_mode)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"Не удалось найти окна для power_mode={target_power_mode}. "
            "Снизь lookback/horizon/window_step или проверь наличие такого режима в целевых месяцах."
        )
    return filtered


def build_train_reference(bundle: ExperimentBundle, cfg: Part2Config) -> np.ndarray:
    return bundle.frame.loc[
        bundle.frame["timestamp"].dt.to_period("M").isin([pd.Period(item, freq="M") for item in bundle.train_months]),
        cfg.active_targets,
    ].to_numpy(dtype=np.float32)


def flatten_finite_arrays(*arrays: np.ndarray) -> list[np.ndarray]:
    flattened = [np.asarray(array, dtype=np.float64).reshape(-1) for array in arrays]
    if not flattened:
        return []
    mask = np.ones(flattened[0].shape[0], dtype=bool)
    for array in flattened:
        mask &= np.isfinite(array)
    return [array[mask] for array in flattened]


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid_true, valid_pred = flatten_finite_arrays(y_true, y_pred)
    if valid_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(valid_true - valid_pred)))


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid_true, valid_pred = flatten_finite_arrays(y_true, y_pred)
    if valid_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(valid_true - valid_pred))))


def mase_metric(y_true: np.ndarray, y_pred: np.ndarray, train_reference: np.ndarray) -> float:
    ref = np.asarray(train_reference, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    if ref.size < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(ref)))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    valid_true, valid_pred = flatten_finite_arrays(y_true, y_pred)
    if valid_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(valid_true - valid_pred)) / scale)


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    valid_true, valid_pred = flatten_finite_arrays(y_true, y_pred)
    if valid_true.size == 0:
        return float("nan")
    errors = valid_true - valid_pred
    loss = np.maximum(quantile * errors, (quantile - 1.0) * errors)
    return float(np.mean(loss))


def weighted_quantile_loss(
    y_true: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
    weights: dict[float, float] | None = None,
) -> float:
    if not quantile_predictions:
        return float("nan")
    if weights is None:
        weights = {quantile: 1.0 for quantile in quantile_predictions}
    scores = []
    total_weight = 0.0
    for quantile, pred in quantile_predictions.items():
        weight = float(weights.get(quantile, 1.0))
        score = quantile_loss(y_true, pred, quantile)
        if not np.isfinite(score):
            continue
        scores.append(weight * score)
        total_weight += weight
    return float(np.sum(scores) / total_weight) if total_weight else float("nan")


def picp_metric(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    valid_true, valid_lower, valid_upper = flatten_finite_arrays(y_true, lower, upper)
    if valid_true.size == 0:
        return float("nan")
    inside = (valid_true >= valid_lower) & (valid_true <= valid_upper)
    return float(np.mean(inside))


def pinaw_metric(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    valid_true, valid_lower, valid_upper = flatten_finite_arrays(y_true, lower, upper)
    if valid_true.size == 0:
        return float("nan")
    denom = float(np.max(valid_true) - np.min(valid_true))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.mean(valid_upper - valid_lower) / denom)


def winkler_metric(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    valid_true, valid_lower, valid_upper = flatten_finite_arrays(y_true, lower, upper)
    if valid_true.size == 0:
        return float("nan")
    width = valid_upper - valid_lower
    below = valid_true < valid_lower
    above = valid_true > valid_upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (valid_lower[below] - valid_true[below])
    score[above] += (2.0 / alpha) * (valid_true[above] - valid_upper[above])
    return float(np.mean(score))


def evaluate_prediction_bundle(
    model_name: str,
    y_true: np.ndarray,
    point_pred: np.ndarray,
    train_reference: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    quantile_predictions: dict[float, np.ndarray] | None = None,
    alpha: float = 0.10,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model": model_name,
        "mae": mae_metric(y_true, point_pred),
        "rmse": rmse_metric(y_true, point_pred),
        "mase": mase_metric(y_true, point_pred, train_reference),
    }
    if lower is not None and upper is not None:
        metrics["picp"] = picp_metric(y_true, lower, upper)
        metrics["pinaw"] = pinaw_metric(y_true, lower, upper)
        metrics["winkler"] = winkler_metric(y_true, lower, upper, alpha=alpha)
    else:
        metrics["picp"] = float("nan")
        metrics["pinaw"] = float("nan")
        metrics["winkler"] = float("nan")
    if quantile_predictions:
        metrics["wql"] = weighted_quantile_loss(y_true, quantile_predictions)
        for quantile, pred in quantile_predictions.items():
            metrics[f"ql_{quantile:.2f}"] = quantile_loss(y_true, pred, quantile)
    else:
        metrics["wql"] = float("nan")
    return metrics


def extract_stat_window(
    frame: pd.DataFrame,
    window_row: pd.Series,
    target_cols: list[str],
    history_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_start = int(window_row["segment_start_pos"])
    origin_pos = int(window_row["origin_pos"])
    target_start = int(window_row["target_start_pos"])
    target_end = int(window_row["target_end_pos"])
    history_start = max(segment_start, origin_pos - history_limit + 1)
    history = frame.loc[history_start:origin_pos, target_cols].to_numpy(dtype=np.float32)
    future = frame.loc[target_start:target_end, target_cols].to_numpy(dtype=np.float32)
    return history, future


def extract_stat_window_with_exog(
    frame: pd.DataFrame,
    window_row: pd.Series,
    target_cols: list[str],
    exog_cols: list[str],
    history_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segment_start = int(window_row["segment_start_pos"])
    origin_pos = int(window_row["origin_pos"])
    target_start = int(window_row["target_start_pos"])
    target_end = int(window_row["target_end_pos"])
    history_start = max(segment_start, origin_pos - history_limit + 1)
    history = frame.loc[history_start:origin_pos, target_cols].to_numpy(dtype=np.float32)
    future = frame.loc[target_start:target_end, target_cols].to_numpy(dtype=np.float32)
    exog_history = frame.loc[history_start:origin_pos, exog_cols].to_numpy(dtype=np.float32)
    exog_future = frame.loc[target_start:target_end, exog_cols].to_numpy(dtype=np.float32)
    return history, future, exog_history, exog_future


def naive_interval_from_history(history: np.ndarray, horizon: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last = history[-1]
    point = np.repeat(last[None, :], repeats=horizon, axis=0)
    scale = history.std(axis=0, ddof=0)
    z = 1.645 if alpha == 0.10 else 1.96
    lower = point - z * scale
    upper = point + z * scale
    return point, lower, upper


def forecast_sarima_single(
    history: np.ndarray,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    alpha: float,
    exog_history: np.ndarray | None = None,
    exog_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if history.size < max(order[0] + order[2] + 2, 8):
        point, lower, upper = naive_interval_from_history(history[:, None], horizon, alpha)
        return point[:, 0], lower[:, 0], upper[:, 0]
    baseline = float(history[-1])
    residual_history = history - baseline
    try:
        model = SARIMAX(
            residual_history,
            exog=exog_history,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon, exog=exog_future)
        summary = forecast.summary_frame(alpha=alpha)
        point = summary["mean"].to_numpy(dtype=np.float32) + baseline
        lower = summary["mean_ci_lower"].to_numpy(dtype=np.float32) + baseline
        upper = summary["mean_ci_upper"].to_numpy(dtype=np.float32) + baseline
        if not (np.isfinite(point).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise FloatingPointError("SARIMA вернула невалидный прогноз")
        return point, lower, upper
    except Exception:
        point, lower, upper = naive_interval_from_history(history[:, None], horizon, alpha)
        return point[:, 0], lower[:, 0], upper[:, 0]


def predict_sarima_windows(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    target_cols: list[str],
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    cfg: Part2Config,
) -> dict[str, Any]:
    y_true_list = []
    point_list = []
    lower_list = []
    upper_list = []
    exog_cols = resolve_known_covariate_cols(frame)
    for _, window_row in windows.iterrows():
        history, future, exog_history, exog_future = extract_stat_window_with_exog(
            frame,
            window_row,
            target_cols,
            exog_cols,
            cfg.stat_history_limit,
        )
        target_point = []
        target_lower = []
        target_upper = []
        for col_idx in range(len(target_cols)):
            point, lower, upper = forecast_sarima_single(
                history[:, col_idx],
                cfg.horizon_steps,
                order,
                seasonal_order,
                cfg.conformal_alpha,
                exog_history=exog_history,
                exog_future=exog_future,
            )
            target_point.append(point)
            target_lower.append(lower)
            target_upper.append(upper)
        y_true_list.append(future.astype(np.float32))
        point_list.append(np.stack(target_point, axis=1))
        lower_list.append(np.stack(target_lower, axis=1))
        upper_list.append(np.stack(target_upper, axis=1))
    return {
        "y_true": np.stack(y_true_list),
        "point": np.stack(point_list),
        "lower": np.stack(lower_list),
        "upper": np.stack(upper_list),
    }


def tune_sarima(bundle: ExperimentBundle, cfg: Part2Config) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = build_validation_folds(bundle, cfg)
    candidate_rows: list[dict[str, Any]] = []
    total = len(cfg.sarima_orders) * len(cfg.sarima_seasonal_orders) * len(folds)
    progress = create_progress(cfg, total=total, desc="Tune SARIMA", leave=False)
    for order in cfg.sarima_orders:
        for seasonal_order in cfg.sarima_seasonal_orders:
            params = {"order": order, "seasonal_order": seasonal_order}
            for _, fold_valid, fold_name in folds:
                progress.set_postfix_str(f"fold={fold_name}, {format_params_for_progress(params)}")
                valid_sample = sample_windows_evenly(fold_valid, cfg.max_search_windows_stat)
                preds = predict_sarima_windows(bundle.frame, valid_sample, cfg.active_targets, order, seasonal_order, cfg)
                score = mae_metric(preds["y_true"], preds["point"])
                winkler = winkler_metric(preds["y_true"], preds["lower"], preds["upper"], alpha=cfg.conformal_alpha)
                candidate_rows.append(
                    {
                        "model": "SARIMA",
                        "fold": fold_name,
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "mae": score,
                        "winkler": winkler,
                    }
                )
                progress.update(1)
    progress.close()
    history = pd.DataFrame(candidate_rows)
    summary = (
        history.groupby(["order", "seasonal_order"], as_index=False)[["mae", "winkler"]]
        .mean()
        .sort_values(["winkler", "mae"])
        .reset_index(drop=True)
    )
    best = summary.iloc[0].to_dict()
    return best, history


def run_sarima_workflow(bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    workflow_start = time.perf_counter()
    tuning_start = time.perf_counter()
    best_params, tuning_history = tune_sarima(bundle, cfg)
    tuning_time = time.perf_counter() - tuning_start
    test_sample = bundle.benchmark_test_windows.reset_index(drop=True).copy()
    inference_start = time.perf_counter()
    preds = predict_sarima_windows(
        bundle.frame,
        test_sample,
        cfg.active_targets,
        tuple(best_params["order"]),
        tuple(best_params["seasonal_order"]),
        cfg,
    )
    inference_time = time.perf_counter() - inference_start
    train_reference = bundle.frame.loc[
        bundle.frame["timestamp"].dt.to_period("M").isin([pd.Period(item, freq="M") for item in bundle.train_months]),
        cfg.active_targets,
    ].to_numpy(dtype=np.float32)
    metrics = evaluate_prediction_bundle(
        model_name="SARIMA",
        y_true=preds["y_true"],
        point_pred=preds["point"],
        train_reference=train_reference,
        lower=preds["lower"],
        upper=preds["upper"],
        alpha=cfg.conformal_alpha,
    )
    metrics["n_test_windows"] = int(len(test_sample))
    return {
        "model": "SARIMA",
        "best_params": {"order": tuple(best_params["order"]), "seasonal_order": tuple(best_params["seasonal_order"])},
        "tuning_history": tuning_history,
        "metrics": metrics,
        "predictions": preds,
        "test_windows": test_sample,
        "timing": {
            "model": "SARIMA",
            "tuning_time_seconds": tuning_time,
            "final_fit_time_seconds": 0.0,
            "calibration_time_seconds": 0.0,
            "inference_time_seconds": inference_time,
            "training_time_seconds": tuning_time,
            "total_workflow_time_seconds": time.perf_counter() - workflow_start,
            "n_test_windows": int(len(test_sample)),
            "note": "SARIMA fits local per-window models during inference.",
        },
    }


def forecast_var_window(history: np.ndarray, horizon: int, maxlags: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(history) <= max(maxlags + 1, 8):
        return naive_interval_from_history(history, horizon, alpha)
    baseline = history[-1].astype(np.float32)
    residual_history = history - baseline[None, :]
    try:
        model = VAR(residual_history)
        fitted = model.fit(maxlags=maxlags, trend="c")
        lag_order = max(1, fitted.k_ar)
        point, lower, upper = fitted.forecast_interval(residual_history[-lag_order:], steps=horizon, alpha=alpha)
        point = point.astype(np.float32) + baseline[None, :]
        lower = lower.astype(np.float32) + baseline[None, :]
        upper = upper.astype(np.float32) + baseline[None, :]
        if not (np.isfinite(point).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise FloatingPointError("VAR вернула невалидный прогноз")
        return point, lower, upper
    except Exception:
        return naive_interval_from_history(history, horizon, alpha)


def predict_var_windows(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    target_cols: list[str],
    maxlags: int,
    cfg: Part2Config,
) -> dict[str, Any]:
    y_true_list = []
    point_list = []
    lower_list = []
    upper_list = []
    for _, window_row in windows.iterrows():
        history, future = extract_stat_window(frame, window_row, target_cols, cfg.stat_history_limit)
        point, lower, upper = forecast_var_window(history, cfg.horizon_steps, maxlags, cfg.conformal_alpha)
        y_true_list.append(future.astype(np.float32))
        point_list.append(point)
        lower_list.append(lower)
        upper_list.append(upper)
    return {
        "y_true": np.stack(y_true_list),
        "point": np.stack(point_list),
        "lower": np.stack(lower_list),
        "upper": np.stack(upper_list),
    }


def tune_var(bundle: ExperimentBundle, cfg: Part2Config) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = build_validation_folds(bundle, cfg)
    rows: list[dict[str, Any]] = []
    total = len(cfg.var_maxlags_grid) * len(folds)
    progress = create_progress(cfg, total=total, desc="Tune VAR", leave=False)
    for maxlags in cfg.var_maxlags_grid:
        params = {"maxlags": maxlags}
        for _, fold_valid, fold_name in folds:
            progress.set_postfix_str(f"fold={fold_name}, {format_params_for_progress(params)}")
            valid_sample = sample_windows_evenly(fold_valid, cfg.max_search_windows_stat)
            preds = predict_var_windows(bundle.frame, valid_sample, cfg.active_targets, maxlags, cfg)
            score = mae_metric(preds["y_true"], preds["point"])
            winkler = winkler_metric(preds["y_true"], preds["lower"], preds["upper"], alpha=cfg.conformal_alpha)
            rows.append({"model": "VAR", "fold": fold_name, "maxlags": maxlags, "mae": score, "winkler": winkler})
            progress.update(1)
    progress.close()
    history = pd.DataFrame(rows)
    summary = (
        history.groupby("maxlags", as_index=False)[["mae", "winkler"]]
        .mean()
        .sort_values(["winkler", "mae"])
        .reset_index(drop=True)
    )
    best = summary.iloc[0].to_dict()
    return best, history


def run_var_workflow(bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    workflow_start = time.perf_counter()
    tuning_start = time.perf_counter()
    best_params, tuning_history = tune_var(bundle, cfg)
    tuning_time = time.perf_counter() - tuning_start
    test_sample = bundle.benchmark_test_windows.reset_index(drop=True).copy()
    inference_start = time.perf_counter()
    preds = predict_var_windows(bundle.frame, test_sample, cfg.active_targets, int(best_params["maxlags"]), cfg)
    inference_time = time.perf_counter() - inference_start
    train_reference = bundle.frame.loc[
        bundle.frame["timestamp"].dt.to_period("M").isin([pd.Period(item, freq="M") for item in bundle.train_months]),
        cfg.active_targets,
    ].to_numpy(dtype=np.float32)
    metrics = evaluate_prediction_bundle(
        model_name="VAR",
        y_true=preds["y_true"],
        point_pred=preds["point"],
        train_reference=train_reference,
        lower=preds["lower"],
        upper=preds["upper"],
        alpha=cfg.conformal_alpha,
    )
    metrics["n_test_windows"] = int(len(test_sample))
    return {
        "model": "VAR",
        "best_params": {"maxlags": int(best_params["maxlags"])},
        "tuning_history": tuning_history,
        "metrics": metrics,
        "predictions": preds,
        "test_windows": test_sample,
        "timing": {
            "model": "VAR",
            "tuning_time_seconds": tuning_time,
            "final_fit_time_seconds": 0.0,
            "calibration_time_seconds": 0.0,
            "inference_time_seconds": inference_time,
            "training_time_seconds": tuning_time,
            "total_workflow_time_seconds": time.perf_counter() - workflow_start,
            "n_test_windows": int(len(test_sample)),
            "note": "VAR fits local per-window models during inference.",
        },
    }


def build_tabular_feature_names(
    input_cols: list[str],
    cfg: Part2Config,
    future_cov_cols: list[str] | None = None,
) -> list[str]:
    names: list[str] = []
    for col in input_cols:
        names.append(f"{col}__last")
        names.append(f"{col}__diff_1")
        for lag in cfg.lag_steps:
            names.append(f"{col}__lag_{lag}")
        for win in cfg.rolling_windows:
            names.append(f"{col}__roll_mean_{win}")
            names.append(f"{col}__roll_std_{win}")
        names.append(f"{col}__lookback_min")
        names.append(f"{col}__lookback_max")
    for col in future_cov_cols or []:
        names.append(f"{col}__future")
    return names


def extract_tabular_features(
    history: np.ndarray,
    cfg: Part2Config,
    future_known_cov: np.ndarray | None = None,
) -> np.ndarray:
    feature_values: list[float] = []
    for col_idx in range(history.shape[1]):
        series = history[:, col_idx]
        feature_values.append(float(series[-1]))
        diff_1 = float(series[-1] - series[-2]) if len(series) >= 2 else 0.0
        feature_values.append(diff_1)
        for lag in cfg.lag_steps:
            feature_values.append(float(series[-1 - lag]) if lag < len(series) else float(series[0]))
        for win in cfg.rolling_windows:
            tail = series[-min(win, len(series)) :]
            feature_values.append(float(tail.mean()))
            feature_values.append(float(tail.std(ddof=0)))
        feature_values.append(float(series.min()))
        feature_values.append(float(series.max()))
    if future_known_cov is not None:
        feature_values.extend(float(value) for value in np.asarray(future_known_cov, dtype=np.float32).reshape(-1))
    return np.asarray(feature_values, dtype=np.float32)


def build_tabular_step_dataset(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Part2Config,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    residual_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_cols = target_cols or cfg.active_targets
    input_cols = input_cols or resolve_sequence_input_cols(frame, cfg, target_cols=target_cols)
    known_covariate_cols = [col for col in input_cols if col not in target_cols]
    input_values = frame[input_cols].to_numpy(dtype=np.float32)
    target_values = frame[target_cols].to_numpy(dtype=np.float32)
    future_cov_cols = [*known_covariate_cols, HORIZON_STEP_FEATURE_COL]
    future_cov_values = frame[known_covariate_cols].to_numpy(dtype=np.float32) if known_covariate_cols else None

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for _, window_row in windows.iterrows():
        origin_pos = int(window_row["origin_pos"])
        target_start = int(window_row["target_start_pos"])
        history = input_values[int(window_row["row_start"]) : origin_pos + 1]
        next_target = target_values[target_start]
        if residual_target:
            next_target = next_target - target_values[origin_pos]
        base_future_cov = (
            future_cov_values[target_start]
            if future_cov_values is not None
            else np.zeros(0, dtype=np.float32)
        )
        horizon_step = build_horizon_step_feature(
            1,
            int(window_row.get("block_start_step", 0)),
            int(cfg.horizon_steps),
        ).reshape(-1)
        future_known_cov = np.concatenate([base_future_cov, horizon_step]).astype(np.float32, copy=False)
        feature_rows.append(extract_tabular_features(history, cfg, future_known_cov=future_known_cov))
        target_rows.append(next_target.astype(np.float32))
        meta_rows.append(
            {
                "segment_id": int(window_row["segment_id"]),
                "origin_ts": window_row["origin_ts"],
                "target_start_ts": window_row["target_start_ts"],
                "target_end_ts": window_row["target_end_ts"],
                "target_month": window_row["target_month"],
                "power_mode_band": window_row["power_mode_band"],
            }
        )

    feature_names = build_tabular_feature_names(input_cols, cfg, future_cov_cols=future_cov_cols)
    X = pd.DataFrame(np.vstack(feature_rows), columns=feature_names)
    y = pd.DataFrame(np.vstack(target_rows), columns=target_cols)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def fit_lightgbm_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict[str, Any],
    quantiles: tuple[float, ...],
    random_state: int,
) -> dict[float, MultiOutputRegressor]:
    models: dict[float, MultiOutputRegressor] = {}
    for quantile in quantiles:
        base_model = LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
            **params,
        )
        wrapper = MultiOutputRegressor(base_model)
        wrapper.fit(X_train, y_train)
        models[quantile] = wrapper
    return models


def predict_lightgbm_quantiles(models: dict[float, MultiOutputRegressor], X: pd.DataFrame, cfg: Part2Config) -> dict[float, np.ndarray]:
    outputs: dict[float, np.ndarray] = {}
    for quantile, model in models.items():
        pred = model.predict(X)
        outputs[quantile] = np.asarray(pred, dtype=np.float32).reshape(len(X), 1, len(cfg.active_targets))
    return outputs


def predict_lightgbm_quantiles_direct(
    models: dict[float, MultiOutputRegressor],
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Part2Config,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> dict[str, Any]:
    target_cols = target_cols or cfg.active_targets
    step_windows = expand_windows_to_direct_blocks(windows, cfg.horizon_steps, 1)
    X_step, y_step, _ = build_tabular_step_dataset(
        frame,
        step_windows,
        cfg,
        input_cols=input_cols,
        target_cols=target_cols,
    )
    step_quantiles = predict_lightgbm_quantiles(models, X_step, cfg)
    baseline = frame.loc[step_windows["origin_pos"], target_cols].to_numpy(dtype=np.float32)
    for quantile in list(step_quantiles):
        step_quantiles[quantile] = (
            step_quantiles[quantile] + baseline[:, None, :]
        ).astype(np.float32, copy=False)
    y_true, quantiles = stitch_direct_block_quantiles(
        step_windows,
        y_step.to_numpy(dtype=np.float32)[:, None, :],
        step_quantiles,
    )
    return {
        "y_true": y_true,
        "quantiles": quantiles,
    }


def tune_lightgbm(bundle: ExperimentBundle, cfg: Part2Config) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = build_validation_folds(bundle, cfg)
    rows: list[dict[str, Any]] = []
    median_quantile = cfg.primary_quantile
    total = len(cfg.lgbm_grid) * len(folds)
    progress = create_progress(cfg, total=total, desc="Tune LightGBM", leave=False)
    for params in cfg.lgbm_grid:
        for fold_train, fold_valid, fold_name in folds:
            progress.set_postfix_str(f"fold={fold_name}, {format_params_for_progress(params)}")
            train_steps, _ = sample_direct_block_windows(
                fold_train,
                cfg.max_train_windows_ml,
                cfg.horizon_steps,
                1,
            )
            valid_sample = sample_windows_evenly(fold_valid, cfg.max_valid_windows_ml)
            X_train, y_train, _ = build_tabular_step_dataset(
                bundle.frame,
                train_steps,
                cfg,
                residual_target=True,
            )
            models = fit_lightgbm_quantile_models(
                X_train,
                y_train,
                params=params,
                quantiles=(median_quantile,),
                random_state=cfg.random_state,
            )
            preds = predict_lightgbm_quantiles_direct(models, bundle.frame, valid_sample, cfg)
            score = mae_metric(preds["y_true"], preds["quantiles"][median_quantile])
            rows.append({"model": "LightGBM", "fold": fold_name, "params": copy.deepcopy(params), "mae": score})
            progress.update(1)
    progress.close()
    history = pd.DataFrame(rows)
    history["params_key"] = history["params"].map(lambda item: repr(item))
    summary = history.groupby("params_key", as_index=False)["mae"].mean().sort_values("mae").reset_index(drop=True)
    best_key = summary.iloc[0]["params_key"]
    best_params = next(item for item in history["params"] if repr(item) == best_key)
    return dict(best_params), history.drop(columns=["params_key"])


def run_lightgbm_workflow(bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    workflow_start = time.perf_counter()
    tuning_start = time.perf_counter()
    best_params, tuning_history = tune_lightgbm(bundle, cfg)
    tuning_time = time.perf_counter() - tuning_start

    fit_start = time.perf_counter()
    train_valid = pd.concat([bundle.train_windows, bundle.valid_windows], ignore_index=True)
    test_sample = bundle.benchmark_test_windows.reset_index(drop=True).copy()

    train_steps, _ = sample_direct_block_windows(
        train_valid,
        cfg.max_train_windows_ml,
        cfg.horizon_steps,
        1,
    )
    X_train, y_train, _ = build_tabular_step_dataset(
        bundle.frame,
        train_steps,
        cfg,
        residual_target=True,
    )

    models = fit_lightgbm_quantile_models(
        X_train,
        y_train,
        params=best_params,
        quantiles=cfg.quantiles,
        random_state=cfg.random_state,
    )
    final_fit_time = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    preds = predict_lightgbm_quantiles_direct(models, bundle.frame, test_sample, cfg)
    inference_time = time.perf_counter() - inference_start
    quantile_preds = preds["quantiles"]
    y_true = preds["y_true"]
    point_pred = quantile_preds[cfg.primary_quantile]
    lower, upper = order_prediction_interval(
        quantile_preds[min(cfg.quantiles)],
        quantile_preds[max(cfg.quantiles)],
    )

    train_reference = build_train_reference(bundle, cfg)
    metrics = evaluate_prediction_bundle(
        model_name="LightGBM",
        y_true=y_true,
        point_pred=point_pred,
        train_reference=train_reference,
        lower=lower,
        upper=upper,
        quantile_predictions=quantile_preds,
        alpha=cfg.conformal_alpha,
    )
    metrics["n_test_windows"] = int(len(test_sample))
    return {
        "model": "LightGBM",
        "best_params": {
            **best_params,
            "forecast_strategy": "direct_multi_horizon",
            "direct_block_steps": 1,
        },
        "tuning_history": tuning_history,
        "metrics": metrics,
        "predictions": {"y_true": y_true, "quantiles": quantile_preds},
        "test_windows": test_sample,
        "timing": {
            "model": "LightGBM",
            "tuning_time_seconds": tuning_time,
            "final_fit_time_seconds": final_fit_time,
            "calibration_time_seconds": 0.0,
            "inference_time_seconds": inference_time,
            "training_time_seconds": tuning_time + final_fit_time,
            "total_workflow_time_seconds": time.perf_counter() - workflow_start,
            "n_test_windows": int(len(test_sample)),
            "note": "",
        },
    }


@dataclass
class ArrayScaler:
    mean_: np.ndarray
    scale_: np.ndarray


def transform_with_scaler(values: np.ndarray, scaler: ArrayScaler) -> np.ndarray:
    return ((values - scaler.mean_) / scaler.scale_).astype(np.float32)


def inverse_with_scaler(values: np.ndarray, scaler: ArrayScaler) -> np.ndarray:
    return (values * scaler.scale_) + scaler.mean_


def transform_window_with_scaler(values: np.ndarray, scaler: ArrayScaler) -> np.ndarray:
    mean = scaler.mean_.reshape(-1)
    scale = scaler.scale_.reshape(-1)
    return ((values - mean) / scale).astype(np.float32, copy=False)


class WindowFutureTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, future_known_covariates: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.future_known_covariates = torch.tensor(future_known_covariates, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.future_known_covariates[idx], self.y[idx]


@dataclass
class SequenceDirectWindowData:
    input_values: np.ndarray
    target_values: np.ndarray
    future_covariate_values: np.ndarray
    windows: pd.DataFrame
    input_cols: list[str]
    target_cols: list[str]
    known_covariate_cols: list[str]
    future_covariate_cols: list[str]
    residual_target: bool = True
    total_horizon_steps: int = 0

    @property
    def input_dim(self) -> int:
        return int(self.input_values.shape[1])

    @property
    def future_cov_dim(self) -> int:
        return int(len(self.future_covariate_cols))

    @property
    def n_targets(self) -> int:
        return int(self.target_values.shape[1])

    @property
    def horizon_steps(self) -> int:
        if self.windows.empty:
            return 0
        first = self.windows.iloc[0]
        return int(first["target_end_pos"]) - int(first["target_start_pos"]) + 1


def build_sequence_direct_window_data(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Part2Config,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> SequenceDirectWindowData:
    target_cols = target_cols or cfg.active_targets
    input_cols = input_cols or resolve_sequence_input_cols(frame, cfg, target_cols=target_cols)
    known_covariate_cols = [col for col in input_cols if col not in target_cols]
    future_covariate_cols = [*known_covariate_cols, HORIZON_STEP_FEATURE_COL]
    residual_target = all(col in input_cols for col in target_cols)
    return SequenceDirectWindowData(
        input_values=frame[input_cols].to_numpy(dtype=np.float32),
        target_values=frame[target_cols].to_numpy(dtype=np.float32),
        future_covariate_values=frame[known_covariate_cols].to_numpy(dtype=np.float32),
        windows=windows.reset_index(drop=True).copy(),
        input_cols=list(input_cols),
        target_cols=list(target_cols),
        known_covariate_cols=list(known_covariate_cols),
        future_covariate_cols=list(future_covariate_cols),
        residual_target=residual_target,
        total_horizon_steps=int(cfg.horizon_steps),
    )


def replace_sequence_direct_windows(data: SequenceDirectWindowData, windows: pd.DataFrame) -> SequenceDirectWindowData:
    return SequenceDirectWindowData(
        input_values=data.input_values,
        target_values=data.target_values,
        future_covariate_values=data.future_covariate_values,
        windows=windows.reset_index(drop=True).copy(),
        input_cols=data.input_cols,
        target_cols=data.target_cols,
        known_covariate_cols=data.known_covariate_cols,
        future_covariate_cols=data.future_covariate_cols,
        residual_target=data.residual_target,
        total_horizon_steps=data.total_horizon_steps,
    )


def build_horizon_step_feature(length: int, block_start_step: int, total_horizon_steps: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    denom = max(int(total_horizon_steps) - 1, 1)
    steps = int(block_start_step) + np.arange(length, dtype=np.float32)
    return (steps / float(denom)).reshape(-1, 1).astype(np.float32)


def build_augmented_future_covariates(
    data: SequenceDirectWindowData,
    target_start: int,
    target_end: int,
    block_start_step: int,
) -> np.ndarray:
    base = data.future_covariate_values[int(target_start) : int(target_end) + 1]
    horizon_step = build_horizon_step_feature(
        len(base),
        int(block_start_step),
        int(data.total_horizon_steps or len(base)),
    )
    return np.concatenate([base, horizon_step], axis=1).astype(np.float32, copy=False)


def build_target_output_block(
    data: SequenceDirectWindowData,
    origin_pos: int,
    target_start: int,
    target_end: int,
) -> np.ndarray:
    target = data.target_values[int(target_start) : int(target_end) + 1]
    if not data.residual_target:
        return target.astype(np.float32, copy=False)
    baseline = data.target_values[int(origin_pos)]
    return (target - baseline[None, :]).astype(np.float32, copy=False)


def fit_window_slice_scaler(
    values: np.ndarray,
    windows: pd.DataFrame,
    start_col: str,
    end_col: str,
) -> ArrayScaler:
    n_features = int(values.shape[1])
    if n_features == 0:
        empty = np.zeros((1, 1, 0), dtype=np.float32)
        return ArrayScaler(mean_=empty, scale_=empty.copy())

    starts = windows[start_col].to_numpy(dtype=np.int64)
    ends = windows[end_col].to_numpy(dtype=np.int64)
    sums = np.zeros(n_features, dtype=np.float64)
    sq_sums = np.zeros(n_features, dtype=np.float64)
    total_count = 0

    for start, end in zip(starts, ends):
        block = values[int(start) : int(end) + 1]
        if block.size == 0:
            continue
        sums += block.sum(axis=0, dtype=np.float64)
        sq_sums += np.einsum("ij,ij->j", block, block, dtype=np.float64)
        total_count += int(block.shape[0])

    if total_count == 0:
        raise ValueError("Не удалось посчитать scaler: окна не содержат данных.")

    mean = sums / total_count
    variance = np.maximum((sq_sums / total_count) - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return ArrayScaler(
        mean_=mean.reshape(1, 1, -1).astype(np.float32),
        scale_=scale.reshape(1, 1, -1).astype(np.float32),
    )


def fit_generated_window_block_scaler(
    data: SequenceDirectWindowData,
    block_builder: Any,
) -> ArrayScaler:
    first_window = data.windows.iloc[0]
    first_block = block_builder(data, first_window)
    n_features = int(first_block.shape[1])
    if n_features == 0:
        empty = np.zeros((1, 1, 0), dtype=np.float32)
        return ArrayScaler(mean_=empty, scale_=empty.copy())

    sums = np.zeros(n_features, dtype=np.float64)
    sq_sums = np.zeros(n_features, dtype=np.float64)
    total_count = 0

    for _, window_row in data.windows.iterrows():
        block = block_builder(data, window_row)
        if block.size == 0:
            continue
        sums += block.sum(axis=0, dtype=np.float64)
        sq_sums += np.einsum("ij,ij->j", block, block, dtype=np.float64)
        total_count += int(block.shape[0])

    if total_count == 0:
        raise ValueError("Не удалось посчитать scaler: окна не содержат данных.")

    mean = sums / total_count
    variance = np.maximum((sq_sums / total_count) - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return ArrayScaler(
        mean_=mean.reshape(1, 1, -1).astype(np.float32),
        scale_=scale.reshape(1, 1, -1).astype(np.float32),
    )


def future_covariate_block_for_scaler(data: SequenceDirectWindowData, window_row: pd.Series) -> np.ndarray:
    return build_augmented_future_covariates(
        data,
        int(window_row["target_start_pos"]),
        int(window_row["target_end_pos"]),
        int(window_row.get("block_start_step", 0)),
    )


def target_output_block_for_scaler(data: SequenceDirectWindowData, window_row: pd.Series) -> np.ndarray:
    return build_target_output_block(
        data,
        int(window_row["origin_pos"]),
        int(window_row["target_start_pos"]),
        int(window_row["target_end_pos"]),
    )


def fit_sequence_direct_scalers(data: SequenceDirectWindowData) -> tuple[ArrayScaler, ArrayScaler, ArrayScaler]:
    feature_scaler = fit_window_slice_scaler(data.input_values, data.windows, "row_start", "origin_pos")
    future_cov_scaler = fit_generated_window_block_scaler(data, future_covariate_block_for_scaler)
    target_scaler = fit_generated_window_block_scaler(data, target_output_block_for_scaler)
    return feature_scaler, future_cov_scaler, target_scaler


class LazyWindowFutureTensorDataset(Dataset):
    def __init__(
        self,
        data: SequenceDirectWindowData,
        feature_scaler: ArrayScaler,
        future_cov_scaler: ArrayScaler,
        target_scaler: ArrayScaler,
    ):
        self.input_values = data.input_values
        self.target_values = data.target_values
        self.future_covariate_values = data.future_covariate_values
        self.row_start = data.windows["row_start"].to_numpy(dtype=np.int64)
        self.origin_pos = data.windows["origin_pos"].to_numpy(dtype=np.int64)
        self.target_start_pos = data.windows["target_start_pos"].to_numpy(dtype=np.int64)
        self.target_end_pos = data.windows["target_end_pos"].to_numpy(dtype=np.int64)
        self.block_start_step = (
            data.windows["block_start_step"].to_numpy(dtype=np.int64)
            if "block_start_step" in data.windows.columns
            else np.zeros(len(data.windows), dtype=np.int64)
        )
        self.residual_target = data.residual_target
        self.total_horizon_steps = data.total_horizon_steps
        self.feature_scaler = feature_scaler
        self.future_cov_scaler = future_cov_scaler
        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return int(len(self.row_start))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_start = int(self.row_start[idx])
        origin_pos = int(self.origin_pos[idx])
        target_start = int(self.target_start_pos[idx])
        target_end = int(self.target_end_pos[idx])
        block_start_step = int(self.block_start_step[idx])

        X = transform_window_with_scaler(
            self.input_values[row_start : origin_pos + 1],
            self.feature_scaler,
        )
        base_future_cov = self.future_covariate_values[target_start : target_end + 1]
        horizon_step = build_horizon_step_feature(
            len(base_future_cov),
            block_start_step,
            int(self.total_horizon_steps or len(base_future_cov)),
        )
        future_cov_values = np.concatenate([base_future_cov, horizon_step], axis=1).astype(np.float32, copy=False)
        future_cov = transform_window_with_scaler(
            future_cov_values,
            self.future_cov_scaler,
        )
        target_values = self.target_values[target_start : target_end + 1]
        if self.residual_target:
            target_values = target_values - self.target_values[origin_pos][None, :]
        y = transform_window_with_scaler(
            target_values,
            self.target_scaler,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(future_cov)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, channels: tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        blocks = []
        in_channels = num_inputs
        for level, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TCNQuantileRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_cov_dim: int,
        horizon_steps: int,
        n_targets: int,
        quantiles: tuple[float, ...],
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.n_targets = n_targets
        self.quantiles = quantiles
        self.future_cov_dim = future_cov_dim
        self.tcn = TemporalConvNet(input_dim, channels, kernel_size, dropout)
        context_dim = channels[-1] * 2
        hidden_dim = channels[-1]
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_proj = (
            nn.Sequential(
                nn.LayerNorm(future_cov_dim),
                nn.Linear(future_cov_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if future_cov_dim > 0
            else None
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets * len(quantiles)),
        )

    def forward(self, x: torch.Tensor, future_cov: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.tcn(x)
        pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=2)], dim=1)
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, future_cov.size(1), -1)
        if self.future_proj is None:
            future_hidden = torch.zeros_like(context)
        else:
            future_hidden = self.future_proj(future_cov)
        out = self.head(torch.cat([context, future_hidden], dim=-1))
        return out.view(-1, future_cov.size(1), self.n_targets, len(self.quantiles))


class TransformerQuantileRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_cov_dim: int,
        horizon_steps: int,
        n_targets: int,
        quantiles: tuple[float, ...],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.n_targets = n_targets
        self.quantiles = quantiles
        self.future_cov_dim = future_cov_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_proj = (
            nn.Sequential(
                nn.LayerNorm(future_cov_dim),
                nn.Linear(future_cov_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if future_cov_dim > 0
            else None
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_targets * len(quantiles)),
        )

    def forward(self, x: torch.Tensor, future_cov: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        cls = self.cls_token.expand(hidden.size(0), -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = self.pos_encoder(hidden)
        encoded = self.encoder(hidden)
        pooled = torch.cat([encoded[:, 0, :], encoded[:, -1, :], encoded[:, 1:, :].mean(dim=1)], dim=1)
        context = self.context_proj(pooled).unsqueeze(1).expand(-1, future_cov.size(1), -1)
        if self.future_proj is None:
            future_hidden = torch.zeros_like(context)
        else:
            future_hidden = self.future_proj(future_cov)
        out = self.head(torch.cat([context, future_hidden], dim=-1))
        return out.view(-1, future_cov.size(1), self.n_targets, len(self.quantiles))


def quantile_loss_torch(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    losses = []
    for idx, quantile in enumerate(quantiles):
        errors = target - pred[..., idx]
        losses.append(torch.maximum(quantile * errors, (quantile - 1.0) * errors).mean())
    return torch.stack(losses).mean()


def build_sequence_direct_arrays(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Part2Config,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_cols = target_cols or cfg.active_targets
    input_cols = input_cols or resolve_sequence_input_cols(frame, cfg, target_cols=target_cols)
    known_covariate_cols = [col for col in input_cols if col not in target_cols]
    input_values = frame[input_cols].to_numpy(dtype=np.float32)
    target_values = frame[target_cols].to_numpy(dtype=np.float32)
    future_covariate_values = frame[known_covariate_cols].to_numpy(dtype=np.float32)

    X_list = []
    future_cov_list = []
    y_list = []
    for _, window_row in windows.iterrows():
        X_list.append(input_values[int(window_row["row_start"]) : int(window_row["origin_pos"]) + 1])
        target_start = int(window_row["target_start_pos"])
        target_end = int(window_row["target_end_pos"])
        block_start_step = int(window_row.get("block_start_step", 0))
        base_future_cov = future_covariate_values[target_start : target_end + 1]
        horizon_step = build_horizon_step_feature(
            len(base_future_cov),
            block_start_step,
            int(cfg.horizon_steps),
        )
        future_cov_list.append(
            np.concatenate([base_future_cov, horizon_step], axis=1).astype(np.float32, copy=False)
        )
        y_list.append(target_values[target_start : target_end + 1])
    return np.stack(X_list), np.stack(future_cov_list), np.stack(y_list)


def resolve_direct_block_steps(total_horizon_steps: int, preferred_steps: int) -> int:
    preferred_steps = max(1, min(int(total_horizon_steps), int(preferred_steps)))
    for candidate in range(preferred_steps, 0, -1):
        if total_horizon_steps % candidate == 0:
            return candidate
    return 1


def expand_windows_to_direct_blocks(
    windows: pd.DataFrame,
    total_horizon_steps: int,
    block_steps: int,
) -> pd.DataFrame:
    windows = windows.reset_index(drop=True).copy()
    if windows.empty:
        return windows.assign(
            base_window_id=pd.Series(dtype=np.int64),
            block_id=pd.Series(dtype=np.int64),
            block_start_step=pd.Series(dtype=np.int64),
        )

    n_blocks = int(total_horizon_steps) // int(block_steps)
    if n_blocks <= 0:
        raise ValueError("total_horizon_steps и block_steps должны задавать хотя бы один direct-блок.")

    base_idx = np.repeat(np.arange(len(windows), dtype=np.int64), n_blocks)
    block_id = np.tile(np.arange(n_blocks, dtype=np.int64), len(windows))
    return build_direct_block_rows(windows, base_idx, block_id, int(block_steps))


def build_direct_block_rows(
    windows: pd.DataFrame,
    base_idx: np.ndarray,
    block_id: np.ndarray,
    block_steps: int,
) -> pd.DataFrame:
    windows = windows.reset_index(drop=True)
    base_idx = np.asarray(base_idx, dtype=np.int64)
    block_id = np.asarray(block_id, dtype=np.int64)
    block_start_steps = block_id * int(block_steps)
    rows = windows.iloc[base_idx].reset_index(drop=True).copy()
    rows["base_window_id"] = base_idx
    rows["block_id"] = block_id
    rows["block_start_step"] = block_start_steps
    rows["target_start_pos"] = rows["target_start_pos"].to_numpy(dtype=np.int64) + block_start_steps
    rows["target_end_pos"] = rows["target_start_pos"].to_numpy(dtype=np.int64) + int(block_steps) - 1
    return rows


def stitch_direct_block_quantiles(
    block_windows: pd.DataFrame,
    y_true_blocks: np.ndarray,
    quantile_blocks: dict[float, np.ndarray],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    ordered_blocks = block_windows.reset_index(drop=True).copy()
    y_true_rows: list[np.ndarray] = []
    quantile_rows: dict[float, list[np.ndarray]] = {quantile: [] for quantile in quantile_blocks}

    for _, block_group in ordered_blocks.groupby("base_window_id", sort=True):
        block_idx = block_group.sort_values("block_start_step").index.to_numpy(dtype=np.int64)
        y_true_rows.append(np.concatenate([y_true_blocks[idx] for idx in block_idx], axis=0))
        for quantile, block_values in quantile_blocks.items():
            quantile_rows[quantile].append(np.concatenate([block_values[idx] for idx in block_idx], axis=0))

    return (
        np.stack(y_true_rows),
        {quantile: np.stack(rows) for quantile, rows in quantile_rows.items()},
    )


def sample_direct_block_windows(
    windows: pd.DataFrame,
    max_samples: int | None,
    total_horizon_steps: int,
    preferred_block_steps: int,
) -> tuple[pd.DataFrame, int]:
    windows = windows.reset_index(drop=True).copy()
    block_steps = resolve_direct_block_steps(total_horizon_steps, preferred_block_steps)
    n_blocks = int(total_horizon_steps) // int(block_steps)
    total_blocks = int(len(windows) * n_blocks)
    if max_samples is not None and total_blocks > int(max_samples):
        positions = np.linspace(0, total_blocks - 1, num=int(max_samples), dtype=np.int64)
        positions = np.unique(positions)
        base_idx = positions // n_blocks
        block_id = positions % n_blocks
        return build_direct_block_rows(windows, base_idx, block_id, block_steps), block_steps

    expanded = expand_windows_to_direct_blocks(windows, total_horizon_steps, block_steps)
    return expanded.reset_index(drop=True), block_steps


def train_quantile_torch_model(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    cfg: Part2Config,
    params: dict[str, Any],
    patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    grad_clip_norm: float,
    desc: str,
) -> float:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
    )
    pin_memory = str(cfg.device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    best_state = None
    best_valid = float("inf")
    bad_epochs = 0
    epoch_progress = create_progress(
        cfg,
        range(1, int(params["epochs"]) + 1),
        desc=desc,
        leave=False,
    )
    epoch_progress.set_postfix_str(format_params_for_progress(params))

    for epoch_idx in epoch_progress:
        model.train()
        train_losses = []
        for batch_X, batch_future_cov, batch_y in train_loader:
            batch_X = batch_X.to(cfg.device)
            batch_future_cov = batch_future_cov.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            optimizer.zero_grad()
            pred = model(batch_X, batch_future_cov)
            loss = quantile_loss_torch(pred, batch_y, cfg.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch_X, batch_future_cov, batch_y in valid_loader:
                batch_X = batch_X.to(cfg.device)
                batch_future_cov = batch_future_cov.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                pred = model(batch_X, batch_future_cov)
                loss = quantile_loss_torch(pred, batch_y, cfg.quantiles)
                valid_losses.append(float(loss.item()))
        mean_valid = float(np.mean(valid_losses))
        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        if not np.isfinite(mean_valid):
            mean_valid = float("inf")
        scheduler.step(mean_valid)
        epoch_progress.set_postfix(
            epoch=f"{epoch_idx}/{int(params['epochs'])}",
            params=format_params_for_progress(params),
            train=f"{mean_train:.4f}" if np.isfinite(mean_train) else "nan",
            valid=f"{mean_valid:.4f}" if np.isfinite(mean_valid) else "inf",
        )

        if mean_valid < best_valid:
            best_valid = mean_valid
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    epoch_progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_valid


def add_residual_baseline_to_quantile(
    values: np.ndarray,
    model_artifact: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    if not bool(model_artifact.get("residual_target", False)):
        return values
    target_input_indices = list(model_artifact.get("target_input_indices", []))
    if not target_input_indices:
        return values
    baseline = X[:, -1, target_input_indices].astype(np.float32)
    return (values + baseline[:, None, :]).astype(np.float32, copy=False)


def sequence_workflow_spec(model_key: str, cfg: Part2Config) -> dict[str, Any]:
    if model_key == "tcn":
        return {
            "model_name": "TCN",
            "grid": cfg.tcn_grid,
            "max_train_windows": cfg.max_train_windows_tcn,
            "max_valid_windows": cfg.max_valid_windows_tcn,
            "block_steps": cfg.tcn_rollout_steps,
            "patience": cfg.tcn_patience,
            "scheduler_patience": cfg.tcn_scheduler_patience,
            "scheduler_factor": cfg.tcn_scheduler_factor,
            "grad_clip_norm": cfg.tcn_grad_clip_norm,
        }
    if model_key == "transformer":
        return {
            "model_name": "Transformer",
            "grid": cfg.transformer_grid,
            "max_train_windows": cfg.max_train_windows_transformer,
            "max_valid_windows": cfg.max_valid_windows_transformer,
            "block_steps": cfg.transformer_rollout_steps,
            "patience": cfg.transformer_patience,
            "scheduler_patience": cfg.transformer_scheduler_patience,
            "scheduler_factor": cfg.transformer_scheduler_factor,
            "grad_clip_norm": cfg.transformer_grad_clip_norm,
        }
    raise ValueError(f"Неподдерживаемая sequence-модель: {model_key}")


def build_sequence_model(model_key: str, data: SequenceDirectWindowData, cfg: Part2Config, params: dict[str, Any]) -> nn.Module:
    common = {
        "input_dim": data.input_dim,
        "future_cov_dim": data.future_cov_dim,
        "horizon_steps": data.horizon_steps,
        "n_targets": data.n_targets,
        "quantiles": cfg.quantiles,
    }
    if model_key == "tcn":
        return TCNQuantileRegressor(
            **common,
            channels=tuple(params["channels"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
        )
    if model_key == "transformer":
        return TransformerQuantileRegressor(
            **common,
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
        )
    raise ValueError(f"Неподдерживаемая sequence-модель: {model_key}")


def fit_sequence_model_lazy(
    model_key: str,
    frame: pd.DataFrame,
    train_windows: pd.DataFrame,
    valid_windows: pd.DataFrame,
    cfg: Part2Config,
    params: dict[str, Any],
) -> dict[str, Any]:
    set_seed(cfg.random_state)
    spec = sequence_workflow_spec(model_key, cfg)
    train_data = build_sequence_direct_window_data(frame, train_windows, cfg)
    valid_data = replace_sequence_direct_windows(train_data, valid_windows)
    feature_scaler, future_cov_scaler, target_scaler = fit_sequence_direct_scalers(train_data)
    model = build_sequence_model(model_key, train_data, cfg, params).to(cfg.device)
    best_valid = train_quantile_torch_model(
        model,
        LazyWindowFutureTensorDataset(train_data, feature_scaler, future_cov_scaler, target_scaler),
        LazyWindowFutureTensorDataset(valid_data, feature_scaler, future_cov_scaler, target_scaler),
        cfg,
        params,
        patience=spec["patience"],
        scheduler_patience=spec["scheduler_patience"],
        scheduler_factor=spec["scheduler_factor"],
        grad_clip_norm=spec["grad_clip_norm"],
        desc=f"Train {spec['model_name']}",
    )
    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "future_cov_scaler": future_cov_scaler,
        "target_scaler": target_scaler,
        "best_valid_loss": best_valid,
        "residual_target": train_data.residual_target,
        "target_input_indices": resolve_target_input_indices(train_data.input_cols, train_data.target_cols),
    }


def predict_sequence_quantiles(
    model_artifact: dict[str, Any],
    X: np.ndarray,
    future_known_cov: np.ndarray,
    cfg: Part2Config,
    batch_size: int = 512,
) -> dict[float, np.ndarray]:
    model: nn.Module = model_artifact["model"]
    X_scaled = transform_with_scaler(X, model_artifact["feature_scaler"])
    future_cov_scaled = transform_with_scaler(future_known_cov, model_artifact["future_cov_scaler"])
    horizon = int(getattr(model, "horizon_steps"))
    loader = DataLoader(
        WindowFutureTensorDataset(
            X_scaled,
            future_cov_scaled,
            np.zeros((len(X_scaled), horizon, len(cfg.active_targets)), dtype=np.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_future_cov, _ in loader:
            pred = model(batch_X.to(cfg.device), batch_future_cov.to(cfg.device)).cpu().numpy()
            preds.append(pred)
    pred_array = np.concatenate(preds, axis=0)
    return {
        quantile: add_residual_baseline_to_quantile(
            inverse_with_scaler(pred_array[..., idx], model_artifact["target_scaler"]),
            model_artifact,
            X,
        )
        for idx, quantile in enumerate(cfg.quantiles)
    }


def predict_sequence_quantiles_direct(
    model_artifact: dict[str, Any],
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Part2Config,
) -> dict[str, Any]:
    block_steps = int(getattr(model_artifact["model"], "horizon_steps"))
    block_windows = expand_windows_to_direct_blocks(windows, cfg.horizon_steps, block_steps)
    X, future_known_cov, y_true_blocks = build_sequence_direct_arrays(frame, block_windows, cfg)
    block_quantiles = predict_sequence_quantiles(model_artifact, X, future_known_cov, cfg)
    y_true, quantiles = stitch_direct_block_quantiles(block_windows, y_true_blocks, block_quantiles)
    return {"y_true": y_true, "quantiles": quantiles}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def rank_metric(values: pd.Series, ascending: bool) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.notna()
    if valid.any():
        ranked = values.loc[valid].rank(method="dense", ascending=ascending).astype("Int64")
        result.loc[valid] = ranked
    return result


def build_comparison_export(outputs: dict[str, Any]) -> pd.DataFrame:
    comparison = outputs["comparison"].copy()
    bundle: ExperimentBundle = outputs["bundle"]
    cfg: Part2Config = outputs["config"]
    enabled_models = ", ".join(resolve_enabled_model_keys(cfg.enabled_models))
    metadata = {
        "power_mode_filter": cfg.power_mode_filter or "",
        "benchmark_target_start": cfg.benchmark_target_start or "",
        "enabled_models": enabled_models,
        "model_rule": cfg.model_rule or f"{bundle.model_step_seconds}s",
        "lookback_steps": int(cfg.lookback_steps),
        "horizon_steps": int(cfg.horizon_steps),
        "window_step": int(cfg.window_step),
        "benchmark_test_windows": int(len(bundle.benchmark_test_windows)),
        "train_months": ", ".join(bundle.train_months),
        "valid_months": ", ".join(bundle.valid_months_list),
        "test_months": ", ".join(bundle.test_months_list),
    }
    for column_name, value in reversed(list(metadata.items())):
        comparison.insert(0, column_name, value)
    return comparison


def order_prediction_interval(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered_lower = np.minimum(lower, upper)
    ordered_upper = np.maximum(lower, upper)
    return ordered_lower.astype(np.float32), ordered_upper.astype(np.float32)


def scale_prediction_interval(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_lower, ordered_upper = order_prediction_interval(lower, upper)
    scale = float(scale)
    scaled_lower = point - (point - ordered_lower) * scale
    scaled_upper = point + (ordered_upper - point) * scale
    return order_prediction_interval(scaled_lower, scaled_upper)


def fit_interval_scale(
    y_true: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    scale_grid: tuple[float, ...],
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for scale in scale_grid:
        scaled_lower, scaled_upper = scale_prediction_interval(point, lower, upper, scale)
        picp = picp_metric(y_true, scaled_lower, scaled_upper)
        pinaw = pinaw_metric(y_true, scaled_lower, scaled_upper)
        winkler = winkler_metric(y_true, scaled_lower, scaled_upper, alpha=alpha)
        candidates.append(
            {
                "scale": float(scale),
                "picp": picp,
                "pinaw": pinaw,
                "winkler": winkler,
                "coverage_gap": abs(picp - (1.0 - alpha)) if np.isfinite(picp) else float("inf"),
            }
        )
    ranked = pd.DataFrame(candidates).sort_values(["winkler", "coverage_gap", "pinaw", "scale"]).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def extract_prediction_arrays(result: dict[str, Any], cfg: Part2Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = result["predictions"]
    if {"y_true", "point", "lower", "upper"}.issubset(predictions):
        return (
            np.asarray(predictions["y_true"], dtype=np.float32),
            np.asarray(predictions["point"], dtype=np.float32),
            np.asarray(predictions["lower"], dtype=np.float32),
            np.asarray(predictions["upper"], dtype=np.float32),
        )
    if "quantiles" in predictions:
        y_true = np.asarray(predictions["y_true"], dtype=np.float32)
        point = np.asarray(predictions["quantiles"][cfg.primary_quantile], dtype=np.float32)
        lower, upper = order_prediction_interval(
            np.asarray(predictions["quantiles"][min(cfg.quantiles)], dtype=np.float32),
            np.asarray(predictions["quantiles"][max(cfg.quantiles)], dtype=np.float32),
        )
        return y_true, point, lower, upper
    return (
        np.asarray(predictions["y_true"], dtype=np.float32),
        np.asarray(predictions["point"], dtype=np.float32),
        np.asarray(predictions["lower"], dtype=np.float32),
        np.asarray(predictions["upper"], dtype=np.float32),
    )


def build_target_timestamp_matrix(frame: pd.DataFrame, test_windows: pd.DataFrame) -> np.ndarray:
    matrix: list[list[str]] = []
    for _, window_row in test_windows.iterrows():
        timestamps = frame.loc[int(window_row["target_start_pos"]) : int(window_row["target_end_pos"]), "timestamp"].astype(str).tolist()
        matrix.append(timestamps)
    return np.asarray(matrix, dtype=object)


def build_model_test_prediction_table(
    result: dict[str, Any],
    bundle: ExperimentBundle,
    cfg: Part2Config,
) -> pd.DataFrame:
    model_name = str(result["model"])
    test_windows = result["test_windows"].reset_index(drop=True)
    y_true, point, lower, upper = extract_prediction_arrays(result, cfg)
    target_timestamps = build_target_timestamp_matrix(bundle.frame, test_windows)

    rows: list[dict[str, Any]] = []
    for window_idx, window_row in test_windows.iterrows():
        for horizon_idx in range(cfg.horizon_steps):
            target_ts = target_timestamps[window_idx, horizon_idx]
            for target_idx, target_name in enumerate(cfg.active_targets):
                rows.append(
                    {
                        "model": model_name,
                        "window_id": int(window_idx),
                        "segment_id": int(window_row["segment_id"]),
                        "power_mode_band": window_row["power_mode_band"],
                        "origin_ts": str(window_row["origin_ts"]),
                        "target_ts": target_ts,
                        "horizon_step": int(horizon_idx + 1),
                        "target_name": target_name,
                        "actual_value": float(y_true[window_idx, horizon_idx, target_idx]),
                        "pred_point": float(point[window_idx, horizon_idx, target_idx]),
                        "pred_lower": float(lower[window_idx, horizon_idx, target_idx]),
                        "pred_upper": float(upper[window_idx, horizon_idx, target_idx]),
                    }
                )
    return pd.DataFrame(rows)


def export_benchmark_results(outputs: dict[str, Any], out_path: str | Path) -> Path:
    target_path = Path(out_path)
    export_frame = build_comparison_export(outputs)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    export_frame.to_csv(target_path, index=False, encoding="utf-8")
    return target_path


def export_model_test_prediction_tables(outputs: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle: ExperimentBundle = outputs["bundle"]
    cfg: Part2Config = outputs["config"]
    exported: dict[str, Path] = {}
    for key in resolve_enabled_model_keys(cfg.enabled_models):
        if key not in outputs:
            continue
        result = outputs[key]
        table = build_model_test_prediction_table(result, bundle, cfg)
        file_name = f"{str(result['model']).lower()}_test_predictions.csv"
        target_path = out_dir / file_name
        table.to_csv(target_path, index=False, encoding="utf-8")
        exported[key] = target_path
    return exported


def build_inference_time_export(outputs: dict[str, Any]) -> pd.DataFrame:
    cfg: Part2Config = outputs["config"]
    bundle: ExperimentBundle = outputs["bundle"]
    rows: list[dict[str, Any]] = []
    metadata = {
        "model_rule": cfg.model_rule or f"{bundle.model_step_seconds}s",
        "lookback_steps": int(cfg.lookback_steps),
        "horizon_steps": int(cfg.horizon_steps),
        "window_step": int(cfg.window_step),
        "benchmark_test_windows": int(len(bundle.benchmark_test_windows)),
    }
    for key in resolve_enabled_model_keys(cfg.enabled_models):
        result = outputs.get(key)
        if not result:
            continue
        timing = dict(result.get("timing", {}))
        if not timing:
            timing = {"model": str(result.get("model", key))}
        rows.append({**metadata, **timing})

    timing_frame = pd.DataFrame(rows)
    preferred_order = [
        "model_rule",
        "lookback_steps",
        "horizon_steps",
        "window_step",
        "benchmark_test_windows",
        "model",
        "training_time_seconds",
        "tuning_time_seconds",
        "final_fit_time_seconds",
        "calibration_time_seconds",
        "inference_time_seconds",
        "total_workflow_time_seconds",
        "n_test_windows",
        "note",
    ]
    ordered_cols = [col for col in preferred_order if col in timing_frame.columns]
    remaining_cols = [col for col in timing_frame.columns if col not in ordered_cols]
    return timing_frame[ordered_cols + remaining_cols]


def export_inference_times(outputs: dict[str, Any], out_path: str | Path = "inference_time.csv") -> Path:
    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    build_inference_time_export(outputs).to_csv(target_path, index=False, encoding="utf-8")
    return target_path


def tune_sequence_model(model_key: str, bundle: ExperimentBundle, cfg: Part2Config) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = sequence_workflow_spec(model_key, cfg)
    folds = build_validation_folds(bundle, cfg)
    rows: list[dict[str, Any]] = []
    progress = create_progress(cfg, total=len(spec["grid"]) * len(folds), desc=f"Tune {spec['model_name']}", leave=False)
    for params in spec["grid"]:
        for fold_train, fold_valid, fold_name in folds:
            progress.set_postfix_str(f"fold={fold_name}, {format_params_for_progress(params)}")
            train_blocks, direct_block_steps = sample_direct_block_windows(
                fold_train,
                spec["max_train_windows"],
                cfg.horizon_steps,
                spec["block_steps"],
            )
            valid_sample = sample_windows_evenly(fold_valid, spec["max_valid_windows"])
            valid_blocks, _ = sample_direct_block_windows(
                valid_sample,
                None,
                cfg.horizon_steps,
                direct_block_steps,
            )
            artifact = fit_sequence_model_lazy(model_key, bundle.frame, train_blocks, valid_blocks, cfg, params)
            preds = predict_sequence_quantiles_direct(artifact, bundle.frame, valid_sample, cfg)
            point = preds["quantiles"][cfg.primary_quantile]
            raw_lower, raw_upper = order_prediction_interval(
                preds["quantiles"][min(cfg.quantiles)],
                preds["quantiles"][max(cfg.quantiles)],
            )
            interval_scale = fit_interval_scale(
                preds["y_true"],
                point,
                raw_lower,
                raw_upper,
                alpha=cfg.conformal_alpha,
                scale_grid=cfg.interval_scale_grid,
            )
            lower, upper = scale_prediction_interval(point, raw_lower, raw_upper, interval_scale["scale"])
            rows.append(
                {
                    "model": spec["model_name"],
                    "fold": fold_name,
                    "params": copy.deepcopy(params),
                    "wql": weighted_quantile_loss(preds["y_true"], preds["quantiles"]),
                    "mae": mae_metric(preds["y_true"], point),
                    "picp": picp_metric(preds["y_true"], lower, upper),
                    "pinaw": pinaw_metric(preds["y_true"], lower, upper),
                    "winkler": winkler_metric(preds["y_true"], lower, upper, alpha=cfg.conformal_alpha),
                    "interval_scale": float(interval_scale["scale"]),
                    "direct_block_steps": int(direct_block_steps),
                    "best_valid_loss": artifact["best_valid_loss"],
                }
            )
            progress.update(1)
    progress.close()
    history = pd.DataFrame(rows)
    history["params_key"] = history["params"].map(lambda item: repr(item))
    summary = (
        history.groupby("params_key", as_index=False)[["wql", "mae", "pinaw", "winkler", "interval_scale", "best_valid_loss"]]
        .mean()
        .sort_values(["winkler", "mae", "wql", "pinaw", "best_valid_loss"])
        .reset_index(drop=True)
    )
    best_key = summary.iloc[0]["params_key"]
    best_params = next(item for item in history["params"] if repr(item) == best_key)
    return dict(best_params), history.drop(columns=["params_key"])


def run_sequence_workflow(model_key: str, bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    workflow_start = time.perf_counter()
    spec = sequence_workflow_spec(model_key, cfg)
    tuning_start = time.perf_counter()
    best_params, tuning_history = tune_sequence_model(model_key, bundle, cfg)
    tuning_time = time.perf_counter() - tuning_start

    test_sample = bundle.benchmark_test_windows.reset_index(drop=True).copy()

    fit_start = time.perf_counter()
    fit_windows, stop_windows = build_final_fit_stop_windows(
        bundle,
        cfg,
        spec["max_train_windows"],
        spec["max_valid_windows"],
    )

    fit_blocks, direct_block_steps = sample_direct_block_windows(
        fit_windows,
        None,
        cfg.horizon_steps,
        int(best_params.get("direct_block_steps", spec["block_steps"])),
    )
    stop_blocks, _ = sample_direct_block_windows(stop_windows, None, cfg.horizon_steps, direct_block_steps)

    artifact = fit_sequence_model_lazy(model_key, bundle.frame, fit_blocks, stop_blocks, cfg, best_params)
    final_fit_time = time.perf_counter() - fit_start

    calibration_start = time.perf_counter()
    stop_preds = predict_sequence_quantiles_direct(artifact, bundle.frame, stop_windows, cfg)
    stop_point = stop_preds["quantiles"][cfg.primary_quantile]
    stop_raw_lower, stop_raw_upper = order_prediction_interval(
        stop_preds["quantiles"][min(cfg.quantiles)],
        stop_preds["quantiles"][max(cfg.quantiles)],
    )
    interval_scale = fit_interval_scale(
        stop_preds["y_true"],
        stop_point,
        stop_raw_lower,
        stop_raw_upper,
        alpha=cfg.conformal_alpha,
        scale_grid=cfg.interval_scale_grid,
    )
    calibration_time = time.perf_counter() - calibration_start

    inference_start = time.perf_counter()
    preds = predict_sequence_quantiles_direct(artifact, bundle.frame, test_sample, cfg)
    inference_time = time.perf_counter() - inference_start
    quantile_preds = preds["quantiles"]
    raw_lower, raw_upper = order_prediction_interval(
        quantile_preds[min(cfg.quantiles)],
        quantile_preds[max(cfg.quantiles)],
    )
    point = quantile_preds[cfg.primary_quantile]
    lower, upper = scale_prediction_interval(point, raw_lower, raw_upper, interval_scale["scale"])

    train_reference = build_train_reference(bundle, cfg)
    metrics = evaluate_prediction_bundle(
        model_name=spec["model_name"],
        y_true=preds["y_true"],
        point_pred=point,
        train_reference=train_reference,
        lower=lower,
        upper=upper,
        quantile_predictions=quantile_preds,
        alpha=cfg.conformal_alpha,
    )
    metrics["n_test_windows"] = int(len(test_sample))
    return {
        "model": spec["model_name"],
        "best_params": {
            **best_params,
            "forecast_strategy": "direct_multi_horizon",
            "direct_block_steps": int(direct_block_steps),
            "interval_scale": float(interval_scale["scale"]),
        },
        "tuning_history": tuning_history,
        "metrics": metrics,
        "predictions": {
            "y_true": preds["y_true"],
            "quantiles": quantile_preds,
            "point": point,
            "lower": lower,
            "upper": upper,
        },
        "test_windows": test_sample,
        "timing": {
            "model": spec["model_name"],
            "tuning_time_seconds": tuning_time,
            "final_fit_time_seconds": final_fit_time,
            "calibration_time_seconds": calibration_time,
            "inference_time_seconds": inference_time,
            "training_time_seconds": tuning_time + final_fit_time + calibration_time,
            "total_workflow_time_seconds": time.perf_counter() - workflow_start,
            "n_test_windows": int(len(test_sample)),
            "note": "",
        },
    }


def run_tcn_workflow(bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    return run_sequence_workflow("tcn", bundle, cfg)


def run_transformer_workflow(bundle: ExperimentBundle, cfg: Part2Config) -> dict[str, Any]:
    return run_sequence_workflow("transformer", bundle, cfg)


def compare_model_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = dict(result["metrics"])
        row["supports_intervals"] = bool(np.isfinite(row.get("winkler", float("nan"))))
        row["supports_probabilistic"] = bool(np.isfinite(row.get("wql", float("nan"))))
        row["best_params"] = json.dumps(make_json_safe(result.get("best_params")), ensure_ascii=False)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["point_rank"] = rank_metric(comparison["mae"], ascending=True)
    comparison["interval_rank"] = rank_metric(comparison["winkler"], ascending=True)
    comparison["probabilistic_rank"] = rank_metric(comparison["wql"], ascending=True)
    return comparison.sort_values(["point_rank", "probabilistic_rank", "interval_rank", "model"], na_position="last").reset_index(drop=True)


def run_benchmark_suite(cfg: Part2Config, bundle: ExperimentBundle | None = None) -> dict[str, Any]:
    bundle = bundle or create_experiment_bundle(cfg)
    enabled_model_keys = resolve_enabled_model_keys(cfg.enabled_models)
    workflow_catalog = {
        "sarima": ("SARIMA", run_sarima_workflow),
        "var": ("VAR", run_var_workflow),
        "lightgbm": ("LightGBM", run_lightgbm_workflow),
        "tcn": ("TCN", run_tcn_workflow),
        "transformer": ("Transformer", run_transformer_workflow),
    }
    workflow_specs = [(key, *workflow_catalog[key]) for key in enabled_model_keys]
    progress = create_progress(cfg, workflow_specs, desc="Benchmark Models", leave=False)
    results: dict[str, Any] = {}
    for key, model_name, workflow in progress:
        progress.set_postfix_str(f"model={model_name}")
        results[key] = workflow(bundle, cfg)
    progress.close()
    comparison = compare_model_results([results[key] for key in enabled_model_keys])
    outputs = {
        "config": copy.deepcopy(cfg),
        "bundle": bundle,
        "comparison": comparison,
    }
    outputs.update(results)
    return outputs
