// Typed client for the FastAPI backend. Used from server components only:
// API_URL is read at request time (http://api:8000 inside docker compose).

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface Metrics {
  mae: number;
  rmse: number;
  mape: number;
  smape?: number | null;
  bias?: number | null;
}

export interface ForecastPoint {
  timestamp: string;
  lead_h: number;
  prediction_mw: number;
  lower_mw: number;
  upper_mw: number;
  model: string;
  bucket_h: number;
  peak_risk: RiskLevel;
}

export interface DemandPoint {
  timestamp: string;
  demand_mw: number | null;
  temperature_c?: number | null;
}

export interface PeakPeriod {
  start: string;
  end: string;
  hours: number;
  max_predicted_mw: number;
  max_risk: RiskLevel;
}

export interface ForecastResponse {
  origin: string;
  horizon: number;
  generated_at: string;
  interval_coverage: number;
  peak_threshold_mw: number;
  forecast: ForecastPoint[];
  history: DemandPoint[];
  peak_periods: PeakPeriod[];
  model_trained_at: string | null;
  dataset_version: string | null;
  disclaimer: string;
}

export interface HorizonMetrics {
  horizon_h: number;
  selected_model: string;
  test: Metrics;
  cv: Metrics;
  best_baseline: string;
  best_baseline_test: Metrics;
  interval_coverage_test: number;
  interval_mean_width_mw: number;
  peak_alerts: Record<string, number>;
}

export interface MetricsResponse {
  trained_at: string;
  dataset_version: string;
  data_is_synthetic: boolean;
  horizons: HorizonMetrics[];
}

export interface ModelsResponse {
  trained_at: string;
  dataset: { source: string; version: string; rows: number; start: string; end: string; is_synthetic: boolean };
  splits: { folds: { name: string; start: string; end: string }[]; test: { start: string; end: string } };
  horizons: {
    horizon_h: number;
    selected_model: string;
    candidates: { model: string; cv: Metrics; test: Metrics }[];
    interval_coverage_test: number | null;
    interval_mean_width_mw: number | null;
  }[];
  mlflow: { tracking_uri: string | null; run_id: string | null; registered_model: string | null; registered_version: string | null } | null;
}

export interface HistoricalResponse {
  start: string;
  end: string;
  resolution: "hourly" | "daily";
  points: DemandPoint[];
  summary: {
    n: number;
    mean_mw: number | null;
    max_mw: number | null;
    max_timestamp: string | null;
    min_mw: number | null;
    min_timestamp: string | null;
    change_vs_previous_period_pct: number | null;
  };
}

export interface BacktestResponse {
  horizon_h: number;
  model: string;
  metrics: Metrics;
  points: { timestamp: string; actual_mw: number; prediction_mw: number; lower_mw: number | null; upper_mw: number | null }[];
}

export type ApiResult<T> = { ok: true; data: T } | { ok: false; status: number; message: string };

export const API_URL = process.env.API_URL ?? "http://localhost:8000";

export async function apiGet<T>(path: string): Promise<ApiResult<T>> {
  try {
    const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
    if (!res.ok) {
      let message = res.statusText;
      try {
        const body = await res.json();
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* keep statusText */
      }
      return { ok: false, status: res.status, message };
    }
    return { ok: true, data: (await res.json()) as T };
  } catch {
    return { ok: false, status: 0, message: `API unreachable at ${API_URL}. Is the api service running?` };
  }
}
