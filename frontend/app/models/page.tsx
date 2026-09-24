import { Card, ErrorPanel, PageHeader } from "@/components/ui";
import { MetricsTable } from "@/components/tables";
import { apiGet, type MetricsResponse, type ModelsResponse } from "@/lib/api";
import { fmtDate, fmtDateTime, fmtNum, fmtPct, horizonLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ModelsPage() {
  const [models, metrics] = await Promise.all([apiGet<ModelsResponse>("/models"), apiGet<MetricsResponse>("/metrics")]);
  if (!models.ok) return <><PageHeader title="Models" /><ErrorPanel status={models.status} message={models.message} /></>;
  const m = models.data;
  const peaks = new Map(metrics.ok ? metrics.data.horizons.map((h) => [h.horizon_h, h.peak_alerts]) : []);

  return (
    <>
      <PageHeader
        title="Models"
        subtitle="Every candidate is evaluated with the same chronological protocol. The model in use is chosen on cross-validation MAE only (simplest model wins within 1 %); the test period is reported, never used for choices."
      />
      <div className="mb-6 grid grid-cols-1 gap-4 text-sm md:grid-cols-3">
        <Card title="Training run">
          <div className="space-y-1 text-ink-2">
            <div>Trained: <span className="text-ink">{fmtDateTime(m.trained_at)}</span></div>
            <div>Dataset: <span className="text-ink">{m.dataset.source}</span> · version <code className="text-ink">{m.dataset.version}</code></div>
            <div>{m.dataset.rows.toLocaleString("en")} rows, {fmtDate(m.dataset.start)} → {fmtDate(m.dataset.end)}</div>
          </div>
        </Card>
        <Card title="Validation">
          <div className="space-y-1 text-ink-2">
            <div>{m.splits.folds.length} expanding-window CV folds</div>
            <div>{fmtDate(m.splits.folds[0].start)} → {fmtDate(m.splits.folds.at(-1)!.end)}</div>
            <div>Test: <span className="text-ink">{fmtDate(m.splits.test.start)} → {fmtDate(m.splits.test.end)}</span></div>
          </div>
        </Card>
        <Card title="MLflow">
          <div className="space-y-1 break-all text-ink-2">
            <div>Registered model: <span className="text-ink">{m.mlflow?.registered_model ?? "–"}</span> {m.mlflow?.registered_version ? `v${m.mlflow.registered_version} (alias champion)` : ""}</div>
            <div>Run: <code className="text-xs">{m.mlflow?.run_id ?? "–"}</code></div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {m.horizons.map((h) => {
          const pk = peaks.get(h.horizon_h);
          return (
            <Card key={h.horizon_h} title={`Horizon: ${horizonLabel(h.horizon_h)}`} subtitle={`In use: ${h.selected_model}`}>
              <MetricsTable rows={h.candidates} selected={h.selected_model} />
              <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-ink-2">
                <div className="rounded-lg bg-surface-2 p-3">
                  <div className="font-medium text-ink">90 % interval on test</div>
                  <div className="tabular">coverage {fmtPct((h.interval_coverage_test ?? NaN) * 100)} · width {fmtNum(h.interval_mean_width_mw, 1)} MW</div>
                </div>
                {pk && (
                  <div className="rounded-lg bg-surface-2 p-3">
                    <div className="font-medium text-ink">Peak alerts on test (HIGH)</div>
                    <div className="tabular">precision {fmtPct(pk.high_precision * 100, 0)} · recall {fmtPct(pk.high_recall * 100, 0)}</div>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </>
  );
}
