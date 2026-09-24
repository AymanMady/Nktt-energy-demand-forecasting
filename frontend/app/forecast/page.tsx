import { BacktestChart, ForecastChart } from "@/components/charts";
import { Card, ErrorPanel, PageHeader, RiskBadge, Segmented, StatTile } from "@/components/ui";
import { apiGet, type BacktestResponse, type ForecastResponse } from "@/lib/api";
import { fmtDateTime, fmtMW, fmtNum, horizonLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

const OPTIONS = [
  { value: "6", label: "6 h" },
  { value: "24", label: "24 h" },
  { value: "168", label: "7 days" },
];

export default async function ForecastPage({ searchParams }: { searchParams: Promise<{ h?: string }> }) {
  const { h } = await searchParams;
  const horizon = OPTIONS.some((o) => o.value === h) ? Number(h) : 24;
  const historyHours = horizon === 168 ? 168 : 48;
  const bucket = horizon === 6 ? 6 : horizon === 24 ? 24 : 168;

  const [fc, bt] = await Promise.all([
    apiGet<ForecastResponse>(`/forecast?horizon=${horizon}&include_history_hours=${historyHours}`),
    apiGet<BacktestResponse>(`/backtest?horizon=${bucket}&days=${horizon === 168 ? 28 : 14}`),
  ]);

  return (
    <>
      <PageHeader title="Forecast" subtitle="Point forecast, prediction interval and experimental peak risk for the selected horizon.">
        <Segmented options={OPTIONS} active={String(horizon)} basePath="/forecast" param="h" />
      </PageHeader>

      {!fc.ok ? (
        <ErrorPanel status={fc.status} message={fc.message} />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StatTile label="Forecast origin" value={<span className="text-lg">{fmtDateTime(fc.data.origin)}</span>} detail="last observed hour" />
            <StatTile
              label={`Max forecast (${horizonLabel(horizon)})`}
              value={fmtMW(Math.max(...fc.data.forecast.map((p) => p.prediction_mw)))}
              detail={`threshold ${fmtMW(fc.data.peak_threshold_mw)}`}
            />
            <StatTile label="Hours at HIGH risk" value={fc.data.forecast.filter((p) => p.peak_risk === "HIGH").length} detail={`${fc.data.forecast.filter((p) => p.peak_risk === "MEDIUM").length} h at MEDIUM risk`} />
          </div>

          <Card className="mt-6" title={`Actual demand and ${horizonLabel(horizon)} forecast`} subtitle={`Band: ${Math.round(fc.data.interval_coverage * 100)} % empirical prediction interval. Dashed line: experimental peak threshold (P95 of the trailing 365 days, not an operator rule).`}>
            <ForecastChart history={fc.data.history} forecast={fc.data.forecast} threshold={fc.data.peak_threshold_mw} height={380} />
          </Card>

          <Card className="mt-6" title="Forecast table" subtitle="Each lead time uses the model of the smallest horizon bucket that covers it.">
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface">
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                    <th className="py-2 pr-4 font-medium">Time</th>
                    <th className="py-2 pr-4 text-right font-medium">Lower</th>
                    <th className="py-2 pr-4 text-right font-medium">Forecast</th>
                    <th className="py-2 pr-4 text-right font-medium">Upper</th>
                    <th className="py-2 pr-4 font-medium">Risk</th>
                    <th className="py-2 pr-4 font-medium">Model</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {fc.data.forecast.map((p) => (
                    <tr key={p.timestamp} className="border-b border-line last:border-0">
                      <td className="py-1.5 pr-4">{fmtDateTime(p.timestamp)}</td>
                      <td className="py-1.5 pr-4 text-right text-ink-2">{fmtNum(p.lower_mw, 1)}</td>
                      <td className="py-1.5 pr-4 text-right font-medium">{fmtNum(p.prediction_mw, 1)}</td>
                      <td className="py-1.5 pr-4 text-right text-ink-2">{fmtNum(p.upper_mw, 1)}</td>
                      <td className="py-1.5 pr-4"><RiskBadge risk={p.peak_risk} /></td>
                      <td className="py-1.5 pr-4 text-ink-2">{p.model} <span className="text-xs text-muted">(h{p.bucket_h})</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      <Card
        className="mt-6"
        title={`How good is the ${horizonLabel(bucket)} model? Backtest on the held-out test period`}
        subtitle="Out-of-sample forecasts issued H hours before each target hour, compared with what actually happened (synthetic data)."
      >
        {bt.ok ? (
          <>
            <div className="mb-3 flex flex-wrap gap-4 text-sm text-ink-2">
              <span>Model: <strong className="text-ink">{bt.data.model}</strong></span>
              <span className="tabular">MAE {fmtNum(bt.data.metrics.mae)} MW</span>
              <span className="tabular">RMSE {fmtNum(bt.data.metrics.rmse)} MW</span>
              <span className="tabular">MAPE {fmtNum(bt.data.metrics.mape)} %</span>
            </div>
            <BacktestChart points={bt.data.points} />
          </>
        ) : (
          <ErrorPanel status={bt.status} message={bt.message} />
        )}
      </Card>
    </>
  );
}
