import Link from "next/link";
import { ForecastChart } from "@/components/charts";
import { Card, ErrorPanel, PageHeader, RiskBadge, StatTile } from "@/components/ui";
import { apiGet, type ForecastResponse, type MetricsResponse, type RiskLevel } from "@/lib/api";
import { fmtDateTime, fmtMW, fmtNum, fmtHour, horizonLabel } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const [fc, metrics] = await Promise.all([
    apiGet<ForecastResponse>("/forecast?horizon=24&include_history_hours=48"),
    apiGet<MetricsResponse>("/metrics"),
  ]);
  if (!fc.ok) return <><PageHeader title="Dashboard" /><ErrorPanel status={fc.status} message={fc.message} /></>;

  const f = fc.data;
  const current = f.history.at(-1);
  const next = f.forecast[0];
  const peak = f.forecast.reduce((a, b) => (b.prediction_mw > a.prediction_mw ? b : a));
  const rank: Record<RiskLevel, number> = { LOW: 0, MEDIUM: 1, HIGH: 2 };
  const maxRisk = f.forecast.reduce<RiskLevel>((a, p) => (rank[p.peak_risk] > rank[a] ? p.peak_risk : a), "LOW");
  const highHours = f.forecast.filter((p) => p.peak_risk === "HIGH").length;
  const h1 = metrics.ok ? metrics.data.horizons.find((h) => h.horizon_h === 1) : undefined;
  const h24 = metrics.ok ? metrics.data.horizons.find((h) => h.horizon_h === 24) : undefined;

  return (
    <>
      <PageHeader
        title="Dashboard"
        subtitle={`Situation as of the last observed hour, ${fmtDateTime(f.origin)} (UTC = Nouakchott time). Forecasts use only data available at that time.`}
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="Current demand" value={fmtMW(current?.demand_mw)} detail={current ? fmtDateTime(current.timestamp) : undefined} />
        <StatTile
          label="Next hour forecast"
          value={fmtMW(next.prediction_mw)}
          detail={`${fmtMW(next.lower_mw)} – ${fmtMW(next.upper_mw)} · ${next.model}`}
        />
        <StatTile label="Peak in next 24 h" value={fmtMW(peak.prediction_mw)} detail={`at ${fmtDateTime(peak.timestamp)} · upper ${fmtMW(peak.upper_mw)}`} />
        <StatTile
          label="Peak risk (24 h)"
          value={<RiskBadge risk={maxRisk} />}
          detail={`${highHours} h forecast above ${fmtMW(f.peak_threshold_mw)} (P95, trailing year)`}
        />
      </div>

      <Card className="mt-6" title="Last 48 hours and next 24 hours" subtitle={`Shaded band: ${Math.round(f.interval_coverage * 100)} % prediction interval (empirical, not a guarantee).`}>
        <ForecastChart history={f.history} forecast={f.forecast} threshold={f.peak_threshold_mw} />
      </Card>

      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Models in use" subtitle="Selected per horizon on cross-validation MAE" className="lg:col-span-2">
          {metrics.ok ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                    <th className="py-2 pr-4 font-medium">Horizon</th>
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 text-right font-medium">MAE (MW)</th>
                    <th className="py-2 pr-4 text-right font-medium">RMSE (MW)</th>
                    <th className="py-2 pr-4 text-right font-medium">MAPE</th>
                    <th className="py-2 pr-4 text-right font-medium">Best baseline MAE</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {metrics.data.horizons.map((h) => (
                    <tr key={h.horizon_h} className="border-b border-line last:border-0">
                      <td className="py-2 pr-4">{horizonLabel(h.horizon_h)}</td>
                      <td className="py-2 pr-4 font-medium">{h.selected_model}</td>
                      <td className="py-2 pr-4 text-right">{fmtNum(h.test.mae)}</td>
                      <td className="py-2 pr-4 text-right">{fmtNum(h.test.rmse)}</td>
                      <td className="py-2 pr-4 text-right">{fmtNum(h.test.mape)} %</td>
                      <td className="py-2 pr-4 text-right text-ink-2">{fmtNum(h.best_baseline_test.mae)} <span className="text-xs">({h.best_baseline})</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-muted">Metrics on the held-out test period (synthetic data). <Link className="underline" href="/models">All models →</Link></p>
            </div>
          ) : (
            <ErrorPanel status={metrics.status} message={metrics.message} />
          )}
        </Card>
        <Card title="Next 24 h at a glance">
          <ul className="space-y-2 text-sm">
            <li className="flex justify-between"><span className="text-ink-2">1 h model</span><span className="font-medium">{h1?.selected_model ?? next.model}</span></li>
            <li className="flex justify-between"><span className="text-ink-2">24 h model</span><span className="font-medium">{h24?.selected_model ?? "–"}</span></li>
            <li className="flex justify-between"><span className="text-ink-2">24 h model MAE</span><span className="tabular">{fmtMW(h24?.test.mae, 1)}</span></li>
            <li className="flex justify-between"><span className="text-ink-2">24 h model RMSE</span><span className="tabular">{fmtMW(h24?.test.rmse, 1)}</span></li>
            <li className="flex justify-between"><span className="text-ink-2">24 h model MAPE</span><span className="tabular">{fmtNum(h24?.test.mape)} %</span></li>
            <li className="flex justify-between"><span className="text-ink-2">Peak periods</span><span className="tabular">{f.peak_periods.length ? f.peak_periods.map((p) => `${fmtHour(p.start)}–${fmtHour(p.end)}`).join(", ") : "none"}</span></li>
          </ul>
        </Card>
      </div>
    </>
  );
}
