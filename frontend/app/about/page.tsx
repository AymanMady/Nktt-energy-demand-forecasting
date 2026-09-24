import { Card, PageHeader } from "@/components/ui";

export default function AboutPage() {
  return (
    <>
      <PageHeader title="About this prototype" subtitle="An end-to-end electricity demand forecasting platform built as a portfolio project for Nouakchott, Mauritania." />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Important: prototype data">
          <p className="text-sm text-ink-2">
            This dashboard does <strong className="text-ink">not</strong> use SOMELEC data. Demand is <strong className="text-ink">synthetic</strong>, produced by a documented
            generator (daily and weekly cycles, seasonality, growth trend, temperature and humidity effects, heatwaves, load shedding and meter glitches). Weather can
            optionally come from the public Open-Meteo archive. The project is not affiliated with or endorsed by SOMELEC, and none of the numbers describe real consumption.
          </p>
        </Card>
        <Card title="Question answered">
          <p className="text-sm text-ink-2">
            <em>What will electricity demand in Nouakchott be over the next hours and days?</em> Forecasts at 1 h, 6 h, 24 h and 7 days, with prediction intervals and
            experimental high-demand alerts, to illustrate how an operator could anticipate peaks and plan production.
          </p>
        </Card>
        <Card title="Method">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            <li>Ingestion with data-quality validation into PostgreSQL; dbt staging → intermediate → marts.</li>
            <li>Leakage-safe features: a horizon-H model only sees data available H hours before the target.</li>
            <li>Candidates: naive, seasonal naive (24 h), weekly naive (168 h), SARIMA, XGBoost, LightGBM.</li>
            <li>Chronological cross-validation over a full year for selection; a held-out final quarter for reporting.</li>
            <li>Intervals: empirical quantiles of out-of-sample errors; coverage is measured, not assumed.</li>
            <li>Peak risk: P95 of the trailing year — a statistical convention, not an operator rule.</li>
            <li>Experiments and the production model are tracked in MLflow.</li>
          </ul>
        </Card>
        <Card title="With real operator data">
          <p className="text-sm text-ink-2">
            The pipeline accepts any hourly load export through a YAML column mapping. With real data it could be extended to forecasts per zone or substation,
            operational peak thresholds, loss analysis, anomaly detection and production monitoring. The real data format would have to be agreed with the data owner.
          </p>
        </Card>
      </div>
    </>
  );
}
