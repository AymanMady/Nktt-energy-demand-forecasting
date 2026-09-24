"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TooltipContentProps } from "recharts";
import type { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";
import { fmtDateTime, fmtDayHour, fmtDate } from "@/lib/format";

type Row = {
  t: number;
  actual?: number | null;
  forecast?: number | null;
  band?: [number, number] | null;
};

const axisProps = {
  stroke: "var(--baseline)",
  tick: { fill: "var(--muted)", fontSize: 11 },
  tickLine: false,
} as const;

const yAxisProps = {
  ...axisProps,
  width: 52,
  label: { value: "MW", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 },
} as const;

/** Legend text uses text ink; the swatch alone carries the series colour. */
const legendText = (value: string) => <span style={{ color: "var(--ink-2)" }}>{value}</span>;

function ChartTooltip({ active, payload, label }: TooltipContentProps<ValueType, NameType>) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as Row;
  const items: [string, string, string][] = [];
  if (row.actual != null) items.push(["Actual", `${row.actual.toFixed(1)} MW`, "var(--series-1)"]);
  if (row.forecast != null) items.push(["Forecast", `${row.forecast.toFixed(1)} MW`, "var(--series-2)"]);
  if (row.band) items.push(["Interval", `${row.band[0].toFixed(0)} – ${row.band[1].toFixed(0)} MW`, "var(--band)"]);
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="mb-1 font-medium text-ink">{fmtDateTime(new Date(Number(label)).toISOString())}</div>
      {items.map(([k, v, c]) => (
        <div key={k} className="flex items-center gap-2 text-ink-2">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: c }} />
          <span>{k}</span>
          <span className="ml-auto tabular pl-4 font-medium text-ink">{v}</span>
        </div>
      ))}
    </div>
  );
}

function yDomain(rows: Row[], extra: number[] = []): [number, number] {
  const vals = rows.flatMap((r) => [r.actual, r.forecast, r.band?.[0], r.band?.[1]]).filter((v): v is number => v != null);
  const all = [...vals, ...extra];
  if (!all.length) return [0, 1];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 5;
  return [Math.floor((lo - pad) / 10) * 10, Math.ceil((hi + pad) / 10) * 10];
}

export function ForecastChart({
  history,
  forecast,
  threshold,
  height = 340,
}: {
  history: { timestamp: string; demand_mw: number | null }[];
  forecast: { timestamp: string; prediction_mw: number; lower_mw: number; upper_mw: number }[];
  threshold?: number;
  height?: number;
}) {
  const rows: Row[] = [
    ...history.map((h) => ({ t: Date.parse(h.timestamp), actual: h.demand_mw })),
    ...forecast.map((f) => ({ t: Date.parse(f.timestamp), forecast: f.prediction_mw, band: [f.lower_mw, f.upper_mw] as [number, number] })),
  ];
  // Connect the forecast line to the last observed point.
  if (history.length && forecast.length) {
    const last = rows[history.length - 1];
    last.forecast = last.actual ?? null;
  }
  const origin = history.length ? Date.parse(history[history.length - 1].timestamp) : undefined;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v) => fmtDayHour(new Date(v).toISOString())} minTickGap={40} {...axisProps} />
        <YAxis domain={yDomain(rows, threshold ? [threshold] : [])} {...yAxisProps} />
        <Tooltip content={ChartTooltip} />
        <Legend verticalAlign="top" align="right" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} formatter={legendText} />
        <Area dataKey="band" name="90 % interval" stroke="none" fill="var(--band)" isAnimationActive={false} connectNulls={false} legendType="square" />
        <Line dataKey="actual" name="Actual demand" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
        <Line dataKey="forecast" name="Forecast" stroke="var(--series-2)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
        {origin && <ReferenceLine x={origin} stroke="var(--baseline)" label={{ value: "now", position: "insideTopLeft", fill: "var(--muted)", fontSize: 11 }} />}
        {threshold && (
          <ReferenceLine y={threshold} stroke="var(--critical)" strokeDasharray="4 4" label={{ value: `Peak threshold ${threshold.toFixed(0)} MW`, position: "insideTopLeft", fill: "var(--ink-2)", fontSize: 11 }} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function BacktestChart({
  points,
  height = 300,
}: {
  points: { timestamp: string; actual_mw: number; prediction_mw: number; lower_mw: number | null; upper_mw: number | null }[];
  height?: number;
}) {
  const rows: Row[] = points.map((p) => ({
    t: Date.parse(p.timestamp),
    actual: p.actual_mw,
    forecast: p.prediction_mw,
    band: p.lower_mw != null && p.upper_mw != null ? [p.lower_mw, p.upper_mw] : null,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v) => fmtDate(new Date(v).toISOString())} minTickGap={50} {...axisProps} />
        <YAxis domain={yDomain(rows)} {...yAxisProps} />
        <Tooltip content={ChartTooltip} />
        <Legend verticalAlign="top" align="right" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} formatter={legendText} />
        <Area dataKey="band" name="90 % interval" stroke="none" fill="var(--band)" isAnimationActive={false} legendType="square" />
        <Line dataKey="actual" name="Actual demand" stroke="var(--series-1)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line dataKey="forecast" name="Forecast (out-of-sample)" stroke="var(--series-2)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function HistoryChart({
  points,
  mean,
  daily,
  height = 340,
}: {
  points: { timestamp: string; demand_mw: number | null }[];
  mean?: number | null;
  daily: boolean;
  height?: number;
}) {
  const rows: Row[] = points.map((p) => ({ t: Date.parse(p.timestamp), actual: p.demand_mw }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v) => (daily ? fmtDate : fmtDayHour)(new Date(v).toISOString())} minTickGap={50} {...axisProps} />
        <YAxis domain={yDomain(rows)} {...yAxisProps} />
        <Tooltip content={ChartTooltip} />
        <Line dataKey="actual" name={daily ? "Daily mean demand" : "Hourly demand"} stroke="var(--series-1)" strokeWidth={daily ? 2 : 1.25} dot={false} isAnimationActive={false} />
        {mean != null && (
          <ReferenceLine y={mean} stroke="var(--muted)" label={{ value: `Mean ${mean.toFixed(0)} MW`, position: "insideTopRight", fill: "var(--ink-2)", fontSize: 11 }} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
