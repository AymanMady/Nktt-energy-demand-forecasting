import { HistoryChart } from "@/components/charts";
import { Card, ErrorPanel, PageHeader, Segmented, StatTile } from "@/components/ui";
import { apiGet, type HistoricalResponse } from "@/lib/api";
import { fmtDate, fmtDateTime, fmtMW, fmtPct } from "@/lib/format";

export const dynamic = "force-dynamic";

const OPTIONS = [
  { value: "7", label: "7 days" },
  { value: "30", label: "30 days" },
  { value: "90", label: "90 days" },
  { value: "365", label: "1 year" },
  { value: "1100", label: "All" },
];

export default async function HistoricalPage({ searchParams }: { searchParams: Promise<{ days?: string }> }) {
  const { days } = await searchParams;
  const d = OPTIONS.some((o) => o.value === days) ? Number(days) : 30;
  const resolution = d > 30 ? "daily" : "hourly";
  const res = await apiGet<HistoricalResponse>(`/historical-demand?days=${d}&resolution=${resolution}`);

  return (
    <>
      <PageHeader title="Historical demand" subtitle="Observed demand from the warehouse (PostgreSQL). Longer ranges are shown as daily means.">
        <Segmented options={OPTIONS} active={String(d)} basePath="/historical" param="days" />
      </PageHeader>
      {!res.ok ? (
        <ErrorPanel status={res.status} message={res.message} />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile label="Mean" value={fmtMW(res.data.summary.mean_mw, 1)} detail={`${fmtDate(res.data.start)} → ${fmtDate(res.data.end)}`} />
            <StatTile label="Maximum" value={fmtMW(res.data.summary.max_mw, 1)} detail={res.data.summary.max_timestamp ? fmtDateTime(res.data.summary.max_timestamp) : undefined} />
            <StatTile label="Minimum" value={fmtMW(res.data.summary.min_mw, 1)} detail={res.data.summary.min_timestamp ? fmtDateTime(res.data.summary.min_timestamp) : undefined} />
            <StatTile
              label="Change vs previous period"
              value={res.data.summary.change_vs_previous_period_pct == null ? "–" : `${res.data.summary.change_vs_previous_period_pct > 0 ? "+" : ""}${fmtPct(res.data.summary.change_vs_previous_period_pct)}`}
              detail="mean demand, same-length period just before"
            />
          </div>
          <Card className="mt-6" title={resolution === "daily" ? "Daily mean demand" : "Hourly demand"} subtitle={`${res.data.summary.n.toLocaleString("en")} hourly observations`}>
            <HistoryChart points={res.data.points} mean={res.data.summary.mean_mw} daily={resolution === "daily"} height={380} />
          </Card>
        </>
      )}
    </>
  );
}
