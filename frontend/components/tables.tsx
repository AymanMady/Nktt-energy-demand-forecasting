import type { Metrics } from "@/lib/api";
import { fmtNum } from "@/lib/format";

export function MetricsTable({
  rows,
  selected,
}: {
  rows: { model: string; cv?: Metrics; test: Metrics }[];
  selected?: string;
}) {
  const hasCv = rows.some((r) => r.cv);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
            <th className="py-2 pr-4 font-medium">Model</th>
            {hasCv && <th className="py-2 pr-4 text-right font-medium">CV MAE</th>}
            <th className="py-2 pr-4 text-right font-medium">Test MAE</th>
            <th className="py-2 pr-4 text-right font-medium">Test RMSE</th>
            <th className="py-2 pr-4 text-right font-medium">Test MAPE</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r) => (
            <tr key={r.model} className="border-b border-line last:border-0">
              <td className="py-2 pr-4">
                <span className={r.model === selected ? "font-semibold" : ""}>{r.model}</span>
                {r.model === selected && (
                  <span className="ml-2 rounded-full bg-accent px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white">in use</span>
                )}
              </td>
              {hasCv && <td className="py-2 pr-4 text-right">{fmtNum(r.cv?.mae)}</td>}
              <td className="py-2 pr-4 text-right">{fmtNum(r.test.mae)}</td>
              <td className="py-2 pr-4 text-right">{fmtNum(r.test.rmse)}</td>
              <td className="py-2 pr-4 text-right">{fmtNum(r.test.mape)} %</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted">MAE / RMSE in MW, MAPE in %. Selection uses CV MAE only; test = held-out period.</p>
    </div>
  );
}
