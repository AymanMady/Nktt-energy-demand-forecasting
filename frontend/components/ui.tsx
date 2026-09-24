import Link from "next/link";
import type { ReactNode } from "react";
import type { RiskLevel } from "@/lib/api";

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-ink-2">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export function Card({ title, subtitle, children, className = "" }: { title?: string; subtitle?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>
      {title && <h2 className="text-sm font-semibold">{title}</h2>}
      {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      <div className={title || subtitle ? "mt-4" : ""}>{children}</div>
    </section>
  );
}

export function StatTile({ label, value, detail }: { label: string; value: ReactNode; detail?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-2 text-2xl font-semibold">{value}</div>
      {detail && <div className="mt-1 text-xs text-ink-2">{detail}</div>}
    </div>
  );
}

const RISK_STYLE: Record<RiskLevel, { icon: string; color: string; label: string }> = {
  HIGH: { icon: "▲", color: "var(--critical)", label: "High" },
  MEDIUM: { icon: "●", color: "var(--warning)", label: "Medium" },
  LOW: { icon: "▼", color: "var(--good)", label: "Low" },
};

/** Status colour is never used alone: icon + text label always accompany it. */
export function RiskBadge({ risk }: { risk: RiskLevel }) {
  const s = RISK_STYLE[risk];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink">
      <span aria-hidden style={{ color: s.color }}>
        {s.icon}
      </span>
      {s.label}
    </span>
  );
}

export function Segmented({ options, active, basePath, param }: { options: { value: string; label: string }[]; active: string; basePath: string; param: string }) {
  return (
    <nav className="inline-flex rounded-lg border border-line bg-surface p-1 text-sm" aria-label="Select range">
      {options.map((o) => (
        <Link
          key={o.value}
          href={`${basePath}?${param}=${o.value}`}
          aria-current={o.value === active ? "page" : undefined}
          className={`rounded-md px-3 py-1.5 transition-colors ${o.value === active ? "bg-accent font-medium text-white" : "text-ink-2 hover:bg-surface-2"}`}
        >
          {o.label}
        </Link>
      ))}
    </nav>
  );
}

export function ErrorPanel({ status, message }: { status: number; message: string }) {
  const hint =
    status === 503
      ? "No trained model or data yet. Run: python -m src.ingestion.generate_data && python -m src.ingestion.load_data && python -m src.training.train"
      : status === 0
        ? "Start the API: docker compose up api (or uvicorn src.api.main:app --port 8000)."
        : null;
  return (
    <div role="alert" className="rounded-xl border border-line bg-surface p-5">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <span aria-hidden style={{ color: "var(--critical)" }}>▲</span> Data unavailable{status ? ` (HTTP ${status})` : ""}
      </div>
      <p className="mt-2 text-sm text-ink-2">{message}</p>
      {hint && <pre className="mt-3 overflow-x-auto rounded-lg bg-surface-2 p-3 text-xs text-ink-2">{hint}</pre>}
    </div>
  );
}

export function SyntheticNotice() {
  return (
    <div className="border-b border-line bg-surface-2 px-6 py-2 text-xs text-ink-2">
      <strong className="font-semibold text-ink">Prototype.</strong> All figures come from <strong>synthetic data</strong> — not SOMELEC data. Forecasts, intervals and peak alerts are experimental.
    </div>
  );
}
