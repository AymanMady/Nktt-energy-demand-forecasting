"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/forecast", label: "Forecast" },
  { href: "/historical", label: "Historical" },
  { href: "/models", label: "Models" },
  { href: "/about", label: "About" },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="border-b border-line bg-surface md:min-h-screen md:w-56 md:border-b-0 md:border-r">
      <div className="px-5 py-5">
        <div className="text-sm font-semibold leading-tight">Nouakchott</div>
        <div className="text-xs text-muted">Energy Demand Forecasting</div>
      </div>
      <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:pb-0">
        {LINKS.map((l) => {
          const active = pathname === l.href || pathname.startsWith(`${l.href}/`);
          return (
            <Link
              key={l.href}
              href={l.href}
              aria-current={active ? "page" : undefined}
              className={`whitespace-nowrap rounded-lg px-3 py-2 text-sm transition-colors ${active ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:bg-surface-2"}`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
