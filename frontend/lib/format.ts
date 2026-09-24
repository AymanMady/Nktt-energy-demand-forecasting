// Formatting helpers. Nouakchott is UTC+0 with no DST: UTC = local time.

const dateTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  weekday: "short",
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
});
const dateOnly = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric" });
const hourOnly = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit" });
const dayHour = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "2-digit", month: "short", hour: "2-digit" });

export const fmtDateTime = (iso: string) => dateTime.format(new Date(iso));
export const fmtDate = (iso: string) => dateOnly.format(new Date(iso));
export const fmtHour = (iso: string) => hourOnly.format(new Date(iso));
export const fmtDayHour = (iso: string) => dayHour.format(new Date(iso));

export function fmtMW(v: number | null | undefined, digits = 0): string {
  return v == null || Number.isNaN(v) ? "–" : `${v.toFixed(digits)} MW`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  return v == null || Number.isNaN(v) ? "–" : v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? "–" : `${v.toFixed(digits)} %`;
}

export function horizonLabel(h: number): string {
  if (h === 168) return "7 days";
  if (h % 24 === 0 && h > 24) return `${h / 24} days`;
  return h === 1 ? "1 hour" : `${h} hours`;
}
