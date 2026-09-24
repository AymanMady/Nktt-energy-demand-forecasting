import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Sidebar } from "@/components/Sidebar";
import { SyntheticNotice } from "@/components/ui";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nouakchott Energy Demand Forecasting",
  description: "Prototype dashboard: hourly electricity demand forecasts for Nouakchott (synthetic data).",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <div className="flex min-h-screen flex-col md:flex-row">
          <Sidebar />
          <div className="min-w-0 flex-1">
            <SyntheticNotice />
            <main className="mx-auto max-w-7xl px-4 py-8 md:px-8">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
