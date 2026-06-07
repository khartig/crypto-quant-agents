import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Quant Trigger Dashboard",
  description: "Interactive dashboard for buy/sell/hold trigger predictions and monitor alerts."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
