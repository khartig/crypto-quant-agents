import { NextResponse } from "next/server";
import { loadDashboardOverview } from "@/lib/quant-data";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const payload = await loadDashboardOverview();
    return NextResponse.json(payload, {
      headers: {
        "Cache-Control": "no-store"
      }
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to load dashboard overview",
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}
