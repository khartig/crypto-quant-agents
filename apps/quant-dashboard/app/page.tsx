"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { DashboardOverview } from "@/lib/types";

export default function HomePage() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbolFilter, setSymbolFilter] = useState<string>("all");
  const [recommendationFilter, setRecommendationFilter] = useState<string>("all");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/overview", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as DashboardOverview;
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const symbolOptions = useMemo(() => {
    const symbols = new Set<string>();
    for (const row of data?.predictions || []) {
      if (row.symbol) {
        symbols.add(row.symbol);
      }
    }
    return ["all", ...Array.from(symbols).sort()];
  }, [data?.predictions]);

  const filteredPredictions = useMemo(() => {
    return (data?.predictions || []).filter((row) => {
      if (symbolFilter !== "all" && row.symbol !== symbolFilter) {
        return false;
      }
      if (recommendationFilter !== "all" && row.recommendation !== recommendationFilter) {
        return false;
      }
      return true;
    });
  }, [data?.predictions, symbolFilter, recommendationFilter]);

  const filteredAlerts = useMemo(() => {
    return (data?.alerts || []).filter((row) => {
      if (symbolFilter !== "all" && row.symbol !== symbolFilter) {
        return false;
      }
      if (recommendationFilter !== "all" && row.recommendation !== recommendationFilter) {
        return false;
      }
      return true;
    });
  }, [data?.alerts, symbolFilter, recommendationFilter]);

  const actionablePredictions = filteredPredictions.filter(
    (row) => row.recommendation === "buy" || row.recommendation === "sell"
  );
  const highConfidence = filteredPredictions.filter((row) => row.confidence >= 0.7);

  return (
    <main>
      <header>
        <h1>Quant Trigger Dashboard</h1>
        <p className="muted">
          Interactive view of model predictions, probabilities, reason traces, and monitor alerts.
        </p>
        <p className="muted">
          Data root: <code>{data?.quantDataRoot || "…"}</code>
        </p>
      </header>

      <div className="toolbar">
        <label>
          Symbol
          <select value={symbolFilter} onChange={(event) => setSymbolFilter(event.target.value)}>
            {symbolOptions.map((value) => (
              <option value={value} key={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Recommendation
          <select
            value={recommendationFilter}
            onChange={(event) => setRecommendationFilter(event.target.value)}
          >
            <option value="all">all</option>
            <option value="buy">buy</option>
            <option value="sell">sell</option>
            <option value="hold">hold</option>
          </select>
        </label>
        <label>
          Data freshness
          <span className="muted">{data?.generatedAtUtc || "loading…"}</span>
        </label>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error ? (
        <section className="section">
          <h2>Load error</h2>
          <p>{error}</p>
        </section>
      ) : null}

      <section className="cards">
        <div className="card">
          <div className="label">Predictions (filtered)</div>
          <div className="value">{filteredPredictions.length}</div>
        </div>
        <div className="card">
          <div className="label">Actionable predictions</div>
          <div className="value">{actionablePredictions.length}</div>
        </div>
        <div className="card">
          <div className="label">High confidence (≥ 0.70)</div>
          <div className="value">{highConfidence.length}</div>
        </div>
        <div className="card">
          <div className="label">Alerts (filtered)</div>
          <div className="value">{filteredAlerts.length}</div>
        </div>
      </section>

      {data?.latestAgentPlane ? (
        <section className="section">
          <h2>Latest agent-plane run</h2>
          <p>
            Run <code>{data.latestAgentPlane.runId}</code> •{" "}
            <strong>{data.latestAgentPlane.exchange}</strong> {data.latestAgentPlane.symbol} (
            {data.latestAgentPlane.timeframe})
          </p>
          <p className="muted">
            riskApproved={String(data.latestAgentPlane.riskApproved)} • intent=
            {data.latestAgentPlane.intentStatus} • execution=
            {data.latestAgentPlane.paperExecutionStatus}
          </p>
        </section>
      ) : null}

      <section className="section">
        <h2>Recent predictions</h2>
        <table>
          <thead>
            <tr>
              <th>Created</th>
              <th>Scope</th>
              <th>Recommendation</th>
              <th>Confidence</th>
              <th>Probabilities</th>
              <th>Why</th>
            </tr>
          </thead>
          <tbody>
            {filteredPredictions.length === 0 ? (
              <tr>
                <td colSpan={6} className="muted">
                  No predictions found for the selected filters.
                </td>
              </tr>
            ) : (
              filteredPredictions.map((row) => (
                <tr key={row.id}>
                  <td>
                    <div>{row.createdAtUtc || "n/a"}</div>
                    <div className="muted">{row.predictionTimestampUtc || "n/a"}</div>
                  </td>
                  <td>
                    <div>{row.exchange}</div>
                    <div>{row.symbol}</div>
                    <div className="muted">{row.timeframe}</div>
                  </td>
                  <td>
                    <span className={`pill ${row.recommendation}`}>{row.recommendation}</span>
                  </td>
                  <td>{row.confidence.toFixed(3)}</td>
                  <td>
                    <div>buy={row.probabilities.buy.toFixed(3)}</div>
                    <div>hold={row.probabilities.hold.toFixed(3)}</div>
                    <div>sell={row.probabilities.sell.toFixed(3)}</div>
                  </td>
                  <td>
                    <ul className="reason-list">
                      {row.topReasons.map((reason) => (
                        <li key={`${row.id}-${reason}`}>{reason}</li>
                      ))}
                    </ul>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>

      <section className="section">
        <h2>Recent alerts</h2>
        <table>
          <thead>
            <tr>
              <th>Created</th>
              <th>Scope</th>
              <th>Recommendation</th>
              <th>Confidence</th>
              <th>Top reasons</th>
            </tr>
          </thead>
          <tbody>
            {filteredAlerts.length === 0 ? (
              <tr>
                <td colSpan={5} className="muted">
                  No alerts found for the selected filters.
                </td>
              </tr>
            ) : (
              filteredAlerts.map((row) => (
                <tr key={row.id}>
                  <td>{row.createdAtUtc || "n/a"}</td>
                  <td>
                    <div>{row.exchange}</div>
                    <div>{row.symbol}</div>
                    <div className="muted">{row.timeframe}</div>
                  </td>
                  <td>
                    <span className={`pill ${row.recommendation}`}>{row.recommendation}</span>
                  </td>
                  <td>{row.confidence.toFixed(3)}</td>
                  <td>
                    <ul className="reason-list">
                      {row.topReasons.map((reason) => (
                        <li key={`${row.id}-${reason}`}>{reason}</li>
                      ))}
                    </ul>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>
    </main>
  );
}
