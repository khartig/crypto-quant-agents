"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type {
  DashboardOverview,
  Recommendation,
  TriggerAlertRow,
  TriggerPredictionRow
} from "@/lib/types";

interface ChartPoint {
  id: string;
  timeMs: number;
  timeLabel: string;
  closePrice: number | null;
  smaFast: number | null;
  smaSlow: number | null;
  macd: number | null;
  macdHist: number | null;
  rsi14: number | null;
  volatility24: number | null;
}

interface MarkerPoint {
  id: string;
  timeMs: number;
  timeLabel: string;
  price: number;
  recommendation: Recommendation;
  confidence: number;
  source: "prediction" | "alert";
  prediction?: TriggerPredictionRow;
  alert?: TriggerAlertRow;
}

type SelectedDatum =
  | { kind: "prediction"; row: TriggerPredictionRow }
  | { kind: "alert"; row: TriggerAlertRow }
  | null;

function formatTimeTick(value: number): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  const date = new Date(value);
  return date.toISOString().slice(5, 16).replace("T", " ");
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function renderPredictionShape(props: any) {
  const { cx, cy, payload } = props;
  if (typeof cx !== "number" || typeof cy !== "number" || !payload) {
    return <g />;
  }
  const recommendation = payload.recommendation as Recommendation;
  if (recommendation === "buy") {
    return (
      <text x={cx} y={cy + 5} fill="#2cc878" textAnchor="middle" fontSize={20}>
        ↑
      </text>
    );
  }
  if (recommendation === "sell") {
    return (
      <text x={cx} y={cy + 5} fill="#f25f5c" textAnchor="middle" fontSize={20}>
        ↓
      </text>
    );
  }
  return <circle cx={cx} cy={cy} r={4} fill="#f2c14e" stroke="#f2c14e" />;
}

function renderAlertShape(props: any) {
  const { cx, cy, payload } = props;
  if (typeof cx !== "number" || typeof cy !== "number" || !payload) {
    return <g />;
  }
  const recommendation = payload.recommendation as Recommendation;
  const fill = recommendation === "buy" ? "#2cc878" : recommendation === "sell" ? "#f25f5c" : "#f2c14e";
  return <circle cx={cx} cy={cy} r={4} fill={fill} stroke="#ffffff" strokeWidth={1} />;
}

export default function HomePage() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbolFilter, setSymbolFilter] = useState<string>("all");
  const [recommendationFilter, setRecommendationFilter] = useState<string>("all");
  const [showPredictionsOnChart, setShowPredictionsOnChart] = useState(true);
  const [showAlertsOnChart, setShowAlertsOnChart] = useState(true);
  const [showCloseLine, setShowCloseLine] = useState(true);
  const [showSmaFastLine, setShowSmaFastLine] = useState(true);
  const [showSmaSlowLine, setShowSmaSlowLine] = useState(true);
  const [showMacdLine, setShowMacdLine] = useState(true);
  const [showMacdHistogram, setShowMacdHistogram] = useState(false);
  const [showRsiLine, setShowRsiLine] = useState(true);
  const [showVolatilityLine, setShowVolatilityLine] = useState(true);
  const [pricePanelHeight, setPricePanelHeight] = useState(330);
  const [oscillatorPanelHeight, setOscillatorPanelHeight] = useState(220);
  const [selectedDatum, setSelectedDatum] = useState<SelectedDatum>(null);

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

  const chartPoints = useMemo<ChartPoint[]>(() => {
    return filteredPredictions
      .map((row) => {
        const timestamp = row.predictionTimestampUtc || row.createdAtUtc;
        const parsed = Date.parse(timestamp);
        if (!Number.isFinite(parsed)) {
          return null;
        }
        return {
          id: row.id,
          timeMs: parsed,
          timeLabel: timestamp,
          closePrice: row.closePrice,
          smaFast: row.smaFast,
          smaSlow: row.smaSlow,
          macd: row.macd,
          macdHist: row.macdHist,
          rsi14: row.rsi14,
          volatility24: row.volatility24
        };
      })
      .filter((row): row is ChartPoint => row !== null)
      .sort((a, b) => a.timeMs - b.timeMs);
  }, [filteredPredictions]);

  const predictionByPath = useMemo(() => {
    const map = new Map<string, TriggerPredictionRow>();
    for (const row of filteredPredictions) {
      map.set(row.predictionPath, row);
    }
    return map;
  }, [filteredPredictions]);

  const predictionMarkers = useMemo<MarkerPoint[]>(() => {
    const rows: MarkerPoint[] = [];
    for (const row of filteredPredictions) {
      const timestamp = row.predictionTimestampUtc || row.createdAtUtc;
      const parsed = Date.parse(timestamp);
      if (!Number.isFinite(parsed) || row.closePrice === null) {
        continue;
      }
      rows.push({
        id: `prediction:${row.id}`,
        timeMs: parsed,
        timeLabel: timestamp,
        price: row.closePrice,
        recommendation: row.recommendation,
        confidence: row.confidence,
        source: "prediction",
        prediction: row
      });
    }
    return rows;
  }, [filteredPredictions]);

  const alertMarkers = useMemo<MarkerPoint[]>(() => {
    const rows: MarkerPoint[] = [];
    for (const row of filteredAlerts) {
      const linkedPrediction = row.predictionPath ? predictionByPath.get(row.predictionPath) : undefined;
      const timestamp =
        row.predictionTimestampUtc ||
        linkedPrediction?.predictionTimestampUtc ||
        row.createdAtUtc;
      const parsed = Date.parse(timestamp);
      const price = row.closePrice ?? linkedPrediction?.closePrice ?? null;
      if (!Number.isFinite(parsed) || price === null) {
        continue;
      }
      rows.push({
        id: `alert:${row.id}`,
        timeMs: parsed,
        timeLabel: timestamp,
        price,
        recommendation: row.recommendation,
        confidence: row.confidence,
        source: "alert",
        alert: row
      });
    }
    return rows;
  }, [filteredAlerts, predictionByPath]);

  const hasChartData = chartPoints.length > 0;

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

      <section className="button-row">
        <button
          type="button"
          className={`toggle-btn large ${showPredictionsOnChart ? "active" : ""}`}
          onClick={() => setShowPredictionsOnChart((prev) => !prev)}
        >
          Predictions ({filteredPredictions.length})
        </button>
        <button
          type="button"
          className={`toggle-btn large ${showAlertsOnChart ? "active" : ""}`}
          onClick={() => setShowAlertsOnChart((prev) => !prev)}
        >
          Alerts ({filteredAlerts.length})
        </button>
        <div className="stats-inline muted">
          actionable={actionablePredictions.length} • high-confidence={highConfidence.length}
        </div>
      </section>

      <section className="button-row compact">
        <button
          type="button"
          className={`toggle-btn ${showCloseLine ? "active" : ""}`}
          onClick={() => setShowCloseLine((prev) => !prev)}
        >
          BTC Price
        </button>
        <button
          type="button"
          className={`toggle-btn ${showSmaFastLine ? "active" : ""}`}
          onClick={() => setShowSmaFastLine((prev) => !prev)}
        >
          SMA Fast
        </button>
        <button
          type="button"
          className={`toggle-btn ${showSmaSlowLine ? "active" : ""}`}
          onClick={() => setShowSmaSlowLine((prev) => !prev)}
        >
          SMA Slow
        </button>
        <button
          type="button"
          className={`toggle-btn ${showMacdLine ? "active" : ""}`}
          onClick={() => setShowMacdLine((prev) => !prev)}
        >
          MACD
        </button>
        <button
          type="button"
          className={`toggle-btn ${showMacdHistogram ? "active" : ""}`}
          onClick={() => setShowMacdHistogram((prev) => !prev)}
        >
          MACD Hist
        </button>
        <button
          type="button"
          className={`toggle-btn ${showRsiLine ? "active" : ""}`}
          onClick={() => setShowRsiLine((prev) => !prev)}
        >
          RSI
        </button>
        <button
          type="button"
          className={`toggle-btn ${showVolatilityLine ? "active" : ""}`}
          onClick={() => setShowVolatilityLine((prev) => !prev)}
        >
          Volatility
        </button>
      </section>
      <section className="button-row compact chart-controls">
        <label className="range-control">
          Price panel height
          <input
            type="range"
            min={260}
            max={520}
            step={10}
            value={pricePanelHeight}
            onChange={(event) => setPricePanelHeight(Number(event.target.value))}
          />
          <span className="muted">{pricePanelHeight}px</span>
        </label>
        <label className="range-control">
          Oscillator panel height
          <input
            type="range"
            min={160}
            max={360}
            step={10}
            value={oscillatorPanelHeight}
            onChange={(event) => setOscillatorPanelHeight(Number(event.target.value))}
          />
          <span className="muted">{oscillatorPanelHeight}px</span>
        </label>
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
        <h2>Market + indicator chart</h2>
        {!hasChartData ? (
          <p className="muted">
            No chartable prediction points yet. Let the monitor run a few cycles to populate price
            and indicator values.
          </p>
        ) : (
          <div className="chart-stack">
            <div className="chart-panel price" style={{ height: `${pricePanelHeight}px` }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartPoints}
                  syncId="market-series"
                  margin={{ top: 18, right: 24, bottom: 8, left: 8 }}
                >
                  <CartesianGrid stroke="#273155" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="timeMs" domain={["dataMin", "dataMax"]} hide />
                  <YAxis
                    yAxisId="price"
                    orientation="left"
                    tick={{ fill: "#a6b0cf", fontSize: 11 }}
                    width={82}
                    domain={["auto", "auto"]}
                    label={{
                      value: "Price (USD)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#a6b0cf",
                      dx: -6
                    }}
                  />
                  <Tooltip
                    labelFormatter={(value) => formatTimeTick(Number(value))}
                    formatter={(value: any, name: string) => [formatNumber(Number(value), 5), name]}
                  />
                  <Legend />
                  {showCloseLine ? (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="closePrice"
                      name="BTC Price"
                      stroke="#5b8cff"
                      dot={false}
                      strokeWidth={2}
                      connectNulls
                    />
                  ) : null}
                  {showSmaFastLine ? (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="smaFast"
                      name="SMA Fast"
                      stroke="#9ad0f5"
                      dot={false}
                      connectNulls
                    />
                  ) : null}
                  {showSmaSlowLine ? (
                    <Line
                      yAxisId="price"
                      type="monotone"
                      dataKey="smaSlow"
                      name="SMA Slow"
                      stroke="#e8a1ff"
                      dot={false}
                      connectNulls
                    />
                  ) : null}
                  {showPredictionsOnChart ? (
                    <Scatter
                      yAxisId="price"
                      name="Predictions"
                      data={predictionMarkers}
                      dataKey="price"
                      shape={renderPredictionShape}
                      onClick={(event: any) => {
                        const payload = event?.payload as MarkerPoint | undefined;
                        if (payload?.prediction) {
                          setSelectedDatum({ kind: "prediction", row: payload.prediction });
                        }
                      }}
                    />
                  ) : null}
                  {showAlertsOnChart ? (
                    <Scatter
                      yAxisId="price"
                      name="Alerts"
                      data={alertMarkers}
                      dataKey="price"
                      shape={renderAlertShape}
                      onClick={(event: any) => {
                        const payload = event?.payload as MarkerPoint | undefined;
                        if (payload?.alert) {
                          setSelectedDatum({ kind: "alert", row: payload.alert });
                        }
                      }}
                    />
                  ) : null}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-panel oscillator" style={{ height: `${oscillatorPanelHeight}px` }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartPoints}
                  syncId="market-series"
                  margin={{ top: 10, right: 24, bottom: 14, left: 8 }}
                >
                  <CartesianGrid stroke="#273155" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="timeMs"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={formatTimeTick}
                    tick={{ fill: "#a6b0cf", fontSize: 11 }}
                    label={{
                      value: "Time (UTC)",
                      position: "insideBottom",
                      offset: -4,
                      fill: "#a6b0cf"
                    }}
                  />
                  <YAxis
                    yAxisId="osc"
                    orientation="left"
                    tick={{ fill: "#a6b0cf", fontSize: 11 }}
                    width={82}
                    domain={["auto", "auto"]}
                    label={{
                      value: "MACD / Vol",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#a6b0cf",
                      dx: -6
                    }}
                  />
                  <YAxis
                    yAxisId="rsi"
                    orientation="right"
                    tick={{ fill: "#a6b0cf", fontSize: 11 }}
                    width={82}
                    domain={[0, 100]}
                    label={{
                      value: "RSI",
                      angle: -90,
                      position: "insideRight",
                      fill: "#a6b0cf",
                      dx: 6
                    }}
                  />
                  <Tooltip
                    labelFormatter={(value) => formatTimeTick(Number(value))}
                    formatter={(value: any, name: string) => [formatNumber(Number(value), 5), name]}
                  />
                  {showMacdHistogram ? (
                    <Bar
                      yAxisId="osc"
                      dataKey="macdHist"
                      name="MACD Hist"
                      fill="#f59e0b"
                      opacity={0.28}
                      barSize={5}
                    />
                  ) : null}
                  {showMacdLine ? (
                    <Line
                      yAxisId="osc"
                      type="monotone"
                      dataKey="macd"
                      name="MACD"
                      stroke="#f59e0b"
                      dot={false}
                      connectNulls
                    />
                  ) : null}
                  {showRsiLine ? (
                    <Line
                      yAxisId="rsi"
                      type="monotone"
                      dataKey="rsi14"
                      name="RSI(14)"
                      stroke="#34d399"
                      dot={false}
                      connectNulls
                    />
                  ) : null}
                  {showVolatilityLine ? (
                    <Line
                      yAxisId="osc"
                      type="monotone"
                      dataKey="volatility24"
                      name="Volatility(24)"
                      stroke="#ef4444"
                      dot={false}
                      connectNulls
                    />
                  ) : null}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </section>

      <section className="section">
        <h2>Selected marker details</h2>
        {!selectedDatum ? (
          <p className="muted">
            Select a prediction arrow or alert marker on the chart to inspect confidence,
            probabilities, and rationale details.
          </p>
        ) : selectedDatum.kind === "prediction" ? (
          <div className="detail-panel">
            <div>
              <span className={`pill ${selectedDatum.row.recommendation}`}>
                {selectedDatum.row.recommendation}
              </span>{" "}
              prediction
            </div>
            <div className="muted">
              {selectedDatum.row.exchange} {selectedDatum.row.symbol} ({selectedDatum.row.timeframe}) •{" "}
              {selectedDatum.row.predictionTimestampUtc || selectedDatum.row.createdAtUtc}
            </div>
            <p>confidence={formatNumber(selectedDatum.row.confidence, 3)}</p>
            <p>
              probabilities: buy={formatNumber(selectedDatum.row.probabilities.buy, 3)} hold=
              {formatNumber(selectedDatum.row.probabilities.hold, 3)} sell=
              {formatNumber(selectedDatum.row.probabilities.sell, 3)}
            </p>
            <p>
              close={formatNumber(selectedDatum.row.closePrice, 2)} smaFast=
              {formatNumber(selectedDatum.row.smaFast, 2)} smaSlow=
              {formatNumber(selectedDatum.row.smaSlow, 2)} macd=
              {formatNumber(selectedDatum.row.macd, 5)} macdHist=
              {formatNumber(selectedDatum.row.macdHist, 5)} rsi={formatNumber(selectedDatum.row.rsi14, 2)}{" "}
              volatility={formatNumber(selectedDatum.row.volatility24, 6)}
            </p>
            <ul className="reason-list">
              {selectedDatum.row.topReasons.map((reason) => (
                <li key={`${selectedDatum.row.id}-${reason}`}>{reason}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="detail-panel">
            <div>
              <span className={`pill ${selectedDatum.row.recommendation}`}>
                {selectedDatum.row.recommendation}
              </span>{" "}
              alert
            </div>
            <div className="muted">
              {selectedDatum.row.exchange} {selectedDatum.row.symbol} ({selectedDatum.row.timeframe}) •{" "}
              {selectedDatum.row.predictionTimestampUtc || selectedDatum.row.createdAtUtc}
            </div>
            <p>confidence={formatNumber(selectedDatum.row.confidence, 3)}</p>
            <p>
              probabilities: buy={formatNumber(selectedDatum.row.probabilities.buy, 3)} hold=
              {formatNumber(selectedDatum.row.probabilities.hold, 3)} sell=
              {formatNumber(selectedDatum.row.probabilities.sell, 3)}
            </p>
            <p>
              close={formatNumber(selectedDatum.row.closePrice, 2)} smaFast=
              {formatNumber(selectedDatum.row.smaFast, 2)} smaSlow=
              {formatNumber(selectedDatum.row.smaSlow, 2)} macd=
              {formatNumber(selectedDatum.row.macd, 5)} macdHist=
              {formatNumber(selectedDatum.row.macdHist, 5)} rsi={formatNumber(selectedDatum.row.rsi14, 2)}{" "}
              volatility={formatNumber(selectedDatum.row.volatility24, 6)}
            </p>
            <ul className="reason-list">
              {selectedDatum.row.topReasons.map((reason) => (
                <li key={`${selectedDatum.row.id}-${reason}`}>{reason}</li>
              ))}
            </ul>
          </div>
        )}
      </section>
    </main>
  );
}