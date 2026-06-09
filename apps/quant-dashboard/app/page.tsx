"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

interface DragPanState {
  startClientX: number;
  initialDomain: [number, number];
  hasPanned: boolean;
}

const ONE_HOUR_MS = 60 * 60 * 1000;

function clampDomainToBounds(
  start: number,
  end: number,
  bounds: [number, number]
): [number, number] {
  const [boundStart, boundEnd] = bounds;
  const boundSpan = Math.max(0, boundEnd - boundStart);
  let nextStart = start;
  let nextEnd = end;
  if (boundSpan <= 0) {
    return [boundStart, boundEnd];
  }
  if (nextEnd - nextStart >= boundSpan) {
    return [boundStart, boundEnd];
  }
  if (nextStart < boundStart) {
    const shift = boundStart - nextStart;
    nextStart += shift;
    nextEnd += shift;
  }
  if (nextEnd > boundEnd) {
    const shift = nextEnd - boundEnd;
    nextStart -= shift;
    nextEnd -= shift;
  }
  nextStart = Math.max(boundStart, nextStart);
  nextEnd = Math.min(boundEnd, nextEnd);
  if (nextEnd <= nextStart) {
    return [boundStart, boundEnd];
  }
  return [nextStart, nextEnd];
}

function formatSpanLabel(spanMs: number): string {
  if (!Number.isFinite(spanMs) || spanMs <= 0) {
    return "n/a";
  }
  const hours = spanMs / ONE_HOUR_MS;
  if (hours < 48) {
    return `${hours.toFixed(1)}h`;
  }
  return `${(hours / 24).toFixed(1)}d`;
}

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

function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(digits)}%`;
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
  const [activeTab, setActiveTab] = useState<"signals" | "performance">("signals");
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);
  const [isDragPanning, setIsDragPanning] = useState(false);
  const chartStackRef = useRef<HTMLDivElement | null>(null);
  const dragPanRef = useRef<DragPanState | null>(null);

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
  const modelPerformance = data?.performance.model;
  const paperTradingPerformance = data?.performance.paperTrading;
  const recentModelRuns = modelPerformance?.runs.slice(0, 12) || [];
  const recentExecutions = paperTradingPerformance?.executions.slice(0, 20) || [];

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

  const fullDomain = useMemo<[number, number] | null>(() => {
    if (!chartPoints.length) {
      return null;
    }
    return [chartPoints[0].timeMs, chartPoints[chartPoints.length - 1].timeMs];
  }, [chartPoints]);

  const minimumZoomSpanMs = useMemo(() => {
    if (chartPoints.length < 2) {
      return ONE_HOUR_MS;
    }
    let minStep = Number.POSITIVE_INFINITY;
    for (let index = 1; index < chartPoints.length; index += 1) {
      const delta = chartPoints[index].timeMs - chartPoints[index - 1].timeMs;
      if (delta > 0) {
        minStep = Math.min(minStep, delta);
      }
    }
    if (!Number.isFinite(minStep)) {
      return ONE_HOUR_MS;
    }
    return Math.max(ONE_HOUR_MS, minStep * 2);
  }, [chartPoints]);

  const visibleDomain = useMemo<[number, number] | null>(() => {
    if (!fullDomain) {
      return null;
    }
    if (!zoomDomain) {
      return fullDomain;
    }
    return clampDomainToBounds(zoomDomain[0], zoomDomain[1], fullDomain);
  }, [zoomDomain, fullDomain]);

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
  const markerFocusDomain = useMemo<[number, number] | null>(() => {
    if (!fullDomain) {
      return null;
    }
    const markerTimes = [...predictionMarkers, ...alertMarkers]
      .map((marker) => marker.timeMs)
      .sort((a, b) => a - b);
    if (!markerTimes.length) {
      return null;
    }
    let start = markerTimes[0];
    let end = markerTimes[markerTimes.length - 1];
    if (end <= start) {
      start -= minimumZoomSpanMs / 2;
      end += minimumZoomSpanMs / 2;
    } else {
      const padding = Math.max(ONE_HOUR_MS, (end - start) * 0.08);
      start -= padding;
      end += padding;
    }
    return clampDomainToBounds(start, end, fullDomain);
  }, [predictionMarkers, alertMarkers, minimumZoomSpanMs, fullDomain]);

  const fullSpanMs = useMemo(() => {
    if (!fullDomain) {
      return 0;
    }
    return Math.max(0, fullDomain[1] - fullDomain[0]);
  }, [fullDomain]);

  const visibleSpanMs = useMemo(() => {
    if (!visibleDomain) {
      return 0;
    }
    return Math.max(0, visibleDomain[1] - visibleDomain[0]);
  }, [visibleDomain]);

  const xDomain = useMemo<[number, number] | ["dataMin", "dataMax"]>(() => {
    if (!visibleDomain) {
      return ["dataMin", "dataMax"];
    }
    return [visibleDomain[0], visibleDomain[1]];
  }, [visibleDomain]);

  const canZoomIn = fullDomain !== null && visibleSpanMs > minimumZoomSpanMs + 1;
  const canZoomOut = fullDomain !== null && visibleSpanMs < fullSpanMs - 1;
  const canDragPan = canZoomOut;
  const fullDomainStart = fullDomain ? fullDomain[0] : null;
  const fullDomainEnd = fullDomain ? fullDomain[1] : null;

  const beginDragPan = useCallback(
    (event: { button: number; clientX: number }) => {
      if (!canDragPan || !visibleDomain) {
        return;
      }
      if (event.button !== 0) {
        return;
      }
      dragPanRef.current = {
        startClientX: event.clientX,
        initialDomain: visibleDomain,
        hasPanned: false
      };
    },
    [canDragPan, visibleDomain]
  );

  const applyZoomFactor = useCallback(
    (factor: number) => {
      if (!fullDomain || factor <= 0) {
        return;
      }
      const baseDomain = visibleDomain ?? fullDomain;
      const currentSpan = Math.max(1, baseDomain[1] - baseDomain[0]);
      const targetSpan = Math.min(fullSpanMs, Math.max(minimumZoomSpanMs, currentSpan * factor));
      const center = (baseDomain[0] + baseDomain[1]) / 2;
      const targetStart = center - targetSpan / 2;
      const targetEnd = center + targetSpan / 2;
      const nextDomain = clampDomainToBounds(targetStart, targetEnd, fullDomain);
      setZoomDomain(nextDomain);
    },
    [fullDomain, fullSpanMs, minimumZoomSpanMs, visibleDomain]
  );

  useEffect(() => {
    setZoomDomain(null);
  }, [fullDomainStart, fullDomainEnd]);

  useEffect(() => {
    if (canDragPan) {
      return;
    }
    dragPanRef.current = null;
    setIsDragPanning(false);
  }, [canDragPan]);

  useEffect(() => {
    const handleWindowMouseMove = (event: MouseEvent) => {
      const dragState = dragPanRef.current;
      if (!dragState || !fullDomain) {
        return;
      }
      const chartWidth = chartStackRef.current?.clientWidth ?? 0;
      if (chartWidth <= 0) {
        return;
      }
      const deltaPx = event.clientX - dragState.startClientX;
      if (!dragState.hasPanned && Math.abs(deltaPx) < 3) {
        return;
      }
      if (!dragState.hasPanned) {
        dragState.hasPanned = true;
        setIsDragPanning(true);
      }
      event.preventDefault();
      const spanMs = Math.max(1, dragState.initialDomain[1] - dragState.initialDomain[0]);
      const shiftMs = -(deltaPx / chartWidth) * spanMs;
      const nextDomain = clampDomainToBounds(
        dragState.initialDomain[0] + shiftMs,
        dragState.initialDomain[1] + shiftMs,
        fullDomain
      );
      setZoomDomain(nextDomain);
    };

    const stopDragPan = () => {
      dragPanRef.current = null;
      setIsDragPanning(false);
    };

    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", stopDragPan);
    window.addEventListener("mouseleave", stopDragPan);
    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", stopDragPan);
      window.removeEventListener("mouseleave", stopDragPan);
    };
  }, [fullDomain]);

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

      <section className="button-row tab-row">
        <button
          type="button"
          className={`toggle-btn large ${activeTab === "signals" ? "active" : ""}`}
          onClick={() => setActiveTab("signals")}
        >
          Signals & Markers
        </button>
        <button
          type="button"
          className={`toggle-btn large ${activeTab === "performance" ? "active" : ""}`}
          onClick={() => setActiveTab("performance")}
        >
          Model & Trade Performance
        </button>
      </section>

      {activeTab === "signals" ? (
        <>

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
      <section className="button-row compact chart-zoom-controls">
        <button
          type="button"
          className="toggle-btn"
          onClick={() => applyZoomFactor(0.5)}
          disabled={!canZoomIn}
        >
          Zoom In
        </button>
        <button
          type="button"
          className="toggle-btn"
          onClick={() => applyZoomFactor(2)}
          disabled={!canZoomOut}
        >
          Zoom Out
        </button>
        <button
          type="button"
          className="toggle-btn"
          onClick={() => setZoomDomain(markerFocusDomain)}
          disabled={!markerFocusDomain}
        >
          Focus Markers
        </button>
        <button
          type="button"
          className="toggle-btn"
          onClick={() => setZoomDomain(null)}
          disabled={!zoomDomain}
        >
          Reset Zoom
        </button>
        <span className="muted zoom-state">Window: {formatSpanLabel(visibleSpanMs)}</span>
        {canDragPan ? <span className="muted zoom-state">Drag chart left/right to pan</span> : null}
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
          <div
            className={`chart-stack ${canDragPan ? "pan-enabled" : ""} ${isDragPanning ? "is-panning" : ""}`}
            ref={chartStackRef}
            onMouseDown={beginDragPan}
          >
            <div className="chart-panel price" style={{ height: `${pricePanelHeight}px` }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartPoints}
                  syncId="market-series"
                  margin={{ top: 18, right: 24, bottom: 8, left: 8 }}
                >
                  <CartesianGrid stroke="#273155" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="timeMs"
                    domain={xDomain}
                    allowDataOverflow
                    hide
                  />
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
                    domain={xDomain}
                    allowDataOverflow
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
        </>
      ) : (
        <>
          <section className="cards">
            <article className="card">
              <div className="label">Model runs tracked</div>
              <div className="value">{modelPerformance?.runCount ?? 0}</div>
            </article>
            <article className="card">
              <div className="label">Latest model accuracy</div>
              <div className="value">{formatPercent(modelPerformance?.latestAccuracy, 2)}</div>
            </article>
            <article className="card">
              <div className="label">Rolling model accuracy</div>
              <div className="value">{formatPercent(modelPerformance?.rollingAccuracy, 2)}</div>
            </article>
            <article className="card">
              <div className="label">Executed paper trades</div>
              <div className="value">{paperTradingPerformance?.executedCount ?? 0}</div>
            </article>
            <article className="card">
              <div className="label">Paper trade win rate</div>
              <div className="value">{formatPercent(paperTradingPerformance?.winRate, 1)}</div>
            </article>
            <article className="card">
              <div className="label">Realized PnL delta (USD)</div>
              <div className="value">{formatNumber(paperTradingPerformance?.totalRealizedPnlDeltaUsd, 2)}</div>
            </article>
            <article className="card">
              <div className="label">Executed notional (USD)</div>
              <div className="value">{formatNumber(paperTradingPerformance?.totalNotionalUsd, 2)}</div>
            </article>
            <article className="card">
              <div className="label">Fees paid (USD)</div>
              <div className="value">{formatNumber(paperTradingPerformance?.totalFeesUsd, 2)}</div>
            </article>
          </section>

          <section className="section">
            <h2>Model training performance</h2>
            {!recentModelRuns.length ? (
              <p className="muted">
                No model training artifacts found yet under the trigger-model output path.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Created (UTC)</th>
                    <th>Exchange</th>
                    <th>Symbol</th>
                    <th>Timeframe</th>
                    <th>Samples</th>
                    <th>Train/Test</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {recentModelRuns.map((row) => (
                    <tr key={row.id}>
                      <td>{row.createdAtUtc || "n/a"}</td>
                      <td>{row.exchange || "n/a"}</td>
                      <td>{row.symbol || "n/a"}</td>
                      <td>{row.timeframe || "n/a"}</td>
                      <td>{row.sampleCount}</td>
                      <td>
                        {row.trainCount}/{row.testCount}
                      </td>
                      <td>{formatPercent(row.accuracy, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="section">
            <h2>Paper execution performance</h2>
            {!recentExecutions.length ? (
              <p className="muted">
                No paper execution records found yet. Run `quant-agents agent-plane` to emit executions.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Created (UTC)</th>
                    <th>Status</th>
                    <th>Intent</th>
                    <th>Executed</th>
                    <th>Notional USD</th>
                    <th>Fee USD</th>
                    <th>PnL Δ USD</th>
                    <th>Cash After USD</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {recentExecutions.map((row) => (
                    <tr key={row.id}>
                      <td>{row.createdAtUtc || "n/a"}</td>
                      <td>{row.executionStatus}</td>
                      <td>
                        <span className={`pill ${row.intentAction}`}>{row.intentAction}</span>
                      </td>
                      <td>
                        <span className={`pill ${row.executedAction}`}>{row.executedAction}</span>
                      </td>
                      <td>{formatNumber(row.executedNotionalUsd, 2)}</td>
                      <td>{formatNumber(row.feeUsd, 4)}</td>
                      <td>{formatNumber(row.realizedPnlDeltaUsd, 4)}</td>
                      <td>{formatNumber(row.cashAfterUsd, 2)}</td>
                      <td>{row.reason || "n/a"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </main>
  );
}