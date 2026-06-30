"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type {
  DashboardOverview,
  PaperTradeExecutionRow,
  RegimeBacktestRunSummary,
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
  source: "prediction" | "alert" | "trade" | "regime";
  prediction?: TriggerPredictionRow;
  alert?: TriggerAlertRow;
  trade?: PaperTradeExecutionRow;
  regimeRun?: RegimeBacktestRunSummary;
}

type SelectedDatum =
  | { kind: "prediction"; row: TriggerPredictionRow }
  | { kind: "alert"; row: TriggerAlertRow }
  | { kind: "trade"; row: PaperTradeExecutionRow }
  | { kind: "regime"; row: RegimeBacktestRunSummary }
  | null;

interface DragPanState {
  startClientX: number;
  initialDomain: [number, number];
  hasPanned: boolean;
}

const ONE_HOUR_MS = 60 * 60 * 1000;
const RANGE_SLIDER_STEPS = 1000;
const RECOMMENDATION_OPTIONS: Recommendation[] = ["buy", "sell", "hold"];

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

function formatDateTimeLabel(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return new Date(value).toISOString().slice(0, 16).replace("T", " ");
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

function formatUsd(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${value.toFixed(digits)} USD`;
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

function renderTradeShape(props: any) {
  const { cx, cy, payload } = props;
  if (typeof cx !== "number" || typeof cy !== "number" || !payload) {
    return <g />;
  }
  const recommendation = payload.recommendation as Recommendation;
  if (recommendation === "buy") {
    return (
      <text x={cx} y={cy + 5} fill="#00e676" textAnchor="middle" fontSize={16}>
        ▲
      </text>
    );
  }
  if (recommendation === "sell") {
    return (
      <text x={cx} y={cy + 5} fill="#ff6b6b" textAnchor="middle" fontSize={16}>
        ▼
      </text>
    );
  }
  return <circle cx={cx} cy={cy} r={4} fill="#f2c14e" stroke="#111830" strokeWidth={1} />;
}

function renderRegimeShape(props: any) {
  const { cx, cy, payload } = props;
  if (typeof cx !== "number" || typeof cy !== "number" || !payload) {
    return <g />;
  }
  const run = payload.regimeRun as RegimeBacktestRunSummary | undefined;
  const pnl = run?.realizedPnlDeltaUsd ?? 0;
  const fill = pnl >= 0 ? "#16a34a" : "#dc2626";
  return (
    <g>
      <rect x={cx - 4} y={cy - 4} width={8} height={8} fill={fill} stroke="#ffffff" strokeWidth={1} />
    </g>
  );
}

function isMarkerPoint(value: unknown): value is MarkerPoint {
  if (!value || typeof value !== "object") {
    return false;
  }
  const payload = value as Record<string, unknown>;
  return (
    typeof payload.id === "string" &&
    typeof payload.source === "string" &&
    typeof payload.timeMs === "number" &&
    typeof payload.price === "number"
  );
}

interface PriceTooltipEntry {
  color?: string;
  name?: string;
  payload?: ChartPoint | MarkerPoint;
  value?: number | string;
}

interface PriceTooltipProps {
  active?: boolean;
  label?: number | string;
  payload?: PriceTooltipEntry[];
}

function PriceChartTooltip({ active, label, payload }: PriceTooltipProps) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const markerById = new Map<string, MarkerPoint>();
  const numericRows: Array<{ color?: string; name: string; value: number }> = [];
  for (const entry of payload) {
    if (isMarkerPoint(entry.payload)) {
      markerById.set(entry.payload.id, entry.payload);
      continue;
    }
    const parsedValue =
      typeof entry.value === "number" ? entry.value : Number.parseFloat(String(entry.value));
    if (!Number.isFinite(parsedValue)) {
      continue;
    }
    numericRows.push({
      color: entry.color,
      name: String(entry.name || "value"),
      value: parsedValue
    });
  }
  const markers = Array.from(markerById.values());
  const labelValue =
    typeof label === "number" ? label : Number.parseFloat(typeof label === "string" ? label : "");

  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-time">{formatTimeTick(labelValue)}</div>
      {numericRows.map((row) => (
        <div className="chart-tooltip-row" key={`${row.name}:${row.color || "none"}`}>
          <span className="chart-tooltip-name">
            {row.color ? <span className="chart-tooltip-color" style={{ background: row.color }} /> : null}
            {row.name}
          </span>
          <span className="chart-tooltip-value">{formatNumber(row.value, 5)}</span>
        </div>
      ))}
      {markers.map((marker) => (
        <div className="chart-tooltip-marker" key={marker.id}>
          <div className="chart-tooltip-row">
            <span className="chart-tooltip-name">
              <span className={`pill ${marker.recommendation}`}>{marker.recommendation}</span>
              {marker.source === "trade" ? "trade marker" : `${marker.source} marker`}
            </span>
            <span className="chart-tooltip-value">@ {formatNumber(marker.price, 2)}</span>
          </div>
          {marker.source === "trade" && marker.trade ? (
            <>
              <div className="chart-tooltip-sub">
                notional={formatUsd(marker.trade.executedNotionalUsd, 2)} • fee=
                {formatUsd(marker.trade.feeUsd, 4)}
              </div>
              {marker.trade.executedAction === "sell" ? (
                <div className="chart-tooltip-sub">
                  sell PnL Δ = {formatUsd(marker.trade.realizedPnlDeltaUsd, 4)}
                </div>
              ) : null}
            </>
          ) : marker.source === "regime" && marker.regimeRun ? (
            <>
              <div className="chart-tooltip-sub">
                {marker.regimeRun.regime} ({marker.regimeRun.split})
              </div>
              <div className="chart-tooltip-sub">
                window: {marker.regimeRun.startUtc || "n/a"} → {marker.regimeRun.endUtc || "n/a"}
              </div>
              <div className="chart-tooltip-sub">
                realized PnL Δ = {formatUsd(marker.regimeRun.realizedPnlDeltaUsd, 2)} • return=
                {formatPercent(marker.regimeRun.equityReturn, 2)}
              </div>
            </>
          ) : (
            <div className="chart-tooltip-sub">confidence={formatNumber(marker.confidence, 3)}</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function HomePage() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbolFilter, setSymbolFilter] = useState<string>("all");
  const [recommendationFilter, setRecommendationFilter] = useState<Recommendation[]>([
    ...RECOMMENDATION_OPTIONS
  ]);
  const [showPredictionsOnChart, setShowPredictionsOnChart] = useState(true);
  const [showAlertsOnChart, setShowAlertsOnChart] = useState(true);
  const [showTradesOnChart, setShowTradesOnChart] = useState(true);
  const [showRegimeBacktestsOnChart, setShowRegimeBacktestsOnChart] = useState(true);
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
  const [dateRangeSlider, setDateRangeSlider] = useState<[number, number]>([
    0,
    RANGE_SLIDER_STEPS
  ]);
  const [priceRangeSlider, setPriceRangeSlider] = useState<[number, number]>([
    0,
    RANGE_SLIDER_STEPS
  ]);
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

  const symbolScopedPredictions = useMemo(() => {
    return (data?.predictions || []).filter((row) => {
      if (symbolFilter !== "all" && row.symbol !== symbolFilter) {
        return false;
      }
      return true;
    });
  }, [data?.predictions, symbolFilter]);

  const filteredPredictions = useMemo(() => {
    return symbolScopedPredictions.filter((row) => {
      if (!recommendationFilter.includes(row.recommendation)) {
        return false;
      }
      return true;
    });
  }, [symbolScopedPredictions, recommendationFilter]);

  const filteredAlerts = useMemo(() => {
    return (data?.alerts || []).filter((row) => {
      if (symbolFilter !== "all" && row.symbol !== symbolFilter) {
        return false;
      }
      if (!recommendationFilter.includes(row.recommendation)) {
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
  const regimeBacktestPerformance = data?.performance.regimeBacktests;
  const recentModelRuns = modelPerformance?.runs.slice(0, 12) || [];
  const recentExecutions = paperTradingPerformance?.executions.slice(0, 20) || [];
  const regimeBacktestRuns = useMemo(
    () => regimeBacktestPerformance?.runs || [],
    [regimeBacktestPerformance?.runs]
  );
  const filteredTradeExecutions = useMemo(() => {
    return (paperTradingPerformance?.executions || []).filter((row) => {
      if (row.executionStatus !== "executed") {
        return false;
      }
      if (row.executedAction !== "buy" && row.executedAction !== "sell") {
        return false;
      }
      if (symbolFilter !== "all" && row.symbol !== symbolFilter) {
        return false;
      }
      if (!recommendationFilter.includes(row.executedAction)) {
        return false;
      }
      return true;
    });
  }, [paperTradingPerformance?.executions, recommendationFilter, symbolFilter]);

  const chartPoints = useMemo<ChartPoint[]>(() => {
    const rows: ChartPoint[] = symbolScopedPredictions
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
      .filter((row): row is ChartPoint => row !== null);
    for (const run of regimeBacktestRuns) {
      const startMs = Date.parse(run.startUtc || "");
      const endMs = Date.parse(run.endUtc || run.startUtc || "");
      if (Number.isFinite(startMs)) {
        rows.push({
          id: `regime-window-start:${run.id}`,
          timeMs: startMs,
          timeLabel: run.startUtc || "",
          closePrice: null,
          smaFast: null,
          smaSlow: null,
          macd: null,
          macdHist: null,
          rsi14: null,
          volatility24: null
        });
      }
      if (Number.isFinite(endMs)) {
        rows.push({
          id: `regime-window-end:${run.id}`,
          timeMs: endMs,
          timeLabel: run.endUtc || run.startUtc || "",
          closePrice: null,
          smaFast: null,
          smaSlow: null,
          macd: null,
          macdHist: null,
          rsi14: null,
          volatility24: null
        });
      }
    }
    return rows.sort((a, b) => a.timeMs - b.timeMs);
  }, [symbolScopedPredictions, regimeBacktestRuns]);

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
    for (const row of symbolScopedPredictions) {
      map.set(row.predictionPath, row);
    }
    return map;
  }, [symbolScopedPredictions]);

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

  const tradeMarkers = useMemo<MarkerPoint[]>(() => {
    const rows: MarkerPoint[] = [];
    for (const row of filteredTradeExecutions) {
      const timestamp = row.predictionTimestampUtc || row.createdAtUtc;
      const parsed = Date.parse(timestamp);
      if (!Number.isFinite(parsed)) {
        continue;
      }
      let markerPrice = row.executionPrice ?? row.markPrice;
      if (markerPrice === null) {
        let nearestPoint: ChartPoint | null = null;
        let nearestDelta = Number.POSITIVE_INFINITY;
        for (const point of chartPoints) {
          if (point.closePrice === null) {
            continue;
          }
          const delta = Math.abs(point.timeMs - parsed);
          if (delta < nearestDelta) {
            nearestDelta = delta;
            nearestPoint = point;
          }
        }
        markerPrice = nearestPoint?.closePrice ?? null;
      }
      if (markerPrice === null || !Number.isFinite(markerPrice)) {
        continue;
      }
      rows.push({
        id: `trade:${row.id}`,
        timeMs: parsed,
        timeLabel: timestamp,
        price: markerPrice,
        recommendation: row.executedAction,
        confidence: 1,
        source: "trade",
        trade: row
      });
    }
    return rows.sort((a, b) => a.timeMs - b.timeMs);
  }, [chartPoints, filteredTradeExecutions]);
  const regimeRanges = useMemo(() => {
    return regimeBacktestRuns
      .map((run) => {
        const parsedStart = Date.parse(run.startUtc || "");
        const parsedEnd = Date.parse(run.endUtc || run.startUtc || "");
        if (!Number.isFinite(parsedStart) || !Number.isFinite(parsedEnd)) {
          return null;
        }
        const startMs = Math.min(parsedStart, parsedEnd);
        const endMs = Math.max(parsedStart, parsedEnd);
        return {
          id: run.id,
          startMs,
          endMs,
          run
        };
      })
      .filter(
        (row): row is { id: string; startMs: number; endMs: number; run: RegimeBacktestRunSummary } =>
          row !== null
      )
      .sort((left, right) => left.startMs - right.startMs);
  }, [regimeBacktestRuns]);
  const regimeMarkers = useMemo<MarkerPoint[]>(() => {
    const rows: MarkerPoint[] = [];
    for (const run of regimeBacktestRuns) {
      const timestamp = run.endUtc || run.startUtc;
      const parsed = Date.parse(timestamp);
      if (!Number.isFinite(parsed)) {
        continue;
      }
      let markerPrice: number | null = null;
      let nearestDelta = Number.POSITIVE_INFINITY;
      for (const point of chartPoints) {
        if (point.closePrice === null) {
          continue;
        }
        const delta = Math.abs(point.timeMs - parsed);
        if (delta < nearestDelta) {
          nearestDelta = delta;
          markerPrice = point.closePrice;
        }
      }
      if (markerPrice === null || !Number.isFinite(markerPrice)) {
        continue;
      }
      rows.push({
        id: `regime:${run.id}`,
        timeMs: parsed,
        timeLabel: timestamp,
        price: markerPrice,
        recommendation: run.realizedPnlDeltaUsd >= 0 ? "buy" : "sell",
        confidence: 1,
        source: "regime",
        regimeRun: run
      });
    }
    return rows.sort((a, b) => a.timeMs - b.timeMs);
  }, [chartPoints, regimeBacktestRuns]);
  const markerFocusDomain = useMemo<[number, number] | null>(() => {
    if (!fullDomain) {
      return null;
    }
    const markerTimes = [...predictionMarkers, ...alertMarkers, ...tradeMarkers, ...regimeMarkers]
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
  }, [predictionMarkers, alertMarkers, tradeMarkers, regimeMarkers, minimumZoomSpanMs, fullDomain]);

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
  const fullDomainStart = fullDomain ? fullDomain[0] : null;
  const fullDomainEnd = fullDomain ? fullDomain[1] : null;
  const visibleDomainStart = visibleDomain ? visibleDomain[0] : null;
  const visibleDomainEnd = visibleDomain ? visibleDomain[1] : null;

  const activePriceBounds = useMemo<[number, number] | null>(() => {
    let minPrice = Number.POSITIVE_INFINITY;
    let maxPrice = Number.NEGATIVE_INFINITY;
    const withinVisibleDomain = (timeMs: number) =>
      !visibleDomain || (timeMs >= visibleDomain[0] && timeMs <= visibleDomain[1]);
    const consider = (value: number | null | undefined) => {
      if (value === null || value === undefined || !Number.isFinite(value)) {
        return;
      }
      minPrice = Math.min(minPrice, value);
      maxPrice = Math.max(maxPrice, value);
    };

    for (const point of chartPoints) {
      if (!withinVisibleDomain(point.timeMs)) {
        continue;
      }
      if (showCloseLine) {
        consider(point.closePrice);
      }
      if (showSmaFastLine) {
        consider(point.smaFast);
      }
      if (showSmaSlowLine) {
        consider(point.smaSlow);
      }
    }
    if (showPredictionsOnChart) {
      for (const marker of predictionMarkers) {
        if (withinVisibleDomain(marker.timeMs)) {
          consider(marker.price);
        }
      }
    }
    if (showAlertsOnChart) {
      for (const marker of alertMarkers) {
        if (withinVisibleDomain(marker.timeMs)) {
          consider(marker.price);
        }
      }
    }
    if (showTradesOnChart) {
      for (const marker of tradeMarkers) {
        if (withinVisibleDomain(marker.timeMs)) {
          consider(marker.price);
        }
      }
    }
    if (showRegimeBacktestsOnChart) {
      for (const marker of regimeMarkers) {
        if (withinVisibleDomain(marker.timeMs)) {
          consider(marker.price);
        }
      }
    }

    if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) {
      return null;
    }
    if (maxPrice <= minPrice) {
      const pad = Math.max(Math.abs(minPrice) * 0.01, 1);
      return [minPrice - pad, maxPrice + pad];
    }
    const pad = (maxPrice - minPrice) * 0.04;
    return [minPrice - pad, maxPrice + pad];
  }, [
    alertMarkers,
    chartPoints,
    predictionMarkers,
    showAlertsOnChart,
    showCloseLine,
    showPredictionsOnChart,
    showRegimeBacktestsOnChart,
    showSmaFastLine,
    showSmaSlowLine,
    showTradesOnChart,
    regimeMarkers,
    tradeMarkers,
    visibleDomain
  ]);

  const priceAxisDomain = useMemo<[number, number] | null>(() => {
    if (!activePriceBounds) {
      return null;
    }
    const [boundMin, boundMax] = activePriceBounds;
    const span = Math.max(1e-9, boundMax - boundMin);
    let selectedMin = boundMin + (priceRangeSlider[0] / RANGE_SLIDER_STEPS) * span;
    let selectedMax = boundMin + (priceRangeSlider[1] / RANGE_SLIDER_STEPS) * span;
    if (selectedMax <= selectedMin) {
      const minGap = Math.max(span / RANGE_SLIDER_STEPS, 1e-6);
      selectedMax = selectedMin + minGap;
    }
    return [selectedMin, selectedMax];
  }, [activePriceBounds, priceRangeSlider]);

  const selectedDateRange = useMemo<[number | null, number | null]>(() => {
    if (!fullDomain) {
      return [null, null];
    }
    const [boundStart] = fullDomain;
    const startMs = boundStart + (dateRangeSlider[0] / RANGE_SLIDER_STEPS) * fullSpanMs;
    const endMs = boundStart + (dateRangeSlider[1] / RANGE_SLIDER_STEPS) * fullSpanMs;
    return [startMs, endMs];
  }, [dateRangeSlider, fullDomain, fullSpanMs]);

  const selectedPriceRange = useMemo<[number | null, number | null]>(() => {
    if (!priceAxisDomain) {
      return [null, null];
    }
    return [priceAxisDomain[0], priceAxisDomain[1]];
  }, [priceAxisDomain]);

  const canZoomIn = fullDomain !== null && visibleSpanMs > minimumZoomSpanMs + 1;
  const canZoomOut = fullDomain !== null && visibleSpanMs < fullSpanMs - 1;
  const canDragPan = canZoomOut;

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

  const applyDateRangeSlider = useCallback(
    (requestedStartStep: number, requestedEndStep: number) => {
      if (!fullDomain) {
        return;
      }
      let startStep = Math.round(requestedStartStep);
      let endStep = Math.round(requestedEndStep);
      startStep = Math.max(0, Math.min(startStep, RANGE_SLIDER_STEPS - 1));
      endStep = Math.min(RANGE_SLIDER_STEPS, Math.max(endStep, startStep + 1));
      const nextSlider: [number, number] = [startStep, endStep];
      setDateRangeSlider(nextSlider);

      if (startStep === 0 && endStep === RANGE_SLIDER_STEPS) {
        setZoomDomain(null);
        return;
      }
      const [boundStart, boundEnd] = fullDomain;
      const span = Math.max(1, boundEnd - boundStart);
      const targetStart = boundStart + (startStep / RANGE_SLIDER_STEPS) * span;
      const targetEnd = boundStart + (endStep / RANGE_SLIDER_STEPS) * span;
      setZoomDomain(clampDomainToBounds(targetStart, targetEnd, fullDomain));
    },
    [fullDomain]
  );

  useEffect(() => {
    setZoomDomain(null);
  }, [fullDomainStart, fullDomainEnd]);

  useEffect(() => {
    if (!fullDomain || !visibleDomain) {
      setDateRangeSlider((prev) =>
        prev[0] === 0 && prev[1] === RANGE_SLIDER_STEPS ? prev : [0, RANGE_SLIDER_STEPS]
      );
      return;
    }
    const [boundStart, boundEnd] = fullDomain;
    const span = Math.max(1, boundEnd - boundStart);
    let startStep = Math.round(((visibleDomain[0] - boundStart) / span) * RANGE_SLIDER_STEPS);
    let endStep = Math.round(((visibleDomain[1] - boundStart) / span) * RANGE_SLIDER_STEPS);
    startStep = Math.max(0, Math.min(startStep, RANGE_SLIDER_STEPS - 1));
    endStep = Math.min(RANGE_SLIDER_STEPS, Math.max(endStep, startStep + 1));
    setDateRangeSlider((prev) =>
      prev[0] === startStep && prev[1] === endStep ? prev : [startStep, endStep]
    );
  }, [fullDomainStart, fullDomainEnd, visibleDomainStart, visibleDomainEnd, fullDomain, visibleDomain]);

  useEffect(() => {
    setPriceRangeSlider((prev) =>
      prev[0] === 0 && prev[1] === RANGE_SLIDER_STEPS ? prev : [0, RANGE_SLIDER_STEPS]
    );
  }, [fullDomainStart, fullDomainEnd, symbolFilter]);

  useEffect(() => {
    setPriceRangeSlider((prev) => {
      let nextMin = Math.max(0, Math.min(prev[0], RANGE_SLIDER_STEPS - 1));
      let nextMax = Math.min(RANGE_SLIDER_STEPS, Math.max(prev[1], nextMin + 1));
      if (nextMin === prev[0] && nextMax === prev[1]) {
        return prev;
      }
      return [nextMin, nextMax];
    });
  }, [activePriceBounds]);

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
            multiple
            value={recommendationFilter}
            onChange={(event) => {
              const selected = Array.from(event.target.selectedOptions).map(
                (option) => option.value as Recommendation
              );
              setRecommendationFilter(
                selected.length > 0 ? selected : [...RECOMMENDATION_OPTIONS]
              );
            }}
          >
            {RECOMMENDATION_OPTIONS.map((value) => (
              <option value={value} key={value}>
                {value}
              </option>
            ))}
          </select>
          <div className="recommendation-actions">
            <button
              type="button"
              className="chip-btn"
              onClick={() => setRecommendationFilter([...RECOMMENDATION_OPTIONS])}
            >
              all
            </button>
            <button
              type="button"
              className="chip-btn"
              onClick={() => setRecommendationFilter(["buy", "sell"])}
            >
              buy + sell
            </button>
          </div>
          <span className="recommendation-helper muted">Ctrl/Cmd-click supports manual multi-select.</span>
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
        <button
          type="button"
          className={`toggle-btn large ${showTradesOnChart ? "active" : ""}`}
          onClick={() => setShowTradesOnChart((prev) => !prev)}
        >
          Executed Trades ({tradeMarkers.length})
        </button>
        <button
          type="button"
          className={`toggle-btn large ${showRegimeBacktestsOnChart ? "active" : ""}`}
          onClick={() => setShowRegimeBacktestsOnChart((prev) => !prev)}
        >
          Regime Backtests ({regimeMarkers.length})
        </button>
        <div className="stats-inline muted">
          actionable={actionablePredictions.length} • high-confidence={highConfidence.length} • trades=
          {tradeMarkers.length} • regime PnL=
          {formatUsd(regimeBacktestPerformance?.totalRealizedPnlDeltaUsd, 2)}
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
      <section className="button-row compact chart-range-controls">
        <label className="range-control wide">
          Date start ({formatDateTimeLabel(selectedDateRange[0])})
          <input
            type="range"
            min={0}
            max={RANGE_SLIDER_STEPS - 1}
            step={1}
            value={dateRangeSlider[0]}
            disabled={!fullDomain}
            onChange={(event) =>
              applyDateRangeSlider(Number(event.target.value), dateRangeSlider[1])
            }
          />
        </label>
        <label className="range-control wide">
          Date end ({formatDateTimeLabel(selectedDateRange[1])})
          <input
            type="range"
            min={1}
            max={RANGE_SLIDER_STEPS}
            step={1}
            value={dateRangeSlider[1]}
            disabled={!fullDomain}
            onChange={(event) =>
              applyDateRangeSlider(dateRangeSlider[0], Number(event.target.value))
            }
          />
        </label>
      </section>
      <section className="button-row compact chart-range-controls">
        <label className="range-control wide">
          Price bottom ({formatNumber(selectedPriceRange[0], 2)})
          <input
            type="range"
            min={0}
            max={RANGE_SLIDER_STEPS - 1}
            step={1}
            value={priceRangeSlider[0]}
            disabled={!activePriceBounds}
            onChange={(event) => {
              const nextBottom = Number(event.target.value);
              setPriceRangeSlider((prev) => {
                const boundedBottom = Math.max(0, Math.min(nextBottom, prev[1] - 1));
                return [boundedBottom, prev[1]];
              });
            }}
          />
        </label>
        <label className="range-control wide">
          Price top ({formatNumber(selectedPriceRange[1], 2)})
          <input
            type="range"
            min={1}
            max={RANGE_SLIDER_STEPS}
            step={1}
            value={priceRangeSlider[1]}
            disabled={!activePriceBounds}
            onChange={(event) => {
              const nextTop = Number(event.target.value);
              setPriceRangeSlider((prev) => {
                const boundedTop = Math.min(RANGE_SLIDER_STEPS, Math.max(nextTop, prev[0] + 1));
                return [prev[0], boundedTop];
              });
            }}
          />
        </label>
        <button
          type="button"
          className="toggle-btn"
          onClick={() => setPriceRangeSlider([0, RANGE_SLIDER_STEPS])}
          disabled={priceRangeSlider[0] === 0 && priceRangeSlider[1] === RANGE_SLIDER_STEPS}
        >
          Reset Price Range
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
                    domain={priceAxisDomain ? [priceAxisDomain[0], priceAxisDomain[1]] : ["auto", "auto"]}
                    label={{
                      value: "Price (USD)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#a6b0cf",
                      dx: -6
                    }}
                  />
                  <Tooltip
                    content={<PriceChartTooltip />}
                  />
                  <Legend />
                  {showRegimeBacktestsOnChart
                    ? regimeRanges.map(({ id, startMs, endMs, run }) => {
                        const normalizedRegime = (run.regime || "").toLowerCase();
                        const fill = normalizedRegime.includes("up")
                          ? "#22c55e"
                          : normalizedRegime.includes("down")
                            ? "#3b82f6"
                            : normalizedRegime.includes("flat")
                              ? "#eab308"
                              : "#64748b";
                        const stroke = run.realizedPnlDeltaUsd >= 0 ? "#22c55e" : "#ef4444";
                        return (
                          <ReferenceArea
                            key={`regime-area:${id}`}
                            x1={startMs}
                            x2={endMs}
                            yAxisId="price"
                            ifOverflow="visible"
                            fill={fill}
                            fillOpacity={0.08}
                            stroke={stroke}
                            strokeOpacity={0.45}
                            strokeWidth={1}
                            onClick={() => setSelectedDatum({ kind: "regime", row: run })}
                          />
                        );
                      })
                    : null}
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
                  {showTradesOnChart ? (
                    <Scatter
                      yAxisId="price"
                      name="Executed trades"
                      data={tradeMarkers}
                      dataKey="price"
                      shape={renderTradeShape}
                      onClick={(event: any) => {
                        const payload = event?.payload as MarkerPoint | undefined;
                        if (payload?.trade) {
                          setSelectedDatum({ kind: "trade", row: payload.trade });
                        }
                      }}
                    />
                  ) : null}
                  {showRegimeBacktestsOnChart ? (
                    <Scatter
                      yAxisId="price"
                      name="Regime backtests"
                      data={regimeMarkers}
                      dataKey="price"
                      shape={renderRegimeShape}
                      onClick={(event: any) => {
                        const payload = event?.payload as MarkerPoint | undefined;
                        if (payload?.regimeRun) {
                          setSelectedDatum({ kind: "regime", row: payload.regimeRun });
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
            Select a prediction/alert marker, executed trade marker, or regime marker on the chart
            to inspect confidence, trade sizing, and realized PnL details.
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
        ) : selectedDatum.kind === "alert" ? (
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
            <p className="muted">alert record: {selectedDatum.row.alertPath}</p>
          </div>
        ) : selectedDatum.kind === "regime" ? (
          <div className="detail-panel">
            <div>
              <span
                className={`pill ${selectedDatum.row.realizedPnlDeltaUsd >= 0 ? "buy" : "sell"}`}
              >
                {selectedDatum.row.realizedPnlDeltaUsd >= 0 ? "profit" : "loss"}
              </span>{" "}
              regime backtest
            </div>
            <div className="muted">
              {selectedDatum.row.scenario} • split={selectedDatum.row.split} • regime=
              {selectedDatum.row.regime || "n/a"}
            </div>
            <p>
              window={selectedDatum.row.startUtc || "n/a"} → {selectedDatum.row.endUtc || "n/a"}
            </p>
            <p>
              samples={selectedDatum.row.sampleCount} • train/test={selectedDatum.row.trainCount}/
              {selectedDatum.row.testCount}
            </p>
            <p>
              realized PnL Δ={formatUsd(selectedDatum.row.realizedPnlDeltaUsd, 2)} • equity return=
              {formatPercent(selectedDatum.row.equityReturn, 2)}
            </p>
            <p className="muted">source: {selectedDatum.row.sourceResultsPath}</p>
          </div>
        ) : (
          <div className="detail-panel">
            <div>
              <span className={`pill ${selectedDatum.row.executedAction}`}>
                {selectedDatum.row.executedAction}
              </span>{" "}
              executed trade
            </div>
            <div className="muted">
              {selectedDatum.row.exchange} {selectedDatum.row.symbol} ({selectedDatum.row.timeframe}) •{" "}
              {selectedDatum.row.predictionTimestampUtc || selectedDatum.row.createdAtUtc}
            </div>
            <p>
              status={selectedDatum.row.executionStatus} • intent={selectedDatum.row.intentAction} • fill=
              {formatPercent(selectedDatum.row.fillRatio, 2)}
            </p>
            <p>
              mark={formatNumber(selectedDatum.row.markPrice, 2)} execution=
              {formatNumber(selectedDatum.row.executionPrice, 2)} notional=
              {formatUsd(selectedDatum.row.executedNotionalUsd, 2)} fee=
              {formatUsd(selectedDatum.row.feeUsd, 4)}
            </p>
            <p>
              realized PnL Δ={formatUsd(selectedDatum.row.realizedPnlDeltaUsd, 4)} cash after=
              {formatUsd(selectedDatum.row.cashAfterUsd, 2)} position qty after=
              {formatNumber(selectedDatum.row.positionQtyAfter, 8)}
            </p>
            <p className="muted">reason={selectedDatum.row.reason || "n/a"}</p>
            <p className="muted">execution record: {selectedDatum.row.executionRecordPath}</p>
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
            <article className="card">
              <div className="label">Regime backtests tracked</div>
              <div className="value">{regimeBacktestPerformance?.runCount ?? 0}</div>
            </article>
            <article className="card">
              <div className="label">Regime realized PnL delta (USD)</div>
              <div className="value">
                {formatNumber(regimeBacktestPerformance?.totalRealizedPnlDeltaUsd, 2)}
              </div>
            </article>
            <article className="card">
              <div className="label">Regime avg equity return</div>
              <div className="value">
                {formatPercent(regimeBacktestPerformance?.averageEquityReturn, 2)}
              </div>
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
          <section className="section">
            <h2>Regime backtest performance</h2>
            {!regimeBacktestRuns.length ? (
              <p className="muted">
                No ranked ablation regime summaries found yet under the analysis output path.
              </p>
            ) : (
              <>
                <p className="muted">
                  Source results: <code>{regimeBacktestPerformance?.sourceResultsPath || "n/a"}</code>
                </p>
                <table>
                  <thead>
                    <tr>
                      <th>Scenario</th>
                      <th>Split</th>
                      <th>Regime</th>
                      <th>Window (UTC)</th>
                      <th>Samples</th>
                      <th>Train/Test</th>
                      <th>Equity Return</th>
                      <th>Realized PnL Δ USD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {regimeBacktestRuns.map((row) => (
                      <tr key={row.id}>
                        <td>{row.scenario}</td>
                        <td>{row.split}</td>
                        <td>{row.regime || "n/a"}</td>
                        <td>
                          {row.startUtc || "n/a"} → {row.endUtc || "n/a"}
                        </td>
                        <td>{row.sampleCount}</td>
                        <td>
                          {row.trainCount}/{row.testCount}
                        </td>
                        <td>{formatPercent(row.equityReturn, 2)}</td>
                        <td>{formatNumber(row.realizedPnlDeltaUsd, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </section>
        </>
      )}
    </main>
  );
}