import { promises as fs } from "fs";
import path from "path";
import type {
  AgentPlaneSummary,
  DashboardOverview,
  ReasonDetail,
  TriggerAlertRow,
  TriggerPredictionRow
} from "@/lib/types";

const DEFAULT_QUANT_ROOT = "/mnt/quant-data";

function quantDataRoot(): string {
  return process.env.QUANT_DATA_ROOT || DEFAULT_QUANT_ROOT;
}

function normalizeReasonDetails(value: unknown): ReasonDetail[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const rows: ReasonDetail[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const payload = item as Record<string, unknown>;
    const feature = String(payload.feature || "").trim();
    if (!feature) {
      continue;
    }
    rows.push({
      feature,
      impact: asNumber(payload.impact, 0),
      supports: String(payload.supports || ""),
      value: asNumber(payload.value, 0),
      vsAlternative: String(payload.vs_alternative || "")
    });
  }
  return rows.slice(0, 8);
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

async function listFilesRecursive(
  dirPath: string,
  matchFile: (name: string) => boolean,
  maxFiles: number,
  collected: string[] = []
): Promise<string[]> {
  if (collected.length >= maxFiles) {
    return collected;
  }
  let entries;
  try {
    entries = await fs.readdir(dirPath, { withFileTypes: true });
  } catch {
    return collected;
  }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (collected.length >= maxFiles) {
      break;
    }
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      await listFilesRecursive(fullPath, matchFile, maxFiles, collected);
      continue;
    }
    if (matchFile(entry.name)) {
      collected.push(fullPath);
    }
  }
  return collected;
}

async function readJson(filePath: string): Promise<unknown | null> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}


function asNullableNumber(value: unknown): number | null {
  const parsed = asNumber(value, Number.NaN);
  return Number.isFinite(parsed) ? parsed : null;
}
function normalizeProbabilities(value: unknown): { buy: number; hold: number; sell: number } {
  const payload = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  return {
    buy: asNumber(payload.buy, 0),
    hold: asNumber(payload.hold, 0),
    sell: asNumber(payload.sell, 0)
  };
}

async function loadPredictions(root: string, limit = 200): Promise<TriggerPredictionRow[]> {
  const base = path.join(root, "logs", "agents", "model-predictor");
  if (!(await pathExists(base))) {
    return [];
  }
  const files = await listFilesRecursive(base, (name) => name.startsWith("prediction_") && name.endsWith(".json"), limit * 4);
  const sorted = files.sort().reverse().slice(0, limit);
  const rows: TriggerPredictionRow[] = [];

  for (const filePath of sorted) {
    const parsed = await readJson(filePath);
    if (!parsed || typeof parsed !== "object") {
      continue;
    }
    const payload = parsed as Record<string, unknown>;
    const recommendation = String(payload.recommendation || "hold").toLowerCase();
    if (recommendation !== "buy" && recommendation !== "sell" && recommendation !== "hold") {
      continue;
    }
    const topReasonsRaw = Array.isArray(payload.top_reasons) ? payload.top_reasons : [];
    const topReasons = topReasonsRaw
      .map((reason) => String(reason))
      .filter((reason) => reason.length > 0)
      .slice(0, 5);
    const featureValues =
      typeof payload.feature_values === "object" && payload.feature_values !== null
        ? (payload.feature_values as Record<string, unknown>)
        : {};
    const closePrice = asNullableNumber(payload.close_price ?? featureValues.close);
    const smaFastSpread = asNullableNumber(featureValues.sma_fast_spread);
    const smaSlowSpread = asNullableNumber(featureValues.sma_slow_spread);
    const inferredSmaFast =
      closePrice !== null &&
      smaFastSpread !== null &&
      Math.abs(1 + smaFastSpread) > 1e-9
        ? closePrice / (1 + smaFastSpread)
        : null;
    const inferredSmaSlow =
      closePrice !== null &&
      smaSlowSpread !== null &&
      Math.abs(1 + smaSlowSpread) > 1e-9
        ? closePrice / (1 + smaSlowSpread)
        : null;

    rows.push({
      id: filePath,
      createdAtUtc: String(payload.created_at_utc || ""),
      predictionTimestampUtc: String(payload.prediction_timestamp_utc || ""),
      exchange: String(payload.exchange || ""),
      symbol: String(payload.symbol || ""),
      timeframe: String(payload.timeframe || ""),
      recommendation,
      confidence: asNumber(payload.confidence, 0),
      probabilities: normalizeProbabilities(payload.probabilities),
      closePrice,
      smaFast: asNullableNumber(payload.sma_fast) ?? inferredSmaFast,
      smaSlow: asNullableNumber(payload.sma_slow) ?? inferredSmaSlow,
      macd: asNullableNumber(payload.macd ?? featureValues.macd),
      macdHist: asNullableNumber(payload.macd_hist ?? featureValues.macd_hist),
      rsi14: asNullableNumber(payload.rsi_14 ?? featureValues.rsi_14),
      volatility24: asNullableNumber(payload.volatility_24 ?? featureValues.volatility_24),
      topReasons,
      reasonDetails: normalizeReasonDetails(payload.reason_details),
      modelPath: payload.model_path ? String(payload.model_path) : null,
      sourceDataPath: payload.source_data_path ? String(payload.source_data_path) : null,
      predictionPath: filePath
    });
  }

  return rows;
}

async function loadAlerts(root: string, limit = 200): Promise<TriggerAlertRow[]> {
  const base = path.join(root, "logs", "agents", "trigger-monitor");
  if (!(await pathExists(base))) {
    return [];
  }
  const alertFiles = await listFilesRecursive(base, (name) => name === "alerts.jsonl", limit * 2);
  const sortedFiles = alertFiles.sort().reverse();
  const rows: TriggerAlertRow[] = [];

  for (const alertPath of sortedFiles) {
    let content = "";
    try {
      content = await fs.readFile(alertPath, "utf-8");
    } catch {
      continue;
    }
    const lines = content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .reverse();

    for (const line of lines) {
      if (rows.length >= limit) {
        break;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        continue;
      }
      if (!parsed || typeof parsed !== "object") {
        continue;
      }
      const payload = parsed as Record<string, unknown>;
      const recommendation = String(payload.recommendation || "hold").toLowerCase();
      if (recommendation !== "buy" && recommendation !== "sell" && recommendation !== "hold") {
        continue;
      }
      const topReasonsRaw = Array.isArray(payload.top_reasons) ? payload.top_reasons : [];
      const topReasons = topReasonsRaw
        .map((reason) => String(reason))
        .filter((reason) => reason.length > 0)
        .slice(0, 5);
      rows.push({
        id: `${alertPath}:${rows.length}`,
        createdAtUtc: String(payload.created_at_utc || ""),
        predictionTimestampUtc: String(payload.prediction_timestamp_utc || ""),
        exchange: String(payload.exchange || ""),
        symbol: String(payload.symbol || ""),
        timeframe: String(payload.timeframe || ""),
        recommendation,
        confidence: asNumber(payload.confidence, 0),
        probabilities: normalizeProbabilities(payload.probabilities),
        closePrice: asNullableNumber(payload.close_price),
        smaFast: asNullableNumber(payload.sma_fast),
        smaSlow: asNullableNumber(payload.sma_slow),
        macd: asNullableNumber(payload.macd),
        macdHist: asNullableNumber(payload.macd_hist),
        rsi14: asNullableNumber(payload.rsi_14),
        volatility24: asNullableNumber(payload.volatility_24),
        topReasons,
        reasonDetails: normalizeReasonDetails(payload.reason_details),
        predictionPath: payload.prediction_path ? String(payload.prediction_path) : null,
        alertPath
      });
    }
    if (rows.length >= limit) {
      break;
    }
  }

  return rows;
}

async function loadLatestAgentPlane(root: string): Promise<AgentPlaneSummary | null> {
  const base = path.join(root, "logs", "agents", "openclaw-orchestrator");
  if (!(await pathExists(base))) {
    return null;
  }
  const manifests = await listFilesRecursive(base, (name) => name === "run_manifest.json", 400);
  if (!manifests.length) {
    return null;
  }
  const latestManifest = manifests.sort().reverse()[0];
  const parsed = await readJson(latestManifest);
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  const payload = parsed as Record<string, any>;
  return {
    runId: String(payload.run_id || path.basename(path.dirname(latestManifest))),
    createdAtUtc: String(payload.created_at_utc || ""),
    exchange: String(payload.scope?.exchange || ""),
    symbol: String(payload.scope?.symbol || ""),
    timeframe: String(payload.scope?.timeframe || ""),
    riskApproved: Boolean(payload.outcome?.risk_approved),
    intentStatus: String(payload.outcome?.intent_status || ""),
    paperExecutionStatus: String(payload.outcome?.paper_trade_execution_status || ""),
    runManifestPath: latestManifest
  };
}

export async function loadDashboardOverview(): Promise<DashboardOverview> {
  const root = quantDataRoot();
  const [predictions, alerts, latestAgentPlane] = await Promise.all([
    loadPredictions(root, 200),
    loadAlerts(root, 200),
    loadLatestAgentPlane(root)
  ]);

  return {
    generatedAtUtc: new Date().toISOString(),
    quantDataRoot: root,
    predictions,
    alerts,
    latestAgentPlane
  };
}
