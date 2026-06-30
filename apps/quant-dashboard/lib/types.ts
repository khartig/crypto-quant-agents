export type Recommendation = "buy" | "sell" | "hold";
export interface ReasonDetail {
  feature: string;
  impact: number;
  supports: string;
  value: number;
  vsAlternative: string;
}

export interface TriggerPredictionRow {
  id: string;
  createdAtUtc: string;
  predictionTimestampUtc: string;
  exchange: string;
  symbol: string;
  timeframe: string;
  recommendation: Recommendation;
  confidence: number;
  probabilities: {
    buy: number;
    hold: number;
    sell: number;
  };
  closePrice: number | null;
  smaFast: number | null;
  smaSlow: number | null;
  macd: number | null;
  macdHist: number | null;
  rsi14: number | null;
  volatility24: number | null;
  topReasons: string[];
  reasonDetails: ReasonDetail[];
  modelPath: string | null;
  sourceDataPath: string | null;
  predictionPath: string;
}

export interface TriggerAlertRow {
  id: string;
  createdAtUtc: string;
  predictionTimestampUtc: string;
  exchange: string;
  symbol: string;
  timeframe: string;
  recommendation: Recommendation;
  confidence: number;
  probabilities: {
    buy: number;
    hold: number;
    sell: number;
  };
  closePrice: number | null;
  smaFast: number | null;
  smaSlow: number | null;
  macd: number | null;
  macdHist: number | null;
  rsi14: number | null;
  volatility24: number | null;
  topReasons: string[];
  reasonDetails: ReasonDetail[];
  predictionPath: string | null;
  alertPath: string;
}

export interface AgentPlaneSummary {
  runId: string;
  createdAtUtc: string;
  exchange: string;
  symbol: string;
  timeframe: string;
  riskApproved: boolean;
  intentStatus: string;
  paperExecutionStatus: string;
  runManifestPath: string;
}
export interface ModelTrainingRunSummary {
  id: string;
  createdAtUtc: string;
  exchange: string;
  symbol: string;
  timeframe: string;
  sampleCount: number;
  trainCount: number;
  testCount: number;
  accuracy: number | null;
  modelPath: string;
}

export interface ModelPerformanceSummary {
  runCount: number;
  latestAccuracy: number | null;
  rollingAccuracy: number | null;
  latestModelPath: string | null;
  runs: ModelTrainingRunSummary[];
}

export interface PaperTradeExecutionRow {
  id: string;
  runId: string;
  createdAtUtc: string;
  predictionTimestampUtc: string;
  exchange: string;
  symbol: string;
  timeframe: string;
  intentAction: Recommendation;
  executedAction: Recommendation;
  executionStatus: "executed" | "skipped" | "rejected";
  executedNotionalUsd: number;
  markPrice: number | null;
  executionPrice: number | null;
  feeUsd: number;
  fillRatio: number;
  realizedPnlDeltaUsd: number;
  cashAfterUsd: number | null;
  positionQtyAfter: number | null;
  reason: string;
  executionRecordPath: string;
}

export interface PaperTradePerformanceSummary {
  totalExecutions: number;
  executedCount: number;
  skippedCount: number;
  rejectedCount: number;
  totalNotionalUsd: number;
  totalFeesUsd: number;
  totalRealizedPnlDeltaUsd: number;
  winRate: number | null;
  executions: PaperTradeExecutionRow[];
}

export interface RegimeBacktestRunSummary {
  id: string;
  scenario: string;
  split: string;
  regime: string;
  startUtc: string;
  endUtc: string;
  sampleCount: number;
  trainCount: number;
  testCount: number;
  equityReturn: number | null;
  realizedPnlDeltaUsd: number;
  sourceResultsPath: string;
}

export interface RegimeBacktestSummary {
  runCount: number;
  totalRealizedPnlDeltaUsd: number;
  averageEquityReturn: number | null;
  sourceResultsPath: string | null;
  runs: RegimeBacktestRunSummary[];
}

export interface PerformanceOverview {
  model: ModelPerformanceSummary;
  paperTrading: PaperTradePerformanceSummary;
  regimeBacktests: RegimeBacktestSummary;
}

export interface DashboardOverview {
  generatedAtUtc: string;
  quantDataRoot: string;
  predictions: TriggerPredictionRow[];
  alerts: TriggerAlertRow[];
  latestAgentPlane: AgentPlaneSummary | null;
  performance: PerformanceOverview;
}
