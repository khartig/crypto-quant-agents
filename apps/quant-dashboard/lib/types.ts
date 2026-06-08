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

export interface DashboardOverview {
  generatedAtUtc: string;
  quantDataRoot: string;
  predictions: TriggerPredictionRow[];
  alerts: TriggerAlertRow[];
  latestAgentPlane: AgentPlaneSummary | null;
}
