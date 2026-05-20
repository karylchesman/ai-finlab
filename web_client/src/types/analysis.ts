export interface FundamentalAnalysis {
  overall_investment_thesis: string;
  investment_grade: string;
  confidence_score: number;
  key_strengths: string[];
  key_concerns: string[];
  recommendation: string;
}

export interface MomentumAnalysis {
  overall_momentum: string;
  momentum_strength: string;
  key_momentum_drivers: string[];
  momentum_risks: string[];
  short_term_outlook: string;
  momentum_score: number;
}

export interface SentimentAnalysis {
  sentiment_score: number;
  sentiment_direction: string;
  key_news_themes: string[];
  recent_catalysts: string[];
  market_outlook: string;
}

export interface FinalRecommendation {
  action: RecommendationAction;
  confidence: number;
  rationale: string;
  key_risks: string[];
  key_opportunities: string[];
  time_horizon: string;
}

export type RecommendationAction = 'BUY' | 'SELL' | 'HOLD';

export interface AgentAnalysis {
  query: string;
  ticker: string;
  execution_time: number;
  fundamental_analysis: FundamentalAnalysis;
  momentum_analysis: MomentumAnalysis;
  sentiment_analysis: SentimentAnalysis;
  final_recommendation: FinalRecommendation;
}

export interface AnalysisState {
  response: string;
  agentResponse: AgentAnalysis | null;
  isLoading: boolean;
  error: string | null;
}