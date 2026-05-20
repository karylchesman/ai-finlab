import { RecommendationAction } from '@/types/analysis';

export const RECOMMENDATION_STYLES: Record<
  RecommendationAction,
  { bg: string; border: string; text: string }
> = {
  BUY: {
    bg: 'bg-green-50',
    border: 'border-green-500',
    text: 'text-green-700',
  },
  SELL: {
    bg: 'bg-red-50',
    border: 'border-red-500',
    text: 'text-red-700',
  },
  HOLD: {
    bg: 'bg-yellow-50',
    border: 'border-yellow-500',
    text: 'text-yellow-700',
  },
} as const;

/**
 * Configurações padrão para as requisições da API
 */
export const API_CONFIG = {
  RAG_ENDPOINT: '/api/rag',
  AGENT_ENDPOINT: '/api/agent',
  DEFAULT_LIMIT: 5,
  AGENT_LIMIT: 3,
} as const;

/**
 * Mensagens padrão da interface
 */
export const UI_MESSAGES = {
  PLACEHOLDER_RAG: 'Ask about any company, stock, or market trend...',
  PLACEHOLDER_AGENT: 'Enter company name or ticker symbol (e.g., Apple, AAPL)',
  EMPTY_RESPONSE: 'Your response will appear here...',
  LOADING: 'Processing...',
  AGENT_DESCRIPTION: '(Comprehensive 3-stream analysis: Fundamental, Momentum, and Market Sentiment)',
} as const;