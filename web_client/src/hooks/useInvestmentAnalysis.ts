import { useState, useCallback } from 'react';
import { AgentAnalysis, AnalysisState } from '@/types/analysis';
import { API_CONFIG } from '@/constants/recommendations';

// ===========================================
// Hook customizado para análise de investimentos
// ===========================================

export function useInvestmentAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    response: '',
    agentResponse: null,
    isLoading: false,
    error: null,
  });

  const resetState = useCallback(() => {
    setState({
      response: '',
      agentResponse: null,
      isLoading: false,
      error: null,
    });
  }, []);

  const analyzeWithRAG = useCallback(async (query: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null, response: '' }));

    try {
      const response = await fetch(API_CONFIG.RAG_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          limit: API_CONFIG.DEFAULT_LIMIT,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      setState(prev => ({ ...prev, response: data.answer, isLoading: false }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setState(prev => ({ ...prev, error: errorMessage, isLoading: false }));
      console.error('RAG Error:', err);
    }
  }, []);

  const analyzeWithAgent = useCallback(async (query: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null, agentResponse: null }));
    const startTime = Date.now();

    try {
      const response = await fetch(API_CONFIG.AGENT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          limit: API_CONFIG.AGENT_LIMIT,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data: AgentAnalysis = await response.json();
      const executionTime = (Date.now() - startTime) / 1000;

      const dataWithTime: AgentAnalysis = {
        ...data,
        execution_time: executionTime,
      };

      setState(prev => ({ ...prev, agentResponse: dataWithTime, isLoading: false }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setState(prev => ({ ...prev, error: errorMessage, isLoading: false }));
      console.error('Agent Analysis Error:', err);
    }
  }, []);

  const retry = useCallback((query: string, isAgentMode: boolean) => {
    if (isAgentMode) {
      analyzeWithAgent(query);
    } else {
      analyzeWithRAG(query);
    }
  }, [analyzeWithAgent, analyzeWithRAG]);

  return {
    ...state,
    
    analyzeWithRAG,
    analyzeWithAgent,
    resetState,
    retry,
  };
}