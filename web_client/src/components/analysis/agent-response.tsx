import React from 'react';
import { AgentAnalysis } from '@/types/analysis';
import { AnalysisCard } from './analysis-card';
import { RiskOpportunityPanel } from './risk-opportunity-panel';
import { RecommendationBanner } from './recommendation-banner';

// ===========================================
// Componente de Resposta do Agente
// ===========================================

/**
 * AgentResponse
 * 
 * Componente que orquestra a exibição completa da análise do agente.
 * Demonstra composição de componentes - cada parte da UI é um
 * componente separado que pode ser testado e reutilizado.
 */

interface AgentResponseProps {
  /** Dados completos da análise */
  analysis: AgentAnalysis;
}

export function AgentResponse({ analysis }: AgentResponseProps) {
  return (
    <div className="space-y-6">
      {/* Header com ticker e tempo de execução */}
      <div className="flex justify-between items-center border-b border-border pb-3">
        <h3 className="text-xl font-bold text-foreground">
          {analysis.ticker} - Complete Analysis
        </h3>
        <span className="text-sm text-muted-foreground">
          Execution time: {analysis.execution_time.toFixed(2)}s
        </span>
      </div>

      {/* Recomendação Final - Destaque principal */}
      <RecommendationBanner recommendation={analysis.final_recommendation} />

      {/* Grid das três análises */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Análise Fundamentalista */}
        <AnalysisCard
          title="Fundamental Analysis"
          items={[
            { label: 'Grade', value: analysis.fundamental_analysis.investment_grade },
            { label: 'Recommendation', value: analysis.fundamental_analysis.recommendation },
          ]}
          description={analysis.fundamental_analysis.overall_investment_thesis}
        />

        {/* Análise de Momentum */}
        <AnalysisCard
          title="Momentum Analysis"
          items={[
            { label: 'Momentum', value: analysis.momentum_analysis.overall_momentum },
            { label: 'Score', value: `${analysis.momentum_analysis.momentum_score}/10` },
            { label: 'Outlook', value: analysis.momentum_analysis.short_term_outlook },
          ]}
        />

        {/* Análise de Sentimento */}
        <AnalysisCard
          title="Market Sentiment"
          items={[
            { label: 'Sentiment', value: analysis.sentiment_analysis.sentiment_direction },
            { label: 'Score', value: `${analysis.sentiment_analysis.sentiment_score}/10` },
          ]}
          description={analysis.sentiment_analysis.market_outlook}
        />
      </div>

      {/* Painel de Riscos e Oportunidades */}
      <RiskOpportunityPanel
        risks={analysis.final_recommendation.key_risks}
        opportunities={analysis.final_recommendation.key_opportunities}
      />
    </div>
  );
}