import React from 'react';
import { FinalRecommendation } from '@/types/analysis';
import { RECOMMENDATION_STYLES } from '@/constants/recommendations';
import { cn } from '@/lib/utils';

// ===========================================
// Componente de Banner de Recomendação
// ===========================================

/**
 * RecommendationBanner
 * 
 * Destaca visualmente a recomendação final (BUY/SELL/HOLD).
 * Usa as constantes de estilo para manter consistência e
 * facilitar manutenção.
 */

interface RecommendationBannerProps {
  /** Dados da recomendação final */
  recommendation: FinalRecommendation;
}

export function RecommendationBanner({ recommendation }: RecommendationBannerProps) {
  // Busca os estilos baseado na ação (BUY, SELL, HOLD)
  // Usa HOLD como fallback se a ação não for reconhecida
  const styles = RECOMMENDATION_STYLES[recommendation.action] || RECOMMENDATION_STYLES.HOLD;

  return (
    <div
      className={cn(
        'p-4 rounded-lg border-2',
        styles.bg,
        styles.border
      )}
    >
      {/* Header com ação e confiança */}
      <div className="flex justify-between items-center mb-3">
        <h4 className={cn('text-lg font-semibold', styles.text)}>
          Final Recommendation: {recommendation.action}
        </h4>
        <span className="text-sm font-medium text-muted-foreground">
          Confidence: {(recommendation.confidence * 100).toFixed(0)}%
        </span>
      </div>

      {/* Rationale */}
      <p className="text-foreground mb-2">{recommendation.rationale}</p>

      {/* Time Horizon */}
      <p className="text-sm text-muted-foreground">
        <strong>Time Horizon:</strong> {recommendation.time_horizon}
      </p>
    </div>
  );
}