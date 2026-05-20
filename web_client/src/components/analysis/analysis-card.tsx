import React from 'react';

// ===========================================
// Componente de Card de Análise
// ===========================================

/**
 * AnalysisCard
 * 
 * Card reutilizável para exibir cada tipo de análise.
 * Demonstra o princípio DRY (Don't Repeat Yourself) -
 * um único componente serve para fundamental, momentum e sentiment.
 */

interface AnalysisCardProps {
  /** Título do card */
  title: string;
  /** Itens a serem exibidos no formato label: value */
  items: Array<{
    label: string;
    value: string | number;
  }>;
  /** Texto adicional opcional (ex: thesis ou outlook) */
  description?: string;
}

export function AnalysisCard({ title, items, description }: AnalysisCardProps) {
  return (
    <div className="bg-card p-4 rounded-lg border border-border">
      <h4 className="font-semibold mb-3 text-foreground">{title}</h4>
      
      <div className="text-sm space-y-2">
        {items.map((item, index) => (
          <p key={index}>
            <strong className="text-foreground">{item.label}:</strong>{' '}
            <span className="text-muted-foreground">{item.value}</span>
          </p>
        ))}
        
        {description && (
          <p className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}