import React from 'react';

// ===========================================
// Componente de Riscos e Oportunidades
// ===========================================

/**
 * RiskOpportunityPanel
 * 
 * Exibe lado a lado os riscos e oportunidades identificados.
 * O uso de cores (vermelho/verde) segue convenções universais
 * de UX financeira.
 */

interface RiskOpportunityPanelProps {
  /** Lista de riscos identificados */
  risks: string[];
  /** Lista de oportunidades identificadas */
  opportunities: string[];
}

export function RiskOpportunityPanel({ risks, opportunities }: RiskOpportunityPanelProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Painel de Riscos */}
      <div className="bg-red-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-3 text-red-700">Key Risks</h4>
        <ul className="space-y-2">
          {risks.map((risk, index) => (
            <li 
              key={index} 
              className="text-sm text-red-600 flex items-start gap-2"
            >
              <span className="text-red-400 mt-1">•</span>
              <span>{risk}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Painel de Oportunidades */}
      <div className="bg-green-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-3 text-green-700">Key Opportunities</h4>
        <ul className="space-y-2">
          {opportunities.map((opportunity, index) => (
            <li 
              key={index} 
              className="text-sm text-green-600 flex items-start gap-2"
            >
              <span className="text-green-400 mt-1">•</span>
              <span>{opportunity}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}