'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, TrendingUp } from 'lucide-react';

import { useInvestmentAnalysis } from '@/hooks/useInvestmentAnalysis';
import { UI_MESSAGES } from '@/constants/recommendations';
import { LoadingSkeleton, ErrorDisplay } from '@/components/ui';
import { AgentResponse } from '@/components/analysis';

export function StreamingChat() {
  const [query, setQuery] = useState('');
  const [isAgentMode, setIsAgentMode] = useState(false);

  const {
    response,
    agentResponse,
    isLoading,
    error,
    analyzeWithRAG,
    analyzeWithAgent,
    retry,
  } = useInvestmentAnalysis();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    if (isAgentMode) {
      await analyzeWithAgent(query);
    } else {
      await analyzeWithRAG(query);
    }
  };

  const handleRetry = () => {
    retry(query, isAgentMode);
  };

  return (
    <div className="w-full">
      <div className="bg-card rounded-lg shadow-sm border border-border">
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          
          {/* Toggle de Modo */}
          <ModeToggle
            isAgentMode={isAgentMode}
            onChange={setIsAgentMode}
            disabled={isLoading}
          />

          {/* Input e Botão de Submit */}
          <SearchInput
            query={query}
            onChange={setQuery}
            isLoading={isLoading}
            isAgentMode={isAgentMode}
          />

          {/* Área de Erro */}
          {error && (
            <ErrorDisplay message={error} onRetry={handleRetry} />
          )}

          {/* Área de Resposta */}
          <ResponseArea
            isLoading={isLoading}
            isAgentMode={isAgentMode}
            response={response}
            agentResponse={agentResponse}
          />
        </form>
      </div>
    </div>
  );
}

/**
 * ModeToggle - Alterna entre modo RAG e Agent
 */
interface ModeToggleProps {
  isAgentMode: boolean;
  onChange: (value: boolean) => void;
  disabled: boolean;
}

function ModeToggle({ isAgentMode, onChange, disabled }: ModeToggleProps) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={isAgentMode}
          onChange={(e) => onChange(e.target.checked)}
          className="w-4 h-4 text-primary bg-background border-border rounded focus:ring-primary"
          disabled={disabled}
        />
        <span className="text-sm font-medium flex items-center gap-1">
          <TrendingUp className="h-4 w-4" />
          Multi-Stream Analysis (agent)
        </span>
      </label>
      {isAgentMode && (
        <span className="text-xs text-muted-foreground">
          {UI_MESSAGES.AGENT_DESCRIPTION}
        </span>
      )}
    </div>
  );
}

/**
 * SearchInput - Campo de busca com botão de submit
 */
interface SearchInputProps {
  query: string;
  onChange: (value: string) => void;
  isLoading: boolean;
  isAgentMode: boolean;
}

function SearchInput({ query, onChange, isLoading, isAgentMode }: SearchInputProps) {
  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={query}
        onChange={(e) => onChange(e.target.value)}
        placeholder={isAgentMode ? UI_MESSAGES.PLACEHOLDER_AGENT : UI_MESSAGES.PLACEHOLDER_RAG}
        className="flex-1 px-4 py-3 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-foreground placeholder:text-muted-foreground"
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        aria-label="Send message"
      >
        {isLoading ? (
          UI_MESSAGES.LOADING
        ) : (
          <>
            <span>{isAgentMode ? 'Analyze' : 'Search'}</span>
            <Send className="h-4 w-4" />
          </>
        )}
      </button>
    </div>
  );
}

/**
 * ResponseArea - Área que exibe a resposta ou loading
 */
interface ResponseAreaProps {
  isLoading: boolean;
  isAgentMode: boolean;
  response: string;
  agentResponse: ReturnType<typeof useInvestmentAnalysis>['agentResponse'];
}

function ResponseArea({ isLoading, isAgentMode, response, agentResponse }: ResponseAreaProps) {
  return (
    <div className="p-4 rounded-lg bg-accent min-h-48">
      {/* Estado de Loading */}
      {isLoading && (
        <LoadingSkeleton variant={isAgentMode ? 'full' : 'simple'} />
      )}

      {/* Resposta do Agent */}
      {!isLoading && isAgentMode && agentResponse && (
        <AgentResponse analysis={agentResponse} />
      )}

      {/* Resposta do RAG */}
      {!isLoading && !isAgentMode && (
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown>
            {response || UI_MESSAGES.EMPTY_RESPONSE}
          </ReactMarkdown>
        </div>
      )}

      {/* Estado vazio no modo Agent */}
      {!isLoading && isAgentMode && !agentResponse && (
        <p className="text-muted-foreground">{UI_MESSAGES.EMPTY_RESPONSE}</p>
      )}
    </div>
  );
}