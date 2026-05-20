import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorDisplayProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorDisplay({ message, onRetry }: ErrorDisplayProps) {
  return (
    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
      <div className="flex items-start gap-3">
        {/* Ícone de erro */}
        <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
        
        <div className="flex-1">
          <p className="text-sm font-medium text-red-800">
            Error processing request
          </p>
          <p className="text-sm text-red-600 mt-1">
            {message}
          </p>
          
          {/* Botão de retry (só aparece se onRetry foi fornecido) */}
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-2 text-sm text-red-700 hover:text-red-800 underline underline-offset-2"
            >
              <RefreshCw className="h-4 w-4" />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}