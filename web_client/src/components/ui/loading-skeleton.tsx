import React from 'react';

interface LoadingSkeletonProps {
  /** Variante do skeleton: simples para RAG, completo para Agent */
  variant?: 'simple' | 'full';
}

export function LoadingSkeleton({ variant = 'simple' }: LoadingSkeletonProps) {
  if (variant === 'simple') {
    return (
      <div className="animate-pulse space-y-3">
        <div className="h-4 bg-muted rounded w-3/4"></div>
        <div className="h-4 bg-muted rounded w-full"></div>
        <div className="h-4 bg-muted rounded w-5/6"></div>
        <div className="h-4 bg-muted rounded w-2/3"></div>
      </div>
    );
  }

  // Skeleton completo para o modo Agent
  return (
    <div className="animate-pulse space-y-6">
      {/* Header skeleton */}
      <div className="flex justify-between items-center border-b pb-2">
        <div className="h-6 bg-muted rounded w-48"></div>
        <div className="h-4 bg-muted rounded w-32"></div>
      </div>

      {/* Recommendation banner skeleton */}
      <div className="p-4 rounded-lg border-2 border-muted bg-muted/30">
        <div className="flex justify-between items-center mb-2">
          <div className="h-5 bg-muted rounded w-56"></div>
          <div className="h-4 bg-muted rounded w-24"></div>
        </div>
        <div className="h-4 bg-muted rounded w-full mb-2"></div>
        <div className="h-4 bg-muted rounded w-3/4"></div>
      </div>

      {/* Three cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-card p-4 rounded-lg border">
            <div className="h-5 bg-muted rounded w-32 mb-3"></div>
            <div className="space-y-2">
              <div className="h-4 bg-muted rounded w-full"></div>
              <div className="h-4 bg-muted rounded w-2/3"></div>
              <div className="h-4 bg-muted rounded w-3/4"></div>
            </div>
          </div>
        ))}
      </div>

      {/* Risk/Opportunity skeleton */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-red-50/50 p-4 rounded-lg">
          <div className="h-5 bg-muted rounded w-24 mb-3"></div>
          <div className="space-y-2">
            <div className="h-3 bg-muted rounded w-full"></div>
            <div className="h-3 bg-muted rounded w-5/6"></div>
          </div>
        </div>
        <div className="bg-green-50/50 p-4 rounded-lg">
          <div className="h-5 bg-muted rounded w-32 mb-3"></div>
          <div className="space-y-2">
            <div className="h-3 bg-muted rounded w-full"></div>
            <div className="h-3 bg-muted rounded w-4/5"></div>
          </div>
        </div>
      </div>
    </div>
  );
}