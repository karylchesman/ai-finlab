'use client';
import { StreamingChat } from "@/components/streaming-chat";
import { LineChart } from "lucide-react";

export default function Home() {

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
        <div className="container px-4 md:px-6 flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <LineChart className="h-6 w-6 text-primary" />
            <span className="font-semibold text-foreground text-xl">Finlab</span>
          </div>
        </div>
      </header>

      <main className="flex-1 py-8">

          <div className="mx-auto max-w-4xl">
            <h1 className="text-3xl font-bold mb-6 text-foreground">Investment Intelligence at Your Fingertips</h1>
            <p className="text-muted-foreground mb-8">
              Ask anything about companies, stocks, or market trends. Enable Multi-Stream Analysis for comprehensive investment insights powered by fundamental, momentum, and sentiment analysis.
            </p>
            <StreamingChat />
          </div>

      </main>

      <footer className="border-t border-border py-6 bg-muted">
        <div className="flex items-center justify-start">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} Dev + Eficiente (Engenharia de IA)
          </p>
        </div>
      </footer>
    </div>
  );
}