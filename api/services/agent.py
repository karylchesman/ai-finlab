import asyncio
from typing import Any

from config.prompts import (
    AGGREGATION_PROMPT,
    FUNDAMENTAL_PROMPT,
    FUNDAMENTAL_QUERIES,
    MOMENTUM_PROMPT,
    MOMENTUM_QUERIES,
    SENTIMENT_PROMPT,
    SENTIMENT_QUERY_TEMPLATE,
)
from config.settings import settings
from models.agent import AgentRequest, AgentResponse

# from groq import AsyncGroq
from openai import AsyncClient as AsyncGroq

from services.search import SearchService


class AgentService:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
        )

    def _run_queries(self, queries: list[str], limit: int):
        all_results = []
        for query in queries:
            search_results = self.search_service.search(query=query, limit=limit)
            all_results.extend([result.text for result in search_results.results])
        return "\n\n".join(all_results)

    async def _generate_completion(self, prompt: str):
        response = await self.client.chat.completions.create(
            model=settings.groq_model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def _analyse_fundamentals(self, limit: int):
        context = self._run_queries(FUNDAMENTAL_QUERIES, limit)
        prompt = FUNDAMENTAL_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyse_momentum(self, limit: int):
        context = self._run_queries(MOMENTUM_QUERIES, limit)
        prompt = MOMENTUM_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def _analyse_sentiment(self, ticker: str, limit: int):
        query = SENTIMENT_QUERY_TEMPLATE.format(ticker=ticker)
        results = self.search_service.search(query=query, limit=limit)
        context = "\n\n".join([result.text for result in results.results])
        prompt = SENTIMENT_PROMPT.format(context=context)
        return await self._generate_completion(prompt)

    async def analyse(self, ticker: str, limit: int):
        fundamental_task = self._analyse_fundamentals(limit)
        momentum_task = self._analyse_momentum(limit)
        sentiment_task = self._analyse_sentiment(ticker, limit)

        (
            fundamental_analysis,
            momentum_analysis,
            sentiment_analysis,
        ) = await asyncio.gather(fundamental_task, momentum_task, sentiment_task)

        aggregation_context = AGGREGATION_PROMPT.format(
            fundamental=fundamental_analysis,
            momentum=momentum_analysis,
            sentiment=sentiment_analysis,
        )

        final_recommendation = await self._generate_completion(aggregation_context)

        return AgentResponse(
            ticker=ticker,
            fundamental_analysis=fundamental_analysis,  # type: ignore
            momentum_analysis=momentum_analysis,  # type: ignore
            sentiment_analysis=sentiment_analysis,  # type: ignore
            final_recommendation=final_recommendation,  # type: ignore
        )
