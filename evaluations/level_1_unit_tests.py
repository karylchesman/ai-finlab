import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()


COMPANY_TICKER_MAPPINGS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "tesla": "TSLA",
    "meta": "META",
}


TICKER_EXTRACTION_PROMPT = """
You are a stock ticker symbol extractor. Given a user message, extract the stock ticker symbol for any publicly traded company mentioned.

Rules:
- Return ONLY the ticker symbol (e.g., AAPL, TSLA, MSFT)
- If no company is mentioned, return "NONE"
- If multiple companies are mentioned, return the first/main one
- Use standard US stock exchange ticker symbols (NYSE, NASDAQ)

Examples:
User: "How is Disney doing?" → DIS
User: "What about Tesla's performance?" → TSLA
User: "Tell me about IBM" → IBM
User: "What's the weather today?" → NONE

User message: {query}
Response:"""


class TickerResult(BaseModel):
    ticker: str = Field(
        pattern="^[A-Z]{1,5}$",
        description="Stock ticker symbol (1-5 uppercase letters)",
    )


def load_test_cases(filename: str) -> dict:
    with open(f"test_cases/{filename}", "r") as file:
        return json.load(file)


def extract_ticker(query: str) -> str | None:
    query_lower = query.lower()
    for company_name, ticker in COMPANY_TICKER_MAPPINGS.items():
        if company_name in query_lower:
            return ticker

    prompt = TICKER_EXTRACTION_PROMPT.format(query=query)
    response = client.responses.parse(
        model="gpt-4o-mini",
        text_format=TickerResult,
        input=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    ticker = response.output_parsed.ticker
    return ticker if ticker != "NONE" else None


def test_static_mapping_apple():
    test_case = load_test_cases("apple_test.json")
    result = extract_ticker(test_case["query"])
    assert result == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {result}"
    )


def test_llm_fallback():
    test_case = load_test_cases("ibm_test.json")
    result = extract_ticker(test_case["query"])
    assert result == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {result}"
    )


def test_no_company_mentioned():
    test_case = load_test_cases("no_company_test.json")
    result = extract_ticker(test_case["query"])
    assert result == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {result}"
    )


def test_natural_language_query():
    test_case = load_test_cases("natural_language_test.json")
    result = extract_ticker(test_case["query"])
    assert result == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {result}"
    )


if __name__ == "__main__":
    tests = [
        test_static_mapping_apple,
        test_llm_fallback,
        test_no_company_mentioned,
        test_natural_language_query,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed!")
