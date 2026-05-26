import json
import os

import requests
from dotenv import load_dotenv
from langfuse import get_client, observe

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def load_test_case(filename: str) -> dict:
    with open(f"test_cases/{filename}", "r") as file:
        return json.load(file)


@observe()
def call_agent_endpoint(query: str, limit: int = 3):
    response = requests.post(
        f"{API_BASE_URL}/agent", json={"query": query, "limit": limit}
    )
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        metadata={
            "status_code": response.status_code,
            "query_length": len(query),
        }
    )
    return response


@observe()
def test_agent_endpoint_apple():
    test_case = load_test_case("apple_test.json")
    response = call_agent_endpoint(test_case["query"], limit=3)

    assert response.status_code == 200, (
        f"Expected status code 200 but got {response.status_code}"
    )
    data = response.json()
    assert data["ticker"] == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {data['ticker']}"
    )
    assert "fundamental_analysis" in data, "Expected 'fundamental_analysis' in response"
    assert "momentum_analysis" in data, "Expected 'momentum_analysis' in response"
    assert "sentiment_analysis" in data, "Expected 'sentiment_analysis' in response"
    assert "final_recommendation" in data, (
        "Expected 'final_recommendation' in response "
    )
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        name="test_agent_apple",
        metadata={
            "test_type": test_case["test_type"],
            "expected_ticker": test_case["expected_ticker"],
            "actual_ticker": data["ticker"],
            "has_all_analyses": True,
        },
        input={"query": test_case["query"]},
        output=data,
    )
    langfuse_client.update_current_trace(
        tags=["evaluation", "integration_test", "apple"]
    )
    return data


@observe()
def test_agent_endpoint_ibm():
    test_case = load_test_case("ibm_test.json")
    response = call_agent_endpoint(test_case["query"], limit=3)
    assert response.status_code == 200, (
        f"Expected status code 200 but got {response.status_code}; Error message: {response.json()}"
    )
    assert response.json()["ticker"] == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {response.json()['ticker']}"
    )
    data = response.json()
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        name="test_agent_ibm",
        metadata={
            "test_type": test_case["test_type"],
            "expected_ticker": test_case["expected_ticker"],
            "actual_ticker": data["ticker"],
            "has_all_analyses": True,
        },
        input={"query": test_case["query"]},
        output=data,
    )
    langfuse_client.update_current_trace(
        tags=["evaluation", "integration_test", "ibm", "llm_fallback"]
    )
    return data


@observe()
def test_agent_endpoint_no_company():
    test_case = load_test_case("no_company_test.json")
    response = call_agent_endpoint(test_case["query"], limit=3)
    assert response.status_code == 400, (
        f"Expected status code 400 but got {response.status_code}"
    )
    data = response.json()
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        name="test_agent_no_company",
        metadata={
            "test_type": test_case["test_type"],
            "expected_status": 400,
            "actual_status": response.status_code,
        },
        input={"query": test_case["query"]},
        output=data,
    )
    langfuse_client.update_current_trace(
        tags=["evaluation", "integration_test", "error_handling"]
    )
    return data


@observe()
def test_agent_endpoint_natural_language():
    test_case = load_test_case("natural_language_test.json")
    response = requests.post(
        f"{API_BASE_URL}/agent", json={"query": test_case["query"], "limit": 3}
    )
    assert response.status_code == 200, (
        f"Expected status code 200 but got {response.status_code}"
    )
    data = response.json()
    assert data["ticker"] == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {data['ticker']}"
    )
    assert data["final_recommendation"]["action"] in ["BUY", "SELL", "HOLD"], (
        f"Expected final recommendation to be one of BUY, SELL, HOLD but got {data['final_recommendation']['action']}"
    )
    data = response.json()
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        name="test_agent_natural_language",
        metadata={
            "test_type": test_case["test_type"],
            "expected_ticker": test_case["expected_ticker"],
            "actual_ticker": data["ticker"],
            "recommendation_action": data["final_recommendation"]["action"],
        },
        input={"query": test_case["query"]},
        output=data,
    )
    langfuse_client.update_current_trace(
        tags=["evaluation", "integration_test", "natural_language"]
    )
    return data


def run_evaluation_pipeline():
    tests = [
        ("Apple (static mapping)", test_agent_endpoint_apple),
        ("IBM (LLM fallback)", test_agent_endpoint_ibm),
        ("No company (error handling)", test_agent_endpoint_no_company),
        ("Natural language", test_agent_endpoint_natural_language),
    ]
    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"{test_name}: PASSED")
            passed += 1
        except AssertionError as e:
            print(f"{test_name}: FAILED - {str(e)}")
    print(f"Passed {passed} out of {len(tests)} tests.")


if __name__ == "__main__":
    run_evaluation_pipeline()
