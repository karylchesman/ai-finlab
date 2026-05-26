import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def load_test_case(filename: str) -> dict:
    with open(f"test_cases/{filename}", "r") as file:
        return json.load(file)


def test_agent_endpoint_apple():
    test_case = load_test_case("apple_test.json")
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
    assert "fundamental_analysis" in data, "Expected 'fundamental_analysis' in response"
    assert "momentum_analysis" in data, "Expected 'momentum_analysis' in response"
    assert "sentiment_analysis" in data, "Expected 'sentiment_analysis' in response"
    assert "final_recommendation" in data, (
        "Expected 'final_recommendation' in response "
    )


def test_agent_endpoint_ibm():
    test_case = load_test_case("ibm_test.json")
    response = requests.post(
        f"{API_BASE_URL}/agent", json={"query": test_case["query"], "limit": 3}
    )
    assert response.status_code == 200, (
        f"Expected status code 200 but got {response.status_code}; Error message: {response.json()}"
    )
    assert response.json()["ticker"] == test_case["expected_ticker"], (
        f"Expected {test_case['expected_ticker']} but got {response.json()['ticker']}"
    )


def test_agent_endpoint_no_company():
    test_case = load_test_case("no_company_test.json")
    response = requests.post(
        f"{API_BASE_URL}/agent", json={"query": test_case["query"], "limit": 3}
    )
    assert response.status_code == 400, (
        f"Expected status code 400 but got {response.status_code}"
    )


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
