"""Tests for mock backend - no LLM required"""

import pytest
import os
from app.audit import LLMAuditor


def test_mock_backend_initialization():
    """Test that mock backend initializes without dependencies"""
    auditor = LLMAuditor(backend="mock")
    assert auditor.backend == "mock"
    assert auditor.model == "mock-model"
    assert auditor.client is None


def test_mock_backend_from_env():
    """Test mock backend initialization from environment variable"""
    os.environ["LLM_BACKEND"] = "mock"
    auditor = LLMAuditor(backend=os.getenv("LLM_BACKEND"))
    assert auditor.backend == "mock"
    backend_info = auditor.get_backend_info()
    assert backend_info["backend"] == "mock"


@pytest.mark.asyncio
async def test_mock_audit_hiring():
    """Test complete audit with mock backend - hiring scenario"""
    auditor = LLMAuditor(backend="mock")
    report = await auditor.run_audit(
        scenario="hiring",
        temperature=0.7,
        max_tokens=300
    )
    
    assert report["backend"] == "mock"
    assert report["model"] == "mock-model"
    assert report["scenario"] == "hiring"
    assert len(report["responses"]) == 24  # 8 groups x 3 names
    assert "metrics" in report
    assert "summary" in report
    
    # Verify all responses are non-empty
    for response in report["responses"]:
        assert len(response["text"]) > 0
        assert "ERROR" not in response["text"]


@pytest.mark.asyncio
async def test_mock_audit_recommendation():
    """Test complete audit with mock backend - recommendation scenario"""
    auditor = LLMAuditor(backend="mock")
    report = await auditor.run_audit(
        scenario="recommendation",
        temperature=0.7,
        max_tokens=300
    )
    
    assert report["backend"] == "mock"
    assert report["scenario"] == "recommendation"
    assert len(report["responses"]) == 24
    
    # Verify names are extracted correctly
    sample_response = report["responses"][0]
    assert sample_response["name"] in sample_response["text"]


@pytest.mark.asyncio
async def test_mock_audit_credit():
    """Test complete audit with mock backend - credit scenario"""
    auditor = LLMAuditor(backend="mock")
    report = await auditor.run_audit(
        scenario="credit",
        temperature=0.7,
        max_tokens=300
    )
    
    assert report["backend"] == "mock"
    assert report["scenario"] == "credit"
    assert len(report["responses"]) == 24


@pytest.mark.asyncio
async def test_mock_deterministic():
    """Test that mock backend provides deterministic responses"""
    auditor = LLMAuditor(backend="mock")
    
    report1 = await auditor.run_audit(scenario="hiring", temperature=0.7, max_tokens=300)
    report2 = await auditor.run_audit(scenario="hiring", temperature=0.7, max_tokens=300)
    
    # Responses should be identical (excluding audit_id and timestamp)
    assert len(report1["responses"]) == len(report2["responses"])
    for r1, r2 in zip(report1["responses"], report2["responses"]):
        assert r1["text"] == r2["text"]
        assert r1["group"] == r2["group"]
        assert r1["name"] == r2["name"]
