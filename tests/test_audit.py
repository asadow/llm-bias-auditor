"""Tests for audit functionality"""

import pytest
from app.prompts import generate_prompts, get_available_scenarios
from app.metrics import (
    calculate_length_disparity,
    calculate_refusal_disparity,
    calculate_sentiment_disparity
)


def test_generate_prompts():
    """Test prompt generation"""
    prompts = generate_prompts("hiring")
    assert len(prompts) > 0
    assert "group" in prompts[0]
    assert "prompt" in prompts[0]
    assert "system" in prompts[0]


def test_get_available_scenarios():
    """Test scenario listing"""
    scenarios = get_available_scenarios()
    assert "hiring" in scenarios
    assert "recommendation" in scenarios
    assert "credit" in scenarios


def test_length_disparity():
    """Test length disparity calculation"""
    responses = [
        {"group": "group_a", "text": "This is a short response."},
        {"group": "group_a", "text": "This is another short response."},
        {"group": "group_b", "text": "This is a much longer response with many more words to analyze."},
        {"group": "group_b", "text": "This is also a longer response with additional content."}
    ]
    
    result = calculate_length_disparity(responses)
    assert "disparity_score" in result
    assert "group_means" in result
    assert result["disparity_score"] > 0


def test_refusal_disparity():
    """Test refusal disparity calculation"""
    responses = [
        {"group": "group_a", "text": "I can help with that."},
        {"group": "group_a", "text": "Sure, here's my answer."},
        {"group": "group_b", "text": "I cannot provide that information."},
        {"group": "group_b", "text": "I'm unable to assist with this."}
    ]
    
    result = calculate_refusal_disparity(responses)
    assert "disparity_score" in result
    assert "group_rates" in result
    assert result["disparity_score"] > 0


def test_sentiment_disparity():
    """Test sentiment disparity calculation"""
    responses = [
        {"group": "group_a", "text": "This candidate is excellent and highly qualified."},
        {"group": "group_a", "text": "Outstanding performance and strong skills."},
        {"group": "group_b", "text": "This candidate has some concerning gaps."},
        {"group": "group_b", "text": "Weak background with limited experience."}
    ]
    
    result = calculate_sentiment_disparity(responses)
    assert "disparity_score" in result
    assert "group_means" in result
    assert result["disparity_score"] > 0
