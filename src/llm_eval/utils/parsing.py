"""
Common parsing utilities for LLM responses.

This module provides reusable functions for extracting structured data
from LLM outputs, reducing code duplication across metric implementations.
"""

import re
import json
from typing import Dict, Optional


def extract_json_from_response(text: str) -> Dict:
    """
    Extract JSON from LLM response text.
    
    Handles responses wrapped in markdown code blocks (```json...```) 
    or plain JSON text.
    
    Args:
        text: Raw text response from LLM
        
    Returns:
        Parsed dictionary from JSON. Returns empty dict if parsing fails.
        
    Examples:
        >>> extract_json_from_response('```json\\n{"score": 5}\\n```')
        {'score': 5}
        >>> extract_json_from_response('{"score": 5}')
        {'score': 5}
    """
    # Try to find JSON wrapped in markdown code blocks
    match = re.search(r'```json\s+(.*?)\s+```', text, re.DOTALL)
    json_text = match.group(1) if match else text.strip()
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM response: {text[:100]}...")
        return {}


def extract_score_from_response(text: str, pattern: str = r"SCORE:\s*([\d\.]+)", 
                                 max_score: Optional[float] = None) -> float:
    """
    Extract a numeric score from LLM response text using regex.
    
    Args:
        text: Raw text response from LLM
        pattern: Regex pattern to extract score. Must have one capture group for the number.
        max_score: If provided, normalize the score by dividing by this value.
                   For example, max_score=10 converts a 0-10 scale to 0-1.
        
    Returns:
        Extracted score as float. Returns 0.0 if no match found.
        If max_score is provided, returns normalized score.
        
    Examples:
        >>> extract_score_from_response("SCORE: 8.5")
        8.5
        >>> extract_score_from_response("SCORE: 8.5", max_score=10.0)
        0.85
    """
    score_match = re.search(pattern, text)
    
    if not score_match:
        return 0.0
    
    score = float(score_match.group(1))
    
    if max_score is not None:
        return score / max_score
    
    return score
