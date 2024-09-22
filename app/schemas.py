from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List, Dict, Set, Union
from enum import Enum

class SentimentTags(Enum):
    excellent = "Excellent"
    good = "Good"
    average = "Average"
    bad = "Bad"
    poor = "Poor"

class Input(BaseModel):
    message: str

class LLMOutput(BaseModel):
    sentiment: SentimentTags
    confidence: float
    reason: str

class Output(BaseModel):
    message: str
    analysis: LLMOutput