"""
Carnatic Music ML - Raga Detection and Generation

A practical approach using:
- Pitch detection + rule matching for raga identification
- Grammar-based generation for authentic raga melodies
"""

from .raga_db import RagaDB
from .detector import RagaDetector
from .generator import RagaGenerator

__all__ = ['RagaDB', 'RagaDetector', 'RagaGenerator']
