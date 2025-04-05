# src/__init__.py
"""Secure V2V Communication with Federated Learning & Blockchain"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"

# Define public API
__all__ = [
    'server',
    'client', 
    'blockchain',
    'utils',
    'analysis'
]

# Initialize package-wide logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)