"""RAG project package."""

import signal
import sys

# Set here so Ctrl+C is caught everywhere, even during the slow imports
# (torch, transformers) done by src/__main__.py.
signal.signal(signal.SIGINT, lambda *_: sys.exit("\nInterrupted"))
