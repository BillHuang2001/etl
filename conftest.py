"""Root conftest: ensures the repo root is on sys.path so `import etl` works."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
