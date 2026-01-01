import os
import pathlib
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot_router import build_router


@pytest.fixture()
def router():
    base = pathlib.Path(__file__).parent / "data"
    intents_path = str(base / "intents.json")
    context_path = str(base / "context.json")
    # config_dir: default ist bot_router/config im Paket
    return build_router(intents_path=intents_path, context_path=context_path, llm_threshold=0.99)
