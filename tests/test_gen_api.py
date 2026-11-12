"""the committed API pages under ``docs/api`` equal the output of ``docs/gen_api.py``"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# %% main
def test_api_pages_are_current() -> None:
    # a fresh interpreter isolates the generator's imports from the test session
    result = subprocess.run(
        [sys.executable, str(ROOT / 'docs' / 'gen_api.py'), '--check'],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
