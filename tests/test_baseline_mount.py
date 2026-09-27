from pathlib import Path

import pytest

BASELINE = Path(__file__).resolve().parents[1] / "baseline"


@pytest.mark.integration
def test_baseline_is_mounted_and_readonly():
    if not BASELINE.exists():
        pytest.skip("baseline/ 未挂载，先运行 scripts/mount_baseline.ps1")
    assert (BASELINE / "python" / "orchestrator" / "graph.py").exists()
    assert (BASELINE / "README.md").exists()
