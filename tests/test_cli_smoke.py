import os
import subprocess
import sys


def test_cli_tests_1_smoke():
    env = os.environ.copy()
    env["TESTING_FLAG"] = "true"
    env["JOB_FETCHER_RUN_NUMBER"] = "3"

    # Run: python -m asset_processing_service.main --tests 1
    cmd = [sys.executable, "-m", "asset_processing_service.main", "--tests", "1"]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
