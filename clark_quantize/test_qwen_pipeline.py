from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "scripts" / "spot" / "run_qwen38_pipeline.sh"


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


class PipelineTerminalReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.state = self.root / "state"
        self.runs = self.root / "runs"
        self.model = self.root / "model"
        self.bin = self.root / "bin"
        self.calls = self.root / "calls.log"
        for path in (self.state, self.runs, self.model, self.bin):
            path.mkdir(parents=True)
        shutil.copytree(REPO / "scripts" / "spot", self.repo / "scripts" / "spot")
        experiment = self.repo / "experiments" / "qwen38_27b_spot"
        experiment.mkdir(parents=True)
        write_executable(self.bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
        write_executable(self.bin / "systemctl", "#!/usr/bin/env bash\nexit 1\n")
        write_executable(
            self.bin / "runuser",
            "#!/usr/bin/env bash\nshift 3\nexec \"$@\"\n",
        )
        write_executable(
            self.repo / "scripts" / "spot" / "run_spot_job.sh",
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "name=$1\nshift 2\n"
            "printf '%s\\n' \"$name\" >>\"$TEST_CALLS\"\n"
            "exec \"$@\"\n",
        )
        write_executable(
            self.repo / "scripts" / "spot" / "prune_qat_checkpoints.py",
            "#!/usr/bin/env python3\nraise SystemExit(0)\n",
        )
        write_executable(
            self.repo / "scripts" / "spot" / "update_clark_scientist.sh",
            "#!/usr/bin/env bash\nexit 73\n",
        )
        for name in ("qwen38-base3-full.json", "qwen38-retained-export.json"):
            (self.state / name).write_text(json.dumps({"promotion": {"passed": True}}))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_pipeline(self) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.bin}:{environment['PATH']}",
            "QUANT_REPO": str(self.repo),
            "QUANT_PYTHON": sys.executable,
            "QUANT_STATE_ROOT": str(self.state),
            "QUANT_RUN_ROOT": str(self.runs),
            "QUANT_MODEL_DIR": str(self.model),
            "QUANT_REQUIRE_NATIVE_BASE3": "0",
            "TEST_CALLS": str(self.calls),
        })
        return subprocess.run(
            ["bash", str(PIPELINE)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_existing_nonpromoting_quality_receipt_is_not_rerun(self) -> None:
        (self.state / "qwen38-base3-quality.json").write_text(json.dumps({
            "schema": "clark_qwen38_base3_behavioral_quality_v1",
            "promotion": {"passed": False},
        }))
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertIn("terminal receipt already exists", result.stdout)
        self.assertEqual(self.calls.read_text().splitlines(), ["clark-worker-refresh"])

    def test_new_nonpromoting_quality_receipt_is_terminal_and_pipeline_advances(self) -> None:
        write_executable(
            self.repo / "experiments" / "qwen38_27b_spot" / "behavioral_quality_gate.py",
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "from pathlib import Path\n"
            "state = Path(os.environ['QUANT_STATE_ROOT'])\n"
            "(state / 'qwen38-base3-quality.json').write_text(json.dumps({\n"
            "  'schema': 'clark_qwen38_base3_behavioral_quality_v1',\n"
            "  'promotion': {'passed': False},\n"
            "}))\n"
            "raise SystemExit(3)\n",
        )
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertIn("accepted terminal scientific receipt after exit 3", result.stdout)
        self.assertEqual(
            self.calls.read_text().splitlines(),
            ["qwen38-base3-quality", "clark-worker-refresh"],
        )


if __name__ == "__main__":
    unittest.main()
