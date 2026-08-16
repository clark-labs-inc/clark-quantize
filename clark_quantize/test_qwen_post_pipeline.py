import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
POST_PIPELINE = REPO / "scripts" / "spot" / "run_qwen38_post_pipeline.sh"


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


class PostPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.state = self.root / "state"
        self.runs = self.root / "runs"
        self.bin = self.root / "bin"
        self.calls = self.root / "calls.log"
        for path in (self.state, self.runs, self.bin):
            path.mkdir(parents=True)

        shutil.copytree(REPO / "scripts" / "spot", self.repo / "scripts" / "spot")
        experiment = self.repo / "experiments" / "qwen38_27b_spot"
        experiment.mkdir(parents=True)

        write_executable(
            self.bin / "systemctl",
            "#!/usr/bin/env bash\nexit 1\n",
        )
        write_executable(
            self.bin / "runuser",
            "#!/usr/bin/env bash\n"
            "[[ \"$1\" == --user && \"$3\" == -- ]]\n"
            "shift 3\n"
            "exec \"$@\"\n",
        )
        write_executable(
            self.repo / "scripts" / "spot" / "run_spot_job.sh",
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "name=$1\n"
            "shift 2\n"
            "printf '%s\\n' \"$name\" >>\"$TEST_CALLS\"\n"
            "exec \"$@\"\n",
        )
        write_executable(
            self.repo / "scripts" / "spot" / "prune_qat_checkpoints.py",
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "with Path(os.environ['TEST_CALLS']).open('a') as handle: handle.write('prune\\n')\n",
        )
        write_executable(
            experiment / "audit_completion.py",
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['TEST_CALLS']).open('a') as handle: handle.write('audit\\n')\n"
            "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
            "out.write_text(json.dumps({'schema': 'clark_qwen38_completion_audit_v1', 'complete': True}))\n",
        )
        write_executable(
            experiment / "one_layer_qat_100.py",
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "from pathlib import Path\n"
            "state = Path(os.environ['QUANT_STATE_ROOT'])\n"
            "(state / 'qwen38-one-layer-qat-100.json').write_text(json.dumps({\n"
            "  'schema': 'clark_qwen38_one_layer_qat_100_v1',\n"
            "  'promotion': {'passed': False},\n"
            "}))\n"
            "raise SystemExit(3)\n",
        )
        write_executable(
            experiment / "target_158_reconstruction_gate.py",
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "from pathlib import Path\n"
            "if os.environ.get('TEST_TARGET_WITHOUT_RECEIPT') == '1': raise SystemExit(0)\n"
            "state = Path(os.environ['QUANT_STATE_ROOT'])\n"
            "(state / 'qwen38-target-158-reconstruction.json').write_text(json.dumps({\n"
            "  'schema': 'clark_qwen38_target_158_reconstruction_v1',\n"
            "  'promotion': {'passed': True},\n"
            "}))\n",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def first_qat(self, promoted: bool) -> None:
        (self.state / "qwen38-one-layer-qat.json").write_text(json.dumps({
            "schema": "clark_qwen38_one_layer_qat_v1",
            "promotion": {"passed": promoted},
        }))

    def upstream_receipts(self, *, terminal_promoted: bool = True) -> None:
        schemas = {
            "qwen38-base3-full.json": "test",
            "qwen38-retained-export.json": "test",
            "qwen38-base3-quality.json": "clark_qwen38_base3_behavioral_quality_v1",
            "qwen38-radix5-performance.json": "clark_qwen38_radix5_performance_v1",
            "qwen38-layer-sensitivity.json": "test",
            "qwen38-qat-teacher-cache.json": "test",
        }
        for name, schema in schemas.items():
            (self.state / name).write_text(json.dumps({
                "schema": schema,
                "promotion": {
                    "passed": terminal_promoted
                    if name in {"qwen38-base3-quality.json", "qwen38-radix5-performance.json"}
                    else True
                },
            }))

    def run_post(self, **extra_environment: str) -> subprocess.CompletedProcess[str]:
        if extra_environment.pop("TEST_SKIP_UPSTREAM_RECEIPTS", "0") != "1":
            self.upstream_receipts()
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.bin}:{environment['PATH']}",
            "QUANT_REPO": str(self.repo),
            "QUANT_PYTHON": sys.executable,
            "QUANT_STATE_ROOT": str(self.state),
            "QUANT_RUN_ROOT": str(self.runs),
            "TEST_CALLS": str(self.calls),
            **extra_environment,
        })
        return subprocess.run(
            ["bash", str(POST_PIPELINE)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls_made(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_nonpromoting_first_rung_skips_qat100_and_completes(self) -> None:
        self.first_qat(False)
        result = self.run_post()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls_made(), ["qwen38-target-158-reconstruction", "prune", "audit"])

    def test_promoting_first_rung_runs_qat100_and_accepts_nonpromotion(self) -> None:
        self.first_qat(True)
        result = self.run_post()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls_made(), [
            "qwen38-one-layer-qat-100",
            "qwen38-target-158-reconstruction",
            "prune",
            "audit",
        ])

    def test_nonpromoting_quality_and_radix_receipts_still_allow_qat_postlude(self) -> None:
        self.first_qat(False)
        self.upstream_receipts(terminal_promoted=False)
        result = self.run_post(TEST_SKIP_UPSTREAM_RECEIPTS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls_made(), ["qwen38-target-158-reconstruction", "prune", "audit"])

    def test_missing_first_rung_refuses_all_post_pipeline_work(self) -> None:
        result = self.run_post()
        self.assertEqual(result.returncode, 4)
        self.assertEqual(self.calls_made(), [])

    def test_missing_upstream_receipt_refuses_all_post_pipeline_work(self) -> None:
        self.first_qat(False)
        result = self.run_post(TEST_SKIP_UPSTREAM_RECEIPTS="1")
        self.assertEqual(result.returncode, 4)
        self.assertEqual(self.calls_made(), [])

    def test_wrong_schema_existing_target_is_not_overwritten(self) -> None:
        self.first_qat(False)
        target = self.state / "qwen38-target-158-reconstruction.json"
        target.write_text(json.dumps({"schema": "wrong"}))
        result = self.run_post()
        self.assertEqual(result.returncode, 4)
        self.assertEqual(self.calls_made(), [])
        self.assertEqual(json.loads(target.read_text()), {"schema": "wrong"})

    def test_zero_exit_without_receipt_is_failure(self) -> None:
        self.first_qat(False)
        result = self.run_post(TEST_TARGET_WITHOUT_RECEIPT="1")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls_made(), ["qwen38-target-158-reconstruction"])


if __name__ == "__main__":
    unittest.main()
