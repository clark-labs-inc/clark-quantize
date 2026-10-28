import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "scripts" / "spot" / "run_spot_job.sh"


class SpotJobHarnessTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("flock"), "requires Linux flock")
    def test_completed_job_does_not_leave_lock_in_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_repo = root / "repo"
            scripts = fake_repo / "scripts" / "spot"
            scripts.mkdir(parents=True)
            shutil.copy2(HARNESS, scripts / "run_spot_job.sh")
            status = scripts / "spot_status.py"
            status.write_text("#!/usr/bin/env python3\n")
            status.chmod(0o755)
            environment = {
                **os.environ,
                "QUANT_REPO": str(fake_repo),
                "QUANT_RUN_ROOT": str(root / "runs"),
                "QUANT_STATE_ROOT": str(root / "state"),
            }
            command = ["bash", str(scripts / "run_spot_job.sh")]
            first = subprocess.run(
                [*command, "first", "--", "/bin/true"],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            second = subprocess.run(
                [*command, "second", "--", "/bin/true"],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)


if __name__ == "__main__":
    unittest.main()
