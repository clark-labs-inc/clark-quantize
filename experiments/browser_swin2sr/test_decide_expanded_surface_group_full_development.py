import tempfile
import unittest
from pathlib import Path

from decide_expanded_surface_group_full_development import (
    paired_report,
    validate_report_bundle,
)


def sample(identity, gt, edge, teacher, source="rico"):
    return {
        "sha256": identity,
        "source": source,
        "psnr_vs_ground_truth": gt,
        "edge_psnr_vs_ground_truth": edge,
        "psnr_vs_teacher": teacher,
    }


class ExpandedSurfaceGroupFullDecisionTests(unittest.TestCase):
    def test_paired_report_uses_exact_sample_identities(self):
        baseline = {
            str(index): sample(str(index), 10.0, 20.0, 30.0)
            for index in range(3040)
        }
        candidate = {
            str(index): sample(str(index), 10.1, 20.2, 30.3)
            for index in range(3040)
        }
        rule = {"bootstrap_seed": 7, "bootstrap_samples": 100}

        paired, by_source = paired_report(baseline, candidate, rule)

        self.assertAlmostEqual(
            paired["psnr_vs_ground_truth"]["mean"], 0.1
        )
        self.assertAlmostEqual(by_source["rico"], 0.1)

    def test_report_bundle_is_bound_by_path_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            bundle_path.write_text("{}")
            protocol = {"bundle": str(bundle_path)}
            bundle = {"manifest_sha256": "sealed-manifest"}
            report = {
                "bundle": str(bundle_path),
                "bundle_manifest_sha256": "sealed-manifest",
            }

            validate_report_bundle(report, protocol, bundle, "candidate")

            report["bundle_manifest_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "bundle identity"):
                validate_report_bundle(report, protocol, bundle, "candidate")


if __name__ == "__main__":
    unittest.main()
