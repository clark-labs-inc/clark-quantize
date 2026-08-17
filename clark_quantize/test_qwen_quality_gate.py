from __future__ import annotations

import copy
import unittest

from clark_quantize.qwen_quality_gate import (
    authorize_gate_b_continuation,
    canonical_digest,
    make_gate_request,
    validate_continuation_authorization,
)


def seal(value):
    value["report_content_sha256"] = canonical_digest(value)
    return value


def report(arm: str, correct: list[bool], *, state: str | None = None):
    value = {
        "schema_version": 1,
        "arm": arm,
        "model": {"id": "Qwen/test", "revision": "abc"},
        "checkpoint": (
            {"state_sha256": state, "manifest_sha256": "manifest-file"}
            if state
            else None
        ),
        "manifest": {"content_sha256": "manifest"},
        "results": [
            {
                "task": "mmlu",
                "n": len(correct),
                "accuracy": sum(correct) / len(correct),
                "samples": [
                    {"sample_id": str(index), "correct": item}
                    for index, item in enumerate(correct)
                ],
            }
        ],
    }
    return seal(value)


class QwenQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.request = make_gate_request(
            step=10,
            tokens=1000,
            checkpoint_path=__import__("pathlib").Path("/tmp/checkpoint-10"),
            checkpoint_state_sha256="state",
            checkpoint_manifest_sha256="manifest-file",
        )
        self.baseline = report("bf16", [True, True, True, True, True])

    def test_pass_is_bound_to_exact_checkpoint(self):
        candidate = report("qat_ternary", [True, True, True, False, False], state="state")
        authorization = authorize_gate_b_continuation(
            self.request, self.baseline, candidate
        )
        self.assertTrue(authorization["pass"])
        validate_continuation_authorization(authorization, self.request)

    def test_catastrophic_forgetting_rejects_continuation(self):
        candidate = report("qat_ternary", [True, True, False, False, False], state="state")
        authorization = authorize_gate_b_continuation(
            self.request, self.baseline, candidate
        )
        self.assertFalse(authorization["pass"])
        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_continuation_authorization(authorization, self.request)

    def test_different_checkpoint_is_rejected(self):
        candidate = report("qat_ternary", [True] * 5, state="other")
        with self.assertRaisesRegex(ValueError, "not bound"):
            authorize_gate_b_continuation(self.request, self.baseline, candidate)

    def test_different_sample_ids_are_rejected(self):
        candidate = report("qat_ternary", [True] * 5, state="state")
        candidate["results"][0]["samples"][0]["sample_id"] = "different"
        candidate["report_content_sha256"] = canonical_digest(
            {key: value for key, value in candidate.items() if key != "report_content_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "sample ids"):
            authorize_gate_b_continuation(self.request, self.baseline, candidate)

    def test_tampered_authorization_is_rejected(self):
        candidate = report("qat_ternary", [True] * 5, state="state")
        authorization = authorize_gate_b_continuation(
            self.request, self.baseline, candidate
        )
        tampered = copy.deepcopy(authorization)
        tampered["pass"] = False
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_continuation_authorization(tampered, self.request)


if __name__ == "__main__":
    unittest.main()
