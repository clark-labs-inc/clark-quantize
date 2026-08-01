from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import decide_optimizer_final_promotion as final_promotion
import evaluate_optimizer_final_candidate as final_evaluator
import prepare_full_onnx_smoke as runtime_fixture
import prepare_optimizer_final_candidate as candidate_preparer
from final_artifact_contract import file_sha256


REPOSITORY = Path(__file__).resolve().parents[2]
LOCAL_IMPORT_ROOTS = (
    REPOSITORY,
    REPOSITORY / "experiments/browser_swin2sr",
)


def resolve_local_module(module: str) -> Path | None:
    relative = Path(*module.split("."))
    for root in LOCAL_IMPORT_ROOTS:
        for candidate in (
            root / f"{relative}.py",
            root / relative / "__init__.py",
        ):
            if candidate.exists() and REPOSITORY in candidate.resolve().parents:
                return candidate.resolve()
    return None


def recursive_local_sources(declared: tuple[str, ...]) -> set[str]:
    pending = [(REPOSITORY / name).resolve() for name in declared]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependency = resolve_local_module(alias.name)
                    if dependency is not None:
                        pending.append(dependency)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                dependency = resolve_local_module(node.module)
                if dependency is not None:
                    pending.append(dependency)
    return {
        str(source.relative_to(REPOSITORY))
        for source in visited
    }


class OptimizerFinalPipelineTests(unittest.TestCase):
    def selected_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        checkpoint = root / "checkpoint.pt"
        dense = root / "dense.pt"
        runtime = root / "runtime.pt"
        materialization = root / "materialization.json"
        study = root / "study.json"
        policy = root / "policy.json"
        decision = root / "decision.json"
        checkpoint.write_bytes(b"checkpoint")
        dense.write_bytes(b"dense")
        runtime.write_bytes(b"runtime")
        materialization.write_text("{}\n")
        study.write_text(json.dumps({
            "format": "backdropkit-swin2sr-optimizer-transfer-v1",
            "selection_source": "development",
            "bundle": "development-bundle.json",
            "bundle_sha256": "development-bundle-sha",
            "q4_reference": "development-q4.json",
            "q4_reference_sha256": "development-q4-sha",
        }))
        policy.write_text(json.dumps({
            "reference": {
                "artifact_bytes": 100,
            },
            "artifact_gate": {
                "maximum_fraction_of_reference": 0.5,
            },
        }))
        evidence = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "materialization": str(materialization),
            "materialization_sha256": file_sha256(materialization),
            "dense_state": str(dense),
            "dense_state_sha256": file_sha256(dense),
            "runtime_state": str(runtime),
            "runtime_state_sha256": file_sha256(runtime),
        }
        candidate = {
            "optimizer_arm": "qat8",
            "eligible": True,
            "development_quality_gate": {"passed": True},
            "canonical_evidence": evidence,
        }
        decision_payload = {
            "format": (
                "backdropkit-swin2sr-optimizer-transfer-decision-v1"
            ),
            "optimizer_recipe_selected": True,
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
            "promotion_policy_sha256": file_sha256(policy),
            "study_manifest": str(study),
            "study_manifest_sha256": file_sha256(study),
            "selected_optimizer": "qat8",
            "selected_checkpoint": evidence["checkpoint"],
            "selected_checkpoint_sha256": evidence["checkpoint_sha256"],
            "selected_materialization": evidence["materialization"],
            "selected_materialization_sha256": evidence[
                "materialization_sha256"
            ],
            "selected_dense_state": evidence["dense_state"],
            "selected_dense_state_sha256": evidence["dense_state_sha256"],
            "selected_runtime_state": evidence["runtime_state"],
            "selected_runtime_state_sha256": evidence[
                "runtime_state_sha256"
            ],
            "candidates": [candidate],
        }
        decision.write_text(json.dumps(decision_payload))
        return decision, policy, decision_payload

    def test_artifact_gate_requires_at_least_two_x_reduction(self):
        policy = {
            "reference": {"artifact_bytes": 100},
            "artifact_gate": {"maximum_fraction_of_reference": 0.5},
        }
        at_boundary = candidate_preparer.artifact_size_gate(policy, 50)
        above_boundary = candidate_preparer.artifact_size_gate(policy, 51)

        self.assertTrue(at_boundary["passed"])
        self.assertEqual(at_boundary["candidate_fraction_of_reference"], 0.5)
        self.assertFalse(above_boundary["passed"])

    def test_preparer_rejects_an_unselected_or_quality_failed_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision, policy, payload = self.selected_fixture(root)
            payload["optimizer_recipe_selected"] = False
            decision.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "selected no candidate"):
                candidate_preparer.selected_optimizer_evidence(
                    decision,
                    policy,
                )

            payload["optimizer_recipe_selected"] = True
            payload["candidates"][0]["development_quality_gate"][
                "passed"
            ] = False
            decision.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                RuntimeError,
                "failed development quality",
            ):
                candidate_preparer.selected_optimizer_evidence(
                    decision,
                    policy,
                )

    def test_preparer_revalidates_the_selected_canonical_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision, policy, payload = self.selected_fixture(root)
            selected = payload["candidates"][0]["canonical_evidence"]
            Path(selected["dense_state"]).write_bytes(b"changed")

            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                candidate_preparer.selected_optimizer_evidence(
                    decision,
                    policy,
                )

    def test_existing_candidate_must_match_the_current_precommit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_path, policy_path, decision = self.selected_fixture(root)
            study = json.loads(Path(decision["study_manifest"]).read_text())
            selected = decision["candidates"][0]
            base_onnx = root / "base.onnx"
            precommit_path = root / "artifact-precommit.json"
            base_onnx.write_bytes(b"base")
            precommit = {
                "base_onnx": str(base_onnx),
                "base_onnx_sha256": file_sha256(base_onnx),
            }
            precommit_path.write_text(json.dumps(precommit))

            with self.assertRaisesRegex(RuntimeError, "lineage differs"):
                candidate_preparer.validate_existing_candidate(
                    {"lane": "stale-arm"},
                    decision_path=decision_path,
                    decision=decision,
                    study=study,
                    selected=selected,
                    policy_path=policy_path,
                    precommit_path=precommit_path,
                    precommit=precommit,
                )

    def final_candidate(self) -> dict:
        return {
            "format": "backdropkit-swin2sr-final-candidate-v1",
            "selection_source": "development",
            "final_evaluation_eligible": True,
            "model_promotion_authorized": False,
            "final_heldout_locked": True,
            "development_quality_gate": {"passed": True},
            "artifact_size_gate": {"passed": True},
            **{
                key: f"value-{key}"
                for key in (
                    "lane",
                    "optimizer_decision",
                    "optimizer_decision_sha256",
                    "optimizer_study_manifest",
                    "optimizer_study_manifest_sha256",
                    "development_bundle",
                    "development_bundle_sha256",
                    "development_q4",
                    "development_q4_sha256",
                    "promotion_policy",
                    "promotion_policy_sha256",
                    "artifact_precommit",
                    "artifact_precommit_sha256",
                    "base_onnx",
                    "base_onnx_sha256",
                    "materialization",
                    "materialization_sha256",
                    "hard_state",
                    "hard_state_sha256",
                    "runtime_state",
                    "runtime_state_sha256",
                    "portable_onnx",
                    "portable_onnx_bytes",
                    "portable_onnx_sha256",
                    "portable_ledger",
                    "portable_ledger_sha256",
                    "runtime_package",
                    "runtime_package_bytes",
                    "runtime_package_sha256",
                    "runtime_package_compression",
                )
            },
        }

    def test_final_evaluator_requires_both_development_gates(self):
        candidate = self.final_candidate()
        self.assertEqual(
            final_evaluator.selected_candidate(candidate)["lane"],
            "value-lane",
        )
        candidate["artifact_size_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "artifact size"):
            final_evaluator.selected_candidate(candidate)

    def test_final_selection_cannot_reuse_development_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_path, policy_path, decision = self.selected_fixture(root)
            development_bundle = root / "development-bundle.json"
            development_q4 = root / "development-q4.json"
            final_bundle = root / "final-bundle.json"
            final_q4 = root / "final-q4.json"
            study_path = Path(decision["study_manifest"])
            development_bundle.write_text(json.dumps({
                "manifest_sha256": "development",
                "samples": 1,
            }))
            development_q4.write_text(json.dumps({
                "bundle_manifest_sha256": "development",
            }))
            final_bundle_payload = {
                "manifest_sha256": "final",
                "samples": 1,
            }
            final_q4_payload = {
                "model": "q4",
                "model_revision": "revision",
                "state_sha256": "q4-state",
                "artifact_bytes": 100,
                "bundle_manifest_sha256": "final",
            }
            final_bundle.write_text(json.dumps(final_bundle_payload))
            final_q4.write_text(json.dumps(final_q4_payload))
            study = json.loads(study_path.read_text())
            study.update({
                "bundle": str(development_bundle),
                "bundle_sha256": file_sha256(development_bundle),
                "q4_reference": str(development_q4),
                "q4_reference_sha256": file_sha256(development_q4),
            })
            study_path.write_text(json.dumps(study))
            decision["study_manifest_sha256"] = file_sha256(study_path)
            decision_path.write_text(json.dumps(decision))
            policy = {
                "reference": {
                    "model": "q4",
                    "revision": "revision",
                    "model_sha256": "q4-state",
                    "artifact_bytes": 100,
                    "bundle_manifest_sha256": "final",
                    "samples": 1,
                },
            }
            policy_path.write_text(json.dumps(policy))
            decision["promotion_policy_sha256"] = file_sha256(policy_path)
            decision_path.write_text(json.dumps(decision))
            evidence = decision["candidates"][0]["canonical_evidence"]
            base_onnx = root / "base.onnx"
            portable_onnx = root / "candidate.onnx"
            artifact_precommit = root / "artifact-precommit.json"
            base_onnx.write_bytes(b"base")
            portable_onnx.write_bytes(b"portable")
            candidate = {
                "optimizer_decision": str(decision_path),
                "optimizer_decision_sha256": file_sha256(decision_path),
                "optimizer_study_manifest": str(study_path),
                "optimizer_study_manifest_sha256": file_sha256(study_path),
                "development_bundle": str(development_bundle),
                "development_bundle_sha256": file_sha256(development_bundle),
                "development_q4": str(development_q4),
                "development_q4_sha256": file_sha256(development_q4),
                "promotion_policy": str(policy_path),
                "promotion_policy_sha256": file_sha256(policy_path),
                "base_onnx": str(base_onnx),
                "base_onnx_sha256": file_sha256(base_onnx),
                "lane": "qat8",
                "materialization": evidence["materialization"],
                "materialization_sha256": evidence[
                    "materialization_sha256"
                ],
                "hard_state": evidence["dense_state"],
                "hard_state_sha256": evidence["dense_state_sha256"],
                "runtime_state": evidence["runtime_state"],
                "runtime_state_sha256": evidence["runtime_state_sha256"],
                "portable_onnx": str(portable_onnx),
            }
            artifact_precommit.write_text(json.dumps({
                "format": "backdropkit-swin2sr-artifact-precommit-v1",
                "optimizer_decision": candidate["optimizer_decision"],
                "optimizer_decision_sha256": candidate[
                    "optimizer_decision_sha256"
                ],
                "promotion_policy": candidate["promotion_policy"],
                "promotion_policy_sha256": candidate[
                    "promotion_policy_sha256"
                ],
                "base_onnx": candidate["base_onnx"],
                "base_onnx_sha256": candidate["base_onnx_sha256"],
                "materialization": candidate["materialization"],
                "materialization_sha256": candidate[
                    "materialization_sha256"
                ],
                "portable_onnx": candidate["portable_onnx"],
                "source_files": {},
                "final_heldout_locked": True,
            }))
            candidate["artifact_precommit"] = str(artifact_precommit)
            candidate["artifact_precommit_sha256"] = file_sha256(
                artifact_precommit
            )
            final_evaluator.validate_development_selection(
                candidate,
                policy_path,
                final_bundle,
                final_bundle_payload,
                final_q4,
                final_q4_payload,
            )
            with self.assertRaisesRegex(
                ValueError,
                "reused final evidence",
            ):
                final_evaluator.validate_development_selection(
                    candidate,
                    policy_path,
                    development_bundle,
                    final_bundle_payload,
                    final_q4,
                    final_q4_payload,
                )

    def test_source_manifests_cover_recursive_local_imports(self):
        for declared in (
            candidate_preparer.SOURCE_FILES,
            final_evaluator.SOURCE_FILES,
            final_promotion.SOURCE_FILES,
            runtime_fixture.SOURCE_FILES,
        ):
            self.assertEqual(set(declared), recursive_local_sources(declared))


if __name__ == "__main__":
    unittest.main()
