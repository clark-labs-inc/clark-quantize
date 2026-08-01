from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_factorial_race import (
    Arm,
    all_arms,
    arm_command,
    jsonable_config_value,
    rank_results,
    run_arm,
)


class FactorialRaceTests(unittest.TestCase):
    def test_config_serialization_handles_every_path_list(self) -> None:
        self.assertEqual(
            jsonable_config_value([Path("/one"), Path("/two")]),
            ["/one", "/two"],
        )

    def test_matrix_has_24_unique_arms(self) -> None:
        arms = all_arms()
        self.assertEqual(len(arms), 24)
        self.assertEqual(len({arm.name for arm in arms}), 24)

    def test_ranking_prefers_ground_truth_then_teacher(self) -> None:
        results = [
            {
                "name": "a",
                "status": "complete",
                "elapsed_seconds": 2.0,
                "final": {
                    "psnr_vs_ground_truth": 40.0,
                    "psnr_vs_teacher": 45.0,
                    "edge": 0.2,
                },
            },
            {
                "name": "b",
                "status": "complete",
                "elapsed_seconds": 1.0,
                "final": {
                    "psnr_vs_ground_truth": 40.0,
                    "psnr_vs_teacher": 46.0,
                    "edge": 0.3,
                },
            },
        ]
        self.assertEqual(rank_results(results)[0]["name"], "b")

    def test_ranking_uses_edge_quality_instead_of_runtime(self) -> None:
        results = [
            {
                "name": "fast",
                "status": "complete",
                "elapsed_seconds": 1.0,
                "final": {
                    "psnr_vs_ground_truth": 40.0,
                    "psnr_vs_teacher": 45.0,
                    "edge": 0.3,
                },
            },
            {
                "name": "sharp",
                "status": "complete",
                "elapsed_seconds": 2.0,
                "final": {
                    "psnr_vs_ground_truth": 40.0,
                    "psnr_vs_teacher": 45.0,
                    "edge": 0.2,
                },
            },
        ]
        self.assertEqual(rank_results(results)[0]["name"], "sharp")

    def test_ranking_uses_heldout_when_every_arm_has_it(self) -> None:
        def result(name: str, mean: float) -> dict:
            return {
                "name": name,
                "status": "complete",
                "final": {
                    "psnr_vs_ground_truth": 50.0 if name == "a" else 40.0,
                    "psnr_vs_teacher": 50.0,
                },
                "heldout": {
                    "overall": {
                        "psnr_vs_ground_truth": {"mean": mean, "p10": 20.0},
                        "edge_psnr_vs_ground_truth": {"mean": 30.0},
                    }
                },
            }

        self.assertEqual(
            rank_results([result("a", 30.0), result("b", 31.0)])[0]["name"],
            "b",
        )

    def test_promoted_command_includes_resume_checkpoint(self) -> None:
        args = argparse.Namespace(
            validation_data=Path("/validation"),
            steps=2000,
            schedule_steps=500,
            batch_size=8,
            crop_size=128,
            learning_rate=2e-5,
            validate_every=100,
            checkpoint_every=500,
            device="mps",
            seed=1,
            data=[Path("/data")],
        )
        resume = Path("/checkpoint-500.pt")
        command = arm_command(
            args,
            Arm("optimal_mse", "none", "adamw"),
            Path("/out"),
            resume,
        )

        self.assertEqual(command[command.index("--resume") + 1], str(resume))

    def test_complete_arm_runs_only_missing_heldout_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = Arm("optimal_mse", "none", "adamw")
            destination = root / "stage1" / arm.name
            destination.mkdir(parents=True)
            final = {
                "event": "validation",
                "step": 500,
                "psnr_vs_ground_truth": 42.0,
                "psnr_vs_teacher": 47.0,
            }
            (destination / "train.jsonl").write_text(json.dumps(final) + "\n")
            args = argparse.Namespace(
                out=root,
                stage="stage1",
                steps=500,
                eval_manifest=[Path("/heldout.jsonl")],
                resume_stage=None,
            )
            heldout = {
                "overall": {
                    "psnr_vs_ground_truth": {"mean": 30.0, "p10": 20.0},
                    "edge_psnr_vs_ground_truth": {"mean": 40.0},
                }
            }
            with mock.patch(
                "run_factorial_race.evaluate_arm",
                mock.AsyncMock(return_value=(0, heldout)),
            ) as evaluate:
                async def evaluate_complete_arm() -> dict:
                    return await run_arm(args, arm, asyncio.Semaphore(1))

                result = asyncio.run(evaluate_complete_arm())

            evaluate.assert_awaited_once()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["heldout"], heldout)
            self.assertFalse((destination / "command.json").exists())


if __name__ == "__main__":
    unittest.main()
