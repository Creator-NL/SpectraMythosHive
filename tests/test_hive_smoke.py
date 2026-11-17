"""Lightweight CPU smoke & security guardrail tests for Spectra Mythos Hive."""

import asyncio
import importlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from fastapi import HTTPException, UploadFile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import hive_common  # noqa: E402
import hive_server  # noqa: E402
from open_mythos.main import MythosConfig, OpenMythos  # noqa: E402
from Tools import checkpoint_diff, eval_mythos, plot_training_log  # noqa: E402


def _make_tiny_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=4,
    )


class HiveCommonTests(unittest.TestCase):
    def test_module_imports(self):
        for name in (
            "hive_common",
            "hive_server",
            "hive_worker",
            "hive_worker_phase2",
            "test_model",
            "test_model_phase2",
        ):
            mod = importlib.import_module(name)
            self.assertIsNotNone(mod)

    def test_dotenv_and_env_parsers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("# comment\nHIVE_TEST_INT=42\nHIVE_TEST_FLOAT=1e-4\nHIVE_TEST_QUOTED='hello'\n")

            with patch.dict(os.environ, {}, clear=True):
                hive_common.load_dotenv(env_path)
                self.assertEqual(hive_common.get_env_int("HIVE_TEST_INT", 0), 42)
                self.assertAlmostEqual(hive_common.get_env_float("HIVE_TEST_FLOAT", 0.0), 1e-4)
                self.assertEqual(os.environ.get("HIVE_TEST_QUOTED"), "hello")
                self.assertEqual(hive_common.get_env_int("MISSING_INT", 9), 9)

    def test_resolve_micro_batch_and_dtype_cpu(self):
        with patch.dict(os.environ, {"HIVE_MICRO_BATCH": "auto"}, clear=False):
            self.assertEqual(hive_common.resolve_micro_batch("cpu"), 1)
        with patch.dict(os.environ, {"HIVE_MICRO_BATCH": "4"}, clear=False):
            self.assertEqual(hive_common.resolve_micro_batch("cpu"), 4)
        with patch.dict(os.environ, {"HIVE_MICRO_BATCH": "not_an_int"}, clear=False):
            self.assertEqual(hive_common.resolve_micro_batch("cpu"), 1)

        self.assertEqual(hive_common.resolve_worker_dtype("cpu"), torch.float32)
        with hive_common.make_amp_context("cpu", torch.float32):
            pass

    def test_tiny_openmythos_forward_and_weights_only_load(self):
        model = OpenMythos(_make_tiny_config())
        x = torch.randint(0, 128, (1, 8))
        logits = model(x, n_loops=2)
        self.assertEqual(logits.shape, (1, 8, 128))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "worker_model.pt")
            torch.save({"model_state": model.state_dict()}, ckpt_path)

            reloaded = OpenMythos(_make_tiny_config())
            hive_common.load_weights_into_model(reloaded, ckpt_path, strict_weights_only=True)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, reloaded.state_dict()[k]))


class HiveServerGuardrailTests(unittest.TestCase):
    def _build_temp_server_state(self, tmpdir):
        tiny_model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        patches = (
            patch.object(hive_server, "HIVE_STATE_DIR", tmpdir),
            patch.object(hive_server, "GLOBAL_MODEL_PATH", os.path.join(tmpdir, "global_model.pt")),
            patch.object(hive_server, "WORKER_MODEL_PATH", os.path.join(tmpdir, "worker_model.pt")),
            patch.object(hive_server, "CHECKPOINT_DIR", os.path.join(tmpdir, "checkpoints")),
            patch.object(hive_server, "HIVE_PASSWORD", "test_secret"),
            patch.object(hive_server, "SAVE_EVERY", 100),
            patch.object(hive_server, "model", tiny_model),
            patch.object(hive_server, "optimizer", opt),
            patch.object(hive_server, "global_step", 0),
            patch.object(hive_server, "dispatched_jobs", 0),
            patch.object(hive_server, "pending_jobs", {}),
        )
        return tiny_model, patches

    def _run_submit(self, payload: object, job_id: str = "0"):
        buf = io.BytesIO()
        torch.save(payload, buf)
        buf.seek(0)
        upload = UploadFile(filename="grads.pt", file=buf)
        return asyncio.run(
            hive_server.submit_gradients(
                file=upload,
                x_hive_password="test_secret",
                x_worker_name="CI-Worker",
                x_job_id=job_id,
                x_job_size="8",
            )
        )

    def test_auth_rejects_invalid_password(self):
        with patch.object(hive_server, "HIVE_PASSWORD", "test_secret"):
            with self.assertRaises(HTTPException) as ctx:
                hive_server.verify_auth("wrong_password")
            self.assertEqual(ctx.exception.status_code, 401)
            hive_server.verify_auth("test_secret")

    def test_valid_and_poisoned_gradients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model, patches = self._build_temp_server_state(tmpdir)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
                # 1. Dispatch a job
                job = asyncio.run(hive_server.get_job(x_hive_password="test_secret"))
                self.assertEqual(job["job_id"], 0)
                self.assertIn(0, hive_server.pending_jobs)

                # 2. NaN payload must be rejected with HTTP 400 and leave step untouched
                bad_grads = {name: torch.full_like(param, float("nan")) for name, param in tiny_model.named_parameters()}
                with self.assertRaises(HTTPException) as nan_ctx:
                    self._run_submit(bad_grads, job_id="0")
                self.assertEqual(nan_ctx.exception.status_code, 400)
                self.assertEqual(hive_server.global_step, 0)

                # 3. Shape mismatch must be rejected with HTTP 400
                mismatch_grads = {"weight": torch.zeros((99, 99))}
                with self.assertRaises(HTTPException) as shape_ctx:
                    self._run_submit(mismatch_grads, job_id="0")
                self.assertEqual(shape_ctx.exception.status_code, 400)

                # 4. Valid finite gradient advances global_step and pops pending job
                good_grads = {name: torch.ones_like(param) * 0.01 for name, param in tiny_model.named_parameters()}
                res = self._run_submit(good_grads, job_id="0")
                self.assertEqual(res["global_step"], 1)
                self.assertNotIn(0, hive_server.pending_jobs)


class DiagnosticToolTests(unittest.TestCase):
    def test_rate_loss_thresholds(self):
        self.assertIn("Master", eval_mythos.rate_loss(2.5))
        self.assertIn("Advanced", eval_mythos.rate_loss(3.0))
        self.assertIn("Proficient", eval_mythos.rate_loss(3.5))
        self.assertIn("Developing", eval_mythos.rate_loss(4.2))
        self.assertIn("Novice", eval_mythos.rate_loss(5.5))
        self.assertIn("Untrained", eval_mythos.rate_loss(7.0))

    def test_parse_training_log(self):
        sample_lines = [
            "    ⚡ micro-batch   1/128 | parallel_texts: 6 | loss 4.4189\n",
            "📉 Job #10 Average Loss: 4.3705\n",
            "✅ Job #10 Accepted! Global Step is now 11\n",
        ]
        micro, jobs, steps = plot_training_log.parse_log(sample_lines)
        self.assertEqual(micro, [4.4189])
        self.assertEqual(jobs, [(10, 4.3705)])
        self.assertEqual(steps, {10: 11})

    def test_checkpoint_diff_loads_worker_state_strictly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "worker_model.pt")
            torch.save({"model_state": {"layer.weight": torch.ones(2, 2)}}, p)
            sd = checkpoint_diff.load_state_dict(p)
            self.assertIn("layer.weight", sd)


if __name__ == "__main__":
    unittest.main()
