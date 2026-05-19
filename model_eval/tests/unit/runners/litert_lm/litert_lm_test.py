# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LiteRT LM runner."""

import unittest
from unittest import mock

from model_eval.runners.litert_lm import litert_lm


class TestLiteRtLmRunner(unittest.TestCase):

  @mock.patch("model_eval.runners.base.requests.post")
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.threading.Thread"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.uvicorn.Server"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.uvicorn.Config"
  )
  @mock.patch(
      "model_eval.runners.litert_lm._litert_lm_server.build_app"
  )
  @mock.patch(
      "model_eval.runners.litert_lm._litert_lm_server.wait_for_server"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.litert_lm.set_min_log_severity"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.litert_lm.Engine"
  )
  def test_initialization(
      self,
      mock_engine,
      mock_set_min_log_severity,
      mock_wait_for_server,
      mock_build_app,
      mock_uvicorn_config,
      mock_uvicorn_server,
      mock_thread,
      mock_post,
  ):
    mock_post.return_value.json.return_value = {
        "choices": [{"score": 0.9, "logprobs": []}]
    }
    config = litert_lm.LiteRtLmRunner.Config(
        runner_type="litert-lm",
        model_path="/path/to/model",
        model_name="my-test-model",
        backend="gpu",
        max_num_tokens=2048,
        host="0.0.0.0",
        port=9090,
    )
    runner = litert_lm.LiteRtLmRunner(config)
    self.assertIs(runner._config, config)
    with mock.patch("os.path.exists", return_value=True):
      runner.start()

    # Verify Engine was initialized correctly.
    mock_engine.assert_called_once_with(
        "/path/to/model",
        backend=litert_lm._resolve_backend(litert_lm.litert_lm.Backend.GPU),
        max_num_tokens=2048,
    )
    mock_set_min_log_severity.assert_called_once_with(mock.ANY)

    # Verify build_app was called.
    mock_build_app.assert_called_once_with(mock_engine.return_value, config)

    # Verify uvicorn.Config.
    mock_uvicorn_config.assert_called_once_with(
        mock_build_app.return_value,
        host="0.0.0.0",
        port=9090,
        log_level="warning",
        access_log=False,
    )

    # Verify wait_for_server.
    mock_wait_for_server.assert_called_once_with("http://0.0.0.0:9090")

    runner.stop()

  @mock.patch("model_eval.runners.base.requests.post")
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.threading.Thread"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.uvicorn.Server"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.uvicorn.Config"
  )
  @mock.patch(
      "model_eval.runners.litert_lm._litert_lm_server.build_app"
  )
  @mock.patch(
      "model_eval.runners.litert_lm._litert_lm_server.wait_for_server"
  )
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.litert_lm.Engine"
  )
  def test_initialization_with_multimodal_backends(
      self,
      mock_engine,
      mock_wait_for_server,
      mock_build_app,
      mock_uvicorn_config,
      mock_uvicorn_server,
      mock_thread,
      mock_post,
  ):
    mock_post.return_value.json.return_value = {
        "choices": [{"score": 0.9, "logprobs": []}]
    }
    config = litert_lm.LiteRtLmRunner.Config(
        runner_type="litert-lm",
        model_path="/path/to/model",
        model_name="my-test-model",
        backend="cpu",
        vision_backend="gpu",
        audio_backend="cpu",
    )
    runner = litert_lm.LiteRtLmRunner(config)
    with mock.patch("os.path.exists", return_value=True):
      runner.start()
    mock_engine.assert_called_once_with(
        "/path/to/model",
        backend=litert_lm._resolve_backend(litert_lm.litert_lm.Backend.CPU),
        max_num_tokens=4096,
        vision_backend=litert_lm._resolve_backend(
            litert_lm.litert_lm.Backend.GPU
        ),
        audio_backend=litert_lm._resolve_backend(
            litert_lm.litert_lm.Backend.CPU
        ),
    )
    runner.stop()

  def test_clamp_log_severity(self):
    self.assertEqual(
        litert_lm._clamp_log_severity(-5), litert_lm.litert_lm.LogSeverity(0)
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(0), litert_lm.litert_lm.LogSeverity(0)
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(3), litert_lm.litert_lm.LogSeverity(3)
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(5), litert_lm.litert_lm.LogSeverity(5)
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(6), litert_lm.litert_lm.LogSeverity(5)
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(1000),
        litert_lm.litert_lm.LogSeverity(1000),
    )
    self.assertEqual(
        litert_lm._clamp_log_severity(1005),
        litert_lm.litert_lm.LogSeverity(1000),
    )

  def test_from_unified_args(self):
    config = litert_lm.LiteRtLmRunner.Config.from_unified_args(
        model_path="/path/to/model",
        device="gpu",
        runner_args={"max_num_tokens": 1234},
    )
    self.assertEqual(config.model_path, "/path/to/model")
    self.assertEqual(config.backend, "gpu")
    self.assertEqual(config.max_num_tokens, 1234)

  def test_from_unified_args_conflicts(self):
    with self.assertRaisesRegex(
        ValueError, "--model-path conflicts with 'model_path'"
    ):
      litert_lm.LiteRtLmRunner.Config.from_unified_args(
          model_path="/foo",
          device="cpu",
          runner_args={"model_path": "/bar"},
      )

    with self.assertRaisesRegex(
        ValueError, "--device conflicts with 'backend'"
    ):
      litert_lm.LiteRtLmRunner.Config.from_unified_args(
          model_path="/foo",
          device="gpu",
          runner_args={"backend": "cpu"},
      )

  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.os.path.exists"
  )
  @mock.patch("huggingface_hub.hf_hub_download")
  @mock.patch(
      "model_eval.runners.litert_lm.litert_lm.litert_lm.Engine"
  )
  def test_hf_path_resolution(self, mock_engine, mock_download, mock_exists):
    mock_exists.return_value = False
    mock_download.return_value = "/cached/download/model.litertlm"
    config = litert_lm.LiteRtLmRunner.Config(
        runner_type="litert-lm", model_path="org/model/file.litertlm"
    )
    runner = litert_lm.LiteRtLmRunner(config)
    try:
      runner.start()
    except Exception:  # pylint: disable=broad-except
      pass
    mock_download.assert_called_once_with(
        repo_id="org/model", filename="file.litertlm"
    )
    mock_engine.assert_called_once_with(
        "/cached/download/model.litertlm",
        backend=mock.ANY,
        max_num_tokens=mock.ANY,
    )


if __name__ == "__main__":
  unittest.main()
