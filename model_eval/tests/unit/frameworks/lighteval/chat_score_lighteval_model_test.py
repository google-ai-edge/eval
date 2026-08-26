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

"""Unit tests for chat score lighteval model."""

from unittest import mock
from absl.testing import absltest
import pytest

pytest.importorskip(
    "lighteval",
    reason='install via `pip install -e ".[lighteval]"` to run lighteval tests',
)

from model_eval.frameworks.lighteval import _chat_score_lighteval_model  # pylint: disable=g-import-not-at-top,g-bad-import-order


class TestChatScoreLightevalModel(absltest.TestCase):

  def test_chat_score_model(self):
    mock_client = mock.MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_resp = mock.MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"score": -1.5, "logprobs": {"is_greedy": True}}]
    }
    mock_client.post.return_value = mock_resp

    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"

    with mock.patch.object(
        _chat_score_lighteval_model.httpx, "Client"
    ) as mock_httpx_cls, mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ):
      mock_httpx_cls.return_value = mock_client
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)

      mock_doc = mock.MagicMock()
      mock_doc.query = "hello"
      mock_doc.choices = ["a", "b", " world"]
      mock_doc.gold_index = 2

      res_ll = model.loglikelihood([mock_doc])
    self.assertLen(res_ll, 1)
    self.assertEqual(res_ll[0].logprobs, [-1.5, -1.5, -1.5])
    self.assertEqual(res_ll[0].argmax_logits_eq_gold, [True, True, True])
    self.assertEqual(mock_client.post.call_count, 3)

    model.loglikelihood_rolling = mock.MagicMock(
        side_effect=NotImplementedError
    )
    with self.assertRaises(NotImplementedError):
      model.loglikelihood_rolling([mock_doc])

  def test_prepare_prompt_api_with_images(self):
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None
    with mock.patch("lighteval.models.endpoints.litellm_model.SampleCache"):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.query = "Describe the image"
      mock_doc.instruction = None
      mock_doc.fewshot_samples = []

      mock_img = mock.MagicMock()
      mock_img.mode = "RGB"

      def fake_save(buf, format=None):
        buf.write(b"fake_jpeg_data")

      mock_img.save.side_effect = fake_save
      mock_doc.images = [mock_img]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 1)
      self.assertEqual(messages[0]["role"], "user")
      self.assertIsInstance(messages[0]["content"], list)
      self.assertEqual(
          messages[0]["content"][0],
          {"type": "text", "text": "Describe the image"},
      )
      self.assertEqual(messages[0]["content"][1]["type"], "image_url")
      self.assertTrue(
          messages[0]["content"][1]["image_url"]["url"].startswith(
              "data:image/jpeg;base64,"
          )
      )

  def test_prepare_prompt_api_without_images(self):
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None

    with mock.patch("lighteval.models.endpoints.litellm_model.SampleCache"):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.query = "Just text"
      mock_doc.instruction = None
      mock_doc.fewshot_samples = []
      mock_doc.images = None

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 1)
      self.assertEqual(messages[0], {"role": "user", "content": "Just text"})

  def test_prepare_prompt_api_converts_rgba_to_rgb(self):
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None
    with mock.patch("lighteval.models.endpoints.litellm_model.SampleCache"):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.query = "Describe the image"
      mock_doc.instruction = None
      mock_doc.fewshot_samples = []

      mock_img = mock.MagicMock()
      mock_img.mode = "RGBA"
      mock_converted_img = mock.MagicMock()
      mock_converted_img.mode = "RGB"

      def fake_save(buf, format=None):
        buf.write(b"fake_jpeg_data")

      mock_converted_img.save.side_effect = fake_save
      mock_img.convert.return_value = mock_converted_img
      mock_doc.images = [mock_img]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      mock_img.convert.assert_called_once_with("RGB")
      mock_converted_img.save.assert_called_once()
      self.assertLen(messages, 1)
      self.assertEqual(messages[0]["content"][1]["type"], "image_url")

  def test_prepare_prompt_api_does_not_mutate_original_messages(self):
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None
    orig_msg = {"role": "user", "content": "Describe the image"}
    with mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ), mock.patch(
        "lighteval.models.endpoints.litellm_model.PromptManager.prepare_prompt_api",
        return_value=[orig_msg],
    ):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_img = mock.MagicMock()
      mock_img.mode = "RGB"
      mock_img.save.side_effect = lambda buf, format=None: buf.write(b"data")
      mock_doc.images = [mock_img]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 1)
      self.assertIsInstance(messages[0]["content"], list)
      # Explicitly verify the original message dictionary was not mutated.
      self.assertEqual(
          orig_msg, {"role": "user", "content": "Describe the image"}
      )

  def test_prepare_prompt_api_with_fewshot_images(self):
    """Verifies image injection for both few-shot samples and the main query."""
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None

    orig_messages = [
        {"role": "user", "content": "Fewshot question"},
        {"role": "assistant", "content": "Fewshot answer"},
        {"role": "user", "content": "Main question"},
    ]
    with mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ), mock.patch(
        "lighteval.models.endpoints.litellm_model.PromptManager.prepare_prompt_api",
        return_value=orig_messages,
    ):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_fewshot = mock.MagicMock(spec=["images"])
      mock_img_fewshot = mock.MagicMock()
      mock_img_fewshot.mode = "RGB"
      mock_img_fewshot.save.side_effect = lambda buf, format=None: buf.write(
          b"fs_data"
      )
      mock_fewshot.images = [mock_img_fewshot]

      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.fewshot_samples = [mock_fewshot]
      mock_img_main = mock.MagicMock()
      mock_img_main.mode = "RGB"
      mock_img_main.save.side_effect = lambda buf, format=None: buf.write(
          b"main_data"
      )
      mock_doc.images = [mock_img_main]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 3)
      self.assertIsInstance(messages[0]["content"], list)
      self.assertEqual(
          messages[0]["content"][0],
          {"type": "text", "text": "Fewshot question"},
      )
      self.assertEqual(messages[0]["content"][1]["type"], "image_url")
      self.assertEqual(messages[1]["content"], "Fewshot answer")
      self.assertIsInstance(messages[2]["content"], list)
      self.assertEqual(
          messages[2]["content"][0], {"type": "text", "text": "Main question"}
      )
      self.assertEqual(messages[2]["content"][1]["type"], "image_url")

  def test_prepare_prompt_api_with_fewshot_images_only(self):
    """Verifies image injection when only few-shot samples contain images."""
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None

    orig_messages = [
        {"role": "user", "content": "Fewshot question"},
        {"role": "assistant", "content": "Fewshot answer"},
        {"role": "user", "content": "Main question"},
    ]
    with mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ), mock.patch(
        "lighteval.models.endpoints.litellm_model.PromptManager.prepare_prompt_api",
        return_value=orig_messages,
    ):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_fewshot = mock.MagicMock(spec=["images"])
      mock_img_fewshot = mock.MagicMock()
      mock_img_fewshot.mode = "RGB"
      mock_img_fewshot.save.side_effect = lambda buf, format=None: buf.write(
          b"fs_data"
      )
      mock_fewshot.images = [mock_img_fewshot]

      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.fewshot_samples = [mock_fewshot]
      mock_doc.images = None

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 3)
      self.assertIsInstance(messages[0]["content"], list)
      self.assertEqual(
          messages[0]["content"][0],
          {"type": "text", "text": "Fewshot question"},
      )
      self.assertEqual(messages[0]["content"][1]["type"], "image_url")
      self.assertEqual(messages[1]["content"], "Fewshot answer")
      self.assertEqual(messages[2]["content"], "Main question")

  def test_prepare_prompt_api_with_system_prompt_and_fewshots(self):
    """Verifies few-shot image indexing when a system prompt offset is present."""
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = "You are a helpful assistant."

    orig_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Fewshot question"},
        {"role": "assistant", "content": "Fewshot answer"},
        {"role": "user", "content": "Main question"},
    ]
    with mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ), mock.patch(
        "lighteval.models.endpoints.litellm_model.PromptManager.prepare_prompt_api",
        return_value=orig_messages,
    ):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_fewshot = mock.MagicMock(spec=["images"])
      mock_img_fewshot = mock.MagicMock()
      mock_img_fewshot.mode = "RGB"
      mock_img_fewshot.save.side_effect = lambda buf, format=None: buf.write(
          b"fs_data"
      )
      mock_fewshot.images = [mock_img_fewshot]

      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.fewshot_samples = [mock_fewshot]
      mock_img_main = mock.MagicMock()
      mock_img_main.mode = "RGB"
      mock_img_main.save.side_effect = lambda buf, format=None: buf.write(
          b"main_data"
      )
      mock_doc.images = [mock_img_main]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 4)
      self.assertEqual(
          messages[0],
          {"role": "system", "content": "You are a helpful assistant."},
      )
      self.assertIsInstance(messages[1]["content"], list)
      self.assertEqual(
          messages[1]["content"][0],
          {"type": "text", "text": "Fewshot question"},
      )
      self.assertEqual(messages[1]["content"][1]["type"], "image_url")
      self.assertEqual(messages[2]["content"], "Fewshot answer")
      self.assertIsInstance(messages[3]["content"], list)
      self.assertEqual(
          messages[3]["content"][0], {"type": "text", "text": "Main question"}
      )
      self.assertEqual(messages[3]["content"][1]["type"], "image_url")

  def test_inject_images_into_content_with_list_and_fallback(self):
    """Verifies _inject_images_into_content with pre-formatted lists and fallbacks."""
    mock_img = mock.MagicMock()
    mock_img.mode = "RGB"
    mock_img.save.side_effect = lambda buf, format=None: buf.write(b"data")

    # Test pre-formatted list content.
    orig_list = [{"type": "text", "text": "Pre-existing block"}]
    res_list = _chat_score_lighteval_model._inject_images_into_content(
        orig_list, [mock_img]
    )
    self.assertLen(res_list, 2)
    self.assertEqual(
        res_list[0], {"type": "text", "text": "Pre-existing block"}
    )
    self.assertEqual(res_list[1]["type"], "image_url")
    # Verify the original list was not mutated.
    self.assertLen(orig_list, 1)

    # Test non-string, non-list content (e.g., None) raises ValueError.
    with self.assertRaises(ValueError):
      _chat_score_lighteval_model._inject_images_into_content(None, [mock_img])

  def test_prepare_prompt_api_with_none_fewshot_samples(self):
    """Verifies prepare_prompt_api gracefully handles None fewshot_samples."""
    config = mock.MagicMock()
    config.base_url = "http://127.0.0.1:8000/"
    config.model_name = "test-model"
    config.system_prompt = None

    orig_messages = [{"role": "user", "content": "Query without fewshots"}]
    with mock.patch(
        "lighteval.models.endpoints.litellm_model.SampleCache"
    ), mock.patch(
        "lighteval.models.endpoints.litellm_model.PromptManager.prepare_prompt_api",
        return_value=orig_messages,
    ):
      model = _chat_score_lighteval_model.ChatScoreLightevalModel(config)
      mock_doc = mock.MagicMock(
          spec=["query", "instruction", "fewshot_samples", "images"]
      )
      mock_doc.fewshot_samples = None
      mock_img = mock.MagicMock()
      mock_img.mode = "RGB"
      mock_img.save.side_effect = lambda buf, format=None: buf.write(b"data")
      mock_doc.images = [mock_img]

      messages = model.prompt_manager.prepare_prompt_api(mock_doc)
      self.assertLen(messages, 1)
      self.assertIsInstance(messages[0]["content"], list)
      self.assertEqual(messages[0]["content"][1]["type"], "image_url")


if __name__ == "__main__":
  absltest.main()
