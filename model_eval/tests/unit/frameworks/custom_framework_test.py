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

"""Unit tests for CustomFramework."""

import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from model_eval import custom_tasks as tasks
from model_eval.frameworks import custom
from model_eval.runners import base as runners_base
from model_eval.runners import http_server
import httpx


class CustomFrameworkTest(absltest.TestCase):

  def test_apply_samples(self):
    rows = [{"id": i} for i in range(5)]
    res_dict = list(custom._apply_samples(rows, {"foo": "1-3"}, "foo"))
    self.assertEqual(res_dict, [{"id": 1}, {"id": 2}, {"id": 3}])

    res_list = list(custom._apply_samples(rows, [0, 4], "foo"))
    self.assertEqual(res_list, [{"id": 0}, {"id": 4}])

  def test_evaluate_raises_value_error_on_both_sample_range_and_samples(self):
    task = tasks.CustomTask(
        name="foo",
        endpoint="op",
        dataset=lambda _: iter([]),
        metric_fn=lambda p, g, r: {},
    )
    registry = mock.MagicMock()
    registry.get_task.return_value = task
    with mock.patch.object(
        custom.tasks_registry.TaskRegistry,
        "global_registry",
        return_value=registry,
    ):
      with self.assertRaisesRegex(
          ValueError,
          "Only one of 'sample_range' or 'samples' can be set, not both.",
      ):
        custom.CustomFramework().evaluate(
            _runner(),
            tasks=["foo"],
            sample_range=(0, 10),
            eval_args={"samples": "0-5"},
        )

  def test_run_task_raises_value_error_on_both_sample_range_and_samples(self):
    framework = custom.CustomFramework()
    with self.assertRaisesRegex(
        ValueError,
        "Only one of 'sample_range' or 'samples' can be set, not both.",
    ):
      framework._run_task(
          _runner(),
          tasks.CustomTask(
              name="t",
              endpoint="op",
              dataset=lambda _: iter([]),
              metric_fn=lambda p, g, r: {},
          ),
          mock.MagicMock(),
          sample_range=(0, 10),
          samples="0-5",
      )

  def test_evaluate_conflicts(self):
    framework = custom.CustomFramework()
    with self.assertRaisesRegex(
        ValueError, "--batch-size conflicts with 'batch_size' in --eval-args"
    ):
      framework.evaluate(
          _runner(),
          ["foo"],
          batch_size=2,
          eval_args={"batch_size": 2},
      )

  def test_describe_eval_args(self):
    fields = custom.CustomFramework.describe_eval_args()
    self.assertIsInstance(fields, list)
    for field in fields:
      self.assertIn("name", field)

  @mock.patch("model_eval.runners.registry.get_all_runners")
  def test_supported_runners(self, mock_get_all_runners):
    mock_get_all_runners.return_value = ["runner1", "runner2"]
    runners = custom.CustomFramework.supported_runners()
    self.assertEqual(runners, ["runner1", "runner2"])

  def test_run_task_posts_each_request_to_endpoint(self):
    runner = _runner()
    rows = [
        {"requests": [{"op": "a"}, {"op": "b"}], "ground_truth": 1},
        {"requests": [{"op": "c"}], "ground_truth": 2},
    ]
    seen = []

    def handler(request):
      seen.append((str(request.url), json.loads(request.content)))
      return httpx.Response(200, json={"echo": json.loads(request.content)})

    captured = {}

    def metric_fn(preds, gts, rs):
      captured["preds"] = list(preds)
      captured["gts"] = list(gts)
      captured["rows"] = list(rs)
      return {"n": float(len(captured["preds"]))}

    task = tasks.CustomTask(
        name="endpoint",
        endpoint="op",
        dataset=lambda _: iter(rows),
        metric_fn=metric_fn,
    )
    framework = custom.CustomFramework()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
      result, samples = framework._run_task(runner, task, client)

    self.assertEqual(
        seen,
        [
            ("http://dummy/op", {"op": "a"}),
            ("http://dummy/op", {"op": "b"}),
            ("http://dummy/op", {"op": "c"}),
        ],
    )
    expected_preds = [
        [{"echo": {"op": "a"}}, {"echo": {"op": "b"}}],
        [{"echo": {"op": "c"}}],
    ]
    self.assertEqual(result, {"n": 2.0})
    self.assertEqual(captured["preds"], expected_preds)
    self.assertEqual(captured["gts"], [1, 2])
    self.assertEqual(captured["rows"], rows)
    self.assertEqual(
        samples,
        [
            {
                "input": rows[0]["requests"],
                "prediction": expected_preds[0],
                "errors": [None, None],
                "ground_truth": 1,
            },
            {
                "input": rows[1]["requests"],
                "prediction": expected_preds[1],
                "errors": [None],
                "ground_truth": 2,
            },
        ],
    )

  def test_post_all_records_failures_and_skips_the_rest(self):
    def handler(request):
      op = json.loads(request.content)["op"]
      if op == "http_error":
        return httpx.Response(409, text="busy")
      if op == "not_json":
        return httpx.Response(200, text="not json")
      if op == "network":
        raise httpx.ConnectError("refused", request=request)
      return httpx.Response(200, json={"ok": op})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
      for failing, error_pattern in (
          ("http_error", r"^HTTP 409: busy$"),
          ("not_json", r"^JSONDecodeError: "),
          ("network", r"^ConnectError: refused$"),
      ):
        with self.subTest(failing=failing):
          responses, errors = custom._post_all(
              client,
              "http://dummy/op",
              [{"op": "first"}, {"op": failing}, {"op": "after"}],
          )
          self.assertEqual(responses, [{"ok": "first"}, None, None])
          self.assertIsNone(errors[0])
          self.assertRegex(errors[1], error_pattern)
          self.assertEqual(errors[2], "skipped: an earlier request failed")

  def test_evaluate_uses_runner_timeout(self):
    task = tasks.CustomTask(
        name="timeout",
        endpoint="op",
        dataset=lambda _: iter([]),
        metric_fn=lambda p, g, r: {},
    )
    registry = mock.MagicMock()
    registry.get_task.return_value = task
    for runner_timeout, expected in ((None, 120.0), (1800.0, 1800.0)):
      with self.subTest(runner_timeout=runner_timeout):
        with mock.patch.object(
            custom.tasks_registry.TaskRegistry,
            "global_registry",
            return_value=registry,
        ), mock.patch.object(custom.httpx, "Client") as mock_client:
          custom.CustomFramework().evaluate(
              _runner(request_timeout_sec=runner_timeout),
              ["timeout"],
          )
        mock_client.assert_called_once_with(timeout=expected)

  def test_evaluate_passes_unconsumed_eval_args_and_server_args_to_task(self):
    rows = [{"requests": [], "ground_truth": 1}]
    dataset = mock.MagicMock(side_effect=lambda _: iter(rows))
    task = tasks.CustomTask(
        name="with_args",
        endpoint="/op/",
        dataset=dataset,
        metric_fn=lambda p, g, r: {"n": float(len(list(p)))},
    )
    registry = mock.MagicMock()
    registry.get_task.return_value = task
    framework = custom.CustomFramework()
    with mock.patch.object(
        custom.tasks_registry.TaskRegistry,
        "global_registry",
        return_value=registry,
    ):
      res = framework.evaluate(
          _runner(
              server_args={"model_name": "gemma", "window_duration": 1},
          ),
          ["with_args"],
          eval_args={
              "samples": "0",
              "window_duration": 6,
              "audio": False,
          },
      )
    # Runner server_args are merged beneath eval args; eval args win.
    dataset.assert_called_once_with(
        {"model_name": "gemma", "window_duration": 6, "audio": False}
    )
    self.assertEqual(res.aggregated_metrics["with_args"], {"n": 1.0})

  def test_run_task_rejects_task_args_with_file_dataset(self):
    path = self.create_tempfile(
        "data.jsonl", content='{"requests": [], "ground_truth": 1}\n'
    ).full_path
    task = tasks.CustomTask(
        name="file_with_args",
        endpoint="op",
        dataset=path,
        metric_fn=lambda p, g, r: {},
    )
    with self.assertRaisesRegex(ValueError, "file datasets do not accept"):
      custom.CustomFramework()._run_task(
          _runner(), task, mock.MagicMock(), task_args={"x": 1}
      )

  def test_run_task_file_dataset_ignores_server_args(self):
    path = self.create_tempfile(
        "data.jsonl", content='{"requests": [], "ground_truth": 1}\n'
    ).full_path
    task = tasks.CustomTask(
        name="file_no_args",
        endpoint="op",
        dataset=path,
        metric_fn=lambda p, g, r: {"n": float(len(list(p)))},
    )
    metrics, _ = custom.CustomFramework()._run_task(
        _runner(server_args={"model_name": "gemma"}),
        task,
        mock.MagicMock(),
    )
    self.assertEqual(metrics, {"n": 1.0})

  def test_apply_limit(self):
    rows = [{"id": i} for i in range(10)]
    self.assertEqual(list(custom._apply_limit(rows, 3)), rows[:3])
    self.assertEqual(
        list(custom._apply_limit(rows, 0.25)), rows[:3]
    )  # ceil(2.5)
    self.assertEqual(list(custom._apply_limit(rows, 2.0)), rows[:2])

  def test_apply_limit_stops_consuming_rows(self):
    consumed = []

    def gen():
      for i in range(10):
        consumed.append(i)
        yield {"id": i}

    self.assertEqual(
        list(custom._apply_limit(gen(), 2)), [{"id": 0}, {"id": 1}]
    )
    self.assertEqual(consumed, [0, 1])

  def test_apply_samples_other_task_and_string_expr(self):
    rows = [{"id": i} for i in range(5)]
    self.assertEqual(
        list(custom._apply_samples(rows, {"bar": "0"}, "foo")), rows
    )
    self.assertEqual(
        list(custom._apply_samples(rows, "0,2", "foo")),
        [{"id": 0}, {"id": 2}],
    )

  def test_apply_samples_edge_cases(self):
    rows = [{"id": i} for i in range(5)]
    for samples_val, expected_ids in (
        ("::2", [0, 2, 4]),  # Slice expression.
        ([4, 0, 4], [4, 0, 4]),  # Requested order and repeats are kept.
        ([-1, 0], [4, 0]),  # Negative indices count from the end.
        ([1, 9], [1]),  # Out-of-range indices are dropped.
        ([4, 9, 0], [4, 0]),
        ([-10, -1], [4]),  # Out-of-range negative indices are dropped too.
        ([], []),
    ):
      with self.subTest(samples_val=str(samples_val)):
        self.assertEqual(
            [r["id"] for r in custom._apply_samples(rows, samples_val, "foo")],
            expected_ids,
        )

  def test_select_rows_empty_samples_and_zero_limit_select_nothing(self):
    rows = [{"id": i} for i in range(5)]
    self.assertEqual(list(custom._select_rows(rows, "foo", None, None, [])), [])  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(list(custom._select_rows(rows, "foo", 0, None, None)), [])  # pyrefly: ignore[bad-argument-type]

  def test_apply_samples_stops_after_largest_index(self):
    consumed = []

    def gen():
      for i in range(10):
        consumed.append(i)
        yield {"id": i}

    self.assertEqual(
        [r["id"] for r in custom._apply_samples(gen(), "1-2", "foo")], [1, 2]
    )
    self.assertEqual(consumed, [0, 1, 2])

  def test_apply_samples_rejects_invalid_type(self):
    with self.assertRaisesRegex(ValueError, "expected dict, str, or list"):
      custom._apply_samples([], 3, "foo")

  def test_evaluate_raises_on_both_limit_and_samples(self):
    task = tasks.CustomTask(
        name="foo",
        endpoint="op",
        dataset=lambda _: iter([]),
        metric_fn=lambda p, g, r: {},
    )
    registry = mock.MagicMock()
    registry.get_task.return_value = task
    with mock.patch.object(
        custom.tasks_registry.TaskRegistry,
        "global_registry",
        return_value=registry,
    ):
      with self.assertRaisesRegex(ValueError, "'limit' or 'samples'"):
        custom.CustomFramework().evaluate(
            _runner(),
            ["foo"],
            limit=2,
            eval_args={"samples": "0"},
        )

  def test_run_task_raises_on_conflicting_slicing(self):
    framework = custom.CustomFramework()
    task = tasks.CustomTask(
        name="t",
        endpoint="op",
        dataset=lambda _: iter([]),
        metric_fn=lambda p, g, r: {},
    )
    for kwargs, pattern in (
        ({"limit": 1, "sample_range": (0, 1)}, "'limit' or 'sample_range'"),
        ({"limit": 1, "samples": "0"}, "'limit' or 'samples'"),
    ):
      with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
        with self.assertRaisesRegex(ValueError, pattern):
          framework._run_task(_runner(), task, mock.MagicMock(), **kwargs)

  def test_run_task_applies_limit_and_sample_range(self):
    rows = [{"requests": [], "ground_truth": i} for i in range(5)]
    task = tasks.CustomTask(
        name="sliced",
        endpoint="op",
        dataset=lambda _: iter(rows),
        metric_fn=lambda p, g, r: {"gts": list(g), "rows": list(r)},
    )
    framework = custom.CustomFramework()
    limited, _ = framework._run_task(_runner(), task, mock.MagicMock(), limit=2)
    ranged, _ = framework._run_task(
        _runner(), task, mock.MagicMock(), sample_range=(1, 3)
    )
    self.assertEqual(limited, {"gts": [0, 1], "rows": rows[:2]})
    self.assertEqual(ranged, {"gts": [1, 2, 3], "rows": rows[1:4]})

  def test_run_task_streams_dataset_and_closes_it(self):
    events = []

    def dataset(_):
      for i in range(5):
        events.append(f"setup_{i}")
        try:
          yield {"requests": [{"id": i}], "ground_truth": i}
        finally:
          events.append(f"cleanup_{i}")

    def handler(request):
      req_id = json.loads(request.content)["id"]
      events.append(f"post_{req_id}")
      return httpx.Response(200, json=req_id)

    task = tasks.CustomTask(
        name="streamed",
        endpoint="op",
        dataset=dataset,
        metric_fn=lambda p, g, r: {"preds": [r[0] for r in p]},
    )
    framework = custom.CustomFramework()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
      result, _ = framework._run_task(_runner(), task, client, limit=2)
    self.assertEqual(result, {"preds": [0, 1]})
    # Rows are produced one at a time, and the row after the limit is never
    # produced; closing the dataset runs the cleanup of the last row.
    self.assertEqual(
        events,
        [
            "setup_0", "post_0", "cleanup_0",
            "setup_1", "post_1", "cleanup_1",
        ],
    )  # fmt: skip

  def test_supported_task_ids(self):
    registry = mock.MagicMock()
    registry.get_all_tasks.return_value = ["a", "b"]
    with mock.patch.object(
        custom.tasks_registry.TaskRegistry,
        "global_registry",
        return_value=registry,
    ):
      self.assertEqual(custom.CustomFramework.supported_task_ids(), ["a", "b"])


def _runner(
    request_timeout_sec: float | None = None,
    server_args: dict[str, Any] | None = None,
) -> http_server.HttpServerRunner:
  return http_server.HttpServerRunner(
      http_server.HttpServerRunner.Config(
          runner_type=runners_base.RunnerType.HTTP_SERVER,
          server_url="http://dummy",
          request_timeout_sec=request_timeout_sec,
          server_args=server_args or {},
      )
  )


if __name__ == "__main__":
  absltest.main()
