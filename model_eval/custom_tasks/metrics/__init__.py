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

"""Export metric layer modules."""

from model_eval.custom_tasks.metrics import normalize
from model_eval.custom_tasks.metrics.normalize import clear_normalizers
from model_eval.custom_tasks.metrics.normalize import get_normalizer
from model_eval.custom_tasks.metrics.normalize import list_normalizers
from model_eval.custom_tasks.metrics.normalize import register_normalizer
from model_eval.custom_tasks.metrics.registry import clear_metrics
from model_eval.custom_tasks.metrics.registry import compose
from model_eval.custom_tasks.metrics.registry import get_metric
from model_eval.custom_tasks.metrics.registry import list_metrics
from model_eval.custom_tasks.metrics.registry import register_metric

__all__ = [
    "clear_metrics",
    "clear_normalizers",
    "compose",
    "get_metric",
    "get_normalizer",
    "list_metrics",
    "list_normalizers",
    "normalize",
    "register_metric",
    "register_normalizer",
]
