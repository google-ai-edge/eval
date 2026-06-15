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

"""Export custom task layer modules."""

from model_eval.custom_tasks import base
from model_eval.custom_tasks import groups
from model_eval.custom_tasks import loaders
from model_eval.custom_tasks import metrics
from model_eval.custom_tasks import registry

OpenAIMessages = base.OpenAIMessages
CustomTask = base.CustomTask
DatasetRow = base.DatasetRow
TaskRegistry = registry.TaskRegistry

# Metrics re-exports.
register_metric = metrics.register_metric
get_metric = metrics.get_metric
list_metrics = metrics.list_metrics
clear_metrics = metrics.clear_metrics
compose = metrics.compose

# Normalizer re-exports.
register_normalizer = metrics.register_normalizer
get_normalizer = metrics.get_normalizer
list_normalizers = metrics.list_normalizers
clear_normalizers = metrics.clear_normalizers

# Loaders re-exports.
register_loader = loaders.register_loader
get_loader = loaders.get_loader
list_loaders = loaders.list_loaders
clear_loaders = loaders.clear_loaders

# Groups re-exports.
is_group = groups.is_group
subtasks_of = groups.subtasks_of
list_groups = groups.list_groups

__all__ = [
    "OpenAIMessages",
    "CustomTask",
    "DatasetRow",
    "TaskRegistry",
    "register_metric",
    "get_metric",
    "list_metrics",
    "clear_metrics",
    "compose",
    "register_normalizer",
    "get_normalizer",
    "list_normalizers",
    "clear_normalizers",
    "register_loader",
    "get_loader",
    "list_loaders",
    "clear_loaders",
    "is_group",
    "subtasks_of",
    "list_groups",
]
