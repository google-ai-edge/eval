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

"""Export dataset loading layer modules."""

from model_eval.custom_tasks.loaders import registry
from model_eval.custom_tasks.loaders import util

CSV_DATA_COLUMN = util.CSV_DATA_COLUMN
parse_samples = util.parse_samples
load_dataset = util.load_dataset

clear_loaders = registry.clear_loaders
get_loader = registry.get_loader
list_loaders = registry.list_loaders
register_loader = registry.register_loader

__all__ = [
    "CSV_DATA_COLUMN",
    "parse_samples",
    "load_dataset",
    "clear_loaders",
    "get_loader",
    "list_loaders",
    "register_loader",
]
