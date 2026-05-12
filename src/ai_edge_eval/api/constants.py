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

"""API constants module."""

# Identifier for the local chat scoring evaluation model.
LOCAL_CHAT_SCORE_MODEL_NAME = "local-chat-score"

# Constants for OpenAI-compatible API endpoints:
# Endpoint for chat generation completions.
CHAT_COMPLETIONS_ENDPOINT = "v1/chat/completions"
# Endpoint for chat scoring completions.
CHAT_SCORE_ENDPOINT = "v1/chat/score"
