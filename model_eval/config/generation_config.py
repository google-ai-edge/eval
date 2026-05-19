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

"""Generation configuration definitions."""

import pydantic


class GenerationConfig(pydantic.BaseModel):
  """Per-request generation parameters.

  This config defines generation parameters sent with each inference call.
  """

  # Controls the randomness of the generated output.
  temperature: float = 1.0
  # The maximum number of new tokens to generate.
  max_new_tokens: int = 256
  # A list of sequences where the generation should stop.
  stop_sequences: list[str] = []
