/**
 * @license
 * Copyright 2026 The AI Edge Model Explorer Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import assert from 'node:assert/strict';
const sandbox = {window: {}};
runInNewContext(
  readFileSync(new URL('../public/graph-arch-preview/model-tree.js', import.meta.url), 'utf8'),
  sandbox,
);
const summarize = sandbox.window.summarizeModelTreeMetrics;
const values = [0.2, null, 0, 0.8, null];
assert.equal(
  summarize(values, null, 'output').value,
  null,
  'Missing final output must not use last available layer',
);
assert.equal(summarize(values, 0, 'output').value, 0, 'Zero is a valid output');
assert.equal(summarize(values, null, 'min').value, 0);
assert.equal(summarize(values, null, 'max').value, 0.8);
assert.equal(summarize(values, null, 'max').available, 3);
assert.equal(summarize(values, null, 'max').total, 5);
assert.equal(summarize([null, undefined, NaN, Infinity], null, 'min').value, null);
assert.equal(summarize([null, undefined], null, 'max').value, null);
console.log('Module summaries: missing outputs, zero, min/max and partial coverage passed.');
