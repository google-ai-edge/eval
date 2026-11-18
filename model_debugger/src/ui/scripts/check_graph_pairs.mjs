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

import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// The automatic pairs of the Graph Diff's pairing editor: index.html's identicalGraphPairs and
// anchorGraphPairs, over a small graph of its own.
const html = fs.readFileSync(
  new URL('../public/graph-arch-preview/index.html', import.meta.url),
  'utf8',
);
const context = {analysisContext: null};
vm.createContext(context);
vm.runInContext(
  html.slice(html.indexOf('function mappingTensor('), html.indexOf('const automaticPairScopes')),
  context,
);

const port = (id, shape, extra = []) => ({
  id,
  attrs: [{key: 'tensor_shape', value: `float32[${shape}]`}, ...extra],
});
const node = (id, label, sources, ports) => ({
  id,
  label,
  incomingEdges: sources.map(([sourceNodeId, sourceNodeOutputId]) => ({
    sourceNodeId,
    sourceNodeOutputId,
  })),
  outputsMetadata: ports,
});
/** A layer region: its inputs, a norm, an attention block of two outputs, an add and its outputs. */
const graph = (id, pairInputs = false) => ({
  id,
  nodes: [
    node('boundary:inputs', 'GraphInputs', [], [port('0', '1,4,8')]),
    node('norm', 'RMSNorm', [['boundary:inputs', '0']], [port('0', '1,4,8')]),
    node(
      'attn',
      'Attention',
      [['norm', '0']],
      [port('0', '1,4,8', [{key: 'point_id', value: 'p-attn'}]), port('1', '1,4,4')],
    ),
    node(
      'add',
      'Add',
      [
        ['boundary:inputs', '0'],
        ['attn', '0'],
      ],
      [port('0', '1,4,8')],
    ),
    node('const', 'pseudo_const', [], [port('0', '8')]),
    node('out', 'GraphOutputs', [['add', '0']], []),
  ],
  mappingPresentation: {
    pairInputs,
    anchors: {
      norm_out: {outputNode: 'norm', outputPort: '0'},
      attn_out: {outputNode: 'attn', outputPort: '0'},
      output: {outputNode: 'add', outputPort: '0'},
    },
  },
});

// The page's values are of the sandbox's realm; compared as plain data.
const plain = (value) => JSON.parse(JSON.stringify(value));
const region = graph('layer');
const pairs = plain(context.identicalGraphPairs(region, region, {anchor: 'output'}));
// Every output of a pairable node pairs with itself; constants, inputs and outputs do not.
assert.deepEqual(
  pairs.map((p) => p.reference.id),
  ['layer::norm::0', 'layer::attn::0', 'layer::attn::1', 'layer::add::0'],
);
assert.ok(pairs.every((p) => p.reference.id === p.target.id && p.origin === 'automatic'));
// Each anchor is on exactly one pair; the context's anchor marks its own.
for (const anchor of Object.keys(region.mappingPresentation.anchors))
  assert.equal(pairs.filter((p) => p.anchors.includes(anchor)).length, 1, anchor);
assert.deepEqual(
  pairs.filter((p) => p.anchor).map((p) => [p.reference.node, p.anchor]),
  [['add', 'output']],
);
assert.deepEqual(pairs[2].reference.shape, ['1', '4', '4']);
assert.equal(pairs[2].reference.dtype, 'float32');
// A region that pairs its inputs too: the inputs' tensors are pairs as well.
const withInputs = graph('layer', true);
assert.equal(
  context.identicalGraphPairs(withInputs, withInputs, {anchor: 'output'}).length,
  pairs.length + 1,
);
assert.equal(
  context.identicalGraphPairs(region, {...region, id: 'other-runtime'}, {anchor: 'output'}).length,
  0,
  'Different graph identity must not be auto-paired',
);
assert.equal(
  context.identicalGraphPairs(region, {...region, nodes: region.nodes.slice(1)}, {anchor: 'output'})
    .length,
  0,
  'Different topology must not be auto-paired',
);

// Two different graphs pair at the anchors both place, and at the points the server pairs.
const other = graph('other-runtime');
other.mappingPresentation.anchors = {
  attn_out: {outputNode: 'attn', outputPort: '0'},
  output: {outputNode: 'add', outputPort: '0'},
  only_here: {outputNode: 'norm', outputPort: '0'},
};
const anchored = plain(context.anchorGraphPairs(region, other, {anchor: 'output'}));
assert.deepEqual(
  anchored.map((p) => [p.reference.id, p.target.id, p.anchor]),
  [
    ['layer::attn::0', 'other-runtime::attn::0', null],
    ['layer::add::0', 'other-runtime::add::0', 'output'],
  ],
);
other.nodes[2].outputsMetadata[1].attrs.push({key: 'point_id', value: 'p-attn-1'});
region.nodes[2].outputsMetadata[1].attrs.push({key: 'point_id', value: 'p-attn-1-ref'});
context.analysisContext = {model: {pointPairs: {'p-attn-1-ref': 'p-attn-1', 'p-attn': 'p-attn'}}};
const pointed = plain(context.anchorGraphPairs(region, other, {anchor: 'output'}));
assert.deepEqual(
  pointed.map((p) => [p.reference.id, p.target.id]),
  [
    ['layer::attn::0', 'other-runtime::attn::0'],
    ['layer::add::0', 'other-runtime::add::0'],
    ['layer::attn::1', 'other-runtime::attn::1'],
  ],
  'a point pair adds its row once; an anchor already paired is not paired again',
);
console.log(
  'PASS: automatic pairs of identical graphs (anchors, inputs, identity and topology) and of two graphs (anchors and point pairs).',
);
