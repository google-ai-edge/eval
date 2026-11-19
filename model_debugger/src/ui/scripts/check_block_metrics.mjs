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

const html = fs.readFileSync(
  new URL('../public/graph-arch-preview/index.html', import.meta.url),
  'utf8',
);

const tag = (point, value, text, bg = '#c9b3f5') => ({
  value,
  text,
  bg,
  fg: 'var(--text)',
  title: `${point} · ${text}`,
  observed: true,
  shape: '',
  source: '',
});

const group = {id: 'attention', name: 'Attention', ops: ['norm', 'attn'], output: 'attn:0'};

const context = {
  NO_STAGE_VALUE: 'No value.',
  state: {layer: 0, selection: null},
  archLayer: {anchors: {}},
  groups: [group],
  closedInfoSections: new Set(),
  escape: (value) => String(value),
  infoIcon: () => '',
  infoValue: (_key, value) => String(value),
  formatShape: (shape) => shape.join('×'),
  ops: {
    norm: {output: ['norm:0']},
    attn: {output: ['attn:0']},
  },
  tensors: {
    'norm:0': {name: 'norm_out', producer: 'norm'},
    'attn:0': {name: 'attn_out', producer: 'attn'},
  },
  analysisContext: null,
};
vm.createContext(context);
const start = html.indexOf('function anchorTag(');
const end = html.indexOf('function blockSummaryWidth(');
assert.ok(start > 0 && end > start, 'Expected blockMetric slice in index.html');
vm.runInContext(html.slice(start, end), context);
const infoStart = html.indexOf('function infoSection(');
const infoEnd = html.indexOf('function infoTable(');
const infoTableEnd = html.indexOf('\n', infoEnd);
assert.ok(infoStart > 0 && infoTableEnd > infoStart, 'Expected infoSection/infoTable in index.html');
vm.runInContext(html.slice(infoStart, infoTableEnd + 1), context);
const detailStart = html.indexOf('function archTensorDetailItem(');
const detailEnd = html.indexOf('function details(');
assert.ok(detailStart > 0 && detailEnd > detailStart, 'Expected archSelectedTensorDetails slice');
vm.runInContext(html.slice(detailStart, detailEnd), context);

// 1. Aggregates observed block-internal points (excluding end anchor decode:39:0).
context.analysisContext = {
  summary: 'max',
  model: {
    signatures: {ref: 'decode'},
    blocks: {
      ref: {
        0: {
          Attention: {
            block: 'Attention',
            by: 'anchors',
            to: 'intermediate.1',
            nodes: ['10', '36', '38', '39'],
          },
        },
      },
    },
    bindings: {
      ref: {
        0: {'intermediate.1': {node: '39', output: '0', point: 'decode:39:0'}},
      },
    },
    points: {
      ref: {
        'decode:10:0': tag('decode:10:0', 0, '0', '#f7f5ff'),
        'decode:36:0': tag('decode:36:0', 5.96e-8, '5.96e-8', '#aa92dd'),
        'decode:38:0': tag('decode:38:0', 0, '0', '#f7f5ff'),
        'decode:39:0': tag('decode:39:0', 0.9, '0.9', '#000000'),
        'decode:43:0': tag('decode:43:0', 9.9, '9.9', '#111111'),
      },
    },
    ops: {},
    notObserved: 'Not observed.',
  },
};

const maxSummary = context.blockMetric(group);
assert.equal(maxSummary.value, 5.96e-8);
assert.equal(maxSummary.text, '5.96e-8');
assert.equal(maxSummary.bg, '#aa92dd');
assert.equal(maxSummary.available, 3);
assert.ok(maxSummary.title.includes('decode:36:0'));

// Selected block summary details show observed internal points without repeating `<point> · `.
context.state.selection = {kind: 'summary', id: 'attention'};
const observedDetails = context.archSelectedTensorDetails();
assert.ok(observedDetails.includes('data-tensor-heading="decode:36:0"'));
assert.ok(observedDetails.includes('>5.96e-8<'));
assert.ok(!observedDetails.includes('decode:36:0 · 5.96e-8'));

context.analysisContext.summary = 'min';
const minSummary = context.blockMetric(group);
assert.equal(minSummary.value, 0);
assert.equal(minSummary.text, '0');
assert.equal(minSummary.bg, '#f7f5ff');

// 1b. Includes formatted operator name in summary title and Tensor Details when present.
context.analysisContext.summary = 'max';
context.analysisContext.model.points.ref['decode:36:0'].operator = 'odml.runtime_bmm';
const namedSummary = context.blockMetric(group);
assert.ok(namedSummary.title.includes('decode:36:0 (odml.runtime_bmm)'));
const operatorDetails = context.archSelectedTensorDetails();
assert.ok(operatorDetails.includes('data-tensor-heading="decode:36:0 · odml.runtime_bmm"'));
assert.ok(operatorDetails.includes('Operator'));
assert.ok(operatorDetails.includes('>odml.runtime_bmm<'));
assert.ok(operatorDetails.includes('data-tensor-heading="decode:10:0"'));

// 1c. Layer-fallback blocks (`by: 'layer'`) must not aggregate layer-wide points.
context.analysisContext.model.blocks.ref[0].Attention.by = 'layer';
const fallbackBlock = context.blockMetric(group);
assert.equal(fallbackBlock.value, null);
assert.equal(fallbackBlock.text, '—');
context.analysisContext.model.blocks.ref[0].Attention.by = 'anchors';

// 2. When only the end anchor is observed, internal summary stays empty and details fall back.
context.analysisContext.model.points.ref = {
  'decode:39:0': tag('decode:39:0', 0.9, '0.9', '#000000'),
};
const emptySummary = context.blockMetric(group);
assert.equal(emptySummary.value, null);
assert.equal(emptySummary.text, '—');
const fallbackDetails = context.archSelectedTensorDetails();
assert.ok(fallbackDetails.includes('data-tensor-heading="norm_out"'));

// 3. Cross-runtime paired ops (model.ops[layer]) are included when ref points aren't node-keyed.
context.analysisContext.summary = 'max';
context.analysisContext.model.ops = {
  0: {norm: tag('layers.0.norm', 1.25e-5, '1.25e-5', '#b9a2e8')},
};
const opSummary = context.blockMetric(group);
assert.equal(opSummary.value, 1.25e-5);
assert.equal(opSummary.text, '1.25e-5');
assert.equal(opSummary.available, 1);

// 4. Composite nodes in mapping-pane.js show their composite operator name.
const paneJs = fs.readFileSync(
  new URL('../public/graph-arch-preview/mapping-pane.js', import.meta.url),
  'utf8',
);
const labelStart = paneJs.indexOf('function nodeLabel(');
const labelEnd = paneJs.indexOf('function prepare(');
assert.ok(labelStart > 0 && labelEnd > labelStart, 'Expected nodeLabel in mapping-pane.js');
const paneContext = {};
vm.createContext(paneContext);
vm.runInContext(paneJs.slice(labelStart, labelEnd), paneContext);
assert.equal(
  paneContext.nodeLabel({
    label: 'stablehlo_composite',
    attrs: [{key: 'name', value: '#vhlo.string_v1<"odml.runtime_bmm">'}],
  }),
  'odml.runtime_bmm',
);
assert.equal(
  paneContext.nodeLabel({
    label: 'stablehlo_composite',
    attrs: [{key: 'name', value: 'odml.rms_norm'}],
  }),
  'odml.rms_norm',
);
assert.equal(paneContext.nodeLabel({label: 'softmax', attrs: []}), 'softmax');

console.log(
  'Layer Arch block summaries: min/max, end-anchor exclusion, empty, op-tag fallback and composite names passed.',
);
