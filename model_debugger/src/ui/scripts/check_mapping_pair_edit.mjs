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
const source = html.slice(
  html.indexOf('function draftPairInfo(){'),
  html.indexOf('function publishMappingView(){'),
);
const tensor = (id) => ({id, displayLabel: id, node: id.replace(/^[rt]/, 'n'), port: '0'});
const initial = () => [
  {reference: tensor('r1'), target: tensor('t1'), origin: 'automatic', anchors: ['project']},
  {reference: tensor('r2'), target: tensor('t2'), origin: 'automatic', anchors: ['combine']},
  {reference: tensor('r3'), target: tensor('t3'), origin: 'automatic', anchors: []},
];
function setup(side = 'reference', manual = []) {
  const list = initial();
  const posted = [];
  const state = {
    context: {store: 'scope'},
    draftSide: side,
    selected: {reference: tensor('r1'), target: tensor('t2')},
    selectionBeforeDraft: {reference: tensor('r3'), target: tensor('t3')},
    pairs: new Map([['scope', list]]),
    graphs: {reference: {signature: 'decode'}, target: {signature: 'prefill'}},
  };
  const sandbox = {
    pairableMappingNode: (g, n) => !n.id.startsWith('boundary:'),
    mappingMetricPreview: {remove() {}, compute() {}},
    mappingState: state,
    currentMappingPairs: () => [...list, ...manual],
    manualMappingPairs: () => manual,
    renderMappingDetails() {},
    syncMapping() {},
    parent: {postMessage: (message) => posted.push(message)},
    location: {origin: 'http://127.0.0.1'},
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  return {list, state, sandbox, posted};
}
{
  const {list, state, sandbox, posted} = setup();
  const before = JSON.stringify(list);
  sandbox.cancelDraftPair();
  assert.equal(JSON.stringify(list), before);
  assert.equal(state.selected.reference, null);
  assert.equal(state.selected.target, null);
  assert.equal(state.draftSide, null);
  assert.equal(posted.length, 0);
}
for (const side of ['reference', 'target']) {
  const {list, state, sandbox, posted} = setup(side);
  assert.equal(sandbox.draftPairInfo().conflicts, 2);
  const before = JSON.stringify(list);
  sandbox.confirmDraftPair();
  assert.equal(
    JSON.stringify(list),
    before,
    'the server, not the frame, keeps a pair made by hand',
  );
  assert.deepEqual(JSON.parse(JSON.stringify(posted)), [
    {
      channel: 'intermediate-graph-arch',
      type: 'pair-save',
      ref: {signature: 'decode', node: 'n1', output: '0'},
      target: {signature: 'prefill', node: 'n2', output: '0'},
    },
  ]);
  assert.equal(state.draftSide, null);
}
{
  const {list, state, sandbox, posted} = setup();
  state.selected.target = tensor('t1');
  const before = JSON.stringify(list);
  sandbox.confirmDraftPair();
  assert.equal(posted.length, 0, 'an existing pair is not asked for again');
  assert.equal(JSON.stringify(list), before);
}
{
  const {list, state, sandbox, posted} = setup();
  state.selected.target = null;
  const before = JSON.stringify(list);
  sandbox.confirmDraftPair();
  assert.equal(JSON.stringify(list), before, 'a partial draft cannot modify pairs');
  assert.equal(posted.length, 0);
}
{
  const {sandbox, posted, state} = setup();
  assert.equal(sandbox.canResetMappingDefaults(), false, 'automatic pairs are the default');
  sandbox.resetMappingDefaults();
  assert.equal(posted.length, 0);
  const manual = [
    {reference: tensor('r4'), target: tensor('t4'), origin: 'manual', manualId: 'p1'},
    {reference: tensor('r5'), target: tensor('t5'), origin: 'manual', manualId: 'p2'},
  ];
  const edited = setup('reference', manual);
  assert.equal(edited.sandbox.canResetMappingDefaults(), true);
  edited.sandbox.resetMappingDefaults();
  assert.deepEqual(JSON.parse(JSON.stringify(edited.posted)), [
    {channel: 'intermediate-graph-arch', type: 'pair-delete', ids: ['p1', 'p2']},
  ]);
  assert.equal(edited.state.draftSide, null);
  assert.equal(edited.state.selected.reference, null);
  assert.equal(state.pairs.get('scope').length, 3, 'reset leaves the automatic pairs');
}
console.log(
  'PASS: draft cancellation, partial draft, existing pair, a pair asked of the server in both directions, and reset deleting the pairs made by hand.',
);

// Native expand/collapse handles mousedown before our pointerup. Layout changes
// must not turn that control/node gesture into a blank-canvas cancellation.
{
  const pane = fs.readFileSync(
    new URL('../public/graph-arch-preview/mapping-pane.js', import.meta.url),
    'utf8',
  );
  const handlers = {},
    sent = [];
  let geo = {
    rect: {left: 0, top: 0},
    worldTransform: {x: 0, y: 0, scale: 1},
    nodes: [{x: 20, y: 20, width: 60, height: 30}],
  };
  const host = {
    addEventListener(type, fn) {
      handlers[type] = fn;
    },
  };
  const sandbox = {
    $: () => host,
    ready: true,
    e: {readPairGeometry: () => geo},
    send: (message) => sent.push(message),
  };
  vm.createContext(sandbox);
  vm.runInContext(
    pane.slice(pane.indexOf('// A blank click'), pane.indexOf('function resetPaneView')),
    sandbox,
  );
  const event = (x, y, control = false) => ({
    button: 0,
    pointerId: 1,
    clientX: x,
    clientY: y,
    composedPath: () => (control ? [{matches: () => true}] : []),
  });
  const click = (down, up = down, change) => {
    sent.length = 0;
    handlers.pointerdown(down);
    change?.();
    handlers.pointerup(up);
    return sent.length;
  };
  assert.equal(
    click(event(30, 30), event(30, 30), () => {
      geo = {...geo, nodes: []};
    }),
    0,
    'collapse moving a node away preserves tensor selection',
  );
  assert.equal(
    click(event(10, 10, true)),
    0,
    'native control outside node bounds preserves selection',
  );
  assert.equal(click(event(150, 150)), 1, 'a genuine blank click cancels');
  assert.equal(click(event(150, 150), event(180, 150)), 0, 'pan preserves selection');
  sent.length = 0;
  handlers.pointerdown(event(150, 150));
  handlers.pointercancel();
  handlers.pointerup(event(150, 150));
  assert.equal(sent.length, 0, 'cancelled pointer never clears tensor selection');
  console.log('PASS: native controls, layout changes, blank click, pan and pointer cancellation.');
}
