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

const navStart = html.indexOf('const pageKey=');
const navEnd = html.indexOf('function renderNavigation(){');
assert.ok(navStart > 0 && navEnd > navStart, 'Expected navigation slice in index.html');

const camStart = html.indexOf('const MIN_ZOOM=.1,MAX_ZOOM=4;');
const camEnd = html.indexOf('function updateLayerExpansion(');
assert.ok(camStart > 0 && camEnd > camStart, 'Expected fit/zoom slice in index.html');

let viewportRect = {width: 960, height: 640};
let graphBox = {x: 80, y: 20, width: 640, height: 1120};
let worldExists = true;
const elements = {
  viewport: {
    getBoundingClientRect: () => ({...viewportRect}),
    focus() {},
  },
  'graph-world': {
    getBBox: () => ({...graphBox}),
    setAttribute() {},
  },
  'pair-view': {hidden: true},
  'graph-search': {value: ''},
  stage: {style: {setProperty() {}}},
  reset: {textContent: ''},
  out: {disabled: false},
  in: {disabled: false},
};

const context = {
  $: (id) => (id === 'graph-world' && !worldExists ? null : elements[id]),
  pageViews: new Map(),
  overviewState: {page: 'model', index: 0},
  modelTree: null,
  layerCount: () => 28,
  closeGraphPopovers() {},
  hideIconTooltip() {},
  activateLayer() {},
  searchGraph() {},
  draw() {
    worldExists = context.state.page === 'layer';
  },
  details() {},
  state: {
    page: 'message',
    layer: 0,
    hiddenIo: new Set(),
    selection: null,
    collapsed: new Set(['attention', 'ffn', 'ple']),
    scope: null,
    zoom: 1,
    x: 0,
    y: 0,
    autoFit: true,
  },
};
vm.createContext(context);
vm.runInContext(html.slice(navStart, navEnd), context);
vm.runInContext(html.slice(camStart, camEnd), context);

// 1. Regression: loading `/architecture` after `'message'` state must mount `#graph-world`
// in `draw()` and fit Layer 0 instead of poisoning `pageViews['layer:0']` with `{zoom: 1, x: 0, y: 0}`.
worldExists = false;
context.navigatePage('layer', 0);
assert.equal(
  context.pageViews.has('layer:0'),
  false,
  'Transition from message state must not record a fake layer:0 camera view',
);
assert.equal(context.state.autoFit, true);
const initialCamera = {zoom: context.state.zoom, x: context.state.x, y: context.state.y};
assert.ok(
  initialCamera.zoom < 1,
  `Expected Layer 0 to scale to fit, got zoom=${initialCamera.zoom}`,
);
assert.equal(context.fit(), true);
assert.deepEqual(initialCamera, {
  zoom: context.state.zoom,
  x: context.state.x,
  y: context.state.y,
});

// 1b. Non-message page without a mounted `#graph-world` also must not poison `pageViews`.
context.pageViews.clear();
worldExists = false;
context.state.page = 'model';
context.state.zoom = 1;
context.state.x = 0;
context.state.y = 0;
context.navigatePage('layer', 0);
assert.equal(
  context.pageViews.has('model'),
  false,
  'Unmounted #graph-world must not record a fake model camera view',
);

// 2. If viewport has 0x0 size on mount, `fit()` does not clamp to MIN_ZOOM and stays `autoFit`.
context.pageViews.clear();
context.state.page = 'message';
context.state.layer = 0;
context.state.zoom = 1;
context.state.x = 0;
context.state.y = 0;
viewportRect = {width: 0, height: 0};
context.navigatePage('layer', 0);
assert.equal(context.state.zoom, 1, '0x0 viewport must not overwrite zoom with MIN_ZOOM');
assert.equal(context.state.autoFit, true);
viewportRect = {width: 960, height: 640};
assert.equal(context.fit(), true);
assert.deepEqual({zoom: context.state.zoom, x: context.state.x, y: context.state.y}, initialCamera);

// 3. Async metric arrival widens graph while `autoFit` is true; manual zoom disables `autoFit`
// and is preserved across layer navigation until explicit `fit()` (`Reset graph view`).
graphBox = {x: 80, y: 20, width: 780, height: 1120};
if (context.state.autoFit) {
  context.fit();
}
const widenedCamera = {zoom: context.state.zoom, x: context.state.x, y: context.state.y};
assert.ok(
  widenedCamera.x < initialCamera.x,
  'Wider graph with badges should re-center horizontally',
);

context.setZoom(1.4);
assert.equal(context.state.autoFit, false, 'Manual zoom must disable autoFit');
context.navigatePage('layer', 1);
context.navigatePage('layer', 0);
assert.equal(context.state.zoom, 1.4, 'Manual zoom on layer 0 should be preserved across switch');
assert.equal(context.state.autoFit, false);

context.fit();
assert.equal(context.state.autoFit, true, 'Explicit Reset view re-enables autoFit');
assert.deepEqual({zoom: context.state.zoom, x: context.state.x, y: context.state.y}, widenedCamera);

console.log('Graph Diff arch initial fit regression checks passed.');
