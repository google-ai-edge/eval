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

import {signal} from '@angular/core';
import {TestBed, type ComponentFixture} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import type {
  PositionTensorRequest,
  PositionTensorResponse,
  PositionTensorRow,
} from '../../server/contracts/position_analysis';
import {PositionAnalysisApi} from '../../server/position_analysis_api';
import {CaptureStore} from '../../server/capture_store';
import type {PlotHost, PlotlyApi} from '../../widgets/plotly_loader';
import {ChannelChart} from '../../widgets/charts/channel_chart';
import {ScatterChart} from '../../widgets/charts/scatter_chart';
import {TensorHeatmap} from '../../widgets/charts/tensor_heatmap';
import {TensorIndexControl} from '../../widgets/charts/tensor_index_control';
import {
  LayerActivationInspection,
  type LayerActivationInspectionState,
} from './layer_activation_inspection';
const NONE = {
  min: null,
  max: null,
  mean: null,
  min_position: null,
  max_position: null,
  valid_count: 0,
};
/** The server's answer for a tensor whose sides' elements `ref` and `target` give, or null for a
 *  side without one; the deltas only where it compared them. */
const vector = (
  query: PositionTensorRequest,
  total = 320,
  sides: {ref: ((i: number) => number) | null; target: ((i: number) => number) | null} = {
    ref: (i) => i / 10,
    target: (i) => i / 10 + 1,
  },
  comparable = sides.ref !== null && sides.target !== null,
): PositionTensorResponse => {
  const source = (dtype: string) => ({
    point: 'decode:3:0',
    signature: 'decode',
    step: query.position + 1,
    row: 0,
    tensor_name: 'tiny/layer_0/out',
    shape: [total],
    source_shape: [1, 1, total],
    dtype,
    comparison_dtype: 'float64',
  });
  return {
    turn: 2,
    stage: 'decode',
    alignment: 'content',
    position: query.position,
    layer: query.layer,
    anchor: query.anchor,
    reference_position: query.position - 1,
    target_position: query.position,
    reference: sides.ref ? source('F16') : null,
    target: sides.target ? source('F32') : null,
    comparable,
    channels: Array.from({length: total}, (_, channel) => {
      const ref = sides.ref ? sides.ref(channel) : null;
      const target = sides.target ? sides.target(channel) : null;
      const delta = comparable && ref !== null && target !== null ? target - ref : null;
      return {
        channel,
        ref,
        target,
        delta,
        abs_delta: delta === null ? null : Math.abs(delta),
        status: delta === null ? 'unavailable' : 'available',
      };
    }),
    statistics: {ref: NONE, target: NONE, delta: NONE, abs_delta: NONE},
    metrics: {},
    status: comparable ? 'ok' : 'unavailable',
    reasons: [],
    provenance: {kind: 'captured', label: 'Test data'},
  };
};
type Read = (
  capture: string | null,
  turn: number,
  phase: string,
  query: PositionTensorRequest,
  signal: AbortSignal,
) => Promise<PositionTensorResponse>;
describe('LayerActivationInspection', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        disconnect() {}
      },
    );
    vi.stubGlobal('matchMedia', () => ({
      matches: true,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
    }));
    vi.spyOn(document, 'createRange').mockImplementation(
      () =>
        ({
          getBoundingClientRect: () => ({width: 0}),
          selectNodeContents() {},
          detach() {},
        }) as unknown as Range,
    );
    (window as Window & {Plotly?: PlotlyApi}).Plotly = {
      react: vi.fn(async (host: HTMLElement) => {
        (host as PlotHost).on = () => {};
      }),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
  });
  afterEach(() => {
    delete (window as Window & {Plotly?: PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  function create(read?: Read, initial: LayerActivationInspectionState | null = null) {
    const api = {
      positionTensor: vi.fn(read ?? (async (_capture, _turn, _phase, query) => vector(query))),
    };
    const captureId = signal<string | null>('capture');
    const revision = signal(0);
    TestBed.configureTestingModule({
      providers: [
        {provide: PositionAnalysisApi, useValue: api},
        {provide: CaptureStore, useValue: {captureId, revision}},
      ],
    });
    const fixture = TestBed.createComponent(LayerActivationInspection);
    fixture.componentRef.setInput('turn', 2);
    fixture.componentRef.setInput('stage', 'decode');
    fixture.componentRef.setInput('fullRange', {start: 32, end: 67});
    fixture.componentRef.setInput('layers', [0, 2, 7]);
    fixture.componentRef.setInput('anchors', ['output', 'pre_attention_norm', 'attention_output']);
    fixture.componentRef.setInput('initial', initial);
    fixture.componentRef.setInput('alignment', {
      stage: 'decode',
      alignment: 'content',
      start: 32,
      pairs: [
        [31, 32],
        [null, 33],
      ],
    });
    fixture.detectChanges();
    return {fixture, view: fixture.componentInstance, api};
  }
  async function settle(fixture: ComponentFixture<LayerActivationInspection>) {
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    TestBed.tick();
  }
  it('requests the exact aligned tensor, renders complete vectors, and switches inspection plots', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    expect(api.positionTensor).toHaveBeenCalledWith(
      'capture',
      2,
      'decode',
      {position: 32, layer: 0, anchor: 'output'},
      expect.any(AbortSignal),
      {
        stage: 'decode',
        alignment: 'content',
        start: 32,
        pairs: [
          [31, 32],
          [null, 33],
        ],
      },
    );
    expect(view.channels()).toHaveLength(320);
    expect(view.channels()[319].ref).toBe(31.9);
    expect(view.data()?.reference?.dtype).toBe('F16');
    expect(view.notices()).toEqual([]);
    expect(fixture.nativeElement.textContent).not.toContain('Test data');
    expect(fixture.debugElement.query(By.directive(ChannelChart))).not.toBeNull();
    const compare = vi.fn();
    view.compare.subscribe(compare);
    view.chooseView('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    expect(fixture.debugElement.query(By.directive(ScatterChart))).not.toBeNull();
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]').disabled).toBe(true);
    view.chooseView('Tensor Comparison · Heatmap');
    expect(compare).toHaveBeenCalledWith('heatmap');
    fixture.destroy();
  });
  it('shows |Target − Ref| as a heatmap of the compared shape in the Difference view', async () => {
    const {fixture, view} = create();
    await settle(fixture);

    view.chooseView('Tensor Inspection · Difference');
    await settle(fixture);

    expect(view.plot()).toBe('delta');
    expect(view.visualizationLabel()).toBe('Tensor Inspection · Difference');
    expect(view.captureView().plot).toBe('delta');
    expect(fixture.debugElement.query(By.directive(ChannelChart))).toBeNull();
    const heatmap = fixture.debugElement.query(By.directive(TensorHeatmap))
      .componentInstance as TensorHeatmap;
    expect(heatmap.rows()).toHaveLength(320);
    expect(heatmap.shape()).toEqual([320]);
    expect(heatmap.differing()).toBe(320);
    expect(fixture.nativeElement.textContent).toContain('320 of 320 elements differ');
    expect(fixture.nativeElement.querySelector('.tensor-line-canvas.difference')).not.toBeNull();
    expect(fixture.nativeElement.textContent).toContain('Side bars');
    expect(fixture.nativeElement.querySelector('[aria-label="Display"]').disabled).toBe(true);
    const reset = fixture.nativeElement.querySelector('[aria-label="Reset view"]');
    expect(reset.disabled).toBe(true);

    heatmap.zoom.set({r0: 0, r1: 1, c0: 0, c1: 128});
    fixture.detectChanges();
    expect(reset.disabled).toBe(false);

    reset.click();
    fixture.detectChanges();
    expect(heatmap.zoom()).toBeNull();

    view.setSideBars('Max |Δ|');
    fixture.detectChanges();
    expect(heatmap.sideBars()).toBe('max');

    view.chooseView('Tensor Inspection · Line chart');
    fixture.detectChanges();
    expect(fixture.debugElement.query(By.directive(TensorHeatmap))).toBeNull();
    expect(fixture.debugElement.query(By.directive(ChannelChart))).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.tensor-line-canvas.difference')).toBeNull();

    view.chooseView('Tensor Inspection · Scatter plot');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.tensor-line-canvas')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.tensor-line-canvas.difference')).toBeNull();
    fixture.destroy();
  });
  it('has no Difference heatmap where the server could not compare the sides', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector(query, 320, {ref: (i) => i, target: (i) => i}, false),
    );
    await settle(fixture);
    view.chooseView('Tensor Inspection · Difference');
    fixture.detectChanges();
    expect(view.deltaShape()).toBeNull();
    expect(fixture.debugElement.query(By.directive(TensorHeatmap))).toBeNull();
    expect(fixture.nativeElement.textContent).toContain(
      'Matching Reference and Target tensor shapes are required for a paired plot.',
    );
    fixture.destroy();
  });
  it('restores a saved Difference view', async () => {
    const {fixture, view} = create(undefined, {plot: 'delta', position: 33, layer: 2});
    await settle(fixture);
    expect(view.plot()).toBe('delta');
    fixture.destroy();
  });
  it('restores valid view state, resolves stale coordinates, and preserves zoom when changing tensors', async () => {
    const {fixture, view} = create(undefined, {
      plot: 'index',
      position: 999,
      layer: 3,
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: {channel: 15},
      reduction: 'max',
    });
    await settle(fixture);
    expect(view.position()).toBe(66);
    expect(view.layer()).toBe(0);
    expect(view.anchor()).toBe('output');
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.selectedIndex()).toBe(15);
    expect(view.reduction()).toBe('max');
    view.setLayer(7);
    await settle(fixture);
    expect(view.layer()).toBe(7);
    expect(view.selectedIndex()).toBeNull();
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.captureView().valueRange).toEqual({start: -2, end: 2});
    view.setPosition(-100);
    await settle(fixture);
    expect(view.position()).toBe(66);
    fixture.destroy();
  });
  it('keeps display changes in drafts and disables bins at individual-element scale', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    view.toggleDisplay();
    view.setDisplayDraft('Maximum');
    expect(view.displayDirty()).toBe(true);
    expect(view.reduction()).toBe('mean');
    view.displayOpen.set(false);
    view.toggleDisplay();
    expect(view.displayDraft()).toBe('mean');
    view.setDisplayDraft('Minimum');
    view.applyDisplay();
    expect(view.reduction()).toBe('min');
    view.zoomElements({start: 1, end: 2});
    expect(view.displayBinsDisabled()).toBe(true);
    view.setDisplayDraft('Mean');
    expect(view.reduction()).toBe('min');
    view.chooseView('Tensor Inspection · Scatter plot');
    view.toggleDisplay();
    expect(view.displayOpen()).toBe(false);
    fixture.destroy();
  });
  it('restores a sync point and changes tensors without losing position, layer, or zoom', async () => {
    const {fixture, view, api} = create(undefined, {
      plot: 'index',
      position: 40,
      layer: 7,
      anchor: 'attention_output',
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: {channel: 15},
    });
    await settle(fixture);
    expect(api.positionTensor.mock.calls.at(-1)?.[3]).toMatchObject({
      position: 40,
      layer: 7,
      anchor: 'attention_output',
    });
    expect(view.selectedIndex()).toBe(15);
    view.channelSelection.set({start: 10, end: 20, plane: null});
    view.setAnchor('pre_attention_norm');
    expect(view.selectedIndex()).toBeNull();
    expect(view.channelSelection()).toBeNull();
    await settle(fixture);
    expect(view.captureView()).toMatchObject({
      position: 40,
      layer: 7,
      anchor: 'pre_attention_norm',
      elementRange: {start: 10, end: 30},
      valueRange: {start: -2, end: 2},
      selection: null,
    });
    expect(api.positionTensor.mock.calls.at(-1)?.[3].anchor).toBe('pre_attention_norm');
    const requestCount = api.positionTensor.mock.calls.length;
    view.setAnchor('unknown');
    await settle(fixture);
    expect(view.anchor()).toBe('pre_attention_norm');
    expect(api.positionTensor).toHaveBeenCalledTimes(requestCount);
    fixture.destroy();
  });
  it('offers each layer its own points, maps a point across layers, and labels it briefly', async () => {
    const {fixture, view, api} = create(undefined, {
      plot: 'index',
      position: 32,
      layer: 0,
      anchor: 'decode:36:0',
    });
    const name = (layer: number, op: string) =>
      `litert_torch.generative.Qwen3/Qwen3DecoderLayer_${layer}/litert_torch.attn.${op}`;
    const info = (layer: number, op: string) => ({
      layers: [layer],
      tensorName: name(layer, op),
      signature: 'decode',
    });
    fixture.componentRef.setInput('anchors', [
      'output',
      'decode:36:0',
      'decode:40:0',
      'decode:73:0',
      'decode:300:0',
    ]);
    fixture.componentRef.setInput('anchorLabels', {
      output: 'Layer output',
      'decode:36:0': `${name(0, 'FusedQwen3Attention_self_attn;4')} · decode:36:0`,
    });
    fixture.componentRef.setInput('anchorPoints', {
      output: {layers: [0, 2, 7], tensorName: name(0, 'add;1'), signature: 'decode'},
      'decode:36:0': info(0, 'FusedQwen3Attention_self_attn;4'),
      'decode:40:0': info(0, 'Linear_o_proj;0'),
      'decode:73:0': info(2, 'FusedQwen3Attention_self_attn;4'),
      'decode:300:0': info(2, 'Linear_down_proj;0'),
    });
    await settle(fixture);

    const syncPoint = () => view.tensorIndexDimensions()[1];
    expect(syncPoint().options).toEqual(['output', 'decode:36:0', 'decode:40:0']);
    expect(syncPoint().optionLabels?.['decode:36:0']).toBe(
      'FusedQwen3Attention_self_attn · decode:36:0',
    );
    expect(syncPoint().optionLabels?.['output']).toBe('Layer output');
    expect(syncPoint().optionDescriptions?.['decode:40:0']).toBe(name(0, 'Linear_o_proj;0'));
    expect(syncPoint().optionDescriptions?.['output']).toBeUndefined();

    // Layer 2 has no point of layer 0: the same operator there takes its place.
    view.setLayer(2);
    await settle(fixture);

    expect(syncPoint().options).toEqual(['output', 'decode:73:0', 'decode:300:0']);
    expect(view.anchor()).toBe('decode:73:0');
    expect(api.positionTensor.mock.calls.at(-1)?.[3]).toMatchObject({
      layer: 2,
      anchor: 'decode:73:0',
    });

    view.setAnchor('decode:36:0');
    expect(view.anchor()).toBe('decode:73:0');

    // Layer 0 has no down_proj point: back to the layer output.
    view.setAnchor('decode:300:0');
    view.setLayer(0);
    await settle(fixture);

    expect(view.anchor()).toBe('output');

    // An anchor of every layer stays when the layer changes.
    view.setLayer(7);
    await settle(fixture);

    expect(view.anchor()).toBe('output');
    expect(syncPoint().options).toEqual(['output']);
    fixture.destroy();
  });
  it('keeps an unavailable saved sync point instead of substituting a different tensor', async () => {
    const {fixture, view, api} = create(
      async (_capture, _turn, _phase, query) => ({
        ...vector(query, 0, {ref: null, target: null}),
        reasons: [
          {code: 'tensor_not_captured', side: 'ref', text: 'Not captured.'},
          {code: 'tensor_not_captured', side: 'target', text: 'Not captured.'},
        ],
      }),
      {plot: 'index', position: 32, layer: 0, anchor: 'missing_capture_point'},
    );
    await settle(fixture);
    expect(api.positionTensor.mock.calls.at(-1)?.[3].anchor).toBe('missing_capture_point');
    expect(view.anchor()).toBe('missing_capture_point');
    expect(view.tensorIndexDimensions()[1].value).toBe('missing_capture_point');
    expect(view.channels()).toEqual([]);
    expect(view.notices()).toHaveLength(2);
    view.setLayer(7);
    await settle(fixture);
    expect(view.captureView().anchor).toBe('missing_capture_point');
    fixture.destroy();
  });
  it('renders a missing side independently and suppresses paired differences and scatter', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) => ({
      ...vector(query, 3, {ref: null, target: (i) => i / 10 + 1}),
      reference_position: null,
      reasons: [
        {code: 'tensor_position_missing', side: 'ref', text: 'The side has no token here.'},
      ],
    }));
    await settle(fixture);
    expect(view.channels().map((row) => row.ref)).toEqual([null, null, null]);
    expect(view.channels().map((row) => row.target)).toEqual([1, 1.1, 1.2]);
    expect(view.channels().every((row) => row.delta === null)).toBe(true);
    expect(view.compatible()).toBe(false);
    expect(view.scatterChannels()).toEqual([]);
    // The server's words, after the side they are of.
    expect(view.notices()).toEqual(['ref: The side has no token here.']);
    fixture.destroy();
  });
  it('shows both sides without deltas where the server could not compare them', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector(query, 3, {ref: (i) => i, target: (i) => i + 1}, false),
    );
    await settle(fixture);
    expect(view.channels().map((row) => row.ref)).toEqual([0, 1, 2]);
    expect(view.compatible()).toBe(false);
    expect(view.channels().every((row) => row.delta === null)).toBe(true);
    fixture.destroy();
  });
  it('cancels outdated requests, clears stale values immediately, and retries errors', async () => {
    const requests: {
      query: PositionTensorRequest;
      signal: AbortSignal;
      resolve: (value: PositionTensorResponse) => void;
      reject: (error: Error) => void;
    }[] = [];
    const {fixture, view} = create(
      async (_capture, _turn, _phase, query, signal) =>
        new Promise<PositionTensorResponse>((resolve, reject) =>
          requests.push({query, signal, resolve, reject}),
        ),
    );
    expect(view.loading()).toBe(true);
    requests[0].resolve(vector(requests[0].query, 3));
    await settle(fixture);
    expect(view.total()).toBe(3);
    view.setPosition(33);
    fixture.detectChanges();
    expect(requests[0].signal.aborted).toBe(true);
    expect(view.data()).toBeNull();
    expect(view.channels()).toEqual([]);
    view.setPosition(34);
    fixture.detectChanges();
    expect(requests[1].signal.aborted).toBe(true);
    requests[1].resolve(vector(requests[1].query, 20));
    requests[2].reject(new Error('Tensor missing'));
    await settle(fixture);
    expect(view.error()).toBe('Tensor missing');
    expect(view.total()).toBe(0);
    view.retry();
    fixture.detectChanges();
    expect(view.error()).toBe('');
    expect(view.loading()).toBe(true);
    requests[3].resolve(vector(requests[3].query, 4));
    await settle(fixture);
    expect(view.total()).toBe(4);
    fixture.destroy();
    expect(requests[3].signal.aborted).toBe(true);
  });
  it('rejects a response for another tensor identity', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector({...query, layer: 99}),
    );
    await settle(fixture);
    expect(view.error()).toContain('did not match');
    expect(view.data()).toBeNull();
    fixture.destroy();
  });
  it('rejects a response from another sync point', async () => {
    const {fixture, view} = create(async (_capture, _turn, _phase, query) =>
      vector({...query, anchor: 'attention_output'}),
    );
    await settle(fixture);
    expect(view.error()).toContain('sync point');
    expect(view.data()).toBeNull();
    fixture.destroy();
  });
  it('ignores a late response for the previous sync point without taking details focus', async () => {
    const requests: {
      query: PositionTensorRequest;
      signal: AbortSignal;
      resolve: (value: PositionTensorResponse) => void;
    }[] = [];
    const {fixture, view} = create(
      async (_capture, _turn, _phase, query, signal) =>
        new Promise<PositionTensorResponse>((resolve) => requests.push({query, signal, resolve})),
    );
    const inspect = vi.fn();
    view.inspect.subscribe(inspect);
    view.setAnchor('attention_output', false);
    fixture.detectChanges();
    expect(requests[0].signal.aborted).toBe(true);
    expect(requests[1].query.anchor).toBe('attention_output');
    requests[1].resolve(vector(requests[1].query, 4));
    await settle(fixture);
    requests[0].resolve(vector(requests[0].query, 20));
    await settle(fixture);
    expect(view.data()?.anchor).toBe('attention_output');
    expect(view.total()).toBe(4);
    expect(inspect).not.toHaveBeenCalled();
    fixture.destroy();
  });
  it('uses the shared Tensor Index picker and rejects invalid position edits like KV', async () => {
    const {fixture, view, api} = create();
    await settle(fixture);
    const picker = fixture.debugElement.query(By.directive(TensorIndexControl))
      .componentInstance as TensorIndexControl;
    expect(picker.summary()).toBe('[Layer: 0, Sync point: Layer output]');
    expect(picker.dimensions().map((dimension) => dimension.id)).toEqual(['layer', 'anchor']);
    picker.choose('layer', '7');
    await settle(fixture);
    expect(view.layer()).toBe(7);
    expect(picker.summary()).toBe('[Layer: 7, Sync point: Layer output]');
    picker.choose('anchor', 'attention_output');
    await settle(fixture);
    expect(view.anchor()).toBe('attention_output');
    expect(picker.summary()).toBe('[Layer: 7, Sync point: attention_output]');
    const input = fixture.nativeElement.querySelector(
      '[aria-label="Token Position"]',
    ) as HTMLInputElement;
    const requestCount = api.positionTensor.mock.calls.length;
    for (const value of ['', '31', '67', '32.5']) {
      input.value = value;
      input.dispatchEvent(new Event('change'));
      await settle(fixture);
      expect(input.value).toBe('32');
      expect(view.position()).toBe(32);
    }
    expect(api.positionTensor).toHaveBeenCalledTimes(requestCount);
    input.value = '40';
    input.dispatchEvent(new Event('change'));
    await settle(fixture);
    expect(view.position()).toBe(40);
    (
      fixture.nativeElement.querySelector(
        '[aria-label="Next tensor position"]',
      ) as HTMLButtonElement
    ).click();
    await settle(fixture);
    expect(view.position()).toBe(41);
    fixture.destroy();
  });
  it('applies runtime visibility only on Apply and retains it across plots and saved view restoration', async () => {
    const {fixture, view} = create(undefined, {
      plot: 'index',
      position: 32,
      layer: 0,
      visibleRuntimes: ['target'],
    });
    await settle(fixture);
    view.showElement(15);
    view.zoomElements({start: 10, end: 30});
    view.toggleDisplay();
    expect(view.displayBinsDisabled()).toBe(true);
    view.toggleDisplayRuntime('ref');
    expect(view.displayDirty()).toBe(true);
    expect(view.visibleRuntimes()).toEqual(['target']);
    view.displayOpen.set(false);
    view.toggleDisplay();
    expect(view.displayRuntimes()).toEqual(['target']);
    view.displayRuntimes.set([]);
    view.applyDisplay();
    fixture.detectChanges();
    expect(view.visibleRuntimes()).toEqual([]);
    expect(view.displayDirty()).toBe(false);
    expect(view.captureView().visibleRuntimes).toEqual([]);
    const chart = fixture.debugElement.query(By.directive(ChannelChart))
      .componentInstance as ChannelChart;
    expect(chart.visibleRuntimes()).toEqual([]);
    expect(fixture.nativeElement.querySelector('runtime-legend').textContent.trim()).toBe('');
    view.chooseView('Tensor Inspection · Scatter plot');
    view.chooseView('Tensor Inspection · Line chart');
    expect(view.visibleRuntimes()).toEqual([]);
    expect(view.selectedIndex()).toBe(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    view.toggleDisplay();
    view.displayRuntimes.set(['ref', 'target']);
    view.applyDisplay();
    expect(view.captureView().visibleRuntimes).toEqual(['ref', 'target']);
    fixture.destroy();
  });
  it('drops only multi-element bin selection on scatter and keeps the representative element and both zoom ranges', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    view.showElement(12);
    view.channelSelection.set({start: 10, end: 14, plane: null});
    view.zoomElements({start: 10, end: 100});
    view.zoomValues({start: -1, end: 2});
    view.chooseView('Tensor Inspection · Scatter plot');
    expect(view.channelSelection()).toBeNull();
    expect(view.selectedIndex()).toBe(12);
    expect(view.valueRange()).toEqual({start: -1, end: 2});
    view.chooseView('Tensor Inspection · Line chart');
    expect(view.elementRange()).toEqual({start: 10, end: 100});
    view.channelSelection.set({start: 12, end: 13, plane: null});
    view.chooseView('Tensor Inspection · Scatter plot');
    expect(view.channelSelection()).toEqual({start: 12, end: 13, plane: null});
    fixture.destroy();
  });
  it('preserves visible individual-element zoom during location and emits background inspection to its parent', async () => {
    const {fixture, view} = create();
    await settle(fixture);
    const chart = fixture.debugElement.query(By.directive(ChannelChart));
    const locate = vi
      .spyOn(chart.componentInstance as ChannelChart, 'locateKind')
      .mockImplementation(() => {});
    view.zoomElements({start: 10, end: 30});
    view.locateElement(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    expect(view.selectedIndex()).toBe(15);
    view.locateTensor();
    expect(view.selectedIndex()).toBe(15);
    expect(view.elementRange()).toEqual({start: 10, end: 30});
    view.locateElement(100);
    expect(view.elementRange()).toEqual({start: 90, end: 110});
    expect(locate).toHaveBeenCalledWith('key');
    const background = vi.fn();
    view.inspectStage.subscribe(background);
    const event = new MouseEvent('click');
    chart.triggerEventHandler('backgroundClick', event);
    expect(background).toHaveBeenCalledWith(event);
    expect(view.selectedIndex()).toBe(100);
    fixture.destroy();
  });
  it('reports asynchronous data changes without taking inspection focus from the active panel', async () => {
    let resolve!: (data: PositionTensorResponse) => void;
    let query!: PositionTensorRequest;
    const {fixture, view} = create(async (_capture, _turn, _phase, request) => {
      query = request;
      return new Promise((done) => {
        resolve = done;
      });
    });
    const inspect = vi.fn();
    const stateChange = vi.fn();
    view.inspect.subscribe(inspect);
    view.stateChange.subscribe(stateChange);
    resolve(vector(query));
    await settle(fixture);
    expect(stateChange).toHaveBeenCalledWith(view);
    expect(inspect).not.toHaveBeenCalled();
    view.showElement(4);
    expect(inspect).toHaveBeenCalledOnce();
    inspect.mockClear();
    view.zoomElements({start: 0, end: 20});
    view.toggleDisplay();
    view.toggleDisplayRuntime('ref');
    view.applyDisplay();
    await settle(fixture);
    expect(inspect).not.toHaveBeenCalled();
    view.setPosition(33, false);
    expect(inspect).not.toHaveBeenCalled();
    view.setPosition(34);
    expect(inspect).toHaveBeenCalledOnce();
    fixture.destroy();
  });
  it('shows each sync point max |Δ| at the active position and layer and highlights non-zero entries', async () => {
    const analysisRow = (
      position: number,
      layer: number,
      anchor: string,
      maxAbs: number | null,
    ): PositionTensorRow => ({
      position,
      layer,
      anchor,
      status: maxAbs === null ? 'unavailable' : 'ok',
      reasons: [],
      shape: [8],
      metrics: {max_abs: {value: maxAbs, reasons: []}},
    });
    const {fixture, view} = create(undefined, {
      plot: 'delta',
      position: 32,
      layer: 0,
      anchor: 'output',
    });
    const name = (layer: number, op: string) =>
      `litert_torch.generative.Qwen3/Qwen3DecoderLayer_${layer}/litert_torch.attn.${op}`;
    fixture.componentRef.setInput('anchors', ['output', 'decode:36:0', 'decode:40:0']);
    fixture.componentRef.setInput('anchorLabels', {
      output: 'Layer output',
      'decode:36:0': `${name(0, 'FusedQwen3Attention_self_attn;23')} · decode:36:0`,
      'decode:40:0': `${name(0, 'Linear_o_proj;2')} · decode:40:0`,
    });
    fixture.componentRef.setInput('anchorPoints', {
      output: {layers: [0, 1], tensorName: name(0, 'add;1'), signature: 'decode'},
      'decode:36:0': {
        layers: [0],
        tensorName: name(0, 'FusedQwen3Attention_self_attn;23'),
        signature: 'decode',
      },
      'decode:40:0': {layers: [0], tensorName: name(0, 'Linear_o_proj;2'), signature: 'decode'},
    });
    fixture.componentRef.setInput('analysisRows', [
      analysisRow(32, 0, 'output', 0),
      analysisRow(32, 0, 'decode:36:0', 5.960464477539063e-8),
      analysisRow(32, 1, 'output', 1.5),
      analysisRow(33, 0, 'output', 0.0025),
      analysisRow(33, 0, 'decode:36:0', 0),
      analysisRow(33, 0, 'decode:40:0', 0),
    ]);
    await settle(fixture);

    expect(view.tensorIndexDimensions()[1].optionBadges).toEqual({
      output: '0',
      'decode:36:0': '5.96e-8',
      'decode:40:0': '—',
    });
    expect(view.tensorIndexDimensions()[1].highlightedOptions).toEqual(['decode:36:0']);

    view.setPosition(33);
    await settle(fixture);

    expect(view.tensorIndexDimensions()[1].optionBadges).toEqual({
      output: '0.0025',
      'decode:36:0': '0',
      'decode:40:0': '0',
    });
    expect(view.tensorIndexDimensions()[1].highlightedOptions).toEqual(['output']);
    fixture.destroy();
  });
});
