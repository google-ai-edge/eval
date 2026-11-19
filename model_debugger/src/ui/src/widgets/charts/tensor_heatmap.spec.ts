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

import {TestBed} from '@angular/core/testing';
import type {KvHeadChannel} from '../../server/contracts/kv';
import * as plotly from '../plotly_loader';
import {TensorHeatmap} from './tensor_heatmap';

/** Elements of `shape` whose |Δ| is 2^-20 where `differs(i)`, else 0. */
function elements(total: number, differs: (i: number) => boolean): KvHeadChannel[] {
  return Array.from({length: total}, (_, channel) => {
    const delta = differs(channel) ? Math.pow(2, -20) : 0;
    return {
      channel,
      ref: 1,
      target: 1 + delta,
      delta,
      abs_delta: delta,
      status: 'available',
    };
  });
}

describe('TensorHeatmap', () => {
  afterEach(() => {
    delete (window as Window & {Plotly?: plotly.PlotlyApi}).Plotly;
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });
  function setup(shape: number[], differs: (i: number) => boolean = (i) => i % 3 === 0) {
    const events: Record<string, (e: Record<string, unknown>) => void> = {};
    const api = {
      react: vi.fn(
        async (host: plotly.PlotHost, _traces: unknown[], _layout: Record<string, unknown>) => {
          host.on = (name, fn) => {
            events[name] = fn;
          };
        },
      ),
      restyle: vi.fn(async () => {}),
      purge: vi.fn(),
    };
    (window as Window & {Plotly?: plotly.PlotlyApi}).Plotly = api;
    const observers: ((e: unknown[]) => void)[] = [];
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(private cb: (e: unknown[]) => void) {
          observers.push(cb);
        }
        observe() {
          this.cb([{contentRect: {width: 900, height: 600}}]);
        }
        disconnect() {}
      },
    );
    const resize = (height: number) =>
      observers.forEach((cb) => cb([{contentRect: {width: 900, height}}]));
    vi.stubGlobal('matchMedia', () => ({addEventListener() {}, removeEventListener() {}}));
    const f = TestBed.createComponent(TensorHeatmap);
    f.componentRef.setInput('shape', shape);
    f.componentRef.setInput(
      'rows',
      elements(
        shape.reduce((a, b) => a * b, 1),
        differs,
      ),
    );
    return {f, api, events, resize};
  }
  async function settle(f: ReturnType<typeof setup>['f']) {
    f.detectChanges();
    await f.whenStable();
    f.detectChanges();
  }

  it('shows the shape as the layout control: fixed dimensions as text, middle ones as buttons', async () => {
    const {f} = setup([1, 8, 2, 128]);
    await settle(f);
    const text = f.nativeElement.querySelector('.shape').textContent.replace(/\s+/g, ' ');
    expect(text).toContain('a=1 ↓ Y');
    expect(text).toContain('d=128 → X');
    const buttons = [...f.nativeElement.querySelectorAll('.shape button')] as HTMLElement[];
    expect(buttons.map((b) => b.getAttribute('data-dim'))).toEqual(['1', '2']);
    expect(buttons.map((b) => b.querySelector('.tag')!.textContent)).toEqual(['↓ Y', '↓ Y']);
    expect(f.nativeElement.querySelector('.dim.fixed[data-dim="0"]')).not.toBeNull();
    expect(f.nativeElement.querySelector('.dim.fixed[data-dim="3"]')).not.toBeNull();
    expect(f.nativeElement.textContent).toContain('683 of 2,048 elements differ');
    f.destroy();
  });

  it('switches a middle dimension between Y and X with one click and draws the new grid', async () => {
    const {f, api} = setup([1, 8, 2, 128]);
    await settle(f);
    const heatmap = f.componentInstance;
    expect(heatmap.grid()).toMatchObject({rows: 16, cols: 128});
    expect(heatmap.changed()).toBe(false);
    const [, traces] = api.react.mock.calls.at(-1)!;
    expect((traces[0] as {z: number[][]}).z).toHaveLength(16);

    api.react.mockClear();
    (f.nativeElement.querySelector('button[data-dim="2"]') as HTMLButtonElement).click();
    await settle(f);

    expect(heatmap.layout()).toEqual({y: [1], x: [2, 3], wrap: 0});
    expect(heatmap.grid()).toMatchObject({rows: 8, cols: 256});
    expect(heatmap.changed()).toBe(true);
    expect(f.nativeElement.querySelector('button[data-dim="2"] .tag').textContent).toBe('→ X');
    expect(api.react).toHaveBeenCalledTimes(1);
    const [, nextTraces, layout] = api.react.mock.calls.at(-1)!;
    expect((nextTraces[0] as {z: number[][]}).z).toHaveLength(8);
    const titles = (layout['annotations'] as {text: string}[]).map((a) => a.text);
    expect(titles).toContain('<b>b</b>');
    expect(titles).toContain('<b>c</b>');
    heatmap.reset();
    await settle(f);
    expect(heatmap.layout()).toEqual({y: [1, 2], x: [3], wrap: 0});
    expect(heatmap.changed()).toBe(false);
    f.destroy();
  });

  it('fits the heatmap to the container height and repaints on resize', async () => {
    const {f, api, resize} = setup([1, 8, 2048, 128]);
    await settle(f);
    const heatmap = f.componentInstance;
    const tall = heatmap.heatmapHeight();
    expect(api.react.mock.calls.at(-1)![2]['height']).toBe(600);
    resize(450);
    await settle(f);
    expect(heatmap.heatmapHeight()).toBe(tall - 150);
    expect(api.react.mock.calls.at(-1)![2]['height']).toBe(450);
    // Rows merge by powers of two to fit the shorter heatmap.
    const rows = (api.react.mock.calls.at(-1)![1][0] as {z: number[][]}).z.length;
    expect(rows).toBeLessThanOrEqual(heatmap.heatmapHeight());
    resize(100);
    await settle(f);
    expect(heatmap.heatmapHeight()).toBe(120);
    f.destroy();
  });

  it('zooms on drag, resets on double-click, and reads out a hovered element', async () => {
    const {f, events} = setup([1, 8, 2, 128]);
    f.componentRef.setInput('dtype', 'F32');
    await settle(f);
    const heatmap = f.componentInstance;
    events['plotly_relayout']({
      'xaxis.range[0]': -0.5,
      'xaxis.range[1]': 15.5,
      'yaxis.range[0]': 3.5,
      'yaxis.range[1]': -0.5,
    });
    await settle(f);
    expect(heatmap.zoom()).toEqual({r0: 0, r1: 4, c0: 0, c1: 16});
    events['plotly_hover']({points: [{data: {name: 'cells'}, pointIndex: [0, 3]}]});
    await settle(f);
    const hover = f.nativeElement.querySelector('chart-hover').textContent;
    expect(hover).toContain('1 element');
    expect(hover).toContain('[0, 0, 0, 3]');
    expect(hover).toContain('ULP');
    events['plotly_doubleclick']({});
    await settle(f);
    expect(heatmap.zoom()).toBeNull();
    f.destroy();
  });

  it('zooms to a clicked outer group annotation and renders max side bars on a log axis', async () => {
    const {f, api, events} = setup([1, 8, 2, 128]);
    f.componentRef.setInput('sideBars', 'max');
    await settle(f);

    const [, , layout] = api.react.mock.calls.at(-1)!;
    expect((layout['xaxis2'] as {type?: string}).type).toBe('log');
    expect((layout['yaxis2'] as {type?: string}).type).toBe('log');

    const annotations = layout['annotations'] as {text: string; captureevents?: boolean}[];
    const groupOneIndex = annotations.findIndex((a) => a.captureevents && a.text === '1');
    expect(groupOneIndex).toBeGreaterThanOrEqual(0);

    events['plotly_clickannotation']({index: groupOneIndex});
    await settle(f);

    expect(f.componentInstance.zoom()).toEqual({r0: 2, r1: 4, c0: 0, c1: 128});
    f.destroy();
  });

  it('highlights non-comparable elements separately from equal NaN pairs and scales ULP by dtype', async () => {
    const {f, api, events} = setup([1, 4]);
    f.componentRef.setInput('dtype', 'F16');
    f.componentRef.setInput('rows', [
      {channel: 0, ref: 1, target: 1 + Math.pow(2, -10), delta: Math.pow(2, -10), abs_delta: Math.pow(2, -10), status: 'ok'},
      {channel: 1, ref: 1, target: 'NaN', delta: null, abs_delta: null, status: 'unavailable'},
      {channel: 2, ref: 'NaN', target: 'NaN', delta: null, abs_delta: null, status: 'unavailable'},
      {channel: 3, ref: 0, target: 0, delta: 0, abs_delta: 0, status: 'ok'},
    ]);
    await settle(f);

    expect(f.componentInstance.differing()).toBe(2);
    expect(f.componentInstance.notComparable()).toBe(1);
    expect(f.nativeElement.textContent).toContain('1 not comparable');
    const [, traces] = api.react.mock.calls.at(-1)!;
    expect(traces.some((t) => (t as {name?: string}).name === 'non_comparable')).toBe(true);

    // 2^-10 at magnitude 1 is exactly 1 ULP in F16.
    events['plotly_hover']({points: [{data: {name: 'cells'}, pointIndex: [0, 0]}]});
    await settle(f);
    const ulpRow = [...f.nativeElement.querySelectorAll('chart-hover tbody tr')].find(
      (tr: Element) => tr.querySelector('th')?.textContent?.trim() === 'ULP',
    );
    expect(ulpRow).toBeDefined();
    expect(ulpRow?.querySelector('td:last-child')?.textContent?.trim()).toBe('1');

    // Unsupported dtype hides the ULP row.
    f.componentRef.setInput('dtype', 'I8');
    await settle(f);
    events['plotly_hover']({points: [{data: {name: 'cells'}, pointIndex: [0, 0]}]});
    await settle(f);
    expect(f.nativeElement.querySelector('chart-hover').textContent).not.toContain('ULP');
    f.destroy();
  });
});
