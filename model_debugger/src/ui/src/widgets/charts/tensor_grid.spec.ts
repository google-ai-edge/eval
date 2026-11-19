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

import {
  autoLayout,
  binGrid,
  cellOf,
  dimensionRole,
  elementAt,
  flatten,
  fullWindow,
  gridPoints,
  headerLevels,
  levelLabels,
  levelSeparators,
  logLevel,
  tensorGrid,
  toggleDimension,
  unflatten,
  visibleLevels,
} from './tensor_grid';

describe('tensor grid layout', () => {
  it('stacks a small middle dimension on Y and keeps the last dimension on X', () => {
    expect(autoLayout([1, 8, 2, 128])).toEqual({y: [1, 2], x: [3], wrap: 0});
  });
  it('puts a middle dimension side by side when Y is already tall', () => {
    expect(autoLayout([1, 8, 2048, 128])).toEqual({y: [2], x: [1, 3], wrap: 0});
  });
  it('always makes shape[0] the outermost Y when it has more than one element', () => {
    expect(autoLayout([4, 8, 2048, 128])).toEqual({y: [0, 2], x: [1, 3], wrap: 0});
  });
  it('wraps a long 1-D tensor into rows of 128', () => {
    const layout = autoLayout([1, 1024]);
    expect(layout).toEqual({y: [], x: [1], wrap: 128});
    const grid = tensorGrid([1, 1024], layout);
    expect([grid.rows, grid.cols]).toEqual([8, 128]);
    expect(cellOf(grid, 300)).toEqual([2, 44]);
    expect(tensorGrid([200], autoLayout([200]))).toMatchObject({rows: 1, cols: 200, wrap: 0});
  });
  it('toggles only middle dimensions, keeping shape order within each side', () => {
    const shape = [1, 8, 2, 128];
    const layout = autoLayout(shape);
    expect(['first', 'toggle', 'toggle', 'last'].map((_, k) => dimensionRole(shape, k))).toEqual([
      'first',
      'toggle',
      'toggle',
      'last',
    ]);
    const c = toggleDimension(shape, layout, 2);
    expect(c).toEqual({y: [1], x: [2, 3], wrap: 0});
    expect(toggleDimension(shape, c, 1)).toEqual({y: [], x: [1, 2, 3], wrap: 0});
    expect(toggleDimension(shape, layout, 0)).toBe(layout);
    expect(toggleDimension(shape, layout, 3)).toBe(layout);
    expect(dimensionRole([4, 1, 16], 1)).toBe('unit');
    expect(toggleDimension([4, 1, 16], autoLayout([4, 1, 16]), 1)).toEqual(autoLayout([4, 1, 16]));
  });
  it('round-trips flat indices through multi-indices and cells', () => {
    const shape = [2, 3, 4, 5];
    const grid = tensorGrid(shape, {y: [0, 2], x: [1, 3], wrap: 0});
    expect([grid.rows, grid.cols]).toEqual([8, 15]);
    for (let flat = 0; flat < grid.total; flat++) {
      expect(flatten(shape, unflatten(shape, flat))).toBe(flat);
      const [r, c] = cellOf(grid, flat);
      expect(elementAt(grid, r, c)).toBe(flat);
    }
    expect(unflatten(shape, 119)).toEqual([1, 2, 3, 4]);
    const wrapped = tensorGrid([300], autoLayout([300]));
    expect(elementAt(wrapped, 2, 43)).toBe(299);
    expect(elementAt(wrapped, 2, 44)).toBeNull();
  });
});

describe('tensor grid cells', () => {
  it('merges by powers of two and keeps each cell’s max |Δ|, not the mean', () => {
    const grid = tensorGrid([4, 8], autoLayout([4, 8]));
    const abs = new Map([
      [0, 1e-7],
      [1, 4e-7],
      [9, 2e-6],
      [31, 1e-3],
    ]);
    const points = gridPoints(grid, (i) => abs.get(i) ?? 0);
    expect(points.length).toBe(4);
    const bins = binGrid({points, window: fullWindow(grid), heightPx: 2, widthPx: 3});
    expect([bins.rowBin, bins.colBin, bins.nr, bins.nc]).toEqual([2, 4, 2, 2]);
    expect(bins.max[0]).toBe(2e-6);
    expect(bins.count[0]).toBe(3);
    expect(bins.argmax[0]).toBe(9);
    expect(bins.max[3]).toBe(1e-3);
    expect([...bins.rowCount]).toEqual([3, 1]);
    expect([...bins.colMax]).toEqual([2e-6, 1e-3]);
    expect([bins.vmin, bins.vmax]).toEqual([1e-7, 1e-3]);
    const single = binGrid({points, window: fullWindow(grid), heightPx: 100, widthPx: 100});
    expect([single.rowBin, single.colBin]).toEqual([1, 1]);
  });
  it('colors log |Δ| on [0.25, 1] and no difference at 0', () => {
    const level = logLevel(1e-6, 1e-2);
    expect(level(0)).toBe(0);
    expect(level(1e-6)).toBeCloseTo(0.25);
    expect(level(1e-4)).toBeCloseTo(0.625);
    expect(level(1e-2)).toBeCloseTo(1);
  });
  it('labels outer levels per group and the innermost level per pow2 step', () => {
    const grid = tensorGrid([1, 8, 2, 128], autoLayout([1, 8, 2, 128]));
    const [b, c] = headerLevels(grid, 'y');
    expect(b).toMatchObject({dimension: 1, title: 'b', period: 2, count: 8, inner: false});
    expect(c).toMatchObject({dimension: 2, title: 'c', period: 1, count: 2, inner: true});
    expect(levelLabels({level: b, start: 0, end: 16, pxPerUnit: 27, side: 'y'}).slice(0, 2)).toEqual([
      {text: '0', at: 0.5, group: {start: 0, end: 2}},
      {text: '1', at: 2.5, group: {start: 2, end: 4}},
    ]);
    expect(
      levelLabels({level: c, start: 0, end: 4, pxPerUnit: 27, side: 'y'}).map((l) => l.text),
    ).toEqual(['0', '1', '0', '1']);
    expect(levelSeparators(b, 0, 16)).toEqual([2, 4, 6, 8, 10, 12, 14]);
    const [d] = headerLevels(grid, 'x');
    expect(
      levelLabels({level: d, start: 0, end: 128, pxPerUnit: 8, side: 'x'}).map((l) => l.text),
    ).toEqual(Array.from({length: 32}, (_, i) => String(i * 4)));
    // Groups thinner than 13 px have no labels; the innermost level needs room for two labels.
    expect(visibleLevels({levels: [b, c], side: 'y', pxPerUnit: 2, wrapped: false})).toEqual([]);
    expect(visibleLevels({levels: [b, c], side: 'y', pxPerUnit: 27, wrapped: false})).toEqual([
      b,
      c,
    ]);
    const wrapped = tensorGrid([1024], autoLayout([1024]));
    expect(headerLevels(wrapped, 'y')[0]).toMatchObject({title: 'start', scale: 128});
    expect(
      levelLabels({
        level: headerLevels(wrapped, 'y')[0],
        start: 0,
        end: 3,
        pxPerUnit: 40,
        side: 'y',
      }).map((l) => l.text),
    ).toEqual(['0', '128', '256']);
    expect(headerLevels(wrapped, 'x')[0].title).toBe('mod 128');
  });
});
