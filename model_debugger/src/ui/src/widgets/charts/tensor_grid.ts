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

/**
 * How an N-D tensor is laid out as one 2-D grid, and what each grid cell shows.
 *
 * Every dimension of size > 1 goes to Y (rows) or X (columns). Within Y and within X the
 * dimensions keep their shape order, earlier = outer. shape[0] is always the outermost Y, the
 * last dimension always the innermost X; only the dimensions in between can switch sides. A
 * tensor with one dimension of size > 1 wraps into rows of 128 when it has more than 256
 * elements. No Plotly or Angular here.
 */

/** The grid aspect (columns / rows) the automatic layout aims for. */
const TARGET_ASPECT = 2.3;
// A tensor with one laid-out dimension wraps into rows of WRAP_WIDTH above WRAP_ABOVE elements.
const WRAP_ABOVE = 256;
const WRAP_WIDTH = 128;

/** Which axis of the 2-D heatmap grid a dimension maps onto. */
export type GridSide = 'y' | 'x';

/** Which dimensions go to Y and to X, each outer → inner (ascending), and the 1-D wrap width. */
export interface TensorLayout {
  readonly y: readonly number[];
  readonly x: readonly number[];
  readonly wrap: number;
}

/** Dimensions are named by position, so dimensions of equal size stay distinguishable. */
export function dimensionName(dimension: number): string {
  return 'abcdefghijklmnopqrstuvwxyz'[dimension] ?? `d${dimension}`;
}

function product(values: readonly number[]): number {
  return values.reduce((a, b) => a * b, 1);
}

/** The dimensions that are laid out: those of size > 1. */
function liveDimensions(shape: readonly number[]): number[] {
  return shape.map((_, k) => k).filter((k) => shape[k] > 1);
}

/** What the shape control shows for one dimension. */
export type DimensionRole = 'first' | 'last' | 'unit' | 'toggle';

/** Returns the layout control role of `dimension` in `shape`. */
export function dimensionRole(shape: readonly number[], dimension: number): DimensionRole {
  const live = liveDimensions(shape);
  const last = live[live.length - 1];
  if (dimension === 0 && dimension !== last) {
    return 'first';
  }
  if (shape[dimension] <= 1) {
    return 'unit';
  }
  return dimension === last ? 'last' : 'toggle';
}

/**
 * The default layout: the last dimension on X, the one before on Y, then each earlier dimension,
 * inner to outer, on whichever side keeps the grid aspect closest to TARGET_ASPECT; shape[0] then
 * goes to Y.
 */
export function autoLayout(shape: readonly number[]): TensorLayout {
  const live = liveDimensions(shape);
  if (!live.length) {
    return {y: [], x: [], wrap: 0};
  }
  const x = [live[live.length - 1]];
  if (live.length === 1) {
    return {y: [], x, wrap: shape[live[0]] > WRAP_ABOVE ? WRAP_WIDTH : 0};
  }
  const y = [live[live.length - 2]];
  for (let i = live.length - 3; i >= 0; i--) {
    const k = live[i];
    const n = shape[k];
    const rows = product(y.map((d) => shape[d]));
    const cols = product(x.map((d) => shape[d]));
    const onY = Math.abs(Math.log(cols / (rows * n) / TARGET_ASPECT));
    const onX = Math.abs(Math.log((cols * n) / rows / TARGET_ASPECT));
    (onX < onY ? x : y).unshift(k);
  }
  const layout = {y, x, wrap: 0};
  return shape[0] > 1 && live[live.length - 1] !== 0 ? moveDimension(layout, 0, 'y') : layout;
}

function moveDimension(layout: TensorLayout, dimension: number, side: GridSide): TensorLayout {
  const y = layout.y.filter((k) => k !== dimension);
  const x = layout.x.filter((k) => k !== dimension);
  (side === 'y' ? y : x).push(dimension);
  return {y: y.sort((a, b) => a - b), x: x.sort((a, b) => a - b), wrap: layout.wrap};
}

/** Returns which grid axis `dimension` is currently placed on, or null for unit dimensions. */
export function sideOf(layout: TensorLayout, dimension: number): GridSide | null {
  return layout.y.includes(dimension) ? 'y' : layout.x.includes(dimension) ? 'x' : null;
}

/** Moves a middle dimension to the other side; any other dimension leaves the layout as it is. */
export function toggleDimension(
  shape: readonly number[],
  layout: TensorLayout,
  dimension: number,
): TensorLayout {
  const side = sideOf(layout, dimension);
  if (dimensionRole(shape, dimension) !== 'toggle' || !side) {
    return layout;
  }
  return moveDimension(layout, dimension, side === 'y' ? 'x' : 'y');
}

/** Returns whether two tensor layouts place the same dimensions on Y, X, and wrap width. */
export function sameLayout(a: TensorLayout, b: TensorLayout): boolean {
  return a.wrap === b.wrap && a.y.join() === b.y.join() && a.x.join() === b.x.join();
}

/** A layout applied to a shape: the grid's size and how elements map to cells. */
export interface TensorGrid {
  readonly shape: readonly number[];
  readonly layout: TensorLayout;
  readonly rows: number;
  readonly cols: number;
  /** The wrap width actually applied, or 0. */
  readonly wrap: number;
  readonly strides: readonly number[];
  readonly total: number;
}

/** Builds a 2-D `TensorGrid` by applying `layout` to `shape`. */
export function tensorGrid(shape: readonly number[], layout: TensorLayout): TensorGrid {
  const strides = shape.map((_, k) => product(shape.slice(k + 1)));
  const total = product(shape);
  let rows = product(layout.y.map((k) => shape[k]));
  let cols = product(layout.x.map((k) => shape[k]));
  const wrap = !layout.y.length && layout.wrap && cols > layout.wrap ? layout.wrap : 0;
  if (wrap) {
    rows = Math.ceil(cols / wrap);
    cols = wrap;
  }
  return {shape, layout, rows, cols, wrap, strides, total};
}

/** The multi-index of a row-major flat index. */
export function unflatten(shape: readonly number[], flat: number): number[] {
  const index = shape.map(() => 0);
  for (let k = shape.length - 1; k >= 0; k--) {
    index[k] = flat % shape[k];
    flat = Math.floor(flat / shape[k]);
  }
  return index;
}

/** Converts a row-major multi-index back to its flat index in `shape`. */
export function flatten(shape: readonly number[], index: readonly number[]): number {
  return index.reduce((flat, i, k) => flat * shape[k] + i, 0);
}

/** The [row, column] of an element. */
export function cellOf(grid: TensorGrid, flat: number): [number, number] {
  const index = unflatten(grid.shape, flat);
  const radix = (dims: readonly number[]) =>
    dims.reduce((acc, k) => acc * grid.shape[k] + index[k], 0);
  let row = radix(grid.layout.y);
  let col = radix(grid.layout.x);
  if (grid.wrap) {
    row = Math.floor(col / grid.wrap);
    col %= grid.wrap;
  }
  return [row, col];
}

/** The flat index of the element at a cell, or null past the end of a wrapped tensor. */
export function elementAt(grid: TensorGrid, row: number, col: number): number | null {
  const index = grid.shape.map(() => 0);
  if (grid.wrap) {
    col = row * grid.wrap + col;
    row = 0;
  }
  const decode = (value: number, dims: readonly number[]) => {
    for (let i = dims.length - 1; i >= 0; i--) {
      const k = dims[i];
      index[k] = value % grid.shape[k];
      value = Math.floor(value / grid.shape[k]);
    }
    return value;
  };
  if (decode(row, grid.layout.y) || decode(col, grid.layout.x)) {
    return null;
  }
  return flatten(grid.shape, index);
}

/** The differing or non-comparable elements with their cells. */
export interface GridPoints {
  readonly rows: Int32Array;
  readonly cols: Int32Array;
  readonly flat: Int32Array;
  readonly abs: Float64Array;
  /** 1 for non-comparable elements (e.g. NaN/Inf mismatch), 0 for finite differing elements. */
  readonly nonComparable: Uint8Array;
  readonly nonComparableCount: number;
  readonly length: number;
}

/**
 * Collects differing (`abs(i) > 0` finite) and non-comparable elements with their grid cells.
 */
export function gridPoints(
  grid: TensorGrid,
  abs: (flat: number) => number | null,
  isNonComparable?: (flat: number) => boolean,
): GridPoints {
  const found: number[] = [];
  const values: number[] = [];
  const flags: number[] = [];
  let nonComparableCount = 0;
  for (let i = 0; i < grid.total; i++) {
    const value = abs(i);
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      found.push(i);
      values.push(value);
      flags.push(0);
    } else if (isNonComparable?.(i)) {
      found.push(i);
      values.push(0);
      flags.push(1);
      nonComparableCount++;
    }
  }
  const points = {
    rows: new Int32Array(found.length),
    cols: new Int32Array(found.length),
    flat: Int32Array.from(found),
    abs: Float64Array.from(values),
    nonComparable: Uint8Array.from(flags),
    nonComparableCount,
    length: found.length,
  };
  found.forEach((i, n) => {
    [points.rows[n], points.cols[n]] = cellOf(grid, i);
  });
  return points;
}

function pow2Ceil(n: number): number {
  let p = 1;
  while (p < n) {
    p *= 2;
  }
  return p;
}

/** A rectangle of grid rows [r0, r1) and columns [c0, c1). */
export interface GridWindow {
  readonly r0: number;
  readonly r1: number;
  readonly c0: number;
  readonly c1: number;
}

/** Returns a `GridWindow` covering the entire `grid`. */
export function fullWindow(grid: TensorGrid): GridWindow {
  return {r0: 0, r1: grid.rows, c0: 0, c1: grid.cols};
}

/**
 * The visible window merged into cells of rowBin × colBin elements, both powers of two, so that
 * it fits `heightPx` × `widthPx`. A cell shows the max |Δ| of its elements, never the mean.
 */
export interface GridBins {
  readonly rowBin: number;
  readonly colBin: number;
  /** The first row and column of cell (0, 0). */
  readonly r0: number;
  readonly c0: number;
  /** Cells down and across. */
  readonly nr: number;
  readonly nc: number;
  readonly max: Float64Array;
  readonly count: Uint32Array;
  /** Number of non-comparable elements (e.g. NaN/Inf mismatch) per cell. */
  readonly nonComparable: Uint32Array;
  /** The flat index of each cell's largest |Δ| (or first non-comparable element), or -1. */
  readonly argmax: Int32Array;
  readonly rowCount: Uint32Array;
  readonly rowMax: Float64Array;
  readonly colCount: Uint32Array;
  readonly colMax: Float64Array;
  /** The smallest and largest finite |Δ| of the whole tensor; 0 when nothing finite differs. */
  readonly vmin: number;
  readonly vmax: number;
}

/** Options for binning a visible window of grid points. */
export interface BinGridOptions {
  readonly points: GridPoints;
  readonly window: GridWindow;
  readonly widthPx: number;
  readonly heightPx: number;
}

/** Bins `points` within `window` into power-of-two cells sized to `widthPx` × `heightPx`. */
export function binGrid({points, window, widthPx, heightPx}: BinGridOptions): GridBins {
  const rowBin = pow2Ceil(Math.ceil((window.r1 - window.r0) / Math.max(1, heightPx)));
  const colBin = pow2Ceil(Math.ceil((window.c1 - window.c0) / Math.max(1, widthPx)));
  const r0 = Math.floor(window.r0 / rowBin) * rowBin;
  const c0 = Math.floor(window.c0 / colBin) * colBin;
  const nr = Math.max(1, Math.ceil((window.r1 - r0) / rowBin));
  const nc = Math.max(1, Math.ceil((window.c1 - c0) / colBin));
  const max = new Float64Array(nr * nc);
  const count = new Uint32Array(nr * nc);
  const nonComparable = new Uint32Array(nr * nc);
  const argmax = new Int32Array(nr * nc).fill(-1);
  const rowCount = new Uint32Array(nr);
  const rowMax = new Float64Array(nr);
  const colCount = new Uint32Array(nc);
  const colMax = new Float64Array(nc);
  let vmin = Infinity;
  let vmax = 0;
  for (let n = 0; n < points.length; n++) {
    const r = points.rows[n];
    const c = points.cols[n];
    const isUncomparable = points.nonComparable[n] === 1;
    const d = points.abs[n];
    if (!isUncomparable) {
      vmax = Math.max(vmax, d);
      vmin = Math.min(vmin, d);
    }
    if (r < r0 || r >= r0 + nr * rowBin || c < c0 || c >= c0 + nc * colBin) {
      continue;
    }
    const i = Math.floor((r - r0) / rowBin);
    const j = Math.floor((c - c0) / colBin);
    const g = i * nc + j;
    count[g]++;
    rowCount[i]++;
    colCount[j]++;
    if (isUncomparable) {
      nonComparable[g]++;
      if (argmax[g] < 0) {
        argmax[g] = points.flat[n];
      }
    } else {
      if (d > max[g]) {
        max[g] = d;
        argmax[g] = points.flat[n];
      }
      rowMax[i] = Math.max(rowMax[i], d);
      colMax[j] = Math.max(colMax[j], d);
    }
  }
  return {
    rowBin,
    colBin,
    r0,
    c0,
    nr,
    nc,
    max,
    count,
    nonComparable,
    argmax,
    rowCount,
    rowMax,
    colCount,
    colMax,
    vmin: vmin === Infinity ? 0 : vmin,
    vmax,
  };
}

/** Maps |Δ| to a color level: log |Δ| onto [0.25, 1], and 0 to 0 (the palette's low end). */
export function logLevel(vmin: number, vmax: number): (value: number) => number {
  const low = vmin ? Math.log10(vmin === vmax ? vmax / 10 : vmin) : 0;
  const high = vmax ? Math.log10(vmax) : 1;
  return (value) => (value ? 0.25 + (0.75 * (Math.log10(value) - low)) / (high - low || 1) : 0);
}

/** One level of axis labels: a dimension of Y or X, or the row start / column of a wrap. */
export interface HeaderLevel {
  /** The dimension, or null for a wrapped tensor's rows and columns. */
  readonly dimension: number | null;
  /** The label written above the level. */
  readonly title: string;
  /** Grid rows or columns per step of this level. */
  readonly period: number;
  readonly count: number;
  /** A label shows (index / period) % count × scale. */
  readonly scale: number;
  readonly inner: boolean;
}

/** Builds the axis header levels for `side` of `grid`, from outermost to innermost. */
export function headerLevels(grid: TensorGrid, side: GridSide): HeaderLevel[] {
  if (grid.wrap) {
    return side === 'y'
      ? [
          {
            dimension: null,
            title: 'start',
            period: 1,
            count: grid.rows,
            scale: grid.wrap,
            inner: true,
          },
        ]
      : [
          {
            dimension: null,
            title: `mod ${grid.wrap}`,
            period: 1,
            count: grid.cols,
            scale: 1,
            inner: true,
          },
        ];
  }
  const dims = side === 'y' ? grid.layout.y : grid.layout.x;
  return dims.map((k, j) => ({
    dimension: k,
    title: dimensionName(k),
    period: product(dims.slice(j + 1).map((d) => grid.shape[d])),
    count: grid.shape[k],
    scale: 1,
    inner: j === dims.length - 1,
  }));
}

/** The smallest power of two of units whose labels are at least `minPx` apart. */
function labelStep(pxPerUnit: number, minPx: number): number {
  if (!(pxPerUnit > 0) || !Number.isFinite(minPx)) {
    return 1;
  }
  let step = 1;
  while (step * pxPerUnit < minPx) {
    step *= 2;
  }
  return step;
}

// Minimum pixels per label: rows, columns, outer row groups, outer column groups.
const ROW_LABEL_PX = 13;
const COLUMN_LABEL_PX = 30;
const ROW_GROUP_PX = 13;
const COLUMN_GROUP_PX = 26;

/** Options for selecting header levels that fit at the current zoom. */
export interface VisibleLevelsOptions {
  readonly levels: readonly HeaderLevel[];
  readonly side: GridSide;
  readonly pxPerUnit: number;
  readonly wrapped: boolean;
}

/** Which levels have room for labels at this zoom; hidden levels take no space. */
export function visibleLevels({
  levels,
  side,
  pxPerUnit,
  wrapped,
}: VisibleLevelsOptions): HeaderLevel[] {
  const single = side === 'y' ? ROW_LABEL_PX : COLUMN_LABEL_PX;
  const group = side === 'y' ? ROW_GROUP_PX : COLUMN_GROUP_PX;
  return levels.filter((level) => {
    if (!level.inner) {
      return level.period * pxPerUnit >= group;
    }
    const step = labelStep(pxPerUnit, single);
    return (
      wrapped ||
      step <= Math.max(1, level.count / 2) ||
      step === 1 ||
      (side === 'y' && level.count === 1)
    );
  });
}

/** One axis label: its text, where it is centered, and for a group the units it spans. */
export interface HeaderLabel {
  readonly text: string;
  readonly at: number;
  readonly group: {readonly start: number; readonly end: number} | null;
}

/** Options for computing axis labels of one header level. */
export interface LevelLabelsOptions {
  readonly level: HeaderLevel;
  readonly start: number;
  readonly end: number;
  readonly pxPerUnit: number;
  readonly side: GridSide;
}

/** The labels of one level within units [start, end). */
export function levelLabels({
  level,
  start,
  end,
  pxPerUnit,
  side,
}: LevelLabelsOptions): HeaderLabel[] {
  const text = (unit: number) =>
    String((Math.floor(unit / level.period) % level.count) * level.scale);
  const labels: HeaderLabel[] = [];
  if (level.inner) {
    const step = labelStep(pxPerUnit, side === 'y' ? ROW_LABEL_PX : COLUMN_LABEL_PX);
    for (let unit = Math.ceil(start / step) * step; unit < end; unit += step) {
      labels.push({text: text(unit), at: unit, group: null});
    }
    return labels;
  }
  const p = level.period;
  for (let g = Math.floor(start / p); g * p < end; g++) {
    const a = Math.max(start, g * p);
    const b = Math.min(end, (g + 1) * p);
    labels.push({
      text: text(g * p),
      at: (a + b) / 2 - 0.5,
      group: {start: g * p, end: (g + 1) * p},
    });
  }
  return labels;
}

/** The units in (start, end) where an outer level's groups meet. */
export function levelSeparators(level: HeaderLevel, start: number, end: number): number[] {
  const at: number[] = [];
  for (let p = Math.ceil(start / level.period) * level.period; p < end; p += level.period) {
    if (p > start) {
      at.push(p);
    }
  }
  return at;
}
