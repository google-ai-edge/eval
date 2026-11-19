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
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  NgZone,
  afterRenderEffect,
  computed,
  inject,
  input,
  linkedSignal,
  signal,
  viewChild,
} from '@angular/core';
import type {KvHeadChannel} from '../../server/contracts/kv';
import {formatMetric} from '../format/format';
import {plotlyColor, type PlotHost} from '../plotly_loader';
import {
  ChartHover,
  hoverInterval,
  type ChartHoverContent,
  type ChartHoverPosition,
} from './chart_hover';
import {PlotlySurface} from './plotly_surface';
import {rangeChartStyle} from './range_chart_style';
import {
  autoLayout,
  binGrid,
  cellOf,
  dimensionName,
  dimensionRole,
  elementAt,
  fullWindow,
  gridPoints,
  headerLevels,
  levelLabels,
  levelSeparators,
  logLevel,
  sameLayout,
  sideOf,
  tensorGrid,
  toggleDimension,
  unflatten,
  visibleLevels,
  type GridBins,
  type GridPoints,
  type GridSide,
  type GridWindow,
  type HeaderLevel,
  type TensorGrid,
  type TensorLayout,
} from './tensor_grid';

/** What the bars beside the heatmap show per row and per column. */
export type SideBarMetric = 'count' | 'max';

// Pixel geometry: least heatmap height, side bar thickness and gap, label column width, label row
// height, bottom margin. The heatmap takes whatever height the container leaves.
const MIN_HEATMAP_HEIGHT = 120;
const BOTTOM_MARGIN = 36;
const BARS_HEIGHT = 72;
const BARS_GAP = 26;
const LABEL_COLUMN = 34;
const LABEL_ROW = 16;
// Horizontal paper fractions of the heatmap and of the per-row bars.
const HEATMAP_DOMAIN: readonly [number, number] = [0, 0.88] as const;
const ROW_BARS_DOMAIN: readonly [number, number] = [0.9, 1] as const;

/** Which label levels fit at a zoom, and the margins and pixel scales that follow. */
interface HeaderGeometry {
  yLevels: HeaderLevel[];
  xLevels: HeaderLevel[];
  yVisible: HeaderLevel[];
  xVisible: HeaderLevel[];
  /** The heatmap's own height, without labels, side bars and margins. */
  heatmapHeight: number;
  pxPerRow: number;
  pxPerCol: number;
  plotWidth: number;
  margin: {l: number; r: number; t: number; b: number; autoexpand: boolean};
}

/** Label row space above the heatmap for `rows` visible X levels. */
function topMargin(rows: number): number {
  return 8 + (Math.max(1, rows) + 1) * LABEL_ROW;
}

/**
 * The geometry of a `width` × `height` plot: the heatmap takes the height the X label rows, the
 * per-column bars and the margins leave, at least MIN_HEATMAP_HEIGHT. Which X levels fit depends
 * on the label columns, which depend on the heatmap height, so this settles in a few rounds.
 */
function headerGeometry(
  grid: TensorGrid,
  window: GridWindow,
  width: number,
  height: number,
): HeaderGeometry {
  const fixed = BARS_GAP + BARS_HEIGHT + BOTTOM_MARGIN;
  let rows = headerLevels(grid, 'x').length;
  let geometry = headerGeometryAt(grid, window, width, height - fixed - topMargin(rows));
  for (let round = 0; round < 3 && geometry.xVisible.length !== rows; round++) {
    rows = geometry.xVisible.length;
    geometry = headerGeometryAt(grid, window, width, height - fixed - topMargin(rows));
  }
  return geometry;
}

function headerGeometryAt(
  grid: TensorGrid,
  window: GridWindow,
  width: number,
  available: number,
): HeaderGeometry {
  const heatmapHeight = Math.max(MIN_HEATMAP_HEIGHT, Math.floor(available));
  const pxPerRow = heatmapHeight / (window.r1 - window.r0);
  const yLevels = headerLevels(grid, 'y');
  const yVisible = visibleLevels({levels: yLevels, side: 'y', pxPerUnit: pxPerRow, wrapped: !!grid.wrap});
  const left = 14 + Math.max(1, yVisible.length) * LABEL_COLUMN;
  const plotWidth = Math.max(1, width - left - 16);
  const pxPerCol = (plotWidth * HEATMAP_DOMAIN[1]) / (window.c1 - window.c0);
  const xLevels = headerLevels(grid, 'x');
  const xVisible = visibleLevels({levels: xLevels, side: 'x', pxPerUnit: pxPerCol, wrapped: !!grid.wrap});
  return {
    yLevels,
    xLevels,
    yVisible,
    xVisible,
    heatmapHeight,
    pxPerRow,
    pxPerCol,
    plotWidth,
    margin: {
      l: left,
      r: 16,
      t: topMargin(xVisible.length),
      b: BOTTOM_MARGIN,
      autoexpand: false,
    },
  };
}

/** What a paint draws. */
interface HeatmapFrame {
  grid: TensorGrid;
  points: GridPoints;
  window: GridWindow;
  metric: SideBarMetric;
  width: number;
  /** The container's height the plot fills. */
  height: number;
}

/** The rows or columns spanned by a clickable outer header group annotation. */
interface GroupSpan {
  readonly side: GridSide;
  readonly start: number;
  readonly end: number;
}

/** What the last paint drew, for hover and annotation clicks. */
interface HeatmapView extends HeatmapFrame {
  bins: GridBins;
  /** Annotation index → the rows or columns its group spans. */
  groups: Map<number, GroupSpan>;
}

/** Floating-point precision parameters for ULP computation. */
const FLOAT_ULP_PARAMS: Readonly<
  Record<string, {mantissaBits: number; minNormalExp: number; minSubnormalExp: number}>
> = {
  F32: {mantissaBits: 23, minNormalExp: -126, minSubnormalExp: -149},
  FLOAT32: {mantissaBits: 23, minNormalExp: -126, minSubnormalExp: -149},
  F16: {mantissaBits: 10, minNormalExp: -14, minSubnormalExp: -24},
  FLOAT16: {mantissaBits: 10, minNormalExp: -14, minSubnormalExp: -24},
  BF16: {mantissaBits: 7, minNormalExp: -126, minSubnormalExp: -133},
  BFLOAT16: {mantissaBits: 7, minNormalExp: -126, minSubnormalExp: -133},
};

/** Returns 1 ULP at `value`'s magnitude for `dtype`, or null when `dtype` is unsupported. */
function floatUlp(value: number, dtype: string | null): number | null {
  if (!dtype) {
    return null;
  }
  const params = FLOAT_ULP_PARAMS[dtype.toUpperCase()];
  if (!params) {
    return null;
  }
  const a = Math.abs(value);
  return a
    ? Math.pow(2, Math.max(Math.floor(Math.log2(a)), params.minNormalExp) - params.mantissaBits)
    : Math.pow(2, params.minSubnormalExp);
}

/** Returns whether an unavailable row represents a non-comparable divergence (e.g. NaN/Inf mismatch). */
function isNonComparableRow(row: KvHeadChannel | undefined): boolean {
  if (!row || row.abs_delta != null || row.status !== 'unavailable') {
    return false;
  }
  if (row.ref === row.target) {
    return false;
  }
  if (String(row.ref) === 'NaN' && String(row.target) === 'NaN') {
    return false;
  }
  return row.ref != null || row.target != null;
}

/** Builds the Plotly heatmap, optional non-comparable overlay, and marginal bar traces. */
function buildHeatmapTraces(options: {
  grid: TensorGrid;
  bins: GridBins;
  metric: SideBarMetric;
  colors: {low: string; high: string; bar: string; alert: string};
}): Record<string, unknown>[] {
  const {grid, bins, metric, colors} = options;
  const level = logLevel(bins.vmin, bins.vmax);
  const z: number[][] = [];
  const nonComparableZ: (number | null)[][] = [];
  let hasNonComparable = false;
  for (let i = 0; i < bins.nr; i++) {
    const row: number[] = [];
    const nonComparableRow: (number | null)[] = [];
    for (let j = 0; j < bins.nc; j++) {
      const g = i * bins.nc + j;
      row.push(level(bins.max[g]));
      if (bins.nonComparable[g] > 0) {
        nonComparableRow.push(1);
        hasNonComparable = true;
      } else {
        nonComparableRow.push(null);
      }
    }
    z.push(row);
    nonComparableZ.push(nonComparableRow);
  }
  const xEdges = Array.from(
    {length: bins.nc + 1},
    (_, j) => Math.min(grid.cols, bins.c0 + j * bins.colBin) - 0.5,
  );
  const yEdges = Array.from(
    {length: bins.nr + 1},
    (_, i) => Math.min(grid.rows, bins.r0 + i * bins.rowBin) - 0.5,
  );
  const xCenters = Array.from(
    {length: bins.nc},
    (_, j) => bins.c0 + (j + 0.5) * bins.colBin - 0.5,
  );
  const yCenters = Array.from(
    {length: bins.nr},
    (_, i) => bins.r0 + (i + 0.5) * bins.rowBin - 0.5,
  );
  const count = metric === 'count';
  const traces: Record<string, unknown>[] = [
    {
      type: 'heatmap',
      name: 'cells',
      x: xEdges,
      y: yEdges,
      z,
      colorscale: [
        [0, colors.low],
        [1, colors.high],
      ],
      zmin: 0,
      zmax: 1,
      showscale: false,
      zsmooth: false,
      xgap: 0,
      ygap: bins.nr <= 64 ? 1 : 0,
      hoverinfo: 'none',
    },
  ];
  if (hasNonComparable) {
    traces.push({
      type: 'heatmap',
      name: 'non_comparable',
      x: xEdges,
      y: yEdges,
      z: nonComparableZ,
      colorscale: [
        [0, colors.alert],
        [1, colors.alert],
      ],
      zmin: 0,
      zmax: 1,
      showscale: false,
      zsmooth: false,
      xgap: 0,
      ygap: bins.nr <= 64 ? 1 : 0,
      hoverinfo: 'none',
    });
  }
  traces.push(
    {
      type: 'bar',
      name: 'columns',
      xaxis: 'x',
      yaxis: 'y2',
      x: xCenters,
      y: count ? [...bins.colCount] : [...bins.colMax],
      width: bins.colBin * 0.9,
      marker: {color: colors.bar},
      hoverinfo: 'none',
    },
    {
      type: 'bar',
      name: 'rows',
      orientation: 'h',
      xaxis: 'x2',
      yaxis: 'y',
      y: yCenters,
      x: count ? [...bins.rowCount] : [...bins.rowMax],
      width: bins.rowBin * 0.9,
      marker: {color: colors.bar},
      hoverinfo: 'none',
    },
  );
  return traces;
}

/** Appends separator lines, pivot-table header annotations, and clickable group spans for one axis. */
function appendSideHeaders(options: {
  side: GridSide;
  levels: readonly HeaderLevel[];
  visible: readonly HeaderLevel[];
  start: number;
  end: number;
  pxPerUnit: number;
  header: HeaderGeometry;
  plotHeight: number;
  barsDomainBottom: number;
  colors: {ink: string; bar: string; strong: string; rule: string; muted: string};
  shapes: Record<string, unknown>[];
  annotations: Record<string, unknown>[];
  groups: Map<number, GroupSpan>;
}): void {
  const {
    side,
    levels,
    visible,
    start,
    end,
    pxPerUnit,
    header,
    plotHeight,
    barsDomainBottom,
    colors,
    shapes,
    annotations,
    groups,
  } = options;
  const nX = Math.max(1, header.xVisible.length);
  const columnRight = (j: number) => 6 + (header.yVisible.length - 1 - j) * LABEL_COLUMN;
  const rowBottom = (i: number) => 3 + (nX - 1 - i) * LABEL_ROW;
  const font = (fill: string) => ({size: 11, color: fill});

  for (const [j, level] of levels.entries()) {
    if (level.inner || level.period * pxPerUnit < 4) {
      continue;
    }
    const visIndex = visible.indexOf(level);
    for (const p of levelSeparators(level, start, end)) {
      if (side === 'y') {
        shapes.push({
          type: 'line',
          xref: 'paper',
          yref: 'y',
          x0: visIndex >= 0 ? -(columnRight(visIndex) + LABEL_COLUMN - 10) / header.plotWidth : 0,
          x1: ROW_BARS_DOMAIN[1],
          y0: p - 0.5,
          y1: p - 0.5,
          line: {color: j === 0 ? colors.strong : colors.rule, width: j === 0 ? 1.4 : 0.8},
        });
      } else {
        shapes.push({
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: p - 0.5,
          x1: p - 0.5,
          y0: barsDomainBottom,
          y1: visIndex >= 0 ? 1 + (rowBottom(visIndex) + LABEL_ROW - 2) / plotHeight : 1,
          line: {color: j === 0 ? colors.strong : colors.rule, width: j === 0 ? 1.4 : 0.8},
        });
      }
    }
  }

  for (const [idx, level] of visible.entries()) {
    annotations.push(
      side === 'y'
        ? {
            text: `<b>${level.title}</b>`,
            x: 0,
            y: 1,
            xref: 'paper',
            yref: 'paper',
            xanchor: 'right',
            yanchor: 'bottom',
            xshift: -columnRight(idx),
            yshift: rowBottom(nX - 1),
            showarrow: false,
            font: font(colors.ink),
          }
        : {
            text: `<b>${level.title}</b>`,
            x: HEATMAP_DOMAIN[1],
            y: 1,
            xref: 'paper',
            yref: 'paper',
            xanchor: 'left',
            yanchor: 'bottom',
            xshift: 4,
            yshift: rowBottom(idx),
            showarrow: false,
            font: font(colors.ink),
          },
    );
    for (const label of levelLabels({level, start, end, pxPerUnit, side})) {
      if (label.group) {
        groups.set(annotations.length, {side, ...label.group});
      }
      annotations.push(
        side === 'y'
          ? {
              text: label.text,
              x: 0,
              y: label.at,
              xref: 'paper',
              yref: 'y',
              xanchor: 'right',
              xshift: -columnRight(idx),
              showarrow: false,
              ...(label.group ? {captureevents: true} : {}),
              font: font(label.group ? colors.bar : colors.muted),
            }
          : {
              text: label.text,
              x: label.at,
              y: 1,
              xref: 'x',
              yref: 'paper',
              yanchor: 'bottom',
              yshift: rowBottom(idx),
              showarrow: false,
              ...(label.group ? {captureevents: true} : {}),
              font: font(label.group ? colors.bar : colors.muted),
            },
      );
    }
  }
}

function valueText(value: number | string | null | undefined): string {
  return typeof value === 'number' ? formatMetric(value) : (value ?? '-');
}

/**
 * The |Target − Ref| of one tensor of any rank as a 2-D heatmap.
 *
 * The tensor's shape is the layout control: shape[0] is the outermost Y, the last dimension the
 * innermost X, and each dimension in between switches between ↓ Y and → X with one click. Axis
 * labels are pivot-table headers, one label column per Y dimension and one label row per X
 * dimension. When zoomed out, rows and columns merge by powers of two and a cell shows the max |Δ|
 * of its elements. Drag to zoom, double-click to reset.
 */
@Component({
  selector: 'tensor-heatmap',
  imports: [ChartHover],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div class="chart-head">
      <span class="shape" role="group" aria-label="Tensor layout">
        <span class="bracket">[</span>
        @for (dim of dimensions(); track dim.index; let last = $last) {
          @if (dim.role === 'toggle') {
            <button
              type="button"
              class="dim toggle"
              [attr.data-dim]="dim.index"
              [attr.aria-label]="dim.ariaLabel"
              title="Click to switch between ↓ Y and → X"
              (click)="toggle(dim.index)"
            >
              <b>{{ dim.name }}</b
              >={{ dim.size }}
              <span class="tag">{{ dim.sideTag }}</span>
              <span class="swap" aria-hidden="true">⇄</span>
            </button>
          } @else {
            <span class="dim fixed" [attr.data-dim]="dim.index" [attr.title]="dim.title">
              <b>{{ dim.name }}</b
              >={{ dim.size }}
              @if (dim.tag) {
                <span class="tag">{{ dim.tag }}</span>
              }
            </span>
          }
          @if (!last) {
            <span class="bracket">,</span>
          }
        }
        <span class="bracket">]</span>
      </span>
      <span class="stat">
        <b>{{ differingText() }}</b> of {{ totalText() }} elements differ
        @if (notComparable() > 0) {
          · <span class="alert">{{ notComparableText() }} not comparable</span>
        }
        · max |Δ| {{ maxText() }}
        @if (isBinned()) {
          ·
          <span class="note"
            >1 cell = {{ cell().rows }} × {{ cell().cols }} elements (max |Δ|), zoom in for single
            elements</span
          >
        }
      </span>
    </div>
    <div #wrap class="plot-wrap">
      <div
        #plot
        class="tensor-heatmap"
        role="img"
        aria-label="Difference heatmap of |Target − Ref|. Drag to zoom; double-click to reset."
      ></div>
      @if (tooltip(); as tip) {
        <chart-hover [info]="tip" />
      }
    </div>
    @if (error()) {
      <p role="alert">{{ error() }}</p>
    }`,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      height: 100%;
      min-width: 0;
      background: var(--chart-canvas);
      color: var(--me-on-surface-color);
      font: 12px/20px var(--me-font-family);
    }
    .chart-head {
      flex: none;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      padding: 10px 12px 0 16px;
    }
    .shape {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .bracket {
      color: var(--me-on-surface-variant-color);
    }
    .dim {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      box-sizing: border-box;
      height: 24px;
      padding: 0 2px;
      font: 12px var(--me-font-family);
      user-select: none;
    }
    .dim b {
      font-weight: 500;
    }
    .dim .tag {
      font-weight: 500;
    }
    .dim.fixed {
      color: var(--me-on-surface-variant-color);
    }
    .dim.fixed .tag {
      opacity: 0.8;
    }
    .dim.toggle {
      padding: 2px 6px;
      border: 1px solid var(--me-outline-color);
      border-radius: 3px;
      background: var(--me-runtime-field-color);
      color: var(--me-on-surface-color);
      cursor: pointer;
    }
    .dim.toggle:hover {
      background: var(--me-surface-container-highest-color);
    }
    .dim.toggle:focus-visible {
      outline: 2px solid var(--me-primary-color);
    }
    .dim.toggle .tag {
      color: var(--me-primary-color);
    }
    .dim.toggle .swap {
      color: var(--me-on-surface-variant-color);
    }
    .stat .note {
      color: var(--me-on-surface-variant-color);
      font-style: italic;
    }
    .stat .alert {
      color: var(--me-error-color, #c5221f);
      font-weight: 500;
    }
    .plot-wrap {
      position: relative;
      flex: 1 1 0;
      min-height: 0;
      overflow: hidden;
    }
    :host ::ng-deep .tensor-heatmap .nsewdrag {
      cursor: crosshair !important;
    }
    p {
      margin: 8px 12px;
      font: 12px var(--me-font-family);
    }
  `,
})
export class TensorHeatmap {
  /** The tensor's elements, `channel` being each one's row-major flat index. */
  readonly rows = input<readonly KvHeadChannel[]>([]);
  /** The compared shape; its product is the number of elements. */
  readonly shape = input.required<readonly number[]>();
  /** The tensor's element dtype (e.g. `F32`, `F16`, `BF16`), used to scale ULP readouts. */
  readonly dtype = input<string | null>(null);
  readonly sideBars = input<SideBarMetric>('count');
  readonly tooltip = signal<ChartHoverPosition | null>(null);
  /** Rows × columns of elements per cell at the last paint. */
  readonly cell = signal({rows: 1, cols: 1});
  private readonly plot = viewChild<ElementRef<PlotHost>>('plot');
  private readonly wrap = viewChild<ElementRef<HTMLElement>>('wrap');
  private readonly zone = inject(NgZone);
  private readonly shapeKey = computed(() => this.shape().join(','));
  readonly layout = linkedSignal<string, TensorLayout>({
    source: this.shapeKey,
    computation: () => autoLayout(this.shape()),
  });
  /** The zoomed window, or null for the whole grid; a new shape or layout shows the whole grid. */
  readonly zoom = linkedSignal<unknown, GridWindow | null>({
    source: () => [this.shapeKey(), this.layout()],
    computation: () => null,
  });
  /** Whether the layout or the zoom differ from the default. */
  readonly changed = computed(
    () => this.zoom() != null || !sameLayout(this.layout(), autoLayout(this.shape())),
  );
  readonly grid = computed(() => tensorGrid(this.shape(), this.layout()));
  /** Each element's row, by flat index. */
  private readonly elements = computed(() => {
    const byFlat: (KvHeadChannel | undefined)[] = [];
    for (const row of this.rows()) {
      byFlat[row.channel] = row;
    }
    return byFlat;
  });
  private readonly points = computed(() => {
    const elements = this.elements();
    return gridPoints(
      this.grid(),
      (i) => elements[i]?.abs_delta ?? null,
      (i) => isNonComparableRow(elements[i]),
    );
  });
  readonly total = computed(() => this.grid().total);
  readonly differing = computed(() => this.points().length);
  readonly notComparable = computed(() => this.points().nonComparableCount);
  readonly differingText = computed(() => this.differing().toLocaleString());
  readonly totalText = computed(() => this.total().toLocaleString());
  readonly notComparableText = computed(() => this.notComparable().toLocaleString());
  readonly isBinned = computed(() => {
    const {rows, cols} = this.cell();
    return rows * cols > 1;
  });
  readonly maxText = computed(() => {
    const abs = this.points().abs;
    return formatMetric(abs.reduce((a, b) => Math.max(a, b), 0));
  });
  readonly dimensions = computed(() => {
    const shape = this.shape();
    const layout = this.layout();
    return shape.map((size, index) => {
      const role = dimensionRole(shape, index);
      const name = dimensionName(index);
      const side = sideOf(layout, index);
      const sideLabel = side === 'y' ? 'Y' : 'X';
      const nextSideLabel = side === 'y' ? 'X' : 'Y';
      return {
        index,
        size,
        role,
        name,
        side,
        sideTag: side === 'y' ? '↓ Y' : '→ X',
        ariaLabel: `Dimension ${name} of size ${size} on ${sideLabel}. Switch to ${nextSideLabel}`,
        tag: role === 'first' ? '↓ Y' : role === 'last' ? '→ X' : '',
        title:
          role === 'first'
            ? 'First dimension: always the outermost rows'
            : role === 'last'
              ? 'Last dimension: always the innermost columns'
              : null,
      };
    });
  });
  private readonly width = computed(() => this.surface.size().width);
  private readonly height = computed(() => this.surface.size().height);
  /** The heatmap's own height at the last paint, without labels, side bars and margins. */
  readonly heatmapHeight = signal(0);
  private view: HeatmapView | null = null;
  private readonly surface: PlotlySurface<HeatmapFrame> = new PlotlySurface({
    plot: () => this.plot()?.nativeElement,
    // The plot fills the space below the head line; the container sets its height.
    measured: () => this.wrap()?.nativeElement,
    ready: () => {
      const width = this.width();
      const grid = this.grid();
      if (width <= 0 || !grid.total) {
        return null;
      }
      return {
        grid,
        points: this.points(),
        window: this.zoom() ?? fullWindow(grid),
        metric: this.sideBars(),
        width,
        height: this.height(),
      };
    },
    draw: (host, frame) => this.draw(host, frame),
    bind: (host) => this.bind(host),
    failure: () => 'Difference heatmap unavailable. Switch the visualization to retry.',
  });
  readonly error = this.surface.error;

  constructor() {
    afterRenderEffect(() => {
      this.grid();
      this.points();
      this.zoom();
      this.sideBars();
      this.width();
      this.height();
      this.surface.theme();
      void this.surface.paint();
    });
  }

  toggle(dimension: number): void {
    this.tooltip.set(null);
    this.layout.set(toggleDimension(this.shape(), this.layout(), dimension));
  }

  /** Back to the automatic layout and the whole grid. */
  reset(): void {
    this.tooltip.set(null);
    this.layout.set(autoLayout(this.shape()));
    this.zoom.set(null);
  }

  /** Zooms the heatmap onto a small window centered on element `flat`. */
  focusElement(flat: number): void {
    const grid = this.grid();
    if (!Number.isInteger(flat) || flat < 0 || flat >= grid.total) {
      return;
    }
    const [row, col] = cellOf(grid, flat);
    const rowSpan = Math.min(grid.rows, 16);
    const colSpan = Math.min(grid.cols, 32);
    const r0 = Math.max(0, Math.min(grid.rows - rowSpan, row - Math.floor(rowSpan / 2)));
    const c0 = Math.max(0, Math.min(grid.cols - colSpan, col - Math.floor(colSpan / 2)));
    this.zoomTo({r0, r1: r0 + rowSpan, c0, c1: c0 + colSpan});
  }

  private async draw(host: PlotHost, frame: HeatmapFrame): Promise<void> {
    const {grid, points, window, metric, width} = frame;
    const header = headerGeometry(grid, window, width, frame.height);
    const heatmapHeight = header.heatmapHeight;
    const bins = binGrid({
      points,
      window,
      heightPx: heatmapHeight,
      widthPx: Math.max(200, header.plotWidth * HEATMAP_DOMAIN[1]),
    });
    const plotHeight = heatmapHeight + BARS_GAP + BARS_HEIGHT;
    const height = header.margin.t + plotHeight + header.margin.b;
    const style = rangeChartStyle(host, height, false);
    const color = (name: string) => plotlyColor(host, `var(${name})`);
    // The panel's heat palette (Intermediate Tensors sets the activation one), else that one.
    const low = plotlyColor(host, 'var(--heat-low, var(--heat-activation-low))');
    const high = plotlyColor(host, 'var(--heat-high, var(--heat-activation-high))');
    const bar = color('--me-primary-color');
    const ink = color('--me-on-surface-color');
    const strong = color('--me-outline-color');
    const alert = plotlyColor(host, 'var(--me-error-color, #c5221f)');
    const traces = buildHeatmapTraces({
      grid,
      bins,
      metric,
      colors: {low, high, bar, alert},
    });

    const yDomain = [(BARS_HEIGHT + BARS_GAP) / plotHeight, 1];
    const barsDomain = [0, BARS_HEIGHT / plotHeight];
    const nX = Math.max(1, header.xVisible.length);
    const rowBottom = (i: number) => 3 + (nX - 1 - i) * LABEL_ROW;
    const shapes: Record<string, unknown>[] = [];
    const annotations: Record<string, unknown>[] = [];
    const groups = new Map<number, GroupSpan>();
    const headerColors = {ink, bar, strong, rule: style.rule, muted: style.muted};
    appendSideHeaders({
      side: 'y',
      levels: header.yLevels,
      visible: header.yVisible,
      start: window.r0,
      end: window.r1,
      pxPerUnit: header.pxPerRow,
      header,
      plotHeight,
      barsDomainBottom: barsDomain[0],
      colors: headerColors,
      shapes,
      annotations,
      groups,
    });
    appendSideHeaders({
      side: 'x',
      levels: header.xLevels,
      visible: header.xVisible,
      start: window.c0,
      end: window.c1,
      pxPerUnit: header.pxPerCol,
      header,
      plotHeight,
      barsDomainBottom: barsDomain[0],
      colors: headerColors,
      shapes,
      annotations,
      groups,
    });
    const count = metric === 'count';
    const metricLabel = count ? 'differing count' : 'max |Δ|';
    annotations.push(
      {
        text: `Per row · ${metricLabel}`,
        x: ROW_BARS_DOMAIN[0],
        y: 1,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'bottom',
        yshift: rowBottom(0),
        showarrow: false,
      },
      {
        text: `Per column · ${metricLabel}`,
        x: 0,
        y: barsDomain[1],
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'bottom',
        yshift: 2,
        showarrow: false,
      },
    );
    const barAxis = {
      fixedrange: true,
      rangemode: 'tozero',
      gridcolor: style.rule,
      zeroline: false,
      ...(count ? {tickformat: 'd'} : {type: 'log', tickformat: '.0e'}),
    };
    await this.surface.react(host, traces, {
      ...style.layout,
      width,
      height,
      margin: header.margin,
      bargap: 0,
      dragmode: 'zoom',
      shapes,
      annotations,
      xaxis: {
        domain: HEATMAP_DOMAIN,
        anchor: 'y2',
        range: [window.c0 - 0.5, window.c1 - 0.5],
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        ticks: '',
      },
      yaxis: {
        domain: yDomain,
        range: [window.r1 - 0.5, window.r0 - 0.5],
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        ticks: '',
      },
      yaxis2: {...barAxis, domain: barsDomain, anchor: 'x', nticks: 3},
      xaxis2: {...barAxis, domain: ROW_BARS_DOMAIN, anchor: 'y', nticks: 2},
    });
    this.view = {...frame, bins, groups};
    if (heatmapHeight !== this.heatmapHeight()) {
      this.zone.run(() => this.heatmapHeight.set(heatmapHeight));
    }
    const cell = {rows: bins.rowBin, cols: bins.colBin};
    if (cell.rows !== this.cell().rows || cell.cols !== this.cell().cols) {
      this.zone.run(() => this.cell.set(cell));
    }
  }

  private bind(host: PlotHost): void {
    host.on('plotly_relayout', (event) => {
      const view = this.view;
      if (this.surface.painting || this.surface.destroyed || !view) {
        return;
      }
      const rangeBound = (key: string): number | undefined => {
        const value = event[key];
        return typeof value === 'number' ? value : undefined;
      };
      const x0 = rangeBound('xaxis.range[0]');
      const x1 = rangeBound('xaxis.range[1]');
      const y0 = rangeBound('yaxis.range[0]');
      const y1 = rangeBound('yaxis.range[1]');
      if (x0 == null && y0 == null) {
        return;
      }
      let {r0, r1, c0, c1} = view.window;
      if (x0 != null && x1 != null) {
        c0 = Math.max(0, Math.floor(Math.min(x0, x1) + 0.5));
        c1 = Math.min(view.grid.cols, Math.ceil(Math.max(x0, x1) + 0.5));
      }
      if (y0 != null && y1 != null) {
        r0 = Math.max(0, Math.floor(Math.min(y0, y1) + 0.5));
        r1 = Math.min(view.grid.rows, Math.ceil(Math.max(y0, y1) + 0.5));
      }
      if (c1 > c0 && r1 > r0) {
        this.zone.run(() => this.zoomTo({r0, r1, c0, c1}));
      }
    });
    host.on('plotly_doubleclick', () => this.zone.run(() => this.zoomTo(null)));
    host.on('plotly_clickannotation', (event) => {
      const view = this.view;
      const index = event['index'];
      const group = typeof index === 'number' ? view?.groups.get(index) : undefined;
      if (!view || !group) {
        return;
      }
      const next: GridWindow =
        group.side === 'x'
          ? {
              ...view.window,
              c0: group.start,
              c1: Math.min(view.grid.cols, group.end),
            }
          : {
              ...view.window,
              r0: group.start,
              r1: Math.min(view.grid.rows, group.end),
            };
      this.zone.run(() => this.zoomTo(next));
    });
    host.on('plotly_hover', (event) =>
      this.zone.run(() => {
        // Plotly's plotly_hover event supplies `points` as an array of hovered point records.
        const point = (event['points'] as {data?: {name?: string}; pointIndex?: unknown}[])?.[0];
        const content = point && this.view ? this.hoverContent(this.view, point) : null;
        if (!content) {
          return;
        }
        const mouse = event['event'] instanceof MouseEvent ? event['event'] : undefined;
        const rect = host.getBoundingClientRect();
        this.tooltip.set({
          left: (mouse?.clientX ?? rect.left + 60) - rect.left,
          top: Math.max(8, (mouse?.clientY ?? rect.top + 30) - rect.top),
          content,
        });
      }),
    );
    host.on('plotly_unhover', () => this.zone.run(() => this.tooltip.set(null)));
  }

  private zoomTo(window: GridWindow | null): void {
    this.tooltip.set(null);
    const view = this.view;
    const full = view && fullWindow(view.grid);
    const whole =
      !window ||
      (full &&
        window.r0 === full.r0 &&
        window.r1 === full.r1 &&
        window.c0 === full.c0 &&
        window.c1 === full.c1);
    this.zoom.set(whole ? null : window);
  }

  /** The hover readout of a heatmap cell or a side bar. */
  private hoverContent(
    view: HeatmapView,
    point: {data?: {name?: string}; pointIndex?: unknown},
  ): ChartHoverContent | null {
    const {grid, bins} = view;
    const name = point.data?.name;
    if (name === 'columns' && typeof point.pointIndex === 'number') {
      const j = point.pointIndex;
      const c0 = bins.c0 + j * bins.colBin;
      return {
        properties: [
          {label: 'Columns', value: hoverInterval(c0, Math.min(grid.cols, c0 + bins.colBin))},
          {label: 'Differing', value: String(bins.colCount[j])},
          {label: 'Max |Δ|', value: formatMetric(bins.colMax[j])},
        ],
      };
    }
    if (name === 'rows' && typeof point.pointIndex === 'number') {
      const i = point.pointIndex;
      const r0 = bins.r0 + i * bins.rowBin;
      return {
        properties: [
          {label: 'Rows', value: hoverInterval(r0, Math.min(grid.rows, r0 + bins.rowBin))},
          {label: 'Differing', value: String(bins.rowCount[i])},
          {label: 'Max |Δ|', value: formatMetric(bins.rowMax[i])},
        ],
      };
    }
    if ((name !== 'cells' && name !== 'non_comparable') || !Array.isArray(point.pointIndex)) {
      return null;
    }
    const [i, j]: unknown[] = point.pointIndex;
    if (typeof i !== 'number' || typeof j !== 'number') {
      return null;
    }
    const g = i * bins.nc + j;
    const r0 = bins.r0 + i * bins.rowBin;
    const c0 = bins.c0 + j * bins.colBin;
    const per = bins.rowBin * bins.colBin;
    const properties = [
      {label: 'Cell', value: per > 1 ? `max of ${per} elements` : '1 element'},
      {label: 'Differing', value: String(bins.count[g])},
    ];
    if (bins.nonComparable[g] > 0) {
      properties.push({label: 'Not comparable', value: String(bins.nonComparable[g])});
    }
    const flat = bins.argmax[g] >= 0 ? bins.argmax[g] : per === 1 ? elementAt(grid, r0, c0) : null;
    if (flat == null || flat >= grid.total) {
      properties.push(
        {label: 'Rows', value: hoverInterval(r0, Math.min(grid.rows, r0 + bins.rowBin))},
        {label: 'Columns', value: hoverInterval(c0, Math.min(grid.cols, c0 + bins.colBin))},
      );
      return {properties};
    }
    properties.push({label: 'Index', value: `[${unflatten(grid.shape, flat).join(', ')}]`});
    const row = this.elements()[flat];
    if (!row) {
      return {properties};
    }
    const rows = [
      {
        label: 'Value',
        values: [
          valueText(row.ref),
          valueText(row.target),
          valueText(row.delta),
          valueText(row.abs_delta),
        ],
      },
    ];
    if (bins.argmax[g] >= 0 && typeof row.ref === 'number' && typeof row.abs_delta === 'number') {
      const ulp = floatUlp(row.ref, this.dtype());
      if (ulp != null && ulp > 0) {
        rows.push({
          label: 'ULP',
          values: ['', '', '', String(Math.round(row.abs_delta / ulp))],
        });
      }
    }
    return {properties, columns: ['Ref', 'Target', 'Δ', '|Δ|'], rows};
  }
}
