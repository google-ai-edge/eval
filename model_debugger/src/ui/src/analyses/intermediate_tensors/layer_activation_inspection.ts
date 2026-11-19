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
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  linkedSignal,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {OverlayModule} from '@angular/cdk/overlay';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import type {
  PositionTensorResponse,
  PositionTensorRow,
} from '../../server/contracts/position_analysis';
import type {KvPositionAlignment, KvRangeBin} from '../../server/contracts/kv';
import {reasonText} from '../../server/reasons';
import {PositionAnalysisApi} from '../../server/position_analysis_api';
import {CaptureStore} from '../../server/capture_store';
import {EM_DASH, formatMetric} from '../../widgets/format/format';
import {ConfigPicker} from '../../widgets/config_picker/config_picker';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {animateAnalysisLocationFeedback} from '../../widgets/locate_feedback';
import {LatestRequest} from '../../server/latest_request';
import {ChannelChart} from '../../widgets/charts/channel_chart';
import {ScatterChart} from '../../widgets/charts/scatter_chart';
import {TensorHeatmap, type SideBarMetric} from '../../widgets/charts/tensor_heatmap';
import {channelBinSize, type ChannelReduction} from '../../widgets/charts/channel_bins';
import {scatterExtent, validValueRange, type ValueRange} from '../../widgets/charts/scatter_model';
import {PositionRangeControl} from '../../widgets/charts/position_range_control';
import {ValueRangeControl} from '../../widgets/charts/value_range_control';
import {ViewActions} from '../../widgets/charts/view_actions';
import {RuntimeLegend} from '../../widgets/charts/runtime_legend';
import {TENSOR_POINT_LABELS} from './tensor_labels';
import {
  hasSyncPointInLayer,
  syncPointDescription,
  syncPointInLayer,
  syncPointLabel,
  syncPointsForLayer,
  type SyncPointInfo,
} from './sync_points';
import {
  TensorIndexControl,
  type TensorIndexChange,
  type TensorIndexDimension,
} from '../../widgets/charts/tensor_index_control';

import {
  INSPECTION_PLOTS,
  INSPECTION_PLOT_LABELS,
  isInspectionPlot,
  type InspectionPlot,
} from './position_tensor_model';

export {INSPECTION_PLOTS, INSPECTION_PLOT_LABELS, isInspectionPlot, type InspectionPlot};

export interface LayerActivationInspectionState {
  plot: InspectionPlot;
  position: number;
  layer: number | null;
  anchor?: string;
  elementRange?: KvRangeBin | null;
  valueRange?: ValueRange | null;
  selection?: {channel: number} | null;
  reduction?: ChannelReduction;
  visibleRuntimes?: readonly string[];
}

@Component({
  selector: 'layer-activation-inspection',
  imports: [
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    ConfigPicker,
    OverlayModule,
    OverlayDialog,
    OverlayPanel,
    ViewActions,
    RuntimeLegend,
    TensorIndexControl,
    PositionRangeControl,
    ValueRangeControl,
    ChannelChart,
    ScatterChart,
    TensorHeatmap,
  ],
  template: ` <div
      class="controls"
      role="toolbar"
      aria-label="Intermediate Tensors inspection controls"
    >
      <div class="toolbar-scroll">
        <div class="toolbar-view">
          <span>View</span>
          <config-picker
            class="view-picker"
            label="Visualization"
            plural="visualizations"
            [value]="visualizationLabel()"
            [attr.title]="visualizationLabel()"
            [options]="visualizationOptions"
            [groups]="visualizationGroups"
            [optionLabels]="visualizationLabels"
            [compact]="true"
            [searchable]="false"
            (valueChange)="chooseView($event)"
          />
        </div>
        <span class="tool-divider" aria-hidden="true"></span>
        <label class="position-control"
          >Token Position
          <button
            type="button"
            class="icon-button"
            aria-label="Previous tensor position"
            [disabled]="position() <= fullRange().start"
            (click)="setPosition(position() - 1)"
          >
            <mat-icon>chevron_left</mat-icon>
          </button>
          <input
            type="number"
            step="1"
            aria-label="Token Position"
            [min]="fullRange().start"
            [max]="fullRange().end - 1"
            [value]="position()"
            (change)="commitPosition($event)"
            (keydown.enter)="commitPosition($event)"
          />
          <button
            type="button"
            class="icon-button"
            aria-label="Next tensor position"
            [disabled]="position() >= fullRange().end - 1"
            (click)="setPosition(position() + 1)"
          >
            <mat-icon>chevron_right</mat-icon>
          </button>
        </label>
        <div class="toolbar-view layer-control">
          <span>Tensor Index</span>
          <tensor-index-control
            [dimensions]="tensorIndexDimensions()"
            (valueChange)="setTensorIndex($event)"
          />
        </div>
        <div
          class="toolbar-range"
          [attr.aria-label]="
            plot() === 'index'
              ? 'Element Range controls'
              : plot() === 'delta'
                ? 'Difference controls'
                : 'Value Range controls'
          "
          role="group"
        >
          @if (plot() === 'index') {
            <position-range-control
              label="Element Range"
              [range]="elementRange()"
              [full]="elementFull()"
              (rangeChange)="zoomElements($event)"
            />
            <view-actions
              [range]="elementRange()"
              [full]="elementFull()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          } @else if (plot() === 'delta') {
            <span>Side bars</span>
            <config-picker
              class="view-picker side-bars-picker"
              label="Side bars"
              plural="metrics"
              [value]="sideBarLabels[sideBars()]"
              [options]="sideBarOptions"
              [compact]="true"
              [searchable]="false"
              (valueChange)="setSideBars($event)"
            />
            <view-actions
              [modified]="heatmapChanged()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          } @else {
            <value-range-control [range]="valueRange()" (rangeChange)="zoomValues($event)" />
            <view-actions
              [range]="valueRange()"
              [full]="valueFull()"
              [expandable]="false"
              (resetView)="resetRange()"
            />
          }
        </div>
      </div>
      <div class="toolbar-actions">
        <div class="toolbar-legend">
          @if (plot() === 'index') {
            <runtime-legend [runtimes]="visibleRuntimes()" />
          } @else if (plot() === 'scatter') {
            <div class="tensor-axis-legend" aria-label="Tensor scatter axes">
              X: Ref · Y: Target
            </div>
          }
        </div>
        <button
          mat-icon-button
          class="view-display-button"
          aria-label="Display"
          [disabled]="plot() !== 'index'"
          [matTooltip]="
            plot() === 'scatter'
              ? 'No display settings for Scatter plot'
              : plot() === 'delta'
                ? 'No display settings for Difference'
                : 'Display · Bins: ' + (displayBinsDisabled() ? '-' : reductionLabel(reduction()))
          "
          cdkOverlayOrigin
          #displayOrigin="cdkOverlayOrigin"
          [attr.aria-expanded]="displayOpen() && plot() === 'index'"
          (click)="toggleDisplay()"
        >
          <mat-icon>visibility</mat-icon>
        </button>
        @if (!expanded()) {
          <view-actions class="panel-expand" [showReset]="false" (expandView)="expand.emit()" />
        }
      </div>
      <ng-template
        overlayDialog
        [origin]="displayOrigin"
        [open]="displayOpen() && plot() === 'index'"
        placement="below-end"
        [width]="360"
        (closed)="displayOpen.set(false)"
      >
        <div class="chart-display">
          <section class="settings-panel" overlayPanel aria-label="Chart display">
            <header>
              <strong>Display</strong>
              <button
                mat-icon-button
                aria-label="Close chart display"
                (click)="displayOpen.set(false)"
              >
                <mat-icon>close</mat-icon>
              </button>
            </header>
            <div class="setting">
              <label>Bins</label>
              <config-picker
                label="Bins"
                plural="functions"
                [value]="displayBinsDisabled() ? '-' : reductionLabel(displayDraft())"
                [disabled]="displayBinsDisabled()"
                [options]="['Minimum', 'Maximum', 'Mean']"
                [compact]="true"
                [searchable]="false"
                (valueChange)="setDisplayDraft($event)"
              />
              <button
                mat-button
                [disabled]="displayBinsDisabled() || displayDraft() === 'mean'"
                aria-label="Reset Bins to default"
                (click)="setDisplayDraft('Mean')"
              >
                Reset to default
              </button>
            </div>
            @if (plot() === 'index') {
              <div class="setting">
                <div class="series-heading">
                  <label>Runtimes</label
                  ><span class="series-count">{{ displayRuntimes().length }}/2</span>
                </div>
                <div class="series-options" role="group" aria-label="Visible runtimes">
                  @for (series of runtimeSeries; track series.id) {
                    <label>
                      <input
                        type="checkbox"
                        [attr.aria-label]="'Show ' + series.label"
                        [checked]="displayRuntimes().includes(series.id)"
                        (change)="toggleDisplayRuntime(series.id)"
                      />
                      <i [style.border-top-color]="series.color" [class.dashed]="series.dashed"></i>
                      <span>{{ series.label }}</span>
                    </label>
                  }
                </div>
                <div class="series-actions">
                  <button
                    mat-button
                    [disabled]="displayRuntimes().length === 2"
                    aria-label="Reset Runtimes to default"
                    (click)="displayRuntimes.set(['ref', 'target'])"
                  >
                    Reset to default
                  </button>
                  <button
                    mat-button
                    [disabled]="!displayRuntimes().length"
                    aria-label="Clear Runtimes"
                    (click)="displayRuntimes.set([])"
                  >
                    Clear
                  </button>
                </div>
              </div>
            }
            <footer>
              <span role="status" class="sr-only">{{
                displayDirty() ? 'Unapplied changes' : 'Display settings applied'
              }}</span>
              <button
                mat-flat-button
                color="primary"
                [disabled]="!displayDirty()"
                (click)="applyDisplay()"
              >
                Apply
              </button>
            </footer>
          </section>
        </div>
      </ng-template>
    </div>
    <section
      class="results"
      aria-label="Intermediate Tensors tensor visualization"
      [attr.aria-busy]="loading()"
    >
      @if (loading()) {
        <p class="state-message" role="status">Loading intermediate tensor…</p>
      } @else if (error()) {
        <p class="state-message" role="alert">
          {{ error() }} <button mat-button (click)="retry()">Retry</button>
        </p>
      } @else if (!total()) {
        <p class="state-message">
          {{ notices().join(' ') || 'No intermediate tensor elements are available.' }}
        </p>
      } @else {
        @for (notice of notices(); track notice) {
          <p class="state-message">{{ notice }}</p>
        }
        @if (plot() === 'delta') {
          @if (deltaShape(); as shape) {
            <div class="tensor-line-canvas difference" [class.expanded]="expanded()">
              <tensor-heatmap
                [rows]="channels()"
                [shape]="shape"
                [dtype]="data()?.target?.dtype ?? data()?.reference?.dtype ?? null"
                [sideBars]="sideBars()"
              />
            </div>
          } @else {
            <p class="state-message">
              {{ scatterMessages()[0] || 'Tensor shape is unavailable for the Difference heatmap.' }}
            </p>
          }
        } @else {
          <div class="tensor-line-canvas" [class.expanded]="expanded()">
            @if (plot() === 'index') {
              <channel-chart
                dimensionLabel="Element index"
                [groupLabels]="['Intermediate Tensors']"
                [comparisonAllowed]="compatible()"
                [rows]="channels()"
                [visibleRuntimes]="visibleRuntimes()"
                [range]="elementRange()"
                [full]="elementFull()"
                [reduction]="reduction()"
                [capacity]="capacity()"
                (capacityChange)="capacity.set($event)"
                [selected]="selectedIndex()"
                [channelSelection]="channelSelection()"
                (channelSelectionChange)="channelSelection.set($event)"
                (zoom)="zoomElements($event)"
                (select)="showElement($event)"
                (backgroundClick)="inspectStage.emit($event)"
              />
            } @else {
              <scatter-chart
                dimensionLabel="Element index"
                [groupLabels]="['Intermediate Tensors']"
                [singlePlane]="true"
                [showGroupingNotice]="true"
                [rows]="scatterChannels()"
                [range]="valueRange()"
                [selected]="selectedIndex()"
                [messages]="scatterMessages()"
                (select)="showElement($event)"
                (zoom)="zoomValues($event)"
                (backgroundClick)="inspectStage.emit($event)"
              />
            }
          </div>
        }
        @if (plot() === 'index' && binSize() > 1) {
          <p class="plot-caption">
            {{ reductionLabel(reduction()) }} · up to {{ binSize() }} elements per bin
          </p>
        }
      }
      <span class="sr-only" aria-live="polite">{{ announcement() }}</span>
    </section>`,
  styleUrl: '../shared/tensor_inspection.scss',
  styles: `
    .layer-control tensor-index-control {
      width: 320px;
    }
    /* The Difference heatmap's labels and side bars need more height than a line chart. */
    .tensor-line-canvas.difference:not(.expanded) {
      height: 560px;
    }
  `,
})
export class LayerActivationInspection {
  readonly turn = input.required<number>();
  readonly stage = input.required<string>();
  readonly alignment = input<KvPositionAlignment | null>(null);
  readonly fullRange = input.required<KvRangeBin>();
  readonly layers = input.required<number[]>();
  readonly anchors = input<readonly string[]>([]);
  /** The server's label of each sync point. */
  readonly anchorLabels = input<Readonly<Record<string, string>>>({});
  /** Which layers each sync point is of, and its tensor; a point chosen inside the layers is one
   *  layer's, so Tensor Index offers it with that layer only. */
  readonly anchorPoints = input<Readonly<Record<string, SyncPointInfo>>>({});
  /** The stage's compared tensor rows, for showing per-sync-point max |Δ| in Tensor Index. */
  readonly analysisRows = input<readonly PositionTensorRow[]>([]);
  readonly initial = input<LayerActivationInspectionState | null>(null);
  readonly initialPosition = input<number | null>(null);
  readonly initialLayer = input<number | null>(null);
  readonly expanded = input(false);
  readonly compare = output<'heatmap' | 'lineChart' | 'graph'>();
  readonly inspect = output<LayerActivationInspection>();
  readonly stateChange = output<LayerActivationInspection>();
  readonly expand = output<void>();
  readonly inspectStage = output<MouseEvent>();
  readonly request = new LatestRequest<PositionTensorResponse>();
  readonly data = this.request.data;
  readonly loading = this.request.loading;
  readonly error = this.request.error;
  readonly valuesLoading = this.loading;
  readonly valuesError = this.error;
  private readonly api = inject(PositionAnalysisApi);
  private readonly state = inject(CaptureStore);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly retryCount = signal(0);
  private readonly chosenPosition = signal<number | null>(null);
  private readonly chosenLayer = signal<number | null>(null);
  readonly anchor = linkedSignal<
    {layer: number | null; points: Readonly<Record<string, SyncPointInfo>>},
    string
  >({
    source: () => ({layer: this.layer(), points: this.anchorPoints()}),
    computation: ({layer, points}, previous) => {
      const current = previous?.value ?? 'output';
      const point = Object.hasOwn(points, current) ? points[current] : undefined;
      if (hasSyncPointInLayer(point, layer)) {
        return current;
      }
      return syncPointInLayer(current, this.syncPointsOf(layer), points) ?? current;
    },
  });
  readonly plot = signal<InspectionPlot>('index');
  readonly position = computed(() => {
    const range = this.fullRange();
    const value = this.chosenPosition();
    return Math.max(
      range.start,
      Math.min(
        Math.max(range.start, range.end - 1),
        value !== null && Number.isFinite(value) ? Math.floor(value) : range.start,
      ),
    );
  });
  readonly layer = computed(() => {
    const selected = this.chosenLayer();
    const layers = this.layers();
    return selected !== null && layers.includes(selected) ? selected : (layers[0] ?? null);
  });
  readonly tensorIndexDimensions = computed<TensorIndexDimension[]>(() => [
    {
      id: 'layer',
      label: 'Layer',
      value: this.layer() === null ? null : String(this.layer()),
      options: this.layers().map(String),
    },
    {
      id: 'anchor',
      label: 'Sync point',
      value: this.anchor(),
      options: this.syncPoints(),
      optionLabels: this.syncPointLabels(),
      optionDescriptions: this.syncPointDescriptions(),
      optionBadges: this.syncPointDiffs().badges,
      highlightedOptions: this.syncPointDiffs().highlighted,
    },
  ]);
  /** Returns the sync points of `layer`: unknown anchors and anchors observed in `layer`. */
  private syncPointsOf(layer: number | null): string[] {
    const anchors = this.anchors().length ? this.anchors() : ['output'];
    return syncPointsForLayer({
      anchors,
      info: this.anchorPoints(),
      layer,
    });
  }
  private readonly syncPoints = computed(() => this.syncPointsOf(this.layer()));
  /** Short labels: `Linear_o_proj · decode:183:0`; named anchors keep the server's label. */
  private readonly syncPointLabels = computed<Record<string, string>>(() => {
    const labels: Record<string, string> = {...TENSOR_POINT_LABELS, ...this.anchorLabels()};
    for (const anchor of this.syncPoints()) {
      labels[anchor] = syncPointLabel({
        anchor,
        info: this.anchorPoints()[anchor],
        serverLabel: labels[anchor],
      });
    }
    return labels;
  });
  private readonly syncPointDescriptions = computed<Record<string, string>>(() => {
    const descriptions: Record<string, string> = {};
    const points = this.anchorPoints();
    for (const anchor of this.syncPoints()) {
      const description = syncPointDescription(anchor, points[anchor]);
      if (description) {
        descriptions[anchor] = description;
      }
    }
    return descriptions;
  });
  private readonly syncPointDiffs = computed<{
    badges: Record<string, string>;
    highlighted: string[];
  }>(() => {
    const rows = this.analysisRows();
    const layer = this.layer();
    if (!rows.length || layer == null) {
      return {badges: {}, highlighted: []};
    }
    const position = this.position();
    const byAnchor = new Map<string, PositionTensorRow>();
    for (const row of rows) {
      if (row.position === position && row.layer === layer) {
        byAnchor.set(row.anchor, row);
      }
    }
    const badges: Record<string, string> = {};
    const highlighted: string[] = [];
    for (const anchor of this.syncPoints()) {
      const value = byAnchor.get(anchor)?.metrics['max_abs']?.value ?? null;
      if (typeof value === 'number' && Number.isFinite(value)) {
        badges[anchor] = formatMetric(value);
        if (value > 0) {
          highlighted.push(anchor);
        }
      } else {
        badges[anchor] = EM_DASH;
      }
    }
    return {badges, highlighted};
  });
  readonly selectedIndex = signal<number | null>(null);
  readonly selection = computed(() => {
    const index = this.selectedIndex();
    return index === null ? null : {channel: index};
  });
  readonly channelSelection = signal<(KvRangeBin & {plane: number | null}) | null>(null);
  readonly announcement = signal('');
  readonly lineChart = viewChild(ChannelChart);
  readonly scatterChart = viewChild(ScatterChart);
  private readonly heatmap = viewChild(TensorHeatmap);
  /** Whether the Difference heatmap's layout or zoom differ from its default. */
  readonly heatmapChanged = computed(() => this.heatmap()?.changed() ?? false);
  readonly sideBars = signal<SideBarMetric>('count');
  readonly sideBarOptions = ['Differing count', 'Max |Δ|'];
  readonly sideBarLabels: Readonly<Record<SideBarMetric, string>> = {
    count: 'Differing count',
    max: 'Max |Δ|',
  };
  readonly capacity = signal(128);
  readonly displayOpen = signal(false);
  readonly displayDraft = signal<ChannelReduction>('mean');
  readonly visibleRuntimes = signal<readonly string[]>(['ref', 'target']);
  readonly displayRuntimes = signal<string[]>(['ref', 'target']);
  readonly runtimeSeries = [
    {id: 'ref', label: 'Ref', color: 'var(--me-primary-color)', dashed: false},
    {id: 'target', label: 'Target', color: 'var(--me-on-surface-variant-color)', dashed: true},
  ];
  readonly reduction = signal<ChannelReduction>('mean');
  private readonly savedElementRange = signal<KvRangeBin | null>(null);
  private readonly savedValueRange = signal<ValueRange | null>(null);
  private locationAnimation?: Animation;
  /** The tensor's elements, both sides' values and their differences, as the server reads them. */
  readonly channels = computed(() => this.data()?.channels ?? []);
  readonly total = computed(() => this.channels().length);
  /** Whether the server compared the two sides element by element. */
  readonly compatible = computed(() => this.data()?.comparable ?? false);
  readonly scatterChannels = computed(() => (this.compatible() ? this.channels() : []));
  /** The compared shape the Difference heatmap lays out, or null when the sides were not compared
   *  element by element or the shape does not match the elements. */
  readonly deltaShape = computed(() => {
    const data = this.data();
    const shape = data?.reference?.shape ?? data?.target?.shape;
    return this.compatible() &&
      shape?.length &&
      shape.every((n) => Number.isSafeInteger(n) && n > 0) &&
      shape.reduce((a, b) => a * b, 1) === this.total()
      ? shape
      : null;
  });
  readonly scatterMessages = computed(() =>
    this.compatible()
      ? []
      : ['Matching Reference and Target tensor shapes are required for a paired plot.'],
  );
  readonly elementFull = computed<KvRangeBin>(() => ({start: 0, end: Math.max(1, this.total())}));
  readonly elementRange = computed(() => {
    const full = this.elementFull();
    const saved = this.savedElementRange();
    if (!saved) {
      return full;
    }
    const start = Math.max(0, Math.min(full.end - 1, Math.floor(saved.start)));
    return {start, end: Math.max(start + 1, Math.min(full.end, Math.ceil(saved.end)))};
  });
  readonly valueFull = computed(() => scatterExtent([this.scatterChannels()]));
  readonly valueRange = computed(() => this.savedValueRange() ?? this.valueFull());
  readonly binSize = computed(() => channelBinSize(this.elementRange(), this.capacity()));
  readonly displayBinsDisabled = computed(() => this.binSize() === 1);
  readonly displayDirty = computed(
    () =>
      (!this.displayBinsDisabled() && this.displayDraft() !== this.reduction()) ||
      this.displayRuntimes().length !== this.visibleRuntimes().length ||
      this.displayRuntimes().some((side) => !this.visibleRuntimes().includes(side)),
  );
  /** Why some of it is missing, in the server's words, one notice per reason. */
  readonly notices = computed(() =>
    (this.data()?.reasons ?? []).map((reason) => reasonText([reason])),
  );
  readonly visualizationLabel = computed(
    () => `Tensor Inspection · ${INSPECTION_PLOT_LABELS[this.plot()]}`,
  );
  readonly visualizationOptions = [
    'Tensor Comparison · Heatmap',
    'Tensor Comparison · Line chart',
    ...INSPECTION_PLOTS.map((plot) => `Tensor Inspection · ${INSPECTION_PLOT_LABELS[plot]}`),
    'Tensor Comparison · Graph Diff',
  ];
  readonly visualizationGroups = [
    {
      label: 'Tensor Comparison',
      options: [
        'Tensor Comparison · Heatmap',
        'Tensor Comparison · Line chart',
        'Tensor Comparison · Graph Diff',
      ],
    },
    {
      label: 'Tensor Inspection',
      options: INSPECTION_PLOTS.map(
        (plot) => `Tensor Inspection · ${INSPECTION_PLOT_LABELS[plot]}`,
      ),
    },
  ];
  readonly visualizationLabels: Readonly<Record<string, string>> = {
    'Tensor Comparison · Graph Diff': 'Graph Diff',
    'Tensor Comparison · Heatmap': 'Heatmap',
    'Tensor Comparison · Line chart': 'Line chart',
    ...Object.fromEntries(
      INSPECTION_PLOTS.map((plot) => [
        `Tensor Inspection · ${INSPECTION_PLOT_LABELS[plot]}`,
        INSPECTION_PLOT_LABELS[plot],
      ]),
    ),
  };
  constructor() {
    effect(() => {
      const saved = this.initial();
      const position = this.initialPosition();
      const layer = this.initialLayer();
      untracked(() => {
        this.plot.set(isInspectionPlot(saved?.plot) ? saved.plot : 'index');
        this.chosenPosition.set(saved?.position ?? position);
        this.chosenLayer.set(saved?.layer ?? layer);
        const initialAnchor = saved?.anchor ?? 'output';
        const activeLayer = this.layer();
        const points = this.anchorPoints();
        const point = Object.hasOwn(points, initialAnchor) ? points[initialAnchor] : undefined;
        this.anchor.set(
          hasSyncPointInLayer(point, activeLayer)
            ? initialAnchor
            : (syncPointInLayer(initialAnchor, this.syncPointsOf(activeLayer), points) ??
                initialAnchor),
        );
        this.savedElementRange.set(
          saved?.elementRange && validValueRange(saved.elementRange) ? saved.elementRange : null,
        );
        this.savedValueRange.set(
          saved?.valueRange && validValueRange(saved.valueRange) ? saved.valueRange : null,
        );
        this.selectedIndex.set(saved?.selection?.channel ?? null);
        this.reduction.set(saved?.reduction ?? 'mean');
        this.visibleRuntimes.set(
          saved?.visibleRuntimes?.filter((side) => side === 'ref' || side === 'target') ?? [
            'ref',
            'target',
          ],
        );
      });
    });
    let previous: string | null = null;
    effect((onCleanup) => {
      const capture = this.state.captureId();
      const turn = this.turn();
      const phase = this.stage();
      const alignment = this.alignment();
      const position = this.position();
      const layer = this.layer();
      const anchor = this.anchor();
      this.retryCount();
      const identity = JSON.stringify([capture, turn, phase, alignment, position, layer, anchor]);
      if (previous !== null && identity !== previous) {
        untracked(() => {
          this.clearElementSelection();
        });
      }
      previous = identity;
      onCleanup(
        this.request.load(
          capture && layer !== null && this.fullRange().end > this.fullRange().start
            ? async (signal) => {
                const response = await this.api.positionTensor(
                  capture,
                  turn,
                  phase,
                  {position, layer, anchor},
                  signal,
                  alignment,
                );
                if (
                  response.position !== position ||
                  response.layer !== layer ||
                  response.anchor !== anchor
                ) {
                  throw new Error(
                    'Tensor response did not match the selected position, layer, and sync point.',
                  );
                }
                return response;
              }
            : null,
          (data) => {
            const selected = this.selectedIndex();
            const total = data.channels.length;
            if (
              selected !== null &&
              (!Number.isInteger(selected) || selected < 0 || selected >= total)
            ) {
              this.clearElementSelection();
            }
          },
        ),
      );
    });
    effect(() => {
      this.captureView();
      this.data();
      this.loading();
      this.error();
      this.channelSelection();
      untracked(() => this.stateChange.emit(this));
    });
    inject(DestroyRef).onDestroy(() => {
      this.request.destroy();
      this.locationAnimation?.cancel();
    });
  }
  captureView(): LayerActivationInspectionState {
    return {
      plot: this.plot(),
      position: this.position(),
      layer: this.layer(),
      anchor: this.anchor(),
      elementRange: this.savedElementRange(),
      valueRange: this.savedValueRange(),
      selection: this.selection(),
      reduction: this.reduction(),
      visibleRuntimes: [...this.visibleRuntimes()],
    };
  }
  chooseView(label: string): void {
    this.displayOpen.set(false);
    if (label === 'Tensor Comparison · Graph Diff') {
      this.compare.emit('graph');
    } else if (label === 'Tensor Comparison · Heatmap') {
      this.compare.emit('heatmap');
    } else if (label === 'Tensor Comparison · Line chart') {
      this.compare.emit('lineChart');
    } else if (label === 'Tensor Inspection · Line chart') {
      this.plot.set('index');
    } else if (label === 'Tensor Inspection · Difference') {
      this.plot.set('delta');
    } else if (label === 'Tensor Inspection · Scatter plot') {
      this.plot.set('scatter');
      const selection = this.channelSelection();
      if (selection && selection.end - selection.start > 1) {
        this.channelSelection.set(null);
      }
    }
    if (label.startsWith('Tensor Inspection · ')) {
      this.inspect.emit(this);
    }
  }
  setPosition(value: number, inspect = true): void {
    const range = this.fullRange();
    if (!Number.isSafeInteger(value) || value < range.start || value >= range.end) {
      return;
    }
    this.chosenPosition.set(value);
    if (inspect) {
      this.inspect.emit(this);
    }
  }
  commitPosition(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.setPosition(input.valueAsNumber);
    input.value = String(this.position());
  }
  setTensorIndex(change: TensorIndexChange): void {
    if (change.dimension === 'layer') {
      this.setLayer(Number(change.value));
    } else if (change.dimension === 'anchor') {
      this.setAnchor(change.value);
    }
  }
  setLayer(layer: number): void {
    if (!this.layers().includes(layer)) {
      return;
    }
    const previousAnchor = this.anchor();
    this.chosenLayer.set(layer);
    // `anchor` is a linkedSignal that synchronously remaps to the matching operator in `layer`.
    if (this.anchor() !== previousAnchor) {
      this.clearElementSelection();
    }
    this.inspect.emit(this);
  }
  setAnchor(anchor: string, inspect = true): void {
    if (!this.syncPoints().includes(anchor)) {
      return;
    }
    if (anchor !== this.anchor()) {
      this.clearElementSelection();
      this.anchor.set(anchor);
    }
    if (inspect) {
      this.inspect.emit(this);
    }
  }
  retry(): void {
    this.retryCount.update((value) => value + 1);
  }
  showElement(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) {
      return;
    }
    this.selectedIndex.set(index);
    this.channelSelection.set(null);
    this.announcement.set(`Element ${index} selected.`);
    this.inspect.emit(this);
  }
  clearElementSelection(): void {
    this.selectedIndex.set(null);
    this.channelSelection.set(null);
    this.announcement.set('');
  }
  zoomElements(range: KvRangeBin): void {
    if (validValueRange(range)) {
      this.savedElementRange.set(range);
    }
  }
  zoomValues(range: ValueRange): void {
    if (validValueRange(range)) {
      this.savedValueRange.set(range);
    }
  }
  resetRange(): void {
    if (this.plot() === 'delta') {
      this.heatmap()?.reset();
    } else if (this.plot() === 'index') {
      this.savedElementRange.set(null);
    } else {
      this.savedValueRange.set(null);
    }
  }
  reductionLabel(value: ChannelReduction): string {
    return value === 'min' ? 'Minimum' : value === 'max' ? 'Maximum' : 'Mean';
  }
  setSideBars(label: string): void {
    this.sideBars.set(label === this.sideBarLabels.max ? 'max' : 'count');
  }
  toggleDisplay(): void {
    if (this.plot() !== 'index') {
      return;
    }
    if (this.displayOpen()) {
      this.displayOpen.set(false);
      return;
    }
    this.displayDraft.set(this.reduction());
    this.displayRuntimes.set([...this.visibleRuntimes()]);
    this.displayOpen.set(true);
  }
  setDisplayDraft(label: string): void {
    if (!this.displayBinsDisabled()) {
      this.displayDraft.set(label === 'Minimum' ? 'min' : label === 'Maximum' ? 'max' : 'mean');
    }
  }
  toggleDisplayRuntime(side: string): void {
    if (side !== 'ref' && side !== 'target') {
      return;
    }
    this.displayRuntimes.update((visible) =>
      visible.includes(side) ? visible.filter((value) => value !== side) : [...visible, side],
    );
  }
  applyDisplay(): void {
    if (!this.displayBinsDisabled()) {
      this.reduction.set(this.displayDraft());
    }
    this.visibleRuntimes.set([...this.displayRuntimes()]);
  }
  locateView(): void {
    this.locateControl(this.host.nativeElement.querySelector('.view-picker > button'));
  }
  locatePosition(): void {
    this.locateControl(this.host.nativeElement.querySelector('.position-control input'));
  }
  locateLayer(): void {
    this.locateControl(this.host.nativeElement.querySelector('tensor-index-control > button'));
  }
  locateElement(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.total()) {
      return;
    }
    this.showElement(index);
    if (this.plot() === 'delta') {
      this.heatmap()?.focusElement(index);
      return;
    }
    if (this.plot() === 'index') {
      const range = this.elementRange();
      if (index < range.start || index >= range.end || this.binSize() > 1) {
        const width = Math.max(1, Math.min(range.end - range.start, this.capacity()));
        const start = Math.max(0, Math.min(this.total() - width, index - Math.floor(width / 2)));
        this.savedElementRange.set({start, end: start + width});
      }
    } else {
      const row = this.channels()[index];
      const range = this.valueRange();
      if (
        typeof row.ref === 'number' &&
        typeof row.target === 'number' &&
        (row.ref < range.start ||
          row.target < range.start ||
          row.ref > range.end ||
          row.target > range.end)
      ) {
        this.savedValueRange.set(null);
      }
    }
    this.locateTensor();
  }
  locateTensor(): void {
    if (this.plot() !== 'delta') {
      (this.plot() === 'index' ? this.lineChart() : this.scatterChart())?.locateKind('key');
    }
  }
  private locateControl(target: HTMLElement | null): void {
    if (!target) {
      return;
    }
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!this.controlVisible(target)) {
      target.scrollIntoView({
        block: 'nearest',
        inline: 'nearest',
        behavior: reduced ? 'instant' : 'smooth',
      });
    }
    this.locationAnimation?.cancel();
    this.locationAnimation = animateAnalysisLocationFeedback(target, reduced);
  }
  private controlVisible(target: HTMLElement): boolean {
    const rect = target.getBoundingClientRect();
    let top = 0;
    let left = 0;
    let right = window.innerWidth;
    let bottom = window.innerHeight;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      const bounds = parent.getBoundingClientRect();
      if (/auto|scroll|hidden|clip/.test(style.overflowX)) {
        left = Math.max(left, bounds.left + parent.clientLeft);
        right = Math.min(right, bounds.left + parent.clientLeft + parent.clientWidth);
      }
      if (/auto|scroll|hidden|clip/.test(style.overflowY)) {
        top = Math.max(top, bounds.top + parent.clientTop);
        bottom = Math.min(bottom, bounds.top + parent.clientTop + parent.clientHeight);
      }
    }
    return rect.top >= top && rect.bottom <= bottom && rect.left >= left && rect.right <= right;
  }
}
