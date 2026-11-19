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

import type {ObservedPoint} from '../../server/contracts/position_analysis';

/**
 * Which layers a sync point is of, as the stage's observed points say. An architecture anchor
 * (`output`, a layer's input) is of every layer; a point chosen inside the layers (`decode:36:0`)
 * is one operator output of one layer.
 */
export interface SyncPointInfo {
  readonly layers: readonly number[];
  /** The tensor's name and signature, of the point's first observation. */
  readonly tensorName: string;
  readonly signature: string;
}

/** Collects each sync point's layers, tensor name, and signature from the stage's observed points. */
export function syncPointInfo(
  points: readonly ObservedPoint[],
): Readonly<Record<string, SyncPointInfo>> {
  const info: Record<string, {layers: Set<number>; tensorName: string; signature: string}> = {};
  const add = (anchor: string, layer: number, point: ObservedPoint) => {
    const entry = (info[anchor] ??= {
      layers: new Set(),
      tensorName: point.tensor_name,
      signature: point.signature,
    });
    entry.layers.add(layer);
  };
  for (const point of points) {
    if (point.places?.length) {
      for (const place of point.places) {
        add(place.anchor, place.layer, point);
      }
    } else if (point.layer != null) {
      add(point.anchor, point.layer, point);
    }
  }
  const result: Record<string, SyncPointInfo> = {};
  for (const [anchor, entry] of Object.entries(info)) {
    result[anchor] = {
      layers: [...entry.layers].sort((a, b) => a - b),
      tensorName: entry.tensorName,
      signature: entry.signature,
    };
  }
  return result;
}

/** Returns whether `anchor` is a custom tap-point id (`<signature>:<op>:<output>`). */
function isCustomTapPoint(anchor: string): boolean {
  return /^[^:]+:\d+:\d+$/.test(anchor);
}

/**
 * Returns whether `info` is valid for `layer`: true when `info` is absent (unknown anchor),
 * `layer` is null, or `info.layers` includes `layer`.
 */
export function hasSyncPointInLayer(
  info: SyncPointInfo | undefined,
  layer: number | null,
): boolean {
  return !info || layer == null || info.layers.includes(layer);
}

/** Options for selecting the sync points available in a layer. */
export interface SyncPointsForLayerOptions {
  readonly anchors: readonly string[];
  readonly info: Readonly<Record<string, SyncPointInfo>>;
  readonly layer: number | null;
}

/** Looks up `key` as an own property of `info`, ignoring `Object.prototype` keys. */
function ownSyncPoint(
  info: Readonly<Record<string, SyncPointInfo>>,
  key: string,
): SyncPointInfo | undefined {
  return Object.hasOwn(info, key) ? info[key] : undefined;
}

/** Returns the sync points of `layer`: every unknown anchor and every anchor observed in `layer`. */
export function syncPointsForLayer({anchors, info, layer}: SyncPointsForLayerOptions): string[] {
  return anchors.filter((anchor) => hasSyncPointInLayer(ownSyncPoint(info, anchor), layer));
}

/** Strips the layer index from a tensor name: `…DecoderLayer_5/…;4` → `…DecoderLayer_#/…;4`. */
function layerlessName(name: string): string {
  return name.replace(/(layers?[._]?)\d+/gi, '$1#');
}

/**
 * Resolves the sync point of `layer` corresponding to `anchor`: `anchor` itself when `layer` has
 * it; else, for a point of another layer, the point of `layer` of the same signature whose tensor
 * name matches apart from the layer index; else `output`, else the first of `options`.
 */
export function syncPointInLayer(
  anchor: string,
  options: readonly string[],
  info: Readonly<Record<string, SyncPointInfo>>,
): string | null {
  if (options.includes(anchor)) {
    return anchor;
  }
  const current = ownSyncPoint(info, anchor);
  if (current) {
    const name = layerlessName(current.tensorName);
    const match = options.find((option) => {
      const other = ownSyncPoint(info, option);
      return (
        !!other && other.signature === current.signature && layerlessName(other.tensorName) === name
      );
    });
    if (match) {
      return match;
    }
  }
  return options.includes('output') ? 'output' : (options[0] ?? null);
}

/**
 * Extracts the last segment of a tensor's primary path (before any `;` fused/output suffix)
 * without its module prefix: `…/litert_torch.generative….Linear_o_proj;4` → `Linear_o_proj`.
 */
function tensorShortName(name: string): string {
  const primary = name.split(';')[0] || name;
  const segment = primary.split('/').filter(Boolean).at(-1) ?? primary;
  return segment.split('.').filter(Boolean).at(-1) ?? segment;
}

/** Options for formatting a sync point's display label. */
export interface SyncPointLabelOptions {
  readonly anchor: string;
  readonly info?: SyncPointInfo;
  readonly serverLabel?: string;
}

/**
 * Formats a short label for a custom tap point (`Linear_o_proj · decode:183:0`); named anchors and
 * manual pairs keep the server's label.
 */
export function syncPointLabel({anchor, info, serverLabel}: SyncPointLabelOptions): string {
  if (!isCustomTapPoint(anchor) || !info?.tensorName) {
    return serverLabel ?? anchor;
  }
  return `${tensorShortName(info.tensorName)} · ${anchor}`;
}

/** Returns the tooltip description for a single-layer custom tap point, or null otherwise. */
export function syncPointDescription(
  anchor: string,
  info: SyncPointInfo | undefined,
): string | null {
  if (!isCustomTapPoint(anchor) || !info?.tensorName || info.layers.length !== 1) {
    return null;
  }
  return info.tensorName;
}
