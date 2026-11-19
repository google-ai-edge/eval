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
import {
  hasSyncPointInLayer,
  syncPointDescription,
  syncPointInfo,
  syncPointInLayer,
  syncPointLabel,
  syncPointsForLayer,
} from './sync_points';

const PREFIX = 'litert_torch.generative.Qwen3/Qwen3DecoderLayer_';
/** An observed point of the stage, as position-analysis lists it. */
function point(
  id: string,
  layer: number | null,
  tensorName: string,
  places?: {layer: number; anchor: string}[],
): ObservedPoint {
  return {
    id: `${id}@${layer}`,
    layer,
    anchor: id,
    signature: 'decode',
    tensor_name: tensorName,
    shape: [1, 1, 8],
    context_axes: [1],
    dtype: 'F32',
    places,
    cells: 1,
    reasons: [],
  };
}
/** Layer 0 and 5 each with an output and two operators inside the layer. */
const POINTS: ObservedPoint[] = [
  point('output', 0, `${PREFIX}0/add;1`),
  point('output', 5, `${PREFIX}5/add;1`),
  point('decode:36:0', 0, `${PREFIX}0/litert_torch.attn.FusedQwen3Attention_self_attn;4`),
  point('decode:40:0', 0, `${PREFIX}0/Linear_o_proj;0`),
  point('decode:189:0', 5, `${PREFIX}5/litert_torch.attn.FusedQwen3Attention_self_attn;4`),
  point('decode:183:0', 5, `${PREFIX}5/Linear_o_proj;0`),
  point('decode:300:0', 5, `${PREFIX}5/Linear_down_proj;0`),
];
const ANCHORS = POINTS.map((p) => p.anchor).filter((a, i, all) => all.indexOf(a) === i);

describe('sync points by layer', () => {
  const info = syncPointInfo(POINTS);

  it('tracks observed layers for single-layer, multi-layer subset, and places points', () => {
    expect(info['output']).toEqual({
      layers: [0, 5],
      tensorName: `${PREFIX}0/add;1`,
      signature: 'decode',
    });
    expect(hasSyncPointInLayer(info['output'], 0)).toBe(true);
    expect(hasSyncPointInLayer(info['output'], 5)).toBe(true);
    expect(hasSyncPointInLayer(info['output'], 3)).toBe(false);
    expect(hasSyncPointInLayer(info['decode:36:0'], 0)).toBe(true);
    expect(hasSyncPointInLayer(info['decode:36:0'], 5)).toBe(false);
    expect(hasSyncPointInLayer(undefined, 2)).toBe(true);

    const shared = syncPointInfo([
      point('input', 1, 'x', [
        {layer: 0, anchor: 'output'},
        {layer: 1, anchor: 'input'},
      ]),
    ]);
    expect(shared['output']).toEqual({layers: [0], tensorName: 'x', signature: 'decode'});
    expect(shared['input']).toEqual({layers: [1], tensorName: 'x', signature: 'decode'});

    const manualNullLayer = syncPointInfo([
      point('manual:1', null, 'y', [{layer: 3, anchor: 'manual:1'}]),
    ]);
    expect(manualNullLayer['manual:1']).toEqual({
      layers: [3],
      tensorName: 'y',
      signature: 'decode',
    });
    expect(hasSyncPointInLayer(manualNullLayer['manual:1'], 3)).toBe(true);
    expect(hasSyncPointInLayer(manualNullLayer['manual:1'], 0)).toBe(false);
  });

  it('offers only anchors observed in the chosen layer plus unknown anchors', () => {
    expect(syncPointsForLayer({anchors: ANCHORS, info, layer: 0})).toEqual([
      'output',
      'decode:36:0',
      'decode:40:0',
    ]);
    expect(syncPointsForLayer({anchors: ANCHORS, info, layer: 5})).toEqual([
      'output',
      'decode:189:0',
      'decode:183:0',
      'decode:300:0',
    ]);

    // Multi-layer subset anchors (e.g. layers [0, 5]) are excluded on layers outside their subset.
    expect(syncPointsForLayer({anchors: ANCHORS, info, layer: 3})).toEqual([]);

    // A sync point no observed point names stays: nothing says which layer it is of.
    expect(
      syncPointsForLayer({
        anchors: ['output', 'attention_output'],
        info,
        layer: 5,
      }),
    ).toEqual(['output', 'attention_output']);
  });

  it('maps a point to the same operator of another layer, else output, else the first', () => {
    const layer5 = syncPointsForLayer({anchors: ANCHORS, info, layer: 5});
    const layer0 = syncPointsForLayer({anchors: ANCHORS, info, layer: 0});

    expect(syncPointInLayer('decode:36:0', layer5, info)).toBe('decode:189:0');
    expect(syncPointInLayer('decode:40:0', layer5, info)).toBe('decode:183:0');
    expect(syncPointInLayer('decode:189:0', layer0, info)).toBe('decode:36:0');
    expect(syncPointInLayer('decode:300:0', layer0, info)).toBe('output');
    expect(syncPointInLayer('decode:300:0', ['decode:36:0'], info)).toBe('decode:36:0');
    expect(syncPointInLayer('output', layer5, info)).toBe('output');

    const otherSignature = syncPointInfo([
      ...POINTS.slice(0, 3),
      {
        ...point(
          'prefill:189:0',
          5,
          `${PREFIX}5/litert_torch.attn.FusedQwen3Attention_self_attn;4`,
        ),
        signature: 'prefill',
      },
    ]);
    expect(syncPointInLayer('decode:36:0', ['output', 'prefill:189:0'], otherSignature)).toBe(
      'output',
    );
  });

  it('labels custom tap points by primary tensor name and id, named anchors by the server', () => {
    expect(
      syncPointLabel({
        anchor: 'decode:183:0',
        info: info['decode:183:0'],
        serverLabel: 'long server label',
      }),
    ).toBe('Linear_o_proj · decode:183:0');
    expect(
      syncPointLabel({
        anchor: 'decode:36:0',
        info: info['decode:36:0'],
      }),
    ).toBe('FusedQwen3Attention_self_attn · decode:36:0');

    // Fused TFLite tensor names split on `;` before `/` so the primary op name is used.
    const fused = syncPointInfo([
      point('decode:5:0', 7, 'stack/layer_7/layer_7.attn/add;fused/other'),
    ]);
    expect(syncPointLabel({anchor: 'decode:5:0', info: fused['decode:5:0']})).toBe(
      'add · decode:5:0',
    );
    expect(syncPointDescription('decode:5:0', fused['decode:5:0'])).toBe(
      'stack/layer_7/layer_7.attn/add;fused/other',
    );

    // Named anchors and manual pairs keep the server label even when observed in only 1 layer.
    const singleLayerNamed = syncPointInfo([point('input', 1, `${PREFIX}1/add;1`)]);
    expect(
      syncPointLabel({
        anchor: 'input',
        info: singleLayerNamed['input'],
        serverLabel: 'Layer input',
      }),
    ).toBe('Layer input');
    expect(syncPointDescription('input', singleLayerNamed['input'])).toBeNull();
    expect(
      syncPointLabel({
        anchor: 'output',
        info: info['output'],
        serverLabel: 'Layer output',
      }),
    ).toBe('Layer output');
    expect(
      syncPointLabel({
        anchor: 'manual:pair-1',
        info: info['decode:183:0'],
        serverLabel: 'Custom pair',
      }),
    ).toBe('Custom pair');
    expect(syncPointLabel({anchor: 'attention_output'})).toBe('attention_output');
  });
});
