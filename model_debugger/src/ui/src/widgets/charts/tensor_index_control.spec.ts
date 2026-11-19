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
import {TensorIndexControl} from './tensor_index_control';

describe('TensorIndexControl', () => {
  it('supports named dimensions beyond layers and heads and emits only valid coordinates', () => {
    const fixture = TestBed.createComponent(TensorIndexControl);
    const control = fixture.componentInstance;
    fixture.componentRef.setInput('dimensions', [
      {id: 'layer', label: 'Layer', value: '2', options: ['2', '3']},
      {id: 'sync', label: 'Sync point', value: 'attention', options: ['attention', 'mlp']},
    ]);
    const changes: unknown[] = [];
    control.valueChange.subscribe((change) => changes.push(change));
    expect(control.summary()).toBe('[Layer: 2, Sync point: attention]');
    control.choose('sync', 'mlp');
    control.choose('head', '0');
    control.choose('layer', '99');
    expect(changes).toEqual([{dimension: 'sync', value: 'mlp'}]);
    fixture.componentRef.setInput('dimensions', [
      {id: 'sync', label: 'Sync point', value: null, options: ['mlp'], disabled: true},
    ]);
    control.choose('sync', 'mlp');
    expect(changes).toHaveLength(1);
    expect(control.summary()).toBe('[Sync point: -]');
  });

  it('renders option badges and highlights non-zero options while keeping summary unchanged', () => {
    const fixture = TestBed.createComponent(TensorIndexControl);
    const control = fixture.componentInstance;
    fixture.componentRef.setInput('dimensions', [
      {id: 'layer', label: 'Layer', value: '0', options: ['0', '1']},
      {
        id: 'anchor',
        label: 'Sync point',
        value: 'output',
        options: ['output', 'decode:36:0'],
        optionLabels: {
          output: 'Layer output',
          'decode:36:0': 'FusedQwen3Attention_self_attn · decode:36:0',
        },
        optionBadges: {output: '0', 'decode:36:0': '5.96e-8'},
        highlightedOptions: ['decode:36:0'],
      },
    ]);
    fixture.detectChanges();

    expect(control.summary()).toBe('[Layer: 0, Sync point: Layer output]');

    (
      fixture.nativeElement.querySelector('button[aria-label="Tensor Index"]') as HTMLButtonElement
    ).click();
    fixture.detectChanges();
    (
      document.body.querySelector('button[aria-label="Tensor sync point"]') as HTMLButtonElement
    ).click();
    fixture.detectChanges();

    const options = Array.from(
      document.body.querySelectorAll<HTMLButtonElement>('[role="option"]'),
    );
    expect(options).toHaveLength(2);
    expect(options[0].classList.contains('option-highlighted')).toBe(false);
    expect(options[0].querySelector('.option-badge')?.textContent?.trim()).toBe('0');
    expect(options[0].querySelector('.option-badge.nonzero')).toBeNull();
    expect(options[1].classList.contains('option-highlighted')).toBe(true);
    expect(options[1].querySelector('.option-name')?.textContent?.trim()).toBe(
      'FusedQwen3Attention_self_attn · decode:36:0',
    );
    expect(options[1].querySelector('.option-badge.nonzero')?.textContent?.trim()).toBe('5.96e-8');
    fixture.destroy();
  });
});
