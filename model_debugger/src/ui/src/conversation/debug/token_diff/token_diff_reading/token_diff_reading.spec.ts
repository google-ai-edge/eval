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
import {TestBed} from '@angular/core/testing';

import {ConversationState} from '../../../conversation_state';
import {CompactPreviews} from '../compact_previews';
import {TokenDiff} from '../token_diff';
import {TokenDiffDom} from '../token_diff_dom';
import {TokenDiffReading} from './token_diff_reading';

describe('TokenDiffReading', () => {
  it('tracks reading intent on pointerdown, wheel, and keydown without canceling native events', () => {
    const Pointer = globalThis.PointerEvent ?? MouseEvent;
    const conversation = {
      runs: () => [],
      turns: () => [],
      rejectedTurns: () => [],
      chatNotice: () => '',
      usesVirtual: () => false,
      trackReadingIntent: vi.fn(() => false),
      readingScrolled: () => undefined,
    };
    const tokens = {
      boundaries: () => true,
      color: () => 'token_match',
      heatPalette: () => 'default',
      details: () => false,
      panelWidth: () => 360,
    };
    const dom = Object.assign(Object.create(TokenDiffDom.prototype), {
      conversation,
      layoutTick: signal(0),
      readingScrolled: () => undefined,
      readingBackgroundClick: () => undefined,
      clearTokenFeedback: () => undefined,
    });
    TestBed.configureTestingModule({
      providers: [
        {provide: ConversationState, useValue: conversation},
        {provide: TokenDiff, useValue: tokens},
        {provide: CompactPreviews, useValue: {}},
        {provide: TokenDiffDom, useValue: dom},
      ],
    });
    const fixture = TestBed.createComponent(TokenDiffReading);
    fixture.detectChanges();
    const target = fixture.nativeElement.querySelector('.reading .column-headings') as HTMLElement;
    expect(target).not.toBeNull();

    const events = [
      new Pointer('pointerdown', {bubbles: true, cancelable: true}),
      new WheelEvent('wheel', {bubbles: true, cancelable: true, deltaY: -40}),
      new KeyboardEvent('keydown', {bubbles: true, cancelable: true, key: 'ArrowUp'}),
    ];
    for (const event of events) {
      target.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(false);
      expect(conversation.trackReadingIntent).toHaveBeenCalledWith(event);
    }
    fixture.destroy();
  });
});
