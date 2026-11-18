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

import {Route, UrlSegment, UrlSegmentGroup} from '@angular/router';

import {matchSessionAddress} from './session_route';

const SESSION_ADDRESSES = [
  '/sessions/s1',
  '/sessions/s1/debug',
  '/sessions/s1/debug/turns/3',
  '/sessions/s1/debug/turns/12/decode/final_logits',
];

describe('matchSessionAddress', () => {
  it('the Session route takes every path under a Session, and nothing else', () => {
    const segments = (path: string) =>
      path
        .split('/')
        .filter(Boolean)
        .map((part) => new UrlSegment(decodeURIComponent(part), {}));
    const match = (path: string) =>
      matchSessionAddress(segments(path), new UrlSegmentGroup([], {}), {} as Route);
    // One that names nothing of the Session is its page too, which says so.
    for (const path of [...SESSION_ADDRESSES, '/sessions/s1/chats/c2/debug']) {
      expect(match(path)?.consumed.length).toBe(segments(path).length);
    }
    for (const path of ['/', '/elsewhere', '/sessions']) {
      expect(match(path), path).toBe(null);
    }
  });
});
