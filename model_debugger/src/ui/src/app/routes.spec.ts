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

import {Component} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {Router, provideRouter} from '@angular/router';
import {RouterTestingHarness} from '@angular/router/testing';

import {ROUTES} from './routes';

@Component({selector: 'stub-page', template: ''})
class StubPage {}

/** Where the router lands from `url`: the application's routes, each page a stub. */
async function landing(url: string): Promise<string> {
  const harness = await RouterTestingHarness.create();
  await harness.navigateByUrl(url);
  return TestBed.inject(Router).url;
}

describe('application routes', () => {
  beforeEach(() =>
    TestBed.configureTestingModule({
      providers: [
        provideRouter(
          ROUTES.map((route) => (route.component ? {...route, component: StubPage} : route)),
        ),
      ],
    }),
  );

  // One harness per test: RouterTestingHarness refuses a second.
  it('redirects unknown paths home, keeping query and fragment', async () => {
    expect(await landing('/nowhere/at/all?debugPalette=1#top')).toBe('/?debugPalette=1#top');
  });

  it('redirects bare /sessions home, keeping the query', async () => {
    expect(await landing('/sessions?debugPalette=1')).toBe('/?debugPalette=1');
  });

  it("stays on a Session's page from a path under it that names nothing: the page says so", async () => {
    expect(await landing('/sessions/s1/elsewhere?debugPalette=1')).toBe(
      '/sessions/s1/elsewhere?debugPalette=1',
    );
  });

  it("stays on a Session's page from an address of a Chat by name", async () => {
    expect(await landing('/sessions/s1/chats/abc/debug')).toBe('/sessions/s1/chats/abc/debug');
  });

  it('matches a deep Session turn address without redirecting', async () => {
    expect(await landing('/sessions/s1/debug/turns/2?debugPalette=1')).toBe(
      '/sessions/s1/debug/turns/2?debugPalette=1',
    );
  });
});
