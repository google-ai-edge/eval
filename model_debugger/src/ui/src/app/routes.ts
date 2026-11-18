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

import type {Routes} from '@angular/router';

import {HomePage} from '../home/home_page/home_page';
import {SessionPage} from './session_page/session_page';
import {matchSessionAddress} from './session_route';

/**
 * The application's pages by address (state/page_address/page_address.ts): the home page, and one
 * route for every address of a Session, so that its page, and the conversation in it, stays while
 * the mode, turn or analysis changes. An address that names no page goes home, keeping its
 * query (?debugPalette) and fragment.
 */
export const ROUTES: Routes = [
  {path: '', pathMatch: 'full', component: HomePage},
  {matcher: matchSessionAddress, component: SessionPage},
  // A relative redirect: the address keeps its query and fragment ('/' would drop them).
  {path: '**', redirectTo: ''},
];
