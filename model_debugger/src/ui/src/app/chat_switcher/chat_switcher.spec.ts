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

import type {ChatSummary, SessionSummary} from '../../server/contracts/sessions';
import {allowed, chat, refused, session} from '../../server/testing/session_fixtures';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {ChatSwitcher, chatState} from './chat_switcher';

describe('ChatSwitcher', () => {
  const first = chat(1, {state: 'ended', has_capture: true, notice: 'A new chat was started.'});
  const second = chat(2);
  const record = (extra: Partial<SessionSummary> = {}) =>
    session('s', {
      chats: [first, second],
      live_chat: 2,
      actions: {...session().actions, new_chat: allowed},
      ...extra,
    });

  function create() {
    const visits: (number | 'new')[] = [];
    const workspace = {
      session: signal(record()),
      chat: signal<ChatSummary | null>(second),
      newChat: signal(false),
      runtime: {busy: signal(false)},
      showChat: (n: number) => visits.push(n),
      showNewChat: () => visits.push('new'),
    };
    TestBed.configureTestingModule({
      providers: [{provide: WorkspaceStateService, useValue: workspace}],
    });
    const switcher = TestBed.runInInjectionContext(() => new ChatSwitcher());
    return {switcher, workspace, visits};
  }

  it('lists every chat with its state: live, or ended and why', () => {
    const {switcher, workspace} = create();
    expect(switcher.chats().map((c) => c.title)).toStrictEqual(['Chat 1', 'Chat 2']);
    expect(chatState(first)).toBe('Ended · A new chat was started.');
    expect(chatState(second)).toBe('Live');
    expect(chatState(chat(3, {state: 'ended', notice: ''}))).toBe('Ended');
    expect(switcher.label()).toBe('Chat 2');
    workspace.chat.set(null);
    workspace.newChat.set(true);
    expect(switcher.label()).toBe('New chat');
  });

  it('goes to the chat chosen, and to the New chat form', () => {
    const {switcher, visits} = create();
    switcher.show(first);
    switcher.openNewChat();
    expect(visits).toStrictEqual([1, 'new']);
  });

  it('New chat is refused while a turn runs, and as the server refuses a chat', () => {
    const {switcher, workspace, visits} = create();
    expect(switcher.newChat()).toStrictEqual(allowed);
    workspace.runtime.busy.set(true);
    expect(switcher.newChat()).toStrictEqual(refused('Wait for the current task to finish.'));
    switcher.openNewChat();
    expect(visits).toStrictEqual([]);
    workspace.runtime.busy.set(false);
    workspace.session.set(
      record({actions: {...session().actions, new_chat: refused('This Session has ended.')}}),
    );
    expect(switcher.newChat()).toStrictEqual(refused('This Session has ended.'));
  });
});
