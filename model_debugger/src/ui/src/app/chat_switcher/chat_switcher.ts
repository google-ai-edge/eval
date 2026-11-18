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

import {ChangeDetectionStrategy, Component, computed, inject} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatTooltipModule} from '@angular/material/tooltip';

import type {ChatSummary, Permission} from '../../server/contracts/sessions';
import {WorkspaceStateService} from '../../state/workspace_state_service';

/** Why no chat starts while a task of the Session runs: the server's word for it. */
const BUSY = 'Wait for the current task to finish.';

/** A chat's state in words, as the switcher lists it: live, or ended and why. */
export function chatState(chat: ChatSummary): string {
  if (chat.state === 'live') {
    return 'Live';
  }
  return chat.notice ? 'Ended · ' + chat.notice : 'Ended';
}

/**
 * The Session's chats in the toolbar: which one the page shows, a menu of them all, Chat 1 to N
 * with their settings and whether each is live or ended and why, and New chat, which opens the
 * New chat form. New chat is refused as the server refuses a chat's start, and while a task of
 * the Session runs, a turn among them: one turn of a Session at a time.
 */
@Component({
  selector: 'chat-switcher',
  imports: [MatButtonModule, MatIconModule, MatMenuModule, MatTooltipModule],
  templateUrl: './chat_switcher.ng.html',
  styleUrl: './chat_switcher.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatSwitcher {
  readonly workspace = inject(WorkspaceStateService);
  readonly chats = computed(() => this.workspace.session()?.chats ?? []);
  readonly state = chatState;
  /** What the switcher's button says: the chat shown, or the New chat form. */
  readonly label = computed(() => {
    const chat = this.workspace.chat();
    return chat ? chat.title : this.workspace.newChat() ? 'New chat' : 'No chat';
  });
  /** Whether New chat may be chosen now, and if not why. */
  readonly newChat = computed<Permission>(() => {
    const allowed = this.workspace.session()?.actions.new_chat ?? {allowed: false, reason: ''};
    return allowed.allowed && this.workspace.runtime.busy()
      ? {allowed: false, reason: BUSY}
      : allowed;
  });
  show(chat: ChatSummary): void {
    this.workspace.showChat(chat.n);
  }
  openNewChat(): void {
    if (this.newChat().allowed) {
      this.workspace.showNewChat();
    }
  }
}
