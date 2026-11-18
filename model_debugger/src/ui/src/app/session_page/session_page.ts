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

import {ConversationPanel} from '../../conversation/conversation_panel/conversation_panel';
import {ChatConfiguration} from '../../home/chat_configuration/chat_configuration';
import {CaptureStore} from '../../server/capture_store';
import {WorkspaceStateService} from '../../state/workspace_state_service';

/** What the page says instead of the conversation when its address names none. */
interface Missing {
  title: string;
  text: string;
  action: string;
  go(): void;
}

/**
 * The page of a Session's address: the conversation of the chat it shows, or the New chat form;
 * while the Session list loads, a notice; and when the address names a Session or a chat the
 * workspace does not have, or nothing of the Session, what to open instead. Which chat it shows and which mode it is read in
 * are WorkspaceStateService's, after the server's rules.
 */
@Component({
  selector: 'session-page',
  imports: [MatButtonModule, ConversationPanel, ChatConfiguration],
  templateUrl: './session_page.ng.html',
  styleUrl: './session_page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SessionPage {
  readonly state = inject(CaptureStore);
  readonly workspace = inject(WorkspaceStateService);
  /** The conversation is made anew for each chat: its reading and saved views are the chat's. */
  readonly conversationKey = computed(() => this.workspace.captureId() ?? '');
  /** The New chat form is made anew for each Session, from its last chat as it opens; it keeps
   *  what was typed while the Session's chats change, as while its own start runs or failed. */
  readonly formKey = computed(() => this.workspace.session()?.id ?? '');
  readonly missing = computed<Missing | null>(() => {
    if (!this.workspace.sessions.listing()) {
      return null;
    }
    const record = this.workspace.session();
    if (!record) {
      return {
        title: 'Session not found',
        text: `This workspace has no Session ${this.workspace.selectedId()}. It may have been deleted.`,
        action: 'Go to Sessions',
        go: () => this.workspace.goHome(),
      };
    }
    const unmatched = this.workspace.address().unmatched;
    if (unmatched !== null) {
      return {
        title: 'Page not found',
        text: `This Session has no page at ${unmatched}.`,
        action: 'Go to the Session',
        go: () => this.workspace.open(record.id),
      };
    }
    if (this.workspace.chat() || this.workspace.newChat()) {
      return null;
    }
    return {
      title: 'Chat not found',
      text: `This Session has no Chat ${this.workspace.address().chat}.`,
      action: 'Go to the Session',
      go: () => this.workspace.open(record.id),
    };
  });
  reloadPage(): void {
    window.location.reload();
  }
}
