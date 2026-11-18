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

import {OverlayModule} from '@angular/cdk/overlay';
import {ChangeDetectionStrategy, Component, inject, signal, viewChild} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule, MatMenuTrigger} from '@angular/material/menu';
import {MatSnackBar} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';

import {ConfigurationOverview} from '../../home/configuration_overview/configuration_overview';
import {ModelServerSwitch} from '../../home/model_server_switch/model_server_switch';
import {CaptureStore} from '../../server/capture_store';
import {WorkspaceStateService} from '../../state/workspace_state_service';
import {ThemeService} from '../../theme/theme_service';
import {OverlayDialog, OverlayPanel} from '../../widgets/overlay_dialog/overlay_dialog';
import {ChatSwitcher} from '../chat_switcher/chat_switcher';

@Component({
  selector: 'app-toolbar',
  imports: [
    OverlayDialog,
    OverlayPanel,
    OverlayModule,
    ConfigurationOverview,
    ModelServerSwitch,
    ChatSwitcher,
    MatButtonModule,
    MatIconModule,
    MatMenuModule,
    MatTooltipModule,
  ],
  templateUrl: './app_toolbar.ng.html',
  styleUrl: './app_toolbar.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppToolbar {
  readonly state = inject(CaptureStore);
  readonly workspace = inject(WorkspaceStateService);
  readonly theme = inject(ThemeService);
  private readonly dialog = inject(MatDialog);
  private readonly snackBar = inject(MatSnackBar);
  private dialogFailure(label: string): void {
    this.snackBar
      .open(label + ' could not be loaded. Reload the page and try again.', 'Reload')
      .onAction()
      .subscribe(() => window.location.reload());
  }
  private readonly aboutTrigger = viewChild<MatMenuTrigger>('aboutTrigger');
  readonly themes = ['system', 'light', 'dark'] as const;
  readonly modeNames = ['Chat', 'Debug'] as const;
  readonly configurationOpen = signal(false);
  async settings(): Promise<void> {
    try {
      const {AppSettings} = await import('../app_settings/app_settings');
      this.dialog.open(AppSettings, {width: '466px', maxWidth: 'calc(100vw - 32px)'});
    } catch {
      this.dialogFailure('Settings');
    }
  }
  async about(): Promise<void> {
    try {
      const {AboutDialog} = await import('../about_dialog/about_dialog');
      this.dialog
        .open(AboutDialog, {width: '560px', maxWidth: 'calc(100vw - 32px)', restoreFocus: false})
        .afterClosed()
        .subscribe(() => this.aboutTrigger()?.focus('keyboard'));
    } catch {
      this.dialogFailure('About');
      this.aboutTrigger()?.focus('keyboard');
    }
  }
}
