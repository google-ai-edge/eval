// Copyright 2026 The AI Edge Model Explorer Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftUI

/// The Runner's window, as the Model Debugger draws it.
///
/// Under the app bar: the device the Runner runs on, a card that says where
/// it stands and what to do, then its facts (where it listens, whom it
/// serves, the model, the turns). Facts not reported yet read as a dash.
struct RunnerView: View {
  let state: RunnerState?
  let failure: String?
  @Environment(\.colorScheme) private var scheme

  var body: some View {
    let theme = Theme.current(scheme)
    let (symbol, title, hint) = standing
    VStack(alignment: .leading, spacing: 24) {
      HStack(spacing: 12) {
        // The Model Debugger logo, the SVG of the Logo image set.
        Image("Logo").resizable().frame(width: 30, height: 30)
        Text("Model Debugger ") + Text("· ").foregroundStyle(theme.muted)
          + Text("Runner").foregroundStyle(theme.accent)
      }
      .font(.system(size: 20, weight: .semibold))
      HStack(spacing: 12) {
        // Monochrome: the outline the Model Debugger's pages draw a device with.
        Image(systemName: onPhone ? "iphone" : "display")
          .symbolRenderingMode(.monochrome)
          .font(.system(size: 22, weight: .light))
          .foregroundStyle(theme.secondary)
        VStack(alignment: .leading, spacing: 2) {
          Text(verbatim: state?.device ?? "—").font(.system(size: 16, weight: .semibold))
          Text(verbatim: state.map { "\($0.platform) \($0.osVersion)" } ?? "—")
            .font(.system(size: 12))
            .foregroundStyle(theme.muted)
        }
      }
      VStack(spacing: 12) {
        Image(systemName: symbol)
          .font(.system(size: 34))
          .foregroundStyle(failure == nil ? theme.accent : theme.negative)
          .frame(width: 76, height: 76)
          .background(
            failure == nil ? theme.accentWash : theme.negativeWash,
            in: RoundedRectangle(cornerRadius: 24)
          )
          .padding(.bottom, 10)
        Text(title).font(.system(size: 24, weight: .semibold))
        // The line is there in every state, empty if there is nothing to do,
        // so that the card, and the facts under it, keep their place.
        Text(hint ?? " ")
          .font(.system(size: 14))
          .foregroundStyle(theme.secondary)
          .frame(maxWidth: 330)
          .textSelection(.enabled)
      }
      .multilineTextAlignment(.center)
      .frame(maxWidth: .infinity)
      .padding(32)
      .background(theme.surface, in: RoundedRectangle(cornerRadius: 24))
      facts(
        [
          ("Listening", state.map { "\($0.address):\($0.port)" }),
          ("Server", state.flatMap { $0.client.isEmpty ? nil : $0.client }),
          (
            "Model",
            state.flatMap { $0.modelSha256.isEmpty ? nil : String($0.modelSha256.prefix(12)) }
          ),
          ("Turns", state.flatMap { $0.modelSha256.isEmpty ? nil : "\($0.turns)" }),
        ], theme)
      Spacer(minLength: 0)
    }
    .padding(24)
    .frame(maxWidth: 688, maxHeight: .infinity)
    .frame(maxWidth: .infinity)
    .background(theme.background)
    .foregroundStyle(theme.text)
  }

  /// Whether the device is a phone: the App is built for one, and the Runner
  /// runs inside the App.
  private var onPhone: Bool {
    #if os(iOS)
      true
    #else
      false
    #endif
  }

  /// Named facts, one per row; a fact not reported yet reads as a dash.
  private func facts(_ rows: [(String, String?)], _ theme: Theme) -> some View {
    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
      ForEach(rows, id: \.0) { name, value in
        GridRow {
          Text(name).foregroundStyle(theme.muted).frame(width: 72, alignment: .leading)
          Text(verbatim: value ?? "—").textSelection(.enabled)
        }
      }
    }
    .font(.system(size: 12, design: .monospaced))
  }

  /// The card's symbol, its title, and what the user should do, if anything.
  private var standing: (String, String, String?) {
    if let failure { return ("exclamationmark.circle", "Failed to start", failure) }
    guard let state else { return ("hourglass", "Starting", nil) }
    if state.chatOver {
      return ("stop.circle", "Chat over", "Start a new chat in Model Debugger.")
    }
    if state.client.isEmpty {
      return ("powerplug", "Waiting for connection", "Select this device in Model Debugger.")
    }
    if state.modelSha256.isEmpty { return ("checkmark.circle", "Connected", nil) }
    return state.turns == 0 ? ("cube", "Model loaded", nil) : ("bolt.circle", "Serving", nil)
  }
}
