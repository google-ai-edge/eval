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

import {existsSync, readdirSync, readFileSync, statSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {strict as assert} from 'node:assert';

// Layers, lowest first: server/, then widgets/, then analyses/, then the pages home/ and
// conversation/, which import nothing of each other, then app/. A layer imports only from the
// layers below it. state/ (where the page is and what is attached there, and the browser's
// preferences) imports only server/ and serves the pages and app/; widgets/ and analyses/ import
// none of it. theme/ imports only state/ (ThemeService keeps its choice with PreferencesService);
// the pages and app/ may use all of it, widgets/ and analyses/ its stylesheets only, server/ and
// state/ none of it. Nothing but app/ itself and main.ts imports app/. Each analysis of analyses/
// (kv, final_logits, sampling, intermediate_tensors) imports no other analysis: what several share
// is in analyses/shared/, which imports none of them, or in widgets/. The pages and app/ reach
// analyses/ only through analyses/registry.ts, or through analyses/panel_state.ts, which reaches
// no component. In the conversation page (conversation/), its Chat reading (chat/) and its Debug
// reading (debug/) import nothing of each other, and its composer (composer/) neither of them;
// what they share is in conversation/ itself, which imports none of the three nor
// conversation_panel/, the page that puts them together. Tests count too, and so do plain
// JavaScript modules, the stylesheets and templates a component names, the files a module names
// with new URL(…, import.meta.url) and the stylesheets a stylesheet uses, forwards or imports.
const root = fileURLToPath(new URL('../src/', import.meta.url));
// Every folder and file directly under src/. One that is not here has no rules yet: give it its
// place among the layers above, then add it.
const LAYOUT = [
  ...['analyses', 'app', 'conversation', 'home', 'server', 'state', 'theme', 'widgets'],
  ...['main.ts', 'index.html', 'test_providers.ts', 'proxy.conf.json'],
];
const unknown = readdirSync(root).filter((name) => !LAYOUT.includes(name));
assert.deepEqual(
  unknown,
  [],
  'src/ holds a folder or file scripts/check_layering.mjs gives no rules: add it to its LAYOUT ' +
    'and to the rules',
);
const above = (...names) => names.map((name) => path.join(root, name) + path.sep);
const ANALYSES = ['kv', 'final_logits', 'sampling', 'intermediate_tensors'];
const registry = path.join(root, 'analyses', 'registry.ts');
const PAGE_PARTS = ['chat', 'debug', 'composer', 'conversation_panel'].map(
  (name) => `conversation/${name}`,
);
/** The conversation page's parts other than `name`. */
const otherParts = (name) => above(...PAGE_PARTS.filter((part) => part !== name));
const panelState = path.join(root, 'analyses', 'panel_state.ts');
const PAGES = ['home', 'conversation'];
const stylesheetOnly = (target) => /\.(scss|sass|css)$/.test(target);
const rules = [
  {from: 'server', forbidden: above('state', 'theme', 'widgets', 'analyses', ...PAGES)},
  {from: 'state', forbidden: above('theme', 'widgets', 'analyses', ...PAGES)},
  {from: 'theme', forbidden: above('server', 'widgets', 'analyses', ...PAGES)},
  {from: 'widgets', forbidden: above('state', 'analyses', ...PAGES)},
  {from: 'analyses', forbidden: above('state', ...PAGES)},
  // theme/'s code reaches state/ (ThemeService keeps its choice with PreferencesService), so below
  // the pages only its stylesheets are used.
  ...['widgets', 'analyses'].map((from) => ({
    from,
    forbidden: above('theme'),
    allowedIf: stylesheetOnly,
  })),
  ...PAGES.map((from) => ({from, forbidden: above(...PAGES.filter((page) => page !== from))})),
  // app/ is the top: no other module imports it, whatever its layer.
  {from: '.', except: [...above('app'), path.join(root, 'main.ts')], forbidden: above('app')},
  ...ANALYSES.map((name) => ({
    from: `analyses/${name}`,
    forbidden: [
      ...above(...ANALYSES.filter((other) => other !== name).map((other) => `analyses/${other}`)),
      registry,
      panelState,
    ],
  })),
  {
    from: 'analyses/shared',
    forbidden: [...above(...ANALYSES.map((name) => `analyses/${name}`)), registry, panelState],
  },
  ...[...PAGES, 'app'].map((from) => ({
    from,
    forbidden: above('analyses'),
    allowed: [registry, panelState],
  })),
  ...PAGE_PARTS.slice(0, 3).map((from) => ({from, forbidden: otherParts(from)})),
  {from: 'conversation', except: above(...PAGE_PARTS), forbidden: above(...PAGE_PARTS)},
];
function files(directory, pattern = /\.(ts|mjs|cjs|js|scss|sass|css)$/) {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return files(file, pattern);
    return pattern.test(entry.name) ? [file] : [];
  });
}
/** Whether `target` is `prefix` or, when `prefix` is a folder (it ends with a separator), the
 *  folder itself or anything in it: '../app' names src/app/ as much as '../app/routes' does. */
const within = (target, prefix) =>
  prefix.endsWith(path.sep) ? (target + path.sep).startsWith(prefix) : target === prefix;
const firstFile = (candidates) =>
  candidates.find((candidate) => existsSync(candidate) && statSync(candidate).isFile());
/** The file a script's relative specifier names, as TypeScript and Node find it. */
function resolveScript(base) {
  return (
    firstFile([
      base,
      ...['.ts', '.mjs', '.js', '.cjs'].map((extension) => base + extension),
      base.replace(/\.js$/, '.ts'),
      ...['index.ts', 'index.mjs', 'index.js'].map((index) => path.join(base, index)),
    ]) ?? base
  );
}
/** The file a stylesheet's specifier names, as Sass finds it relative to the file first: a partial
 *  (`_name`), the name itself, then the folder's index. Null when it is not under the file's folder
 *  and so comes from a package. */
function resolveSass(file, specifier) {
  const base = path.resolve(path.dirname(file), specifier);
  const name = path.basename(base);
  const found = firstFile([
    ...(/\.(scss|sass|css)$/.test(name) ? [base, path.join(path.dirname(base), `_${name}`)] : []),
    ...['scss', 'sass', 'css'].flatMap((extension) => [
      path.join(path.dirname(base), `_${name}.${extension}`),
      `${base}.${extension}`,
    ]),
    ...['_index.scss', 'index.scss', '_index.sass', 'index.sass'].map((index) =>
      path.join(base, index),
    ),
  ]);
  return found ?? (specifier.startsWith('.') ? base : null);
}
/** A string in one of `quotes`: its text is the group `text`. */
const quoted = (quotes) => String.raw`(?<q>[${quotes}])(?<text>(?:(?!\k<q>)[^\\\n]|\\.)*)\k<q>`;
/** In a static import or export, and in Sass, a string is quoted '…' or "…". */
const STRING = quoted(`'"`);
/** In code, a string may also be a template literal. */
const LITERAL = quoted('\'"\x60');
/** `import(…)` whose argument the check reads, and those it cannot: a specifier made at run time
 *  from a relative path or from nothing but variables (a template literal with ${…}, a
 *  concatenation, a variable). One made from a literal that is not a path, such as a data: URL
 *  bundled from an entry the module names with new URL(…), names no module. */
function dynamicImports(source) {
  const named = [];
  const unread = [];
  for (const match of source.matchAll(/\bimport\s*\(\s*/g)) {
    const rest = source.slice(match.index + match[0].length);
    const literal = rest.match(new RegExp(String.raw`^${LITERAL}\s*(?<end>[,)]?)`))?.groups;
    const template = literal?.q === '`';
    const head = !literal ? '' : template ? literal.text.split('${')[0] : literal.text;
    const computed = !literal || (template && literal.text.includes('${')) || !literal.end;
    if (!computed) named.push(literal.text);
    else if (head === '' || head.startsWith('.') || head.startsWith('/'))
      unread.push(source.slice(0, match.index).split('\n').length);
  }
  return {named, unread};
}
/** Each file a source names, with whether the import is of types only: its imports and exports
 *  (`from`, side-effect `import '…'`, `import(…)`, `require(…)`), the stylesheets and templates
 *  it names (`styleUrl`, `templateUrl`, each of `styleUrls: [...]`), the files it names with
 *  new URL(…, import.meta.url), and the stylesheets it uses, forwards or imports. A script's specifier that is no relative path is a package's, as
 *  is a stylesheet's that Sass does not find from the file (`@angular/material`, `sass:math`). */
function references(file) {
  const source = readFileSync(file, 'utf8');
  const named = [];
  const add = (kind, specifier, typeOnly = false) => named.push({kind, specifier, typeOnly});
  for (const match of source.matchAll(
    new RegExp(String.raw`(import\s+type\s[^;]*?|export\s+type\s[^;]*?)?\bfrom\s*${STRING}`, 'g'),
  ))
    add('script', match.groups.text, !!match[1]);
  for (const match of source.matchAll(new RegExp(String.raw`(?:^|;)\s*import\s*${STRING}`, 'gm')))
    add('script', match.groups.text);
  for (const specifier of dynamicImports(source).named) add('script', specifier);
  for (const match of source.matchAll(new RegExp(String.raw`\brequire\(\s*${LITERAL}\s*\)`, 'g')))
    add('script', match.groups.text);
  for (const match of source.matchAll(
    new RegExp(String.raw`\b(?:styleUrl|templateUrl)\s*:\s*${LITERAL}`, 'g'),
  ))
    add('url', match.groups.text);
  for (const [, list] of source.matchAll(/\bstyleUrls\s*:\s*\[([^\]]*)\]/g))
    for (const match of list.matchAll(new RegExp(LITERAL, 'g'))) add('url', match.groups.text);
  for (const match of source.matchAll(
    new RegExp(String.raw`\bnew\s+URL\(\s*${LITERAL}\s*,\s*import\.meta\.url\s*\)`, 'g'),
  ))
    add('url', match.groups.text);
  for (const [, list] of source.matchAll(
    /@(?:use|forward|import)\s+((?:(['"])(?:(?!\2).)*\2\s*,?\s*)+)/g,
  ))
    for (const match of list.matchAll(/(['"])((?:(?!\1).)*)\1/g)) add('sass', match[2]);
  return named.flatMap(({kind, specifier, typeOnly}) => {
    if (kind === 'sass') {
      if (/^[\w-]+:/.test(specifier)) return []; // sass:math, a URL
      const target = resolveSass(file, specifier);
      return target ? [{specifier, typeOnly, target}] : [];
    }
    // A URL a component or new URL(…) names is relative to the module unless it has a scheme or
    // starts at the root; a script's is a package's unless it starts with '.'.
    if (kind === 'url' ? /^([\w+.-]+:|\/)/.test(specifier) : !specifier.startsWith('.')) return [];
    const base = path.resolve(path.dirname(file), specifier);
    return [{specifier, typeOnly, target: kind === 'script' ? resolveScript(base) : base}];
  });
}
const unread = files(root).flatMap((file) =>
  dynamicImports(readFileSync(file, 'utf8')).unread.map(
    (line) => `${path.relative(root, file)}:${line}`,
  ),
);
assert.deepEqual(
  unread,
  [],
  'a dynamic import builds its specifier at run time (a template literal with ${…}, a ' +
    'concatenation or a variable), which this check cannot follow: name the module with a literal',
);
const violations = [];
for (const {from, forbidden, allowed = [], allowedIf = () => false, except = []} of rules)
  for (const file of files(path.join(root, from)))
    if (!except.some((prefix) => file.startsWith(prefix)))
      for (const {specifier, target} of references(file))
        if (
          forbidden.some((prefix) => within(target, prefix)) &&
          !allowed.includes(target) &&
          !allowedIf(target)
        )
          violations.push(`${path.relative(root, file)} → ${specifier}`);
assert.deepEqual(
  violations,
  [],
  'a layer imports code above it, widgets/ or analyses/ imports theme/ code, a module outside ' +
    'app/ imports app/, a page imports the other, ' +
    'an analysis imports another, a page passes the registry, or a part of the conversation page ' +
    'imports another',
);

// The details panel holds text and tables only: an analysis' details body
// (analyses/<name>/<name>_details.ts) and what the bodies share (analyses/shared/) import no chart
// for more than types, and neither does any module of analyses/ that they import for more than
// types, followed as far as it goes: a body that shows a component of its own analysis which
// draws a chart draws that chart. A chart is Plotly (widgets/plotly_loader.ts) or a module of
// widgets/charts/ that declares a component or directive or imports a chart; the pure models
// there (heatmap_model.ts, runtime_series.ts and the like) are not.
const plotly = path.join(root, 'widgets', 'plotly_loader.ts');
const chartsRoot = path.join(root, 'widgets', 'charts') + path.sep;
const chartModules = new Map();
/** Whether a module is a chart, by the definition above. */
function isChart(file, seen = new Set()) {
  if (file === plotly) return true;
  if (!file.startsWith(chartsRoot) || !existsSync(file)) return false;
  if (chartModules.has(file)) return chartModules.get(file);
  if (seen.has(file)) return false;
  seen.add(file);
  const chart =
    /@(Component|Directive)\(/.test(readFileSync(file, 'utf8')) ||
    references(file).some(({typeOnly, target}) => !typeOnly && isChart(target, seen));
  chartModules.set(file, chart);
  return chart;
}
const analysesRoot = path.join(root, 'analyses') + path.sep;
const bodies = [
  ...ANALYSES.map((name) => path.join(root, 'analyses', name, `${name}_details.ts`)),
  ...files(path.join(root, 'analyses', 'shared'), /\.ts$/),
];
const drawing = [];
for (const body of bodies) {
  const via = new Map([[body, []]]);
  for (const queue = [body]; queue.length; ) {
    const file = queue.shift();
    for (const {specifier, typeOnly, target} of references(file)) {
      if (typeOnly) continue;
      if (isChart(target))
        drawing.push(
          [...via.get(file), file].map((step) => path.relative(root, step)).join(' → ') +
            ` → ${specifier}`,
        );
      else if (target.startsWith(analysesRoot) && existsSync(target) && !via.has(target)) {
        via.set(target, [...via.get(file), file]);
        queue.push(target);
      }
    }
  }
}
assert.deepEqual(drawing, [], 'a details body imports a chart, itself or through analyses/');

// What analyses/panel_state.ts runs, it and every module it imports for more than types, declares
// no component or directive, so the conversation's models that import it stay free of them.
const reached = new Set();
const components = [];
for (const queue = [panelState]; queue.length; ) {
  const file = queue.pop();
  if (reached.has(file) || !existsSync(file)) continue;
  reached.add(file);
  if (/@(Component|Directive)\(/.test(readFileSync(file, 'utf8')))
    components.push(path.relative(root, file));
  for (const {typeOnly, target} of references(file)) if (!typeOnly) queue.push(target);
}
assert.deepEqual(components, [], 'analyses/panel_state.ts reaches a component');

// server/ is the UI's only door to the Python server: no other code sends a request itself.
const requests = [];
const server = path.join(root, 'server') + path.sep;
for (const file of files(root, /\.(ts|mjs|cjs|js|html)$/))
  if (!file.startsWith(server))
    for (const [line, text] of readFileSync(file, 'utf8').split('\n').entries())
      if (/\bfetch\s*\(|\.fetch\b|\bEventSource\b|\bXMLHttpRequest\b/.test(text))
        requests.push(`${path.relative(root, file)}:${line + 1}`);
assert.deepEqual(requests, [], 'only src/server/ may use fetch, EventSource or XMLHttpRequest');

// CSS custom properties count too. A property that server/, widgets/, analyses/ or one of the
// low modules below reads without a fallback (var(--name), or '--name' in code) must be set on the
// root element (:root or html) in theme/, or at or below the reader, not only in a page or app/
// (nor, for what widgets/ and the low modules read, in the analyses). A declaration inside a Sass
// mixin is set where the mixin is included, not where it is written. var(--name, fallback) is a
// setting a caller may give; properties set nowhere in src/ come from libraries such as Angular
// Material. Neither is checked. The rest of src/ is not checked: theme/'s partials read what the
// pages that include them set.
/** Where a file is, as a path under src/ with '/'. */
const placeOf = (file) => path.relative(root, file).split(path.sep).join('/');
/** Whether `place` is one of `places` or in one of them. */
const inPlaces = (place, places) =>
  places.some((other) => place === other || place.startsWith(other + '/'));
// The modules below the pages that live outside widgets/, as stylesheets of theme/ and
// analyses/shared/ or beside their one caller, and are held to its rule.
const LOW_MODULES = [
  ...['theme/_data_table.scss', 'theme/_find_navigation.scss', 'theme/_mode_switch.scss'],
  ...['analyses/shared/_analysis_toolbar.scss', 'analyses/shared/_analysis_view_switch.scss'],
  'conversation/metric_query',
  'conversation/debug/token_diff/token_diff_find/find_navigator',
  'conversation/debug/details_panel/resize_handle',
  'state/page_address',
];
/** [name, place, set on the root element] for every declaration in src/. */
const declarations = [];
for (const file of files(root, /\.(ts|html)$/))
  for (const [, name] of readFileSync(file, 'utf8').matchAll(/(--[a-z][\w-]*)['"]?\s*:/g))
    declarations.push([name, placeOf(file), false]);
const stylesheets = new Map();
/** A stylesheet as nested blocks and statements, with its @use aliases and its mixins. */
function stylesheet(file) {
  if (stylesheets.has(file)) return stylesheets.get(file);
  const text = readFileSync(file, 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
  let at = 0;
  const parse = () => {
    const nodes = [];
    let buffer = '';
    while (at < text.length) {
      const char = text[at++];
      if (char === '{' && text[at - 2] !== '#') {
        nodes.push({prelude: buffer.trim(), body: parse()});
        buffer = '';
      } else if (char === '}' || char === ';') {
        if (buffer.trim()) nodes.push({statement: buffer.trim()});
        buffer = '';
        if (char === '}') return nodes;
      } else buffer += char;
    }
    if (buffer.trim()) nodes.push({statement: buffer.trim()});
    return nodes;
  };
  const sheet = {nodes: parse(), uses: new Map(), mixins: new Map()};
  stylesheets.set(file, sheet);
  for (const node of sheet.nodes) {
    const use = node.statement?.match(/^@use\s+(['"])((?:(?!\1).)*)\1(?:\s+as\s+([\w-]+))?/);
    const target = use && !/^[\w-]+:/.test(use[2]) && resolveSass(file, use[2]);
    if (target && existsSync(target))
      sheet.uses.set(use[3] ?? path.basename(use[2]).replace(/^_|\.(scss|sass|css)$/g, ''), target);
    const mixin = node.prelude?.match(/^@mixin\s+([\w-]+)/);
    if (mixin) sheet.mixins.set(mixin[1], node.body);
  }
  return sheet;
}
const onRoot = (selectors) =>
  selectors.length > 0 &&
  selectors.every(
    (selector) =>
      /^@(media|supports|layer)\b/.test(selector) ||
      selector.split(',').every((part) => /^(:root|html)(\[[^\]]*\])*$/.test(part.trim())),
  );
function declare(file, nodes, selectors, place, depth = 0) {
  assert.ok(depth < 32, `${path.relative(root, file)}: mixins include each other endlessly`);
  for (const node of nodes) {
    const text = node.statement ?? node.prelude;
    if (/^@mixin\b/.test(text)) continue;
    const name = node.statement?.match(/^(--[a-z][\w-]*)\s*:/)?.[1];
    if (name) declarations.push([name, place, onRoot(selectors)]);
    const include = text.match(/^@include\s+(?:([\w-]+)\.)?([\w-]+)/);
    if (include) {
      // A mixin of this stylesheet, or of one it uses; library mixins are not followed.
      const from = include[1] ? stylesheet(file).uses.get(include[1]) : file;
      const body = from && stylesheet(from).mixins.get(include[2]);
      if (body) declare(from, body, selectors, place, depth + 1);
    }
    if (node.body)
      declare(file, node.body, include ? selectors : [...selectors, node.prelude], place, depth);
  }
}
for (const file of files(root, /\.(scss|css)$/))
  declare(file, stylesheet(file).nodes, [], placeOf(file));
/** Each reader and where what it reads may be set, besides the root element in theme/. */
const below = [
  [['server'], ['server']],
  [LOW_MODULES, ['server', ...LOW_MODULES]],
  [['widgets'], ['server', ...LOW_MODULES, 'widgets']],
  [['analyses'], ['server', ...LOW_MODULES, 'widgets', 'analyses']],
];
const unset = [];
for (const [readers, allowed] of below)
  for (const file of readers.flatMap((reader) => {
    const place = path.join(root, reader);
    assert.ok(existsSync(place), `src/${reader}, a reader of CSS custom properties, is gone`);
    return statSync(place).isFile() ? [place] : files(place, /\.(ts|scss|css|html)$/);
  })) {
    const source = readFileSync(file, 'utf8');
    const read = new Set(
      [...source.matchAll(/var\(\s*(--[a-z][\w-]*)\s*\)|['"`](--[a-z][\w-]*)['"`]/g)].map(
        (match) => match[1] ?? match[2],
      ),
    );
    for (const name of read) {
      const places = declarations.filter(([set]) => set === name);
      if (
        places.length &&
        !places.some(
          ([, place, rooted]) => inPlaces(place, allowed) || (place.startsWith('theme/') && rooted),
        )
      )
        unset.push(
          `${path.relative(root, file)} reads ${name}, set only in ` +
            [
              ...new Set(
                places.map(([, place, rooted]) =>
                  place.startsWith('theme/') && !rooted ? `${place} (not on :root)` : place,
                ),
              ),
            ].join(', '),
        );
    }
  }
assert.deepEqual(unset, [], 'a layer reads a CSS custom property that only a layer above sets');
console.log(
  'PASS: src/ holds only folders and files with rules; server/ imports no layer above it; ' +
    'state/ imports only server/, theme/ only state/; widgets/ imports no state/, analyses/ or ' +
    'page and of theme/ only its stylesheets, analyses/ no state/ or page and of theme/ only its ' +
    'stylesheets, and its analyses import no other analysis; home/ and conversation/ import ' +
    'nothing of each other, nothing outside app/ but main.ts imports app/, and the pages and app/ ' +
    'reach analyses/ only through its registry or its component-free panel_state; the Chat and ' +
    'Debug readings and the composer of the conversation page import nothing of each other, and ' +
    'what they share none of them; no dynamic import builds its specifier at run time; details ' +
    'bodies import no chart; only server/ sends requests; server/, widgets/, analyses/ and the low ' +
    'modules read no CSS custom property set only above them or only below the root element',
);

const uiRoot = fileURLToPath(new URL('..', import.meta.url));
const allowedHyphenatedPaths = new Set([
  'public/debugger-logo.svg',
  'public/fonts/LICENSE-material-icons.txt',
  'public/graph-arch-preview',
  'public/licenses/model-explorer.txt',
  'public/vendor/model-explorer',
  'scripts/copy-license-notices.mjs',
]);
const hyphenatedPaths = [];
function checkNoHyphens(directory) {
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === '.angular') {
      continue;
    }
    const full = path.join(directory, entry.name);
    const relPath = path.relative(uiRoot, full).split(path.sep).join('/');
    if (allowedHyphenatedPaths.has(relPath)) {
      continue;
    }
    if (entry.name.includes('-')) {
      hyphenatedPaths.push(relPath);
    }
    if (entry.isDirectory()) {
      checkNoHyphens(full);
    }
  }
}
for (const dir of ['src', 'public', 'scripts', 'e2e']) {
  const target = path.join(uiRoot, dir);
  if (existsSync(target)) {
    checkNoHyphens(target);
  }
}
assert.deepEqual(hyphenatedPaths, [], 'UI paths must not contain hyphens unless allowlisted');
console.log('PASS: UI paths under src/, public/, scripts/ and e2e/ are hyphen-free');
