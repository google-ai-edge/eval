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

// The Layer Arch's data from the server's semantic skeleton (model_arch_explore's schema), in the
// shapes the Arch drawing reads: blocks (a namespace's stages), the residual ops between them,
// each tensor and its anchor. The skeleton and the execution graph are aligned only at the
// anchors; a stage between two anchors has no value of its own.
window.skeletonArch = (() => {
  const slug = (name) => String(name).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  const attr = (node, key) => (node.attrs || []).find((a) => a.key === key)?.value;
  const capital = (text) => String(text).charAt(0).toUpperCase() + String(text).slice(1);

  /** One layer of a skeleton: its definition, blocks, ops, tensors and anchors. */
  function layer(skeleton, index) {
    const entry = skeleton.layers[index],
      definition = skeleton.semantic_graph[entry.def];
    const inputs = new Set(definition.inputs.map((i) => i.id));
    const key = (node, port) => (inputs.has(node) ? node : `${node}:${port ?? '0'}`);
    const anchorTensor = (anchor) => {
      const [node, port] = anchor.of.split(':');
      return key(node, port);
    };
    const anchors = Object.fromEntries(definition.anchors.map((a) => [anchorTensor(a), a]));
    const groups = [],
      byId = new Map(),
      ops = {},
      tensors = {},
      order = [];
    for (const input of definition.inputs)
      tensors[input.id] = {
        name: input.label || capital(input.id.replace(/_/g, ' ')),
        producer: null,
        consumers: [],
        external: input.source !== 'residual',
        source: input.source,
        shape: input.shape,
      };
    for (const node of definition.nodes) {
      const group = slug(node.namespace),
        elementwise = attr(node, 'category') === 'elementwise';
      const output = key(node.id, node.outputsMetadata?.[0]?.id);
      const op = node.ops?.[0]?.op;
      ops[node.id] = {
        // Residual Adds and the layer scale are drawn as the Arch's math nodes.
        name: elementwise ? (op === 'Mul' ? 'Multiply' : op || node.label) : node.label,
        role: node.label,
        group,
        input: node.incomingEdges.map((e) => key(e.sourceNodeId, e.sourceNodeOutputId)),
        output: [output],
        purpose: `${node.label} · ${(node.ops || []).map((o) => o.op).join(', ')}`,
        category: attr(node, 'category'),
      };
      tensors[output] = {name: '', producer: node.id, consumers: [], group};
      if (elementwise) {
        ops[node.id].layout = {row: 0, column: 0};
        order.push(node.id);
        continue;
      }
      let block = byId.get(group);
      if (!block) {
        block = {id: group, name: node.namespace, ops: [], input: null, output: null, description: node.namespace};
        byId.set(group, block);
        groups.push(block);
        order.push(group);
      }
      // Stages stack in their block, one row each, as the Arch lays a folded block out.
      ops[node.id].layout = {row: block.ops.length, column: 0};
      block.ops.push(node.id);
      block.output = output;
      if (!block.input) block.input = ops[node.id].input.find((t) => !tensors[t]?.external) ?? ops[node.id].input[0];
    }
    for (const [id, op] of Object.entries(ops)) for (const input of op.input) tensors[input]?.consumers.push(id);
    // Names: an anchor's semantic, a block's output, a stage's output within its block.
    for (const [id, tensor] of Object.entries(tensors)) {
      if (anchors[id]) tensor.name = `${capital(anchors[id].semantic)} · ${anchors[id].id}`;
      else if (tensor.producer) {
        const op = ops[tensor.producer],
          block = byId.get(op.group);
        tensor.name = block?.output === id ? `${block.name} output` : `${block?.name ?? op.group} · ${op.role} output`;
      }
    }
    // Residuals: an op between blocks that reads a tensor of the spine besides the one before it.
    const residuals = [];
    let previous = null,
      before = null;
    for (const item of order) {
      const block = byId.get(item);
      if (block) {
        previous = block.output;
        before = block.id;
        continue;
      }
      const op = ops[item];
      for (const tensor of op.input)
        if (tensor !== previous && !tensors[tensor]?.external) residuals.push([before, item, tensor]);
      previous = op.output[0];
    }
    const externals = Object.entries(tensors)
      .filter(([, t]) => t.external)
      .map(([id, t]) => ({tensor: id, op: t.consumers[0], group: t.consumers[0] ? ops[t.consumers[0]].group : null}));
    return {entry, definition, groups, ops, tensors, order, anchors, residuals, externals};
  }

  /** A layer's attention and KV facts, as the skeleton's layer attrs give them. */
  function layerFacts(skeleton, index) {
    const entry = skeleton.layers[index];
    const attend = Object.values(entry.attrs || {})
      .flatMap((a) => Object.values(a.ops || {}))
      .find((op) => op && op.span);
    const span = attend?.span;
    return {
      attention: !span ? '—' : span.kind === 'sliding' ? `Sliding window · ${span.window}` : span.kind === 'global' ? 'Full' : capital(span.kind),
      kv: entry.from?.kv ? `Layer ${entry.from.kv.layer}` : 'This layer',
    };
  }

  /** A block of a signature's graph, as the Mapping pane shows it: the operators of the server's
   *  region of the block (`block.nodes`: those between its anchors, or the layer's with the
   *  server's `reason`), the tensors read from outside as inputs and those read outside as
   *  outputs, and where each anchor is. It opens with the end anchor's group expanded. */
  function region(graph, component, anchors, entryAnchor, block, end) {
    const scope = component.split('/');
    if (!block) return {...graph, nodes: [], mappingPresentation: {anchors: {}, note: 'The server gives no region of this block.'}};
    const wanted = new Set(block.nodes);
    // The region's operators; constants and everything else read from outside are its inputs.
    const selected = graph.nodes.filter(
      (node) => wanted.has(node.id) && !['GraphInputs', 'GraphOutputs', 'pseudo_const'].includes(node.label),
    );
    // Groups start at the layer's own scope, the scopes above it holding nothing else here.
    // The layer's scope, the server's (`layer_3`, or a module path's `layers/3`).
    const local = (namespace) => {
      const parts = (namespace || '').split('/');
      const at = parts.findIndex((_, i) => scope.every((part, j) => parts[i + j] === part));
      return at < 0 ? namespace || '' : parts.slice(at).join('/');
    };
    const ids = new Set(selected.map((n) => n.id)),
      byId = new Map(graph.nodes.map((n) => [n.id, n])),
      inputs = new Map();
    const note = block.reason || '';
    if (!selected.length) return {...graph, nodes: [], mappingPresentation: {anchors: {}, note}};
    const nodes = selected.map((node) => ({
      ...node,
      namespace: local(node.namespace),
      incomingEdges: (node.incomingEdges || []).map((edge) => {
        if (ids.has(edge.sourceNodeId)) return edge;
        const key = edge.sourceNodeId + '::' + edge.sourceNodeOutputId;
        if (!inputs.has(key)) {
          const meta = byId.get(edge.sourceNodeId)?.outputsMetadata?.find((p) => p.id === edge.sourceNodeOutputId);
          inputs.set(key, {...meta, id: String(inputs.size)});
        }
        return {...edge, sourceNodeId: 'boundary:inputs', sourceNodeOutputId: inputs.get(key).id};
      }),
    }));
    const outputs = new Map();
    for (const node of graph.nodes)
      if (!ids.has(node.id))
        for (const edge of node.incomingEdges || [])
          if (ids.has(edge.sourceNodeId)) outputs.set(edge.sourceNodeId + '::' + edge.sourceNodeOutputId, edge);
    nodes.unshift({id: 'boundary:inputs', label: 'Inputs', namespace: '', attrs: [], incomingEdges: [], outputsMetadata: [...inputs.values()]});
    nodes.push({
      id: 'boundary:outputs',
      label: 'Outputs',
      namespace: '',
      attrs: [],
      outputsMetadata: [],
      incomingEdges: [...outputs.values()].map((e, i) => ({sourceNodeId: e.sourceNodeId, sourceNodeOutputId: e.sourceNodeOutputId, targetNodeInputId: String(i)})),
    });
    const placed = {};
    for (const [id, at] of Object.entries(anchors || {}))
      if (ids.has(at.node)) placed[id] = {outputNode: at.node, outputPort: at.output};
    const inside = (at) => (at && ids.has(at.node) ? at : null);
    const shown = inside(end) || Object.values(anchors || {}).find((a) => ids.has(a.node)) || null;
    const namespace = shown ? local(byId.get(shown.node)?.namespace) : '';
    if (entryAnchor)
      placed[entryAnchor] = {
        ...(shown ? {outputNode: shown.node, outputPort: shown.output} : {}),
        expandedGroups: namespace ? [namespace] : [],
      };
    // Without an anchor to open at, the layer's own group is opened.
    return {
      ...graph,
      nodes,
      mappingPresentation: {anchors: placed, note, ...(shown ? {} : {expandedGroups: [component]})},
    };
  }

  return {slug, layer, layerFacts, region};
})();
