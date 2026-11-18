# Graph Diff

The Intermediate Tensors Graph Diff view, a same-origin page the Angular
panel shows in a frame (`IntermediateGraphArch`). Packaged from the
graph-style-preview design prototype (revision 142); its look is the
prototype's, its data the server's. It omits the prototype's model name,
application toolbar and Details pane: the Angular panel owns token position,
metric and Details. Graph Diff shows one aligned token position; Heatmap and
Line chart keep their range and aggregation.

## Three levels, all the server's

1. **Layer list** (the module tree): the layers of the model's skeleton (or,
   without one, of the column's signature's graph), each with its layer
   output at the position, as the server's position-range compares it. Each
   layer's attention span and K / V source are the skeleton's
   (`layers[].attrs`, `layers[].from.kv`), shown in Analysis Info. The
   modules the server names as outside the skeleton (`architecture.outside`:
   Gemma 4's Embedding, Per-Layer Embedding, Final Norm and Output Head) keep
   their rows in the stages the server says they run in (`stages`: Final Norm
   and Output Head only in decode, for prefill produces no logits), each with
   no value and the server's words on hover; opening one does nothing.
2. **Layer architecture** (Layer Arch): drawn from the server's skeleton
   (`GET /api/artifacts/{id}/architecture`, model_arch_explore's schema) by
   `skeleton-arch.js`: each namespace of the skeleton's stages is a folded
   block, the elementwise ops between them (residual Adds, the layer scale)
   stay on the overview, an input bound in from outside the layer (Gemma 4's
   `per_layer_input`) enters its block from the left. The skeleton and the
   execution graph are aligned only at the anchors: an anchor's tag is the
   server's T2 cell of its bound point at the column, or the server's reason
   it has none (not bound, as for a layer the signature does not compute; not
   observable; not observed by the Session; observed without a value there).
   A stage between anchors, and a block's internal summary, have no value,
   but for a semantic operator's output the server pairs across two runtimes
   (a LiteRT-LM point chosen beside the PyTorch module of its operator,
   `model.ops`): a norm's output tensor shows its row's value as an anchor's
   tag does.
3. **Execution graph** (Mapping): the signature that ran for the column on
   each side (`columns` of `GET /api/position-analysis`), from
   `GET /api/artifacts/{id}/graph/{signature}`, which the frame asks the host
   for (`graph-request`, answered with `execution-graph`): the frame sends no
   request of its own. The pane shows the block's region as the server found
   it (`bindings[].blocks`): the operators on a path from the anchor the
   block reads to the anchor it ends at (Attention: `input` to
   `intermediate.1`), what they read from outside as the inputs node, what is
   read of them outside as the outputs node, fitted to the pane; a block an
   unbound anchor bounds is its layer's operators, the server's reason beside
   its title in Analysis Info. Only a block's gear opens it; a click on a
   tensor, an anchor's too, selects it in place. Each operator output's tag
   has one of four states: not observable (grey, the server's reason), not
   observed (outline), observed without a value at the column (a dash, the
   server's reason) and a value (metric colour). Fused operators are not
   expanded. A PyTorch side's graph is the execution graph its runner
   exported (pytorch-collector), each module LiteRT-LM runs as one operator
   folded into one node by the server; its groups start at the layer's scope
   the server names (`bindings[].scope`: `layer_3`, or a module path's
   `layers/3`), which the frame never derives.

A model without an architecture template goes from the layer list to its
execution graph; the Layer Arch shows the server's text instead.

## Messages

`IntermediateGraphArch` exchanges same-origin, source-checked messages:

- `context`: stage, metric, position, the layers' outputs with their
  formatted values and colours, the panel's summary choice, palette and
  theme, and `model` (`graph_arch_data.ts`): the skeleton, each anchor's tag
  and binding, each observed point's tag, the column's signatures.
- `graph-request` / `execution-graph`: an execution graph asked for, and the
  server's answer.
- `pair-save` / `pair-delete`: a pair made by hand to save, pairs made by
  hand to delete.
- `selection`, `mapping-view`: what the host's Details panel shows.

The pairing editor (Advanced configuration) shows the automatic pairs as
before: by node and output identity of identical graphs; two different graphs
(two models, or a PyTorch side and a LiteRT-LM side) pair at the anchors the
server places inside both regions, and at each semantic operator the server
pairs across the two runtimes there (a LiteRT-LM point chosen and the PyTorch
module of its operator: the two points of one row, `pointPairs`). A pair made
by hand (a tensor clicked on one side, then one on the other) is the
server's: the frame asks the host to save it (`pair-save`, `{ref, target}`
each `{signature, node, output}`), the host posts it to `/api/mappings`,
which checks it and keeps it for the two models, and the pair comes back in
the next context (`model.manualPairs`, each with its own row's tag at the
column or the server's reason); Reset default asks the host to delete the
pairs made by hand in the range (`pair-delete`), and each one's row in the
host's Metric List has a delete. The server's refusal, and its words when one
model is on both sides and no pair is needed, are the host's note above that
list. A pair has the value of its point only when the server compares the two
(the same point, the two points of one anchor's row, `pointPairs`, or a pair
made by hand's own row); any other pair of two different tensors has none.

## Presentation

The module tree uses full-row metric colors; the Summary control (default
Max) governs collapsed groups. Arch tensor tags show Icon + Metric; names are
in Details. Layer Arch blocks are fixed overview cards; their gear enters
Mapping. Configuration hover outlines the tensors of a block's range.
Coverage edges and outlines share the native subgraph indicator blue
(#135cbb); ordinary connectors use the native Model Explorer edge color (#aaa
in light mode); an edge whose tensor has no sync point (no observed point) is
dashed.
