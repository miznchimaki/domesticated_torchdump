from collections import defaultdict, namedtuple
from contextlib import nullcontext

import torch
import torch.nn as nn


_BoundaryEdge = namedtuple("_BoundaryEdge", ["node", "input_index", "edge_key"])
_InternalOutputEdge = namedtuple("_InternalOutputEdge", ["node", "input_index", "output_id"])


def _iter_public_tensors(obj, seen=None):
    if seen is None:
        seen = set()

    if isinstance(obj, torch.Tensor):
        if obj.requires_grad and id(obj) not in seen:
            seen.add(id(obj))
            yield obj
        return

    if isinstance(obj, nn.Module):
        for tensor in obj.parameters(recurse=True):
            if tensor.requires_grad and id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor
        for tensor in obj.buffers(recurse=True):
            if tensor.requires_grad and id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor
        return

    if isinstance(obj, torch.optim.Optimizer):
        for group in obj.param_groups:
            for tensor in group.get("params", []):
                yield from _iter_public_tensors(tensor, seen)
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_public_tensors(value, seen)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_public_tensors(value, seen)


def _iter_output_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_output_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_output_tensors(value)


def _get_gradient_edge_key(tensor):
    try:
        edge = torch.autograd.graph.get_gradient_edge(tensor)
        return edge.node, edge.output_nr
    except Exception:
        if tensor.grad_fn is not None:
            return tensor.grad_fn, getattr(tensor, "output_nr", 0)
        try:
            grad_fn = tensor.expand_as(tensor).grad_fn
            if grad_fn is not None and grad_fn.next_functions:
                return grad_fn.next_functions[0][0], 0
        except Exception:
            return None
    return None


def _get_output_tensors_and_roots(outputs):
    output_tensors = []
    roots = []
    seen_roots = set()

    for tensor in _iter_output_tensors(outputs):
        if not tensor.requires_grad:
            continue
        output_tensors.append(tensor)
        if tensor.grad_fn is not None and tensor.grad_fn not in seen_roots:
            seen_roots.add(tensor.grad_fn)
            roots.append(tensor.grad_fn)

    return output_tensors, roots


def _find_edges(root, public_edge_keys, output_edge_ids_by_key):
    boundary_edges = []
    internal_output_edges = []
    visited = set()
    queue = [root]

    while queue:
        node = queue.pop(0)
        if node is None or node in visited:
            continue
        visited.add(node)

        for input_index, (next_node, output_nr) in enumerate(getattr(node, "next_functions", ())):
            if next_node is None:
                continue
            edge_key = (next_node, output_nr)
            if edge_key in public_edge_keys:
                boundary_edges.append(_BoundaryEdge(node, input_index, edge_key))
            elif edge_key in output_edge_ids_by_key:
                for output_id in output_edge_ids_by_key[edge_key]:
                    internal_output_edges.append(_InternalOutputEdge(node, input_index, output_id))
                queue.append(next_node)
            else:
                queue.append(next_node)

    return boundary_edges, internal_output_edges


def _accumulate_grad(old_grad, new_grad):
    if new_grad is None:
        return old_grad
    if old_grad is None:
        return new_grad
    return old_grad + new_grad


class BackwardBoundaryCapture:
    def __init__(self, outputs, public_inputs, on_backward_input, on_backward_output, internal_op_context=None):
        self.output_tensors, self.roots = _get_output_tensors_and_roots(outputs)
        self.output_edge_ids_by_key = defaultdict(list)
        self.output_ids_by_node = defaultdict(list)
        for output_id, tensor in enumerate(self.output_tensors):
            edge_key = _get_gradient_edge_key(tensor)
            if edge_key is None:
                continue
            self.output_edge_ids_by_key[edge_key].append(output_id)
            node, output_nr = edge_key
            self.output_ids_by_node[node].append((output_nr, output_id))

        public_edge_keys = {
            edge_key
            for edge_key in (_get_gradient_edge_key(tensor) for tensor in _iter_public_tensors(public_inputs))
            if edge_key is not None
        }
        self.boundary_edges = []
        self.internal_output_edges = []
        self.boundary_occ_id_by_key = {}
        self.internal_output_edge_keys = set()
        self.root_boundary_occ_ids = defaultdict(list)
        for root in self.roots:
            boundary_edges, internal_output_edges = _find_edges(
                root, public_edge_keys, self.output_edge_ids_by_key)
            for edge in boundary_edges:
                occ_key = (edge.node, edge.input_index, edge.edge_key)
                if occ_key not in self.boundary_occ_id_by_key:
                    self.boundary_occ_id_by_key[occ_key] = len(self.boundary_edges)
                    self.boundary_edges.append(edge)
                self.root_boundary_occ_ids[root].append(self.boundary_occ_id_by_key[occ_key])
            for edge in internal_output_edges:
                occ_key = (edge.node, edge.input_index, edge.output_id)
                if occ_key not in self.internal_output_edge_keys:
                    self.internal_output_edge_keys.add(occ_key)
                    self.internal_output_edges.append(edge)

        self.boundary_edges_by_node = defaultdict(list)
        self.boundary_edge_keys = []
        self.boundary_edge_id_by_key = {}
        for occ_id, edge in enumerate(self.boundary_edges):
            if edge.edge_key not in self.boundary_edge_id_by_key:
                self.boundary_edge_id_by_key[edge.edge_key] = len(self.boundary_edge_keys)
                self.boundary_edge_keys.append(edge.edge_key)
            edge_id = self.boundary_edge_id_by_key[edge.edge_key]
            self.boundary_edges_by_node[edge.node].append((edge.input_index, edge_id, occ_id))

        self.internal_output_edges_by_node = defaultdict(list)
        for edge in self.internal_output_edges:
            self.internal_output_edges_by_node[edge.node].append((edge.input_index, edge.output_id))

        self.output_total_grads = [None] * len(self.output_tensors)
        self.output_internal_grads = [None] * len(self.output_tensors)
        self.boundary_grads = [None] * len(self.boundary_edge_keys)
        self.active_boundary_occ_ids = set()
        self.completed_boundary_occ_ids = set()
        self.pending_edges = 0
        self.seen_root = False
        self.output_emitted = False
        self.on_backward_input = on_backward_input
        self.on_backward_output = on_backward_output
        self.internal_op_context = internal_op_context or nullcontext

    def register(self):
        if not self.roots:
            return False

        nodes = set(self.boundary_edges_by_node.keys())
        nodes.update(self.internal_output_edges_by_node.keys())
        nodes.update(self.output_ids_by_node.keys())
        nodes.update(self.roots)
        no_boundary_roots = set(self.roots) if not self.boundary_edge_keys else set()
        for node in nodes:
            node.register_hook(self._make_hook(
                node in no_boundary_roots,
                tuple(self.root_boundary_occ_ids.get(node, ())),
                tuple(self.output_ids_by_node.get(node, ())),
                tuple(self.internal_output_edges_by_node.get(node, ())),
                tuple(self.boundary_edges_by_node.get(node, ())),
            ))
        self._clear_registration_state()
        return True

    def _make_hook(self, no_boundary_root, root_boundary_occ_ids, output_specs, internal_output_specs, boundary_edge_specs):
        def hook(grad_inputs, grad_outputs):
            for output_nr, output_id in output_specs:
                grad = grad_outputs[output_nr] if output_nr < len(grad_outputs) else None
                self.output_total_grads[output_id] = _accumulate_grad(
                    self.output_total_grads[output_id], grad)

            for input_index, output_id in internal_output_specs:
                grad = grad_inputs[input_index] if input_index < len(grad_inputs) else None
                self.output_internal_grads[output_id] = _accumulate_grad(
                    self.output_internal_grads[output_id], grad)

            if no_boundary_root:
                self.seen_root = True
                self._emit_output(grad_inputs)
                return

            if root_boundary_occ_ids:
                self.seen_root = True
                for occ_id in root_boundary_occ_ids:
                    if occ_id not in self.active_boundary_occ_ids:
                        self.active_boundary_occ_ids.add(occ_id)
                        self.pending_edges += 1

            for input_index, edge_id, occ_id in boundary_edge_specs:
                grad = grad_inputs[input_index] if input_index < len(grad_inputs) else None
                self.boundary_grads[edge_id] = _accumulate_grad(self.boundary_grads[edge_id], grad)
                if occ_id in self.active_boundary_occ_ids and occ_id not in self.completed_boundary_occ_ids:
                    self.completed_boundary_occ_ids.add(occ_id)
                    self.pending_edges -= 1

            if self.seen_root and self.pending_edges == 0:
                self._emit_output(tuple(self.boundary_grads))

        return hook

    def _get_external_output_grads(self):
        external_grads = []
        for total_grad, internal_grad in zip(self.output_total_grads, self.output_internal_grads):
            if total_grad is None:
                external_grads.append(None)
            elif internal_grad is None:
                external_grads.append(total_grad)
            else:
                with self.internal_op_context():
                    external_grads.append(total_grad - internal_grad)
        return tuple(external_grads)

    def _emit_output(self, grad_inputs):
        if self.output_emitted:
            return
        self.output_emitted = True
        self.on_backward_input(self._get_external_output_grads())
        self.on_backward_output(grad_inputs)
        self._clear_registration_state()
        self.output_total_grads = None
        self.output_internal_grads = None
        self.boundary_grads = None
        self.active_boundary_occ_ids = None
        self.completed_boundary_occ_ids = None
        self.on_backward_input = None
        self.on_backward_output = None
        self.internal_op_context = None

    def _clear_registration_state(self):
        self.output_tensors = None
        self.roots = None
        self.output_edge_ids_by_key = None
        self.output_ids_by_node = None
        self.boundary_edges = None
        self.internal_output_edges = None
        self.boundary_occ_id_by_key = None
        self.internal_output_edge_keys = None
        self.root_boundary_occ_ids = None
        self.boundary_edges_by_node = None
        self.internal_output_edges_by_node = None
        self.boundary_edge_id_by_key = None


def register_backward_boundary_hook(outputs, public_inputs, on_backward_input, on_backward_output, internal_op_context=None):
    capture = BackwardBoundaryCapture(
        outputs, public_inputs, on_backward_input, on_backward_output, internal_op_context)
    return capture.register()
