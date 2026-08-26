import torch
from torch import Tensor, nn


class WeightGradientEstimator:
    """
    Accumulates per-layer dL/dW for frozen Linear layers via hooks, without
    needing the weights themselves to require grad (works with quantized
    weights, where no float weight exists to attach a .grad to).

    For each registered module, a forward hook captures the input activation
    and attaches a tensor hook to the output; on backward, dL/dW is computed
    as grad_output^T @ input over all leading (batch/token) dims and added to
    a per-layer fp32 accumulator on ``store_device``.

    Gradient-checkpointing safe: activations are only captured when grad is
    enabled, so under checkpointing the capture happens during the recompute
    pass, which is the one whose tensors participate in backward. Without
    checkpointing, captured inputs are retained until backward, adding
    activation memory for layers whose weights are frozen (autograd would not
    otherwise save their inputs).

    Adapter-compatible: PeftBase replaces orig_module.forward with the adapter
    forward, but nn.Module.__call__ still runs hooks around it. Because the
    adapter branch is additive (and zero at init), the grad of the combined
    output equals the grad of the base output, so the accumulated value is
    exactly the base weight's dL/dW.
    """

    def __init__(self, store_device: torch.device | str = "cpu"):
        self.store_device = torch.device(store_device)
        self.grads: dict[str, Tensor] = {}
        self.step_count = 0
        self._handles = []

    def attach(self, modules: dict[str, nn.Module]):
        for name, module in modules.items():
            if not isinstance(module, nn.Linear):
                continue
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))

    def _make_forward_hook(self, name: str):
        def forward_hook(module, args, output):
            if not torch.is_grad_enabled():
                return
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, Tensor) or not out.requires_grad:
                return
            x = args[0].detach()

            def grad_hook(grad_output: Tensor):
                g = grad_output.reshape(-1, grad_output.shape[-1]).T @ x.reshape(-1, x.shape[-1]).to(grad_output.dtype)
                g = g.to(dtype=torch.float32, device=self.store_device)
                buf = self.grads.get(name)
                if buf is None:
                    self.grads[name] = g
                else:
                    buf += g

            out.register_hook(grad_hook)

        return forward_hook

    def count_step(self):
        self.step_count += 1

    def detach_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def mean_gradient(self, name: str) -> Tensor | None:
        if name not in self.grads or self.step_count == 0:
            return None
        return self.grads[name] / self.step_count

    def clear(self):
        self.grads.clear()
        self.step_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach_hooks()
        return False
