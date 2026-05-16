# example for defining custom ops
import torch


@torch.library.custom_op("mylib::gru_opaque", mutates_args=())
def gru_opaque(
    x: torch.Tensor,
    h0: torch.Tensor,
    flat_weights: list[torch.Tensor],
    has_biases: bool,
    num_layers: int,
    batch_first: bool,
    bidirectional: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, hn = torch.ops.aten.gru.input(
        x,
        h0,
        flat_weights,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )
    return out.contiguous(), hn.contiguous()


@gru_opaque.register_fake
def _(x, h0, flat_weights, has_biases, num_layers, batch_first, bidirectional):
    num_directions = 2 if bidirectional else 1
    hidden_size = h0.shape[-1]
    if batch_first:
        batch, seq, _ = x.shape
        out = x.new_empty((batch, seq, num_directions * hidden_size))
    else:
        seq, batch, _ = x.shape
        out = x.new_empty((seq, batch, num_directions * hidden_size))
    return out, torch.empty_like(h0)
