# Full NetAttn

Full 256-path aggregation experiments:

- Uniform coefficients fixed at `1/256`.
- Learned global, input-independent coefficients normalized with softmax.
- Uniform-training VRAM measurement used by the paper efficiency table.

The learned seed-42 run is also the checkpoint consumed by `exp03_topk`.

`subnet_idx` retains the historical checkpoint order. Displayed path strings
use the canonical shallow-to-deep block order; use `subnet_index_to_bits()` and
`subnet_bits_to_index()` instead of formatting the integer index directly.
