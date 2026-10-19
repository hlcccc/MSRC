# Vendored PlugGuard Head

`models.py` is copied without modification from
[Alibaba-AAIG/Kelp, commit de7abe9](https://github.com/Alibaba-AAIG/Kelp/tree/de7abe9bb1af1186e12b3fe5bf7fab65c2fdc550).
Its SHA256 is
`c16d237c5634bcaae24f6bedaf5075479fcf22e395c9ecfff081715fdf5d0479`.
The upstream Apache-2.0 [license](LICENSE) is included.

Our Qwen2.5-VL generation adapter, fixed causal timestep, development-only
training and calibration, and output buffer live in `pipeline/plugguard.py`.
This is not the upstream post-generation demo or an exact reproduction of the
paper's response-risk experiment. Base-model weights and our trained head are
not included in this directory or in Git.

See the [acceptance flow](../../docs/PLUGGUARD_TEST_FLOW.md) for this project's
entry points. Follow the [upstream README](https://github.com/Alibaba-AAIG/Kelp/blob/de7abe9bb1af1186e12b3fe5bf7fab65c2fdc550/README.md)
only when running the separate upstream project.
