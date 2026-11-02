# MSRC Fast Review

This isolated branch implements full historical evidence review plus a separate
fresh fixed-sample demonstration. It is not the original fresh full-inference
acceptance procedure. No base-model training or new calibrator fitting is done.

Start with [the Chinese test procedure](docs/FAST_REVIEW_FLOW.md).
Publication and netdisk contents are described in [the delivery guide](docs/FAST_REVIEW_DELIVERY.md).

The original delivery branch and all original runs remain unchanged. Large model
weights and historical evidence belong in a separately shared delivery folder,
not Git. The evidence release is bound by SHA256 in
`resources/fast_review/release.json`.

CPU-only review:

```bash
bash scripts/fast_01_setup.sh
bash scripts/fast_02_review.sh
```

The full numerical review can pass while the overall procedure remains
`EVIDENCE_REVIEW_ONLY_DEMO_PENDING`. Fresh demonstration and acceptance-party
agreement are required before `FAST_REVIEW_PASS` can be emitted.
