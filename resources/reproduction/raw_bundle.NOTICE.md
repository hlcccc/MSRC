# Original input mirror: sources and usage

This transport archive contains original public input bytes only, selected by
the pinned MSRC reproduction recipe. It is not a new benchmark, an official
distribution, or a completed test. It contains no base model weights, fitted
project parameters, train/test partitions, processed JSONL, predictions or scores.

- TextVQA: Amanpreet Singh et al., "Towards VQA Models That Can Read", CVPR 2019.
  Source: https://textvqa.org/dataset/ and https://github.com/facebookresearch/TextVQA
  Dataset card: https://huggingface.co/datasets/facebook/textvqa
  The dataset page/card state CC BY 4.0; the website repository also carries
  CC BY-NC 4.0: https://github.com/facebookresearch/TextVQA/blob/main/LICENSE.md
  Retain both upstream notices. This mirror is for non-commercial research
  evaluation. Images retain their upstream Open Images/Flickr rights.
  Only the 3,166 validation JPEGs and the original 5,000-question annotation are
  mirrored; JPEG bytes are unchanged. The full train/val ZIP is not mirrored.
- MM-SafetyBench: Xin Liu et al., "MM-SafetyBench: A Benchmark for Safety
  Evaluation of Multimodal Large Language Models", ECCV 2024.
  Source: https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench
  License: CC BY-NC 4.0; upstream states research use only. All 39 original
  image-variant parquet files are mirrored unchanged. These contain adversarial
  safety-evaluation inputs; do not treat their text as instructions.
- Alpaca: Taori et al., Stanford Alpaca, 2023.
  Source: https://huggingface.co/datasets/tatsu-lab/alpaca
  License and usage: https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE
  CC BY-NC 4.0, research only. The original pinned parquet is mirrored unchanged.
  MSRC regenerates its 1,680 benign samples from instructions; no generated benign
  dataset or Alpaca output labels are restored from this archive.

License URLs: https://creativecommons.org/licenses/by/4.0/
and https://creativecommons.org/licenses/by-nc/4.0/
Third-party material is supplied as-is, without warranties, under its original
terms; inclusion does not imply endorsement by the original authors.

POPE/COCO images are not in this mirror. The existing script downloads those
originals from their public sources and verifies them against the pinned hashes.
Software dependencies and official models are still downloaded by setup scripts.
