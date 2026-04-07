---
id: preprocess-postprocess
title: Pre/Post Processing
---

IPXml supports **typed ops** and **expression ops** for preprocessing and postprocessing. Pipelines run on CPU using `ndarray`.

## Example: Image Preprocess

```yaml
preprocess:
  - op: resize
    width: 342
    height: 342
    layout: nchw
  - op: center_crop
    width: 299
    height: 299
    layout: nchw
  - op: normalize
    scale: 0.003921569
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
```

## Example: Postprocess + Decode

```yaml
postprocess:
  - op: softmax
    axis: 1
decode:
  type: top_k
  k: 5
  axis: 1
```

## Expression Ops (Rhai)

```yaml
- op: expr
  code: "softmax(matmul(x, reshape([1000, 1000])))"
```

Available helper functions:
`reshape`, `transpose`, `matmul`, `sum`, `mean`, `std`, `softmax`, `argmax`, `topk`, `clip`, `normalize`, `scale`.
