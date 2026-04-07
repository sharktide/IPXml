---
id: schema
title: Schema Reference
---

This page describes the `.ipxml` schema.

## Top-Level

```yaml
name: My App
version: "1.0"
model:
  path: my_model.onnx
inputs: []
outputs: []
layout:
  rows: []
```

## InputSpec

```yaml
- id: input_id
  label: Input Label
  type: image | text | number | bool | file | path
  tensor:
    shape: [1, 3, 224, 224]
    layout: nchw | nhwc | chw | hwc
    normalize:
      scale: 0.003921569
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  preprocess:
    - op: resize
      width: 256
      height: 256
```

## OutputSpec

```yaml
- id: output_id
  label: Output Label
  type: text | number | scores | image | file | path
  source: output_name_in_onnx   # defaults to id
  postprocess:
    - op: softmax
      axis: 1
  decode:
    type: top_k
    k: 5
    axis: 1
  labels:
    path: labels.txt
```

## DecodeSpec

```yaml
decode:
  type: softmax | arg_max | top_k | identity
  axis: 1
  k: 5
```

## LabelsSpec

```yaml
labels:
  inline: ["cat", "dog", "bird"]
# or
labels:
  path: labels.txt
```

`labels.path` can be either a newline-delimited text file or a JSON array of strings.

## OpSpec (Typed Ops)

Supported `op` values:

- `resize`, `center_crop`
- `normalize`, `scale`, `cast`, `clip`
- `transpose`, `reshape`, `squeeze`, `unsqueeze`
- `softmax`, `arg_max`, `top_k`
- `mat_mul`, `add`, `mul`, `div`, `sub`
- `mean`, `std`, `sum`
- `expr` (advanced Rhai expressions)

## OpSpec (Expression)

```yaml
- op: expr
  code: "scale(x, 2.0)"
```

`x` is the input tensor. Helper functions include `reshape`, `transpose`, `matmul`, `sum`, `mean`, `std`, `softmax`, `argmax`, `topk`, `clip`, `normalize`, `scale`.
