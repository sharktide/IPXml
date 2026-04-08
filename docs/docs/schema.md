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

`inputs`, `outputs`, and `layout` may be omitted. If missing, IPXml will infer inputs/outputs from the ONNX model and generate a default layout.

### Multiple Models

```yaml
models:
  - id: encoder
    path: encoder.onnx
    inputs:
      - name: input
        source: image
  - id: classifier
    path: classifier.onnx
    inputs:
      - name: features
        source: encoder:output
```

## InputSpec

```yaml
- id: input_id
  label: Input Label
  type: image | text | number | number_list | checkbox | multiple_choice | multi_select | audio | video | bool | file | path
  fields:
    - id: temperature
      label: Temperature
      default: 20.0
    - id: humidity
      label: Humidity
      default: 45.0
  when: "enable_ui == true"
  rules:
    - if_expr: "mode == \"advanced\""
      then:
        visible: true
      otherwise:
        visible: false
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

If your model input names do not match your UI input ids, use **model input bindings**:

```yaml
model:
  path: model.onnx
  inputs:
    - name: actual_model_input_name
      source: input_id
```

For grouped numeric model inputs (for example 5 weather values passed as one tensor), define one `number_list` input and set a tensor shape such as `[1, 5]`.

Choice and media examples:

```yaml
- id: mode
  label: Mode
  type: multiple_choice
  choices:
    - id: fast
      label: Fast
    - id: full
      label: Full

- id: mic
  label: Microphone
  type: audio
  media:
    sample_rate: 16000
    channels: 1
```

## OutputSpec

```yaml
- id: output_id
  label: Output Label
  type: text | number | scores | image | audio | video | file | path
  source: output_name_in_onnx   # defaults to id
  model: model_id               # required if multiple models are present
  postprocess:
    - op: softmax
      axis: 1
  decode:
    type: top_k
    k: 5
    axis: 1
  labels:
    path: labels.txt
  when: "enable_output == true"
```

Audio/video output pass-through can target input ids via `source`.

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
- `apply_if` (conditional op block)

## OpSpec (Expression)

```yaml
- op: expr
  code: "scale(x, 2.0)"
```

`x` is the input tensor. Helper functions include `reshape`, `transpose`, `matmul`, `sum`, `mean`, `std`, `softmax`, `argmax`, `topk`, `clip`, `normalize`, `scale`.

## Conditional Ops

```yaml
preprocess:
  - op: apply_if
    when: "mode == \"full\""
    then:
      - op: scale
        factor: 0.5
    otherwise:
      - op: scale
        factor: 1.0
```
