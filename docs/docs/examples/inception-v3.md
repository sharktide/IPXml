---
id: inception-v3
title: Inception v3 Example
---

This example uses the ONNX Inception v3 model in `examples/inception_v3` and demonstrates preprocessing + label mapping.

```yaml
name: Inception v3 Demo
version: "1.0"
model:
  path: adv_inception_v3_Opset18.onnx
inputs:
  - id: x
    label: Image
    type: image
    tensor:
      shape: [1, 3, 299, 299]
      layout: nchw
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
outputs:
  - id: scores
    label: Top Predictions
    type: scores
    source: "875"
    labels:
      path: labels_imagenet_1k.txt
    postprocess:
      - op: softmax
        axis: 1
    decode:
      type: top_k
      k: 5
      axis: 1
layout:
  rows:
    - components: [x]
    - components: [scores]
```
