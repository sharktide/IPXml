---
id: getting-started
title: Getting Started
---

## Quick Start

1. Write an `.ipxml` file describing inputs, outputs, and layout.
2. Bundle the ONNX model + schema into a `.ipxmodel.import` file.
3. Run the bundle with the runtime.

```bash
ipxml bundle examples/mnist/app.ipxml -o mnist.ipxmodel.import
ipxml run mnist.ipxmodel.import
```

## Minimal Example

```yaml
name: Demo
version: "0.1.0"
model:
  path: model.onnx
inputs:
  - id: input
    label: Image
    type: image
    tensor:
      shape: [1, 3, 224, 224]
      layout: nchw
outputs:
  - id: output
    label: Scores
    type: scores
layout:
  rows:
    - components: [input]
    - components: [output]
```

## Project Structure

- `.ipxml` file (schema)
- `.onnx` file (model)
- optional assets (labels, images, example inputs)
