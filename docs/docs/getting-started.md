---
id: getting-started
title: Getting Started
---

## Quick Start

1. Write an `.ipxml` file describing inputs, outputs, and layout.
2. Bundle the ONNX model + schema into a `.ipxmodel.import` file.
3. Run the bundle with the runtime.

```bash
ipxml cc --ipxml examples/mnist/app.ipxml --model examples/mnist/mnist-8.onnx --out mnist.ipxmodel.import
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

### Auto-Infer (Fastest)

If you omit `inputs` and `outputs`, IPXml will infer them from the model and generate a basic layout automatically.

## Project Structure

- `.ipxml` file (schema)
- `.onnx` file (model)
- optional assets (labels, images, example inputs)

## If You’re New to Tensors

You don’t need to think about math. For images, just remember:

- `[1, 3, 224, 224]` means 1 image, 3 color channels (RGB), 224x224 pixels.
- IPXml converts your upload into that tensor for you.
