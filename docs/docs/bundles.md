---
id: bundles
title: Bundles
---

A bundle is a single file with the `.ipxmodel.import` extension. It contains:

- one or more model files (`*.onnx`)
- `schema.ipxml`
- assets (labels, images, sample inputs)

## Create a Bundle

```bash
ipxml cc --ipxml examples/mnist/app.ipxml --model examples/mnist/mnist-8.onnx --out mnist.ipxmodel.import
```

## Bundle Assets

If your schema references `labels.path`, the bundler will include it automatically.

```yaml
labels:
  path: labels.txt
```
