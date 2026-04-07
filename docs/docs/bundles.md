---
id: bundles
title: Bundles
---

A bundle is a single file with the `.ipxmodel.import` extension. It contains:

- `model.onnx`
- `schema.ipxml`
- assets (labels, images, sample inputs)

## Create a Bundle

```bash
ipxml bundle examples/mnist/app.ipxml -o mnist.ipxmodel.import
```

## Bundle Assets

If your schema references `labels.path`, the bundler will include it automatically.

```yaml
labels:
  path: labels.txt
```
