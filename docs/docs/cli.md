---
id: cli
title: CLI Reference
---

## Commands

```bash
ipxml cc --ipxml <app.ipxml> --model <model.onnx> --out <output.ipxmodel.import>
ipxml run <bundle.ipxmodel.import>
```

## Flags

- `--ipxml`: path to `.ipxml`
- `--model`: path to ONNX model (optional if the schema already defines model paths)
- `--out`: bundle output path
- `--help`: usage
