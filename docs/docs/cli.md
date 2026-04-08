---
id: cli
title: CLI Reference
---

## Commands

```bash
ipxml cc --ipxml <app.ipxml> --model <model.onnx> --out <output.ipxmodel.import>
ipxml run <bundle.ipxmodel.import>
```

Run in the browser:

```bash
ipxml run <bundle.ipxmodel.import> --serve --port 7860
```

## Flags

- `--ipxml`: path to `.ipxml`
- `--model`: path to ONNX model (optional if the schema already defines model paths)
- `--out`: bundle output path
- `--serve`: launch a browser UI
- `--port`: HTTP port for the browser UI
- `--help`: usage
