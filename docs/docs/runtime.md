---
id: runtime
title: Runtime
---

The runtime loads a bundle, builds the UI, runs preprocessing, invokes ONNX Runtime, and applies postprocessing/decoding.

## Run

```bash
ipxml run path/to/model.ipxmodel.import
```

## Execution Flow

1. Parse schema
2. Load model bytes
3. Load assets (labels)
4. Build UI
5. On run: preprocess → ONNX → postprocess → decode → render

## Multiple Models

If `models` is present, the runtime executes them **in order**. Outputs can feed later models via `source: model_id:output_name`.
