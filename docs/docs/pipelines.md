---
id: pipelines
title: Multiple Models (Ensembles & Chaining)
---

IPXml supports **multiple models** inside one app. This enables:

- **Ensembles** (run several models and compare or average outputs)
- **Chaining** (feed output of one model into the next)

## Define Multiple Models

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

### Source Rules

A `source` can be:

- A **UI input id** (e.g., `image`)
- A **prior model output** in the form `model_id:output_name` (e.g., `encoder:output`)

Models run **in the order they are listed**.

## Multiple Outputs

You can expose multiple outputs from any model:

```yaml
outputs:
  - id: top_scores
    label: Top Scores
    type: scores
    model: classifier
    source: logits
  - id: raw_logits
    label: Raw Logits
    type: text
    model: classifier
    source: logits
```

## Ensemble Example (Two Models)

```yaml
models:
  - id: model_a
    path: model_a.onnx
    inputs:
      - name: input
        source: image
  - id: model_b
    path: model_b.onnx
    inputs:
      - name: input
        source: image

outputs:
  - id: scores_a
    label: Model A
    type: scores
    model: model_a
    source: logits
  - id: scores_b
    label: Model B
    type: scores
    model: model_b
    source: logits
```
