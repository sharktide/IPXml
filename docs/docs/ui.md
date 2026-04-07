---
id: ui
title: UI Rendering
---

The UI is generated dynamically based on the schema.

## Input Types

- `text`, `textarea`
- `number`, `int`, `float`
- `bool`
- `image`, `file`, `path` (with upload + text entry)

## Output Types

- `text`, `label`
- `number`
- `image`, `file`, `path`
- `scores` / `classes` (ranked labels)

## Layout

```yaml
layout:
  rows:
    - components: [input_id]
    - components: [output_id]
```
