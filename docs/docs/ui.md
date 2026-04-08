---
id: ui
title: UI Rendering
---

The UI is generated dynamically based on the schema.

## Input Types

- `text`, `textarea`
- `number`, `int`, `float`
- `number_list` / `number_vector` / `vector` (grouped numeric fields)
- `bool` / `checkbox`
- `multiple_choice`, `multi_select`
- `image`, `audio`, `video`, `file`, `path` (with upload + text entry)

## Output Types

- `text`, `label`
- `number`
- `image`, `audio`, `video`, `file`, `path`
- `scores` / `classes` (ranked labels)

You can declare **multiple outputs** and they will render in the order listed in the schema.

## Conditional Rendering

Components can be shown/hidden with `when` or `rules`.

```yaml
- id: advanced_input
  label: Advanced Input
  type: number
  when: "mode == \"advanced\""
```

## Layout

```yaml
layout:
  rows:
    - components: [input_id]
    - components: [output_id]
```
