---
id: cli
title: CLI Reference
---

## Commands

```bash
ipxml cc --ipxml <app.ipxml> [--out <output.ipxmodel.import>]
ipxml run <bundle.ipxmodel.import>
ipxml editor [app.ipxml]
```

Run in the browser:

```bash
ipxml run <bundle.ipxmodel.import> --serve --port 7860
```

## Flags

- `--ipxml`: path to `.ipxml`
- `--out`: optional bundle output path
- `--serve`: launch a browser UI
- `--port`: HTTP port for the browser UI
- `--help`: usage

`ipxml editor` launches a desktop GUI for creating and editing `.ipxml` files.

## Defaults + Discovery

- If `--out` is omitted, output defaults to `<model_name>.ipxmodel.import` (or the `.ipxml` filename stem if model names are unavailable).
- Model files are discovered from `model.path` / `models[].path` in schema.
- Referenced assets are auto-discovered from schema references (for example `labels.path` and media decode paths).

## Compatibility Binaries

- `ipxml-cc` and `ipxml-runtime` are still available as compatibility aliases.
- Prefer unified commands through `ipxml`.
