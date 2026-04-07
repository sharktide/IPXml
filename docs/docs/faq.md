---
id: faq
title: FAQ
---

## Does IPXml infer inputs automatically?
No. IPXml is explicit by design. The schema is the source of truth.

## Can I use custom labels?
Yes. Provide `labels.inline` or `labels.path`.

## Is GPU supported?
The runtime uses ONNX Runtime for inference. Pre/Post ops in this phase run on CPU using `ndarray`.

## Can I embed this UI?
Yes. The UI backend is pluggable; current examples use egui.
