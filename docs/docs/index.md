---
id: intro
title: IPXml Overview
sidebar_label: Overview
---

# IPXml

IPXml is a declarative UI + preprocessing format for ONNX models. It lets you package a model, its inputs, preprocessing, outputs, and UI layout in a single bundle and run it with a standalone runtime.

**Core ideas**
- **Declarative UI**: inputs/outputs and layout are defined in a simple `.ipxml` file.
- **Pre/Post pipelines**: typed ops and optional expressions power preprocessing and postprocessing.
- **Portable bundles**: `.ipxmodel.import` is a zip-like bundle containing model + schema + assets.
- **Runtime UI**: the `ipxml` runtime builds the UI and handles inference.

If you have used Gradio or Streamlit, think of IPXml as the **portable ONNX-first version** that is language-agnostic and fully declarative.
