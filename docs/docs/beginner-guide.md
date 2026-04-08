---
id: beginner-guide
title: Beginner Guide (No Tensor Experience Needed)
---

Welcome! If you have never worked with tensors before, this page is for you.

## Step 1: Think in Pictures

Most vision models accept an **image** as input. A tensor is just a **grid of numbers**:

- **Height** = number of rows (pixels)
- **Width** = number of columns (pixels)
- **Channels** = color layers (usually 3 for RGB)

So an image tensor with shape `[1, 3, 224, 224]` means:

- `1` = batch size (one image)
- `3` = RGB channels
- `224 x 224` = image size

You don’t need to manually build tensors. IPXml turns your uploaded image into a tensor automatically.

## Step 2: Create the App File

Create a file named `app.ipxml`:

```yaml
name: My First App
version: "0.1"
model:
  path: model.onnx
inputs:
  - id: image
    label: Upload an image
    type: image
    tensor:
      shape: [1, 3, 224, 224]
      layout: nchw
outputs:
  - id: scores
    label: Predictions
    type: scores
layout:
  rows:
    - components: [image]
    - components: [scores]
```

## Step 3: Bundle the Model

```bash
ipxml cc --ipxml app.ipxml --model model.onnx --out my_app.ipxmodel.import
```

## Step 4: Run It

```bash
ipxml run my_app.ipxmodel.import
```

## That’s It

You now have a working ML app with a UI—no code required.
