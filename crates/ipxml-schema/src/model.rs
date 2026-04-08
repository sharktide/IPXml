use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct IpxmlApp {
    pub name: String,
    pub version: Option<String>,
    #[serde(default)]
    pub model: Option<ModelSpec>,
    #[serde(default)]
    pub models: Option<Vec<ModelSpec>>,
    #[serde(default)]
    pub inputs: Vec<InputSpec>,
    #[serde(default)]
    pub outputs: Vec<OutputSpec>,
    #[serde(default)]
    pub layout: LayoutSpec,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelSpec {
    /// Optional identifier used when multiple models are present.
    pub id: Option<String>,
    pub path: String, // relative path inside bundle
    /// Optional explicit input bindings for this model.
    #[serde(default)]
    pub inputs: Option<Vec<ModelInputBinding>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInputBinding {
    /// Model input name.
    pub name: String,
    /// Source identifier. Can be a UI input id or a prior model output in the form
    /// "model_id:output_name".
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputSpec {
    pub id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind: String, // "image", "text", "number", etc.
    #[serde(default)]
    pub tensor: Option<TensorSpec>,
    #[serde(default)]
    pub preprocess: Option<Vec<OpSpec>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputSpec {
    pub id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind: String, // "label", "image", "plot", etc.
    #[serde(default)]
    pub tensor: Option<TensorSpec>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub postprocess: Option<Vec<OpSpec>>,
    #[serde(default)]
    pub labels: Option<LabelsSpec>,
    #[serde(default)]
    pub decode: Option<DecodeSpec>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct LayoutSpec {
    pub rows: Vec<LayoutRow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayoutRow {
    pub components: Vec<String>, // input/output ids
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct TensorSpec {
    /// Optional explicit shape, e.g. [1, 3, 224, 224]
    pub shape: Option<Vec<usize>>,
    /// Optional layout hint like "nchw" or "nhwc"
    pub layout: Option<String>,
    /// Optional normalization params for image tensors
    pub normalize: Option<NormalizeSpec>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct NormalizeSpec {
    /// Scale factor applied before mean/std, e.g. 1.0/255.0
    pub scale: Option<f32>,
    /// Per-channel mean
    pub mean: Option<Vec<f32>>,
    /// Per-channel std
    pub std: Option<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LabelsSpec {
    pub inline: Option<Vec<String>>,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecodeSpec {
    Softmax { axis: Option<isize> },
    ArgMax { axis: Option<isize> },
    TopK { k: usize, axis: Option<isize> },
    Identity,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TensorLiteral {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OpSpec {
    Resize {
        width: usize,
        height: usize,
        layout: Option<String>,
    },
    CenterCrop {
        width: usize,
        height: usize,
        layout: Option<String>,
    },
    Normalize {
        scale: Option<f32>,
        mean: Option<Vec<f32>>,
        std: Option<Vec<f32>>,
    },
    Scale {
        factor: f32,
    },
    Cast {
        dtype: String,
    },
    Clip {
        min: f32,
        max: f32,
    },
    Transpose {
        axes: Vec<usize>,
    },
    Reshape {
        shape: Vec<i64>,
    },
    Squeeze {
        axes: Option<Vec<usize>>,
    },
    Unsqueeze {
        axes: Vec<usize>,
    },
    Softmax {
        axis: Option<isize>,
    },
    ArgMax {
        axis: Option<isize>,
    },
    TopK {
        k: usize,
        axis: Option<isize>,
        largest: Option<bool>,
    },
    MatMul {
        rhs: TensorLiteral,
    },
    Add {
        value: Option<f32>,
        tensor: Option<TensorLiteral>,
    },
    Mul {
        value: Option<f32>,
        tensor: Option<TensorLiteral>,
    },
    Div {
        value: Option<f32>,
        tensor: Option<TensorLiteral>,
    },
    Sub {
        value: Option<f32>,
        tensor: Option<TensorLiteral>,
    },
    Mean {
        axis: Option<isize>,
        keepdims: Option<bool>,
    },
    Std {
        axis: Option<isize>,
        keepdims: Option<bool>,
    },
    Sum {
        axis: Option<isize>,
        keepdims: Option<bool>,
    },
    Expr {
        code: String,
    },
}
