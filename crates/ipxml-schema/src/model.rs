use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct IpxmlApp {
    pub name: String,
    pub version: Option<String>,
    pub model: ModelSpec,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
    pub layout: LayoutSpec,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub path: String, // relative path inside bundle
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputSpec {
    pub id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind: String, // "image", "text", "number", etc.
    #[serde(default)]
    pub tensor: Option<TensorSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputSpec {
    pub id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub kind: String, // "label", "image", "plot", etc.
    #[serde(default)]
    pub tensor: Option<TensorSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
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
