mod model;

pub use model::*;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("YAML parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
}

pub fn load_ipxml_from_str(s: &str) -> Result<IpxmlApp, SchemaError> {
    let app: IpxmlApp = serde_yaml::from_str(s)?;
    Ok(app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pre_post_labels_decode() {
        let yaml = r#"
name: Demo
model:
  path: model.onnx
inputs:
  - id: input
    label: Input
    type: image
    preprocess:
      - op: resize
        width: 224
        height: 224
      - op: normalize
        mean: [0.5, 0.5, 0.5]
        std: [0.25, 0.25, 0.25]
outputs:
  - id: output
    label: Output
    type: text
    source: logits
    postprocess:
      - op: softmax
        axis: -1
    decode:
      type: top_k
      k: 5
    labels:
      inline: ["a", "b", "c"]
layout:
  rows: []
"#;

        let app = load_ipxml_from_str(yaml).expect("parse ipxml");
        assert_eq!(app.inputs.len(), 1);
        assert!(app.inputs[0].preprocess.as_ref().unwrap().len() > 0);
        assert_eq!(app.outputs[0].source.as_deref(), Some("logits"));
        assert!(app.outputs[0].postprocess.as_ref().unwrap().len() > 0);
        assert!(app.outputs[0].labels.is_some());
    }
}
