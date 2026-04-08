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

    #[test]
    fn parse_multiple_models() {
        let yaml = r#"
name: Multi Demo
models:
  - id: encoder
    path: encoder.onnx
    inputs:
      - name: input
        source: user_image
  - id: classifier
    path: classifier.onnx
    inputs:
      - name: features
        source: encoder:output
inputs:
  - id: user_image
    label: Image
    type: image
outputs:
  - id: scores
    label: Scores
    type: scores
    model: classifier
    source: logits
layout:
  rows: []
"#;

        let app = load_ipxml_from_str(yaml).expect("parse ipxml");
        assert!(app.models.is_some());
        let models = app.models.as_ref().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id.as_deref(), Some("encoder"));
        assert!(models[0].inputs.as_ref().unwrap().len() == 1);
        assert_eq!(app.outputs[0].model.as_deref(), Some("classifier"));
    }

    #[test]
    fn parse_grouped_numeric_input() {
        let yaml = r#"
name: FireNet
model:
  path: firenet.onnx
inputs:
  - id: weather_features
    label: Weather Features
    type: number_list
    fields:
      - id: temperature
        label: Temperature
        default: 21.0
      - id: humidity
        label: Humidity
        default: 40.0
    tensor:
      shape: [1, 2]
outputs:
  - id: out
    label: Out
    type: number
layout:
  rows: []
"#;
        let app = load_ipxml_from_str(yaml).expect("parse grouped numeric input");
        assert_eq!(app.inputs.len(), 1);
        let fields = app.inputs[0].fields.as_ref().expect("fields should exist");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].id, "temperature");
    }

    #[test]
    fn parse_multimodal_and_conditions() {
        let yaml = r#"
name: Multi Modal
model:
  path: model.onnx
  when: "run_mode == \"full\""
  rules:
    - if_expr: "enable_model == true"
      then:
        run: true
      otherwise:
        run: false
inputs:
  - id: enable_model
    label: Enable
    type: checkbox
  - id: mode
    label: Mode
    type: multiple_choice
    choices:
      - id: fast
        label: Fast
      - id: full
        label: Full
  - id: mic
    label: Microphone
    type: audio
    media:
      sample_rate: 16000
      channels: 1
outputs:
  - id: out_video
    label: Video
    type: video
    source: out_video
    when: "enable_model == true"
layout:
  rows: []
"#;

        let app = load_ipxml_from_str(yaml).expect("parse multimodal");
        assert_eq!(app.inputs.len(), 3);
        assert!(app.model.as_ref().unwrap().when.is_some());
        assert!(app.model.as_ref().unwrap().rules.is_some());
        assert!(app.inputs[1].choices.is_some());
        assert!(app.inputs[2].media.is_some());
        assert_eq!(app.outputs[0].kind, "video");
    }

    #[test]
    fn parse_apply_if_op() {
        let yaml = r#"
name: ApplyIf Demo
model:
  path: model.onnx
inputs:
  - id: x
    label: X
    type: number
    preprocess:
      - op: apply_if
        when: "enable == true"
        then:
          - op: scale
            factor: 2.0
outputs:
  - id: y
    label: Y
    type: number
layout:
  rows: []
"#;
        let app = load_ipxml_from_str(yaml).expect("parse apply_if");
        let preprocess = app.inputs[0].preprocess.as_ref().unwrap();
        assert!(matches!(
            &preprocess[0],
            OpSpec::ApplyIf { when: Some(_), .. }
        ));
    }
}
