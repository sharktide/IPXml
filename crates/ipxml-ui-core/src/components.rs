use ipxml_schema::{InputSpec, IpxmlApp, OutputSpec};

pub struct UiContext {
    pub app_name: String,
}

pub trait UiBackend {
    fn run(self: Box<Self>, ctx: UiContext);
}

pub trait UiFactory {
    type Backend: UiBackend;

    fn create_backend(app: &IpxmlApp) -> Self::Backend;
}

pub fn find_input<'a>(app: &'a IpxmlApp, id: &str) -> Option<&'a InputSpec> {
    app.inputs.iter().find(|i| i.id == id)
}

pub fn find_output<'a>(app: &'a IpxmlApp, id: &str) -> Option<&'a OutputSpec> {
    app.outputs.iter().find(|o| o.id == id)
}

#[derive(Debug, Clone)]
pub enum InputValue {
    Text(String),
    Number(f64),
    NumberList(Vec<f64>),
    Choice(String),
    Choices(Vec<String>),
    Bool(bool),
    ImagePath(String),
    AudioPath(String),
    VideoPath(String),
}

#[derive(Debug, Clone)]
pub enum OutputValue {
    Text(String),
    Number(f64),
    ImagePath(String),
    AudioPath(String),
    VideoPath(String),
    ClassScores(Vec<(String, f32)>),
}

pub fn input_value_for_spec(spec: &InputSpec) -> InputValue {
    if let Some(fields) = &spec.fields {
        let values = fields
            .iter()
            .map(|field| field.default.unwrap_or(0.0))
            .collect::<Vec<_>>();
        return InputValue::NumberList(values);
    }

    match normalize_kind(&spec.kind).as_str() {
        "checkbox" => InputValue::Bool(false),
        "multiple_choice" => {
            let first = spec
                .choices
                .as_ref()
                .and_then(|c| c.first())
                .map(|c| c.value.clone().unwrap_or_else(|| c.id.clone()))
                .unwrap_or_default();
            InputValue::Choice(first)
        }
        "multi_select" => InputValue::Choices(Vec::new()),
        "audio" => InputValue::AudioPath(String::new()),
        "video" => InputValue::VideoPath(String::new()),
        "number_list" | "number_vector" | "vector" => InputValue::NumberList(Vec::new()),
        "text" | "string" | "textarea" => InputValue::Text(String::new()),
        "number" | "float" | "int" | "integer" => InputValue::Number(0.0),
        "bool" | "boolean" => InputValue::Bool(false),
        "image" | "file" | "path" => InputValue::ImagePath(String::new()),
        _ => InputValue::Text(String::new()),
    }
}

pub fn output_value_for_spec(spec: &OutputSpec) -> OutputValue {
    match normalize_kind(&spec.kind).as_str() {
        "text" | "label" | "string" => OutputValue::Text(String::new()),
        "number" | "float" | "int" | "integer" => OutputValue::Number(0.0),
        "image" | "file" | "path" => OutputValue::ImagePath(String::new()),
        "audio" => OutputValue::AudioPath(String::new()),
        "video" => OutputValue::VideoPath(String::new()),
        "scores" | "classes" => OutputValue::ClassScores(Vec::new()),
        _ => OutputValue::Text(String::new()),
    }
}

fn normalize_kind(kind: &str) -> String {
    kind.trim().to_ascii_lowercase()
}
