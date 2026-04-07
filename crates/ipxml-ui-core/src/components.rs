use ipxml_schema::{IpxmlApp, InputSpec, OutputSpec};

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
    Bool(bool),
    ImagePath(String),
}

#[derive(Debug, Clone)]
pub enum OutputValue {
    Text(String),
    Number(f64),
    ImagePath(String),
    ClassScores(Vec<(String, f32)>),
}

pub fn input_value_for_spec(spec: &InputSpec) -> InputValue {
    match normalize_kind(&spec.kind).as_str() {
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
        "scores" | "classes" => OutputValue::ClassScores(Vec::new()),
        _ => OutputValue::Text(String::new()),
    }
}

fn normalize_kind(kind: &str) -> String {
    kind.trim().to_ascii_lowercase()
}
