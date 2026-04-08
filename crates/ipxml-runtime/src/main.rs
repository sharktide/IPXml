use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use std::collections::HashMap;

use ipxml_bundle::{read_asset_from_bundle, read_ipxml_from_bundle, read_model_from_bundle};
use anyhow::Context;
use ipxml_schema::load_ipxml_from_str;
use ipxml_ui_core::{InputValue, OutputValue, UiBackend, UiContext};
use ipxml_ui_egui::{EguiBackend, ModelEntry, PipelineRunner};
use ort::session::Session;
use ort::value::ValueType;

use axum::{
    extract::Multipart,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};

#[derive(Parser)]
#[command(name = "ipxml")]
#[command(about = "IPXml runtime", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a .ipxmodel.import bundle
    Run {
        /// Path to bundle
        bundle: PathBuf,
        /// Serve the UI over HTTP
        #[arg(long)]
        serve: bool,
        /// Port for the web UI
        #[arg(long, default_value = "7860")]
        port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { bundle, serve, port } => {
            let ipxml_source = read_ipxml_from_bundle(&bundle)?;
            let mut app = load_ipxml_from_str(&ipxml_source)?;

            let ctx = UiContext {
                app_name: app.name.clone(),
            };

            let models = resolve_models(&app)?;
            let model_entries = models
                .into_iter()
                .map(|model| {
                    let bytes = read_model_from_bundle(&bundle, &model.path)?;
                    Ok(ModelEntry {
                        id: model.id,
                        bytes,
                        inputs: model.inputs,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            infer_missing_io(&mut app, &model_entries)?;

            let labels = load_labels_from_bundle(&bundle, &app)?;
            if serve {
                run_web(app, model_entries, labels, port)?;
            } else {
                let backend = EguiBackend::new(app, model_entries, labels);
                Box::new(backend).run(ctx);
            }

        }
    }

    Ok(())
}

fn load_labels_from_bundle(
    bundle: &PathBuf,
    app: &ipxml_schema::IpxmlApp,
) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let mut map = HashMap::new();
    for output in &app.outputs {
        if let Some(labels) = &output.labels {
            if let Some(inline) = &labels.inline {
                map.insert(output.id.clone(), inline.clone());
            } else if let Some(path) = &labels.path {
                let bytes = read_asset_from_bundle(bundle, path)?;
                let text = String::from_utf8(bytes)?;
                let trimmed = text.trim_start();
                let list = if trimmed.starts_with('[') {
                    serde_json::from_str::<Vec<String>>(trimmed)?
                } else {
                    text.lines()
                        .map(|line| line.trim().to_string())
                        .filter(|line| !line.is_empty())
                        .collect::<Vec<_>>()
                };
                map.insert(output.id.clone(), list);
            }
        }
    }
    Ok(map)
}

#[derive(Debug, Clone)]
struct ModelResolved {
    id: String,
    path: String,
    inputs: Option<Vec<ipxml_schema::ModelInputBinding>>,
}

fn resolve_models(app: &ipxml_schema::IpxmlApp) -> anyhow::Result<Vec<ModelResolved>> {
    if let Some(models) = &app.models {
        if models.is_empty() {
            return Err(anyhow::anyhow!("models is empty"));
        }
        let mut resolved = Vec::new();
        for (idx, model) in models.iter().enumerate() {
            let id = model
                .id
                .clone()
                .unwrap_or_else(|| format!("model_{}", idx + 1));
            resolved.push(ModelResolved {
                id,
                path: model.path.clone(),
                inputs: model.inputs.clone(),
            });
        }
        return Ok(resolved);
    }

    if let Some(model) = &app.model {
        let id = model.id.clone().unwrap_or_else(|| "model".to_string());
        return Ok(vec![ModelResolved {
            id,
            path: model.path.clone(),
            inputs: model.inputs.clone(),
        }]);
    }

    Err(anyhow::anyhow!(
        "No model defined. Provide `model` or `models` in the schema."
    ))
}

fn infer_missing_io(app: &mut ipxml_schema::IpxmlApp, models: &[ModelEntry]) -> anyhow::Result<()> {
    if !app.inputs.is_empty() && !app.outputs.is_empty() {
        return Ok(());
    }

    let mut inferred_inputs = Vec::new();
    let mut inferred_outputs = Vec::new();
    let mut used_input_ids = std::collections::HashSet::new();
    let mut used_output_ids = std::collections::HashSet::new();

    for model in models {
        let session = Session::builder()?
            .commit_from_memory(&model.bytes)
            .context("load ONNX model for inference")?;

        if app.inputs.is_empty() {
            for input in session.inputs() {
                let name = input.name().to_string();
                let mut id = name.clone();
                if used_input_ids.contains(&id) {
                    id = format!("{}_{}", model.id, id);
                }
                used_input_ids.insert(id.clone());

                let shape = outlet_shape(input.dtype())
                    .into_iter()
                    .map(|d| if d > 0 { d as usize } else { 0 })
                    .collect::<Vec<_>>();
                let (kind, layout) = infer_input_kind(&shape);
                let label = pretty_label(&id);

                inferred_inputs.push(ipxml_schema::InputSpec {
                    id,
                    label,
                    kind: kind.to_string(),
                    tensor: Some(ipxml_schema::TensorSpec {
                        shape: Some(shape),
                        layout,
                        normalize: None,
                    }),
                    preprocess: None,
                });
            }
        }

        if app.outputs.is_empty() {
            for output in session.outputs() {
                let name = output.name().to_string();
                let mut id = name.clone();
                if used_output_ids.contains(&id) {
                    id = format!("{}_{}", model.id, id);
                }
                used_output_ids.insert(id.clone());

                let shape = outlet_shape(output.dtype())
                    .into_iter()
                    .map(|d| if d > 0 { d as usize } else { 0 })
                    .collect::<Vec<_>>();
                let kind = infer_output_kind(&shape);
                let label = pretty_label(&id);

                inferred_outputs.push(ipxml_schema::OutputSpec {
                    id,
                    label,
                    kind: kind.to_string(),
                    tensor: Some(ipxml_schema::TensorSpec {
                        shape: Some(shape),
                        layout: None,
                        normalize: None,
                    }),
                    source: Some(name),
                    model: if models.len() > 1 {
                        Some(model.id.clone())
                    } else {
                        None
                    },
                    postprocess: None,
                    labels: None,
                    decode: None,
                });
            }
        }
    }

    if app.inputs.is_empty() {
        app.inputs = inferred_inputs;
    }
    if app.outputs.is_empty() {
        app.outputs = inferred_outputs;
    }

    if app.layout.rows.is_empty() {
        app.layout.rows.push(ipxml_schema::LayoutRow {
            components: app.inputs.iter().map(|i| i.id.clone()).collect(),
        });
        app.layout.rows.push(ipxml_schema::LayoutRow {
            components: app.outputs.iter().map(|o| o.id.clone()).collect(),
        });
    }

    Ok(())
}

fn infer_input_kind(shape: &[usize]) -> (&'static str, Option<String>) {
    if shape.len() == 4 {
        if shape.get(1).copied().unwrap_or(0) == 3 || shape.get(1).copied().unwrap_or(0) == 1
        {
            return ("image", Some("nchw".to_string()));
        }
        if shape.get(3).copied().unwrap_or(0) == 3 || shape.get(3).copied().unwrap_or(0) == 1
        {
            return ("image", Some("nhwc".to_string()));
        }
    }
    let total = shape.iter().copied().filter(|d| *d > 0).product::<usize>();
    if total <= 1 {
        ("number", None)
    } else {
        ("text", None)
    }
}

fn infer_output_kind(shape: &[usize]) -> &'static str {
    let total = shape.iter().copied().filter(|d| *d > 0).product::<usize>();
    if total <= 1 {
        return "number";
    }
    if shape.len() >= 2 {
        let last = shape[shape.len() - 1];
        if last > 1 {
            return "scores";
        }
    }
    "text"
}

fn pretty_label(id: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for ch in id.chars() {
        if ch == '_' || ch == '-' {
            out.push(' ');
            capitalize = true;
        } else if capitalize {
            out.push(ch.to_ascii_uppercase());
            capitalize = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn outlet_shape(dtype: &ValueType) -> Vec<i64> {
    match dtype {
        ValueType::Tensor { shape, .. } => shape.to_vec(),
        _ => Vec::new(),
    }
}

struct WebState {
    app: ipxml_schema::IpxmlApp,
    labels: HashMap<String, Vec<String>>,
    runner: Mutex<PipelineRunner>,
}

fn run_web(
    app: ipxml_schema::IpxmlApp,
    models: Vec<ModelEntry>,
    labels: HashMap<String, Vec<String>>,
    port: u16,
) -> anyhow::Result<()> {
    let runner = PipelineRunner::new(models)?;
    let state = Arc::new(WebState {
        app,
        labels,
        runner: Mutex::new(runner),
    });

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/run", post(run_handler))
            .with_state(state);

        let addr = format!("0.0.0.0:{port}");
        println!("Web UI available at http://localhost:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    Ok(())
}

async fn index_handler(State(state): State<Arc<WebState>>) -> Html<String> {
    Html(render_page(&state.app, None, None))
}

async fn run_handler(
    State(state): State<Arc<WebState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut input_map: HashMap<String, InputValue> = HashMap::new();
    for input in &state.app.inputs {
        if matches!(input.kind.as_str(), "bool" | "boolean") {
            input_map.insert(input.id.clone(), InputValue::Bool(false));
        }
    }

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        let Some(spec) = state.app.inputs.iter().find(|i| i.id == name) else {
            continue;
        };

        let kind = spec.kind.trim().to_ascii_lowercase();
        let file_name = field.file_name().map(|s| s.to_string());
        let bytes = match field.bytes().await {
            Ok(data) => data,
            Err(_) => continue,
        };

        if matches!(kind.as_str(), "image" | "file" | "path") {
            if let Some(file_name) = file_name {
                if !bytes.is_empty() {
                    let path = write_temp_file(&bytes, &file_name);
                    input_map.insert(name, InputValue::ImagePath(path));
                    continue;
                }
            }

            let text = String::from_utf8_lossy(&bytes).trim().to_string();
            if !text.is_empty() {
                input_map.insert(name, InputValue::ImagePath(text));
            }
            continue;
        }

        if matches!(kind.as_str(), "bool" | "boolean") {
            input_map.insert(name, InputValue::Bool(true));
            continue;
        }

        let text = String::from_utf8_lossy(&bytes).to_string();
        if matches!(kind.as_str(), "number" | "float" | "int" | "integer") {
            let parsed = text.trim().parse::<f64>().unwrap_or(0.0);
            input_map.insert(name, InputValue::Number(parsed));
        } else {
            input_map.insert(name, InputValue::Text(text));
        }
    }

    let outputs = {
        let mut runner = state.runner.lock().unwrap();
        runner.run(&state.app, &input_map, &state.labels)
    };

    match outputs {
        Ok(outputs) => Html(render_page(&state.app, Some(&input_map), Some(&outputs))).into_response(),
        Err(err) => {
            let html = render_page(&state.app, Some(&input_map), None);
            let html = format!("{html}<p style=\"color:#b91c1c;\">Run failed: {err}</p>");
            (StatusCode::INTERNAL_SERVER_ERROR, Html(html)).into_response()
        }
    }
}

fn write_temp_file(bytes: &[u8], filename: &str) -> String {
    let mut path = std::env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let safe_name = filename.replace(['\\', '/', ':'], "_");
    path.push(format!("ipxml_{ts}_{safe_name}"));
    if let Ok(mut file) = std::fs::File::create(&path) {
        let _ = std::io::Write::write_all(&mut file, bytes);
    }
    path.display().to_string()
}

fn render_page(
    app: &ipxml_schema::IpxmlApp,
    inputs: Option<&HashMap<String, InputValue>>,
    outputs: Option<&HashMap<String, OutputValue>>,
) -> String {
    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\" />");
    html.push_str("<title>IPXml</title>");
    html.push_str("<style>");
    html.push_str(
        "body{font-family:Inter,system-ui,Arial,sans-serif;background:#f5f7fb;padding:24px;}\
         .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:16px;}\
         label{display:block;font-weight:600;margin-bottom:6px;}\
         input[type=text],input[type=number],textarea{width:100%;padding:8px;border:1px solid #cbd5f5;border-radius:8px;}\
         table{width:100%;border-collapse:collapse;}\
         th,td{padding:6px 8px;border-bottom:1px solid #e5e7eb;text-align:left;}\
         .btn{background:#2563eb;color:#fff;border:none;padding:10px 16px;border-radius:8px;font-weight:600;cursor:pointer;}",
    );
    html.push_str("</style></head><body>");
    html.push_str(&format!("<h1>{}</h1>", app.name));
    html.push_str("<form method=\"post\" action=\"/run\" enctype=\"multipart/form-data\">");

    for input in &app.inputs {
        html.push_str("<div class=\"card\">");
        html.push_str(&format!("<label for=\"{}\">{}</label>", input.id, input.label));
        let kind = input.kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "textarea" => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v { InputValue::Text(t) => Some(t.as_str()), _ => None })
                    .unwrap_or("");
                html.push_str(&format!("<textarea name=\"{}\" rows=\"4\">{}</textarea>", input.id, value));
            }
            "number" | "float" | "int" | "integer" => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v { InputValue::Number(n) => Some(*n), _ => None })
                    .unwrap_or(0.0);
                html.push_str(&format!("<input type=\"number\" step=\"any\" name=\"{}\" value=\"{}\" />", input.id, value));
            }
            "bool" | "boolean" => {
                html.push_str(&format!("<input type=\"checkbox\" name=\"{}\" />", input.id));
            }
            "image" | "file" | "path" => {
                html.push_str(&format!("<input type=\"file\" name=\"{}\" />", input.id));
                html.push_str(&format!("<input type=\"text\" name=\"{}\" placeholder=\"or paste a file path\" />", input.id));
            }
            _ => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v { InputValue::Text(t) => Some(t.as_str()), _ => None })
                    .unwrap_or("");
                html.push_str(&format!("<input type=\"text\" name=\"{}\" value=\"{}\" />", input.id, value));
            }
        }
        html.push_str("</div>");
    }

    html.push_str("<button class=\"btn\" type=\"submit\">Run Model</button>");
    html.push_str("</form>");

    if let Some(outputs) = outputs {
        html.push_str("<h2>Outputs</h2>");
        for output in &app.outputs {
            html.push_str("<div class=\"card\">");
            html.push_str(&format!("<strong>{}</strong><br/>", output.label));
            if let Some(value) = outputs.get(&output.id) {
                match value {
                    OutputValue::Text(text) => html.push_str(&format!("<pre>{}</pre>", text)),
                    OutputValue::Number(num) => html.push_str(&format!("<p>{:.4}</p>", num)),
                    OutputValue::ImagePath(path) => html.push_str(&format!("<p>{}</p>", path)),
                    OutputValue::ClassScores(scores) => {
                        html.push_str("<table><tr><th>Label</th><th>Score</th></tr>");
                        for (label, score) in scores.iter().take(20) {
                            html.push_str(&format!("<tr><td>{}</td><td>{:.4}</td></tr>", label, score));
                        }
                        html.push_str("</table>");
                    }
                }
            }
            html.push_str("</div>");
        }
    }

    html.push_str("</body></html>");
    html
}
