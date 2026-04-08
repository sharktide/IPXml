use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use rhai::{Engine, Scope};
use std::collections::HashMap;

use anyhow::Context;
use ipxml_bundle::{
    BundleAsset, BundleModel, create_bundle, read_asset_from_bundle, read_ipxml_from_bundle,
    read_model_from_bundle,
};
use ipxml_schema::load_ipxml_from_str;
use ipxml_ui_core::{InputValue, OutputValue, UiBackend, UiContext};
use ipxml_ui_egui::{EguiBackend, ModelEntry, PipelineRunner, run_editor};
use ort::session::Session;
use ort::value::ValueType;

use axum::{
    Router,
    extract::Multipart,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
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
    /// Compile an .ipxml app + assets into a .ipxmodel.import bundle
    Cc {
        /// Path to .ipxml file
        #[arg(long)]
        ipxml: PathBuf,
        /// Optional output bundle path
        #[arg(long)]
        out: Option<PathBuf>,
    },
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
    /// Open graphical editor for .ipxml files
    Editor {
        /// Optional .ipxml file to open
        file: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    if let Some(bin) = std::env::args().next() {
        if bin.ends_with("ipxml-runtime") || bin.ends_with("ipxml-runtime.exe") {
            eprintln!("ipxml-runtime is kept for compatibility. Prefer: ipxml run <bundle>");
        }
    }
    let cli = Cli::parse();

    match cli.command {
        Commands::Cc { ipxml, out } => {
            let ipxml_source = fs::read_to_string(&ipxml)?;
            let app = load_ipxml_from_str(&ipxml_source)?;
            let base_dir = ipxml.parent().unwrap_or_else(|| std::path::Path::new("."));
            let models = collect_models_for_cc(&app, base_dir)?;
            let assets = collect_assets_for_cc(&app, base_dir)?;
            let out_path = out.unwrap_or_else(|| default_out_path(&ipxml, &app));
            create_bundle(&out_path, &app, &ipxml_source, &models, &assets)?;
            println!("Bundle created at {}", out_path.display());
        }
        Commands::Run {
            bundle,
            serve,
            port,
        } => {
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
                        when: model.when,
                        rules: model.rules,
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
        Commands::Editor { file } => {
            run_editor(file)?;
        }
    }

    Ok(())
}

fn collect_assets_for_cc(
    app: &ipxml_schema::IpxmlApp,
    base_dir: &std::path::Path,
) -> anyhow::Result<Vec<BundleAsset>> {
    let mut assets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut add_asset = |path: &str| -> anyhow::Result<()> {
        if !seen.insert(path.to_string()) {
            return Ok(());
        }
        let full_path = base_dir.join(path);
        let bytes = fs::read(&full_path)?;
        assets.push(BundleAsset {
            path: path.to_string(),
            bytes,
        });
        Ok(())
    };
    for output in &app.outputs {
        if let Some(labels) = &output.labels {
            if let Some(path) = &labels.path {
                add_asset(path)?;
            }
        }
        if let Some(media) = &output.media {
            if let Some(path) = &media.decode {
                add_asset(path)?;
            }
        }
    }
    for input in &app.inputs {
        if let Some(media) = &input.media {
            if let Some(path) = &media.decode {
                add_asset(path)?;
            }
        }
    }
    Ok(assets)
}

fn collect_models_for_cc(
    app: &ipxml_schema::IpxmlApp,
    base_dir: &std::path::Path,
) -> anyhow::Result<Vec<BundleModel>> {
    if let Some(models) = &app.models {
        if models.is_empty() {
            return Err(anyhow::anyhow!("models is empty"));
        }
        let mut out = Vec::new();
        for model in models {
            let full_path = base_dir.join(&model.path);
            let bytes = fs::read(&full_path)?;
            out.push(BundleModel {
                path: model.path.clone(),
                bytes,
            });
        }
        return Ok(out);
    }
    let Some(model) = &app.model else {
        return Err(anyhow::anyhow!(
            "No model defined. Provide `model` or `models` in the schema."
        ));
    };
    let full_path = base_dir.join(&model.path);
    let bytes = fs::read(&full_path)?;
    Ok(vec![BundleModel {
        path: model.path.clone(),
        bytes,
    }])
}

fn default_out_path(ipxml: &PathBuf, app: &ipxml_schema::IpxmlApp) -> PathBuf {
    let base_dir = ipxml
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let stem = if let Some(model) = &app.model {
        PathBuf::from(&model.path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "bundle".to_string())
    } else if let Some(models) = &app.models {
        models
            .first()
            .map(|m| {
                PathBuf::from(&m.path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "bundle".to_string())
            })
            .unwrap_or_else(|| "bundle".to_string())
    } else {
        ipxml
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "bundle".to_string())
    };
    base_dir.join(format!("{stem}.ipxmodel.import"))
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
    when: Option<String>,
    rules: Option<Vec<ipxml_schema::RuleSpec>>,
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
                when: model.when.clone(),
                rules: model.rules.clone(),
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
            when: model.when.clone(),
            rules: model.rules.clone(),
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
                    choices: None,
                    media: None,
                    fields: None,
                    when: None,
                    rules: None,
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
                    media: None,
                    when: None,
                    rules: None,
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
        if shape.get(1).copied().unwrap_or(0) == 3 || shape.get(1).copied().unwrap_or(0) == 1 {
            return ("image", Some("nchw".to_string()));
        }
        if shape.get(3).copied().unwrap_or(0) == 3 || shape.get(3).copied().unwrap_or(0) == 1 {
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
        input_map.insert(input.id.clone(), ipxml_ui_core::input_value_for_spec(input));
    }

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        let file_name = field.file_name().map(|s| s.to_string());
        let bytes = match field.bytes().await {
            Ok(data) => data,
            Err(_) => continue,
        };
        if let Some((base_id, idx)) = parse_number_list_field_name(&name) {
            if let Some(spec) = state.app.inputs.iter().find(|i| i.id == base_id) {
                let kind = spec.kind.trim().to_ascii_lowercase();
                if matches!(kind.as_str(), "number_list" | "number_vector" | "vector")
                    || spec.fields.is_some()
                {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let parsed = text.trim().parse::<f64>().unwrap_or(0.0);
                    set_number_list_value(&mut input_map, &base_id, idx, parsed);
                    continue;
                }
            }
        }

        let Some(spec) = state.app.inputs.iter().find(|i| i.id == name) else {
            continue;
        };

        let kind = spec.kind.trim().to_ascii_lowercase();

        if matches!(kind.as_str(), "image" | "file" | "path" | "audio" | "video") {
            if let Some(file_name) = file_name {
                if !bytes.is_empty() {
                    let path = write_temp_file(&bytes, &file_name);
                    if kind == "audio" {
                        input_map.insert(name, InputValue::AudioPath(path));
                    } else if kind == "video" {
                        input_map.insert(name, InputValue::VideoPath(path));
                    } else {
                        input_map.insert(name, InputValue::ImagePath(path));
                    }
                    continue;
                }
            }

            let text = String::from_utf8_lossy(&bytes).trim().to_string();
            if !text.is_empty() {
                if kind == "audio" {
                    input_map.insert(name, InputValue::AudioPath(text));
                } else if kind == "video" {
                    input_map.insert(name, InputValue::VideoPath(text));
                } else {
                    input_map.insert(name, InputValue::ImagePath(text));
                }
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
        } else if kind == "multiple_choice" {
            input_map.insert(name, InputValue::Choice(text.trim().to_string()));
        } else if matches!(kind.as_str(), "number_list" | "number_vector" | "vector")
            || spec.fields.is_some()
        {
            let parsed = text.trim().parse::<f64>().unwrap_or(0.0);
            input_map.insert(name, InputValue::NumberList(vec![parsed]));
        } else {
            input_map.insert(name, InputValue::Text(text));
        }
    }

    let outputs = {
        let mut runner = state.runner.lock().unwrap();
        runner.run(&state.app, &input_map, &state.labels)
    };

    match outputs {
        Ok(outputs) => {
            Html(render_page(&state.app, Some(&input_map), Some(&outputs))).into_response()
        }
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

fn parse_number_list_field_name(name: &str) -> Option<(String, usize)> {
    let (id, idx) = name.rsplit_once("__")?;
    let idx = idx.parse::<usize>().ok()?;
    Some((id.to_string(), idx))
}

fn set_number_list_value(
    input_map: &mut HashMap<String, InputValue>,
    input_id: &str,
    idx: usize,
    value: f64,
) {
    let entry = input_map
        .entry(input_id.to_string())
        .or_insert_with(|| InputValue::NumberList(Vec::new()));
    if let InputValue::NumberList(values) = entry {
        if values.len() <= idx {
            values.resize(idx + 1, 0.0);
        }
        values[idx] = value;
    }
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
        if !is_visible_web(input.when.as_deref(), input.rules.as_deref(), inputs) {
            continue;
        }
        html.push_str("<div class=\"card\">");
        html.push_str(&format!(
            "<label for=\"{}\">{}</label>",
            input.id, input.label
        ));
        let kind = input.kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "number_list" | "number_vector" | "vector" => {
                let values = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v {
                        InputValue::NumberList(values) => Some(values.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        input
                            .fields
                            .as_ref()
                            .map(|fields| {
                                fields
                                    .iter()
                                    .map(|field| field.default.unwrap_or(0.0))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_else(|| vec![0.0])
                    });
                if let Some(fields) = &input.fields {
                    for (idx, field) in fields.iter().enumerate() {
                        let value = values.get(idx).copied().unwrap_or(0.0);
                        html.push_str(&format!(
                            "<label>{}</label><input type=\"number\" step=\"any\" name=\"{}__{}\" value=\"{}\" />",
                            field.label, input.id, idx, value
                        ));
                    }
                } else {
                    for (idx, value) in values.iter().enumerate() {
                        html.push_str(&format!(
                            "<label>Value {}</label><input type=\"number\" step=\"any\" name=\"{}__{}\" value=\"{}\" />",
                            idx + 1,
                            input.id,
                            idx,
                            value
                        ));
                    }
                }
            }
            "textarea" => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v {
                        InputValue::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .unwrap_or("");
                html.push_str(&format!(
                    "<textarea name=\"{}\" rows=\"4\">{}</textarea>",
                    input.id, value
                ));
            }
            "number" | "float" | "int" | "integer" => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v {
                        InputValue::Number(n) => Some(*n),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                html.push_str(&format!(
                    "<input type=\"number\" step=\"any\" name=\"{}\" value=\"{}\" />",
                    input.id, value
                ));
            }
            "bool" | "boolean" => {
                html.push_str(&format!(
                    "<input type=\"checkbox\" name=\"{}\" />",
                    input.id
                ));
            }
            "multiple_choice" => {
                html.push_str(&format!("<select name=\"{}\">", input.id));
                if let Some(choices) = &input.choices {
                    for choice in choices {
                        let value = choice.value.clone().unwrap_or_else(|| choice.id.clone());
                        html.push_str(&format!(
                            "<option value=\"{}\">{}</option>",
                            value, choice.label
                        ));
                    }
                }
                html.push_str("</select>");
            }
            "image" | "file" | "path" | "audio" | "video" => {
                html.push_str(&format!("<input type=\"file\" name=\"{}\" />", input.id));
                html.push_str(&format!(
                    "<input type=\"text\" name=\"{}\" placeholder=\"or paste a file path\" />",
                    input.id
                ));
            }
            _ => {
                let value = inputs
                    .and_then(|map| map.get(&input.id))
                    .and_then(|v| match v {
                        InputValue::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .unwrap_or("");
                html.push_str(&format!(
                    "<input type=\"text\" name=\"{}\" value=\"{}\" />",
                    input.id, value
                ));
            }
        }
        html.push_str("</div>");
    }

    html.push_str("<button class=\"btn\" type=\"submit\">Run Model</button>");
    html.push_str("</form>");

    if let Some(outputs) = outputs {
        html.push_str("<h2>Outputs</h2>");
        for output in &app.outputs {
            if !is_visible_web(output.when.as_deref(), output.rules.as_deref(), inputs) {
                continue;
            }
            html.push_str("<div class=\"card\">");
            html.push_str(&format!("<strong>{}</strong><br/>", output.label));
            if let Some(value) = outputs.get(&output.id) {
                match value {
                    OutputValue::Text(text) => html.push_str(&format!("<pre>{}</pre>", text)),
                    OutputValue::Number(num) => html.push_str(&format!("<p>{:.4}</p>", num)),
                    OutputValue::ImagePath(path) => html.push_str(&format!("<p>{}</p>", path)),
                    OutputValue::AudioPath(path) => {
                        html.push_str(&format!("<audio controls src=\"{}\"></audio>", path))
                    }
                    OutputValue::VideoPath(path) => html.push_str(&format!(
                        "<video controls style=\"max-width:100%;\" src=\"{}\"></video>",
                        path
                    )),
                    OutputValue::ClassScores(scores) => {
                        html.push_str("<table><tr><th>Label</th><th>Score</th></tr>");
                        for (label, score) in scores.iter().take(20) {
                            html.push_str(&format!(
                                "<tr><td>{}</td><td>{:.4}</td></tr>",
                                label, score
                            ));
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

fn is_visible_web(
    when: Option<&str>,
    rules: Option<&[ipxml_schema::RuleSpec]>,
    inputs: Option<&HashMap<String, InputValue>>,
) -> bool {
    if let Some(expr) = when {
        return eval_condition_web(expr, inputs);
    }
    if let Some(rules) = rules {
        for rule in rules {
            if eval_condition_web(&rule.if_expr, inputs) {
                return rule.then.as_ref().and_then(|a| a.visible).unwrap_or(true);
            }
            if let Some(otherwise) = &rule.otherwise {
                if let Some(visible) = otherwise.visible {
                    return visible;
                }
            }
        }
    }
    true
}

fn eval_condition_web(expr: &str, inputs: Option<&HashMap<String, InputValue>>) -> bool {
    let engine = Engine::new();
    let mut scope = Scope::new();
    if let Some(inputs) = inputs {
        for (id, value) in inputs {
            match value {
                InputValue::Bool(v) => {
                    scope.push(id.as_str(), *v);
                }
                InputValue::Number(v) => {
                    scope.push(id.as_str(), *v);
                }
                InputValue::Text(v)
                | InputValue::ImagePath(v)
                | InputValue::AudioPath(v)
                | InputValue::VideoPath(v)
                | InputValue::Choice(v) => {
                    scope.push(id.as_str(), v.clone());
                }
                InputValue::NumberList(v) => {
                    scope.push(id.as_str(), v.len() as i64);
                }
                InputValue::Choices(v) => {
                    scope.push(id.as_str(), v.len() as i64);
                }
            }
        }
    }
    engine
        .eval_expression_with_scope::<bool>(&mut scope, expr)
        .unwrap_or(false)
}
