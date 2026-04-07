use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use eframe::egui;
use image::{DynamicImage, GenericImageView};
use ipxml_schema::{IpxmlApp, InputSpec, OutputSpec, TensorSpec};
use ipxml_ui_core::{
    find_input, find_output, input_value_for_spec, output_value_for_spec, InputValue, OutputValue,
    UiBackend, UiContext,
};
use ndarray::{Array3, ArrayD, Axis, IxDyn};
use ort::{
    session::Session,
    value::{Tensor, ValueType},
};
use rfd::FileDialog;

pub struct EguiBackend {
    app: IpxmlApp,
    model_bytes: Vec<u8>,
}

impl EguiBackend {
    pub fn new(app: IpxmlApp, model_bytes: Vec<u8>) -> Self {
        Self { app, model_bytes }
    }
}

impl UiBackend for EguiBackend {
    fn run(self: Box<Self>, _ctx: UiContext) {
        let app = self.app;
        let model_bytes = self.model_bytes;

        let title = app.name.clone();

        let native_options = eframe::NativeOptions::default();

        eframe::run_native(
            &title,
            native_options,
            Box::new(|_cc| Box::new(EguiApp::new(app, model_bytes))),
        )
        .unwrap();
    }
}

struct EguiApp {
    app: IpxmlApp,
    state: AppState,
    runner: Option<OnnxRunner>,
    style_applied: bool,
}

struct AppState {
    inputs: HashMap<String, InputValue>,
    outputs: HashMap<String, OutputValue>,
    status: String,
}

impl EguiApp {
    fn new(app: IpxmlApp, model_bytes: Vec<u8>) -> Self {
        let mut inputs = HashMap::new();
        for spec in &app.inputs {
            inputs.insert(spec.id.clone(), input_value_for_spec(spec));
        }

        let mut outputs = HashMap::new();
        for spec in &app.outputs {
            let mut value = output_value_for_spec(spec);
            set_output_status(&mut value, "Awaiting model output.");
            outputs.insert(spec.id.clone(), value);
        }

        let mut state = AppState {
            inputs,
            outputs,
            status: String::new(),
        };

        let runner = match OnnxRunner::new(model_bytes) {
            Ok(runner) => Some(runner),
            Err(err) => {
                state.status = format!("Failed to init ONNX runtime: {err}");
                None
            }
        };

        Self {
            app,
            state,
            runner,
            style_applied: false,
        }
    }
}

impl eframe::App for EguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.style_applied {
            apply_style(ctx);
            self.style_applied = true;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let (app, state) = (&self.app, &mut self.state);

            header(ui, app);

            ui.separator();

            if app.layout.rows.is_empty() {
                section_title(ui, "Inputs");
                for input in &app.inputs {
                    render_input(ui, state, input);
                    ui.add_space(6.0);
                }
                ui.add_space(8.0);
                section_title(ui, "Outputs");
                for output in &app.outputs {
                    render_output(ui, state, output);
                    ui.add_space(6.0);
                }
            } else {
                for (row_idx, row) in app.layout.rows.iter().enumerate() {
                    if row_idx > 0 {
                        ui.add_space(8.0);
                    }
                    ui.horizontal_wrapped(|ui| {
                        for id in &row.components {
                            render_component(ui, app, state, id);
                        }
                    });
                }
            }

            ui.separator();

            run_section(ui, app, state, &mut self.runner);

            if !state.status.is_empty() {
                ui.add_space(6.0);
                ui.colored_label(egui::Color32::from_rgb(180, 80, 40), &state.status);
            }
        });
    }
}

fn render_component(ui: &mut egui::Ui, app: &IpxmlApp, state: &mut AppState, id: &str) {
    if let Some(input) = find_input(app, id) {
        render_input(ui, state, input);
    } else if let Some(output) = find_output(app, id) {
        render_output(ui, state, output);
    } else {
        ui.colored_label(
            egui::Color32::YELLOW,
            format!("Unknown component id: {id}"),
        );
    }
}

fn render_input(ui: &mut egui::Ui, state: &mut AppState, spec: &InputSpec) {
    let value = state
        .inputs
        .entry(spec.id.clone())
        .or_insert_with(|| input_value_for_spec(spec));

    card(ui, |ui| match (spec.kind.trim().to_ascii_lowercase().as_str(), value) {
        ("text" | "string", InputValue::Text(current)) => {
            ui.label(&spec.label);
            ui.text_edit_singleline(current);
        }
        ("textarea", InputValue::Text(current)) => {
            ui.label(&spec.label);
            ui.text_edit_multiline(current);
        }
        ("number" | "float" | "int" | "integer", InputValue::Number(current)) => {
            ui.label(&spec.label);
            ui.add(egui::DragValue::new(current).speed(0.05));
        }
        ("bool" | "boolean", InputValue::Bool(current)) => {
            ui.checkbox(current, &spec.label);
        }
        ("image" | "file" | "path", InputValue::ImagePath(current)) => {
            ui.label(&spec.label);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add_sized(
                        [90.0, 32.0],
                        egui::Button::new("Upload").fill(accent_color()),
                    )
                    .clicked()
                {
                    if let Some(file) = FileDialog::new().pick_file() {
                        *current = file.display().to_string();
                    }
                }
                ui.text_edit_singleline(current);
            });
        }
        (_, InputValue::Text(current)) => {
            ui.label(format!("{} ({})", spec.label, spec.kind));
            ui.text_edit_singleline(current);
        }
        _ => {
            ui.label(format!("{} ({})", spec.label, spec.kind));
            ui.label("Unsupported input type.");
        }
    });
}

fn render_output(ui: &mut egui::Ui, state: &mut AppState, spec: &OutputSpec) {
    let value = state
        .outputs
        .entry(spec.id.clone())
        .or_insert_with(|| output_value_for_spec(spec));

    card(ui, |ui| match (spec.kind.trim().to_ascii_lowercase().as_str(), value) {
        ("text" | "label" | "string", OutputValue::Text(current)) => {
            ui.label(&spec.label);
            ui.label(current.as_str());
        }
        ("number" | "float" | "int" | "integer", OutputValue::Number(current)) => {
            ui.label(&spec.label);
            ui.label(format!("{current:.4}"));
        }
        ("image" | "file" | "path", OutputValue::ImagePath(current)) => {
            ui.label(&spec.label);
            ui.label(current.as_str());
        }
        (_, OutputValue::Text(current)) => {
            ui.label(format!("{} ({})", spec.label, spec.kind));
            ui.label(current.as_str());
        }
        _ => {
            ui.label(format!("{} ({})", spec.label, spec.kind));
            ui.label("Unsupported output type.");
        }
    });
}

fn run_section(
    ui: &mut egui::Ui,
    app: &IpxmlApp,
    state: &mut AppState,
    runner: &mut Option<OnnxRunner>,
) {
    ui.horizontal(|ui| {
        let run_button = egui::Button::new("Run Model")
            .fill(accent_color())
            .rounding(egui::Rounding::same(10.0));
        if ui.add_sized([140.0, 36.0], run_button).clicked() {
            if let Some(runner) = runner {
                match runner.run(app, &state.inputs) {
                    Ok(outputs) => {
                        state.outputs = outputs;
                        state.status.clear();
                    }
                    Err(err) => {
                        state.status = format!("Run failed: {err}");
                    }
                }
            } else {
                state.status = "ONNX runtime not available.".to_string();
            }
        }
        ui.label("Preprocessing and inference execute locally.");
    });
}

fn header(ui: &mut egui::Ui, app: &IpxmlApp) {
    ui.horizontal(|ui| {
        ui.heading(&app.name);
        if let Some(version) = &app.version {
            ui.label(format!("v{version}"));
        }
    });
    ui.label("IPXml runtime");
}

fn section_title(ui: &mut egui::Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(title)
            .size(16.0)
            .color(egui::Color32::from_rgb(60, 70, 90))
            .strong(),
    );
    ui.add_space(4.0);
}

fn card(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) {
    let frame = egui::Frame::none()
        .fill(egui::Color32::from_rgb(255, 255, 255))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(220, 226, 235)))
        .rounding(egui::Rounding::same(12.0))
        .inner_margin(egui::Margin::same(12.0));
    frame.show(ui, content);
}

fn accent_color() -> egui::Color32 {
    egui::Color32::from_rgb(64, 130, 255)
}

fn apply_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(12.0, 10.0);
    style.spacing.button_padding = egui::vec2(12.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(16.0);
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::proportional(28.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::proportional(15.0),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(15.0),
    );

    let mut visuals = egui::Visuals::light();
    visuals.window_rounding = egui::Rounding::same(12.0);
    visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
    visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
    visuals.widgets.active.rounding = egui::Rounding::same(8.0);
    visuals.panel_fill = egui::Color32::from_rgb(245, 247, 250);
    visuals.window_fill = egui::Color32::from_rgb(250, 252, 255);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(255, 255, 255);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(220, 226, 235));
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(238, 244, 255);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(230, 238, 255);
    visuals.selection.bg_fill = egui::Color32::from_rgb(207, 225, 255);
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 160, 255));

    style.visuals = visuals;
    ctx.set_style(style);
}

fn set_output_status(value: &mut OutputValue, status: &str) {
    match value {
        OutputValue::Text(current) => {
            current.clear();
            current.push_str(status);
        }
        OutputValue::Number(current) => {
            *current = 0.0;
        }
        OutputValue::ImagePath(current) => {
            current.clear();
        }
    }
}

struct OnnxRunner {
    session: Session,
    input_meta: Vec<InputMeta>,
    output_meta: Vec<OutputMeta>,
}

#[derive(Debug, Clone)]
struct InputMeta {
    name: String,
    shape: Vec<i64>,
}

#[derive(Debug, Clone)]
struct OutputMeta {
    name: String,
}

impl OnnxRunner {
    fn new(model_bytes: Vec<u8>) -> Result<Self> {
        let mut builder = Session::builder()?;
        let session = builder
            .commit_from_memory(&model_bytes)
            .context("load ONNX model")?;

        let input_meta = session
            .inputs()
            .iter()
            .map(|input| InputMeta {
                name: input.name().to_string(),
                shape: outlet_shape(input.dtype()),
            })
            .collect();

        let output_meta = session
            .outputs()
            .iter()
            .map(|output| OutputMeta {
                name: output.name().to_string(),
            })
            .collect();

        Ok(Self {
            session,
            input_meta,
            output_meta,
        })
    }

    fn run(
        &mut self,
        app: &IpxmlApp,
        inputs: &HashMap<String, InputValue>,
    ) -> Result<HashMap<String, OutputValue>> {
        let mut input_values: Vec<(String, Tensor<f32>)> = Vec::new();
        for input in &self.input_meta {
            let spec = find_input(app, &input.name)
                .ok_or_else(|| anyhow!("Missing input spec for {}", input.name))?;
            let value = inputs
                .get(&input.name)
                .ok_or_else(|| anyhow!("Missing input value for {}", input.name))?;
            let tensor = build_input_tensor(input, spec, value)?;
            input_values.push((input.name.clone(), tensor));
        }

        let outputs = self
            .session
            .run(input_values)
            .context("execute ONNX model")?;

        let mut result = HashMap::new();
        for (name, output) in outputs.iter() {
            let spec = find_output(app, name);
            let output_value = output_to_value(&output, spec)?;
            result.insert(name.to_string(), output_value);
        }

        for meta in &self.output_meta {
            if !result.contains_key(&meta.name) {
                let spec = find_output(app, &meta.name);
                let mut value = spec
                    .map(output_value_for_spec)
                    .unwrap_or(OutputValue::Text(String::new()));
                set_output_status(&mut value, "Missing output from model.");
                result.insert(meta.name.clone(), value);
            }
        }

        Ok(result)
    }
}

fn build_input_tensor(input: &InputMeta, spec: &InputSpec, value: &InputValue) -> Result<Tensor<f32>> {
    let shape = resolve_shape(spec.tensor.as_ref(), &input.shape, value)?;
    let array = match value {
        InputValue::Number(num) => scalar_to_array(*num as f32, &shape),
        InputValue::Bool(flag) => scalar_to_array(if *flag { 1.0 } else { 0.0 }, &shape),
        InputValue::Text(text) => {
            return Err(anyhow!(
                "Text input '{}' is not supported by the ONNX runner (value: {})",
                spec.id,
                text
            ));
        }
        InputValue::ImagePath(path) => image_to_array(path, spec, &shape)?,
    };

    Tensor::from_array(array).context("create input tensor")
}

fn resolve_shape(
    tensor: Option<&TensorSpec>,
    model_shape: &[i64],
    value: &InputValue,
) -> Result<Vec<usize>> {
    if let Some(shape) = tensor.and_then(|spec| spec.shape.clone()) {
        return Ok(shape);
    }

    let mut shape = Vec::new();
    for dim in model_shape {
        let resolved = if *dim > 0 {
            *dim as usize
        } else if matches!(value, InputValue::ImagePath(_)) {
            0
        } else {
            1
        };
        shape.push(resolved);
    }

    if shape.is_empty() {
        shape = match value {
            InputValue::ImagePath(_) => vec![1, 1, 1, 1],
            _ => vec![1],
        };
    }

    Ok(shape)
}

fn outlet_shape(dtype: &ValueType) -> Vec<i64> {
    match dtype {
        ValueType::Tensor { shape, .. } => shape.to_vec(),
        _ => Vec::new(),
    }
}

fn scalar_to_array(value: f32, shape: &[usize]) -> ArrayD<f32> {
    if shape.is_empty() {
        return ArrayD::from_elem(IxDyn(&[]), value);
    }
    ArrayD::from_elem(IxDyn(shape), value)
}

fn image_to_array(path: &str, spec: &InputSpec, shape: &[usize]) -> Result<ArrayD<f32>> {
    let image = image::open(path).with_context(|| format!("open image at {path}"))?;
    let (layout, channels, height, width) = resolve_image_layout(spec, shape, &image)?;
    let image = resize_if_needed(image, height, width);

    let array = match channels {
        1 => image_to_grayscale(&image),
        _ => image_to_rgb(&image),
    }?;

    let array = apply_normalize(spec.tensor.as_ref(), array)?;

    let array = if layout == "nhwc" {
        array.insert_axis(Axis(0)).into_dyn()
    } else {
        array
            .permuted_axes([2, 0, 1])
            .insert_axis(Axis(0))
            .into_dyn()
    };
    Ok(array)
}

fn resolve_image_layout(
    spec: &InputSpec,
    shape: &[usize],
    image: &DynamicImage,
) -> Result<(String, usize, u32, u32)> {
    let layout = spec
        .tensor
        .as_ref()
        .and_then(|tensor| tensor.layout.clone())
        .unwrap_or_else(|| infer_layout(shape).to_string());

    let (mut height, mut width, mut channels) = match shape.len() {
        4 if layout == "nhwc" => (shape[1], shape[2], shape[3]),
        4 => (shape[2], shape[3], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            let (w, h) = image.dimensions();
            (h as usize, w as usize, 3)
        }
    };

    if channels == 0 {
        channels = 3;
    }

    if height == 0 || width == 0 {
        let (w, h) = image.dimensions();
        height = h as usize;
        width = w as usize;
    }

    Ok((layout, channels, height as u32, width as u32))
}

fn infer_layout(shape: &[usize]) -> &'static str {
    if shape.len() == 4 {
        if shape[1] == 3 || shape[1] == 1 {
            "nchw"
        } else if shape[3] == 3 || shape[3] == 1 {
            "nhwc"
        } else {
            "nchw"
        }
    } else {
        "nchw"
    }
}

fn resize_if_needed(image: DynamicImage, height: u32, width: u32) -> DynamicImage {
    if height == 0 || width == 0 {
        return image;
    }
    let (w, h) = image.dimensions();
    if w == width && h == height {
        image
    } else {
        image.resize_exact(width, height, image::imageops::FilterType::Triangle)
    }
}

fn image_to_rgb(image: &DynamicImage) -> Result<Array3<f32>> {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let data: Vec<f32> = rgb.into_raw().into_iter().map(|v| v as f32).collect();
    Array3::from_shape_vec((h as usize, w as usize, 3), data)
        .context("convert rgb image to ndarray")
}

fn image_to_grayscale(image: &DynamicImage) -> Result<Array3<f32>> {
    let gray = image.to_luma8();
    let (w, h) = gray.dimensions();
    let data: Vec<f32> = gray.into_raw().into_iter().map(|v| v as f32).collect();
    Array3::from_shape_vec((h as usize, w as usize, 1), data)
        .context("convert grayscale image to ndarray")
}

fn apply_normalize(tensor: Option<&TensorSpec>, mut array: Array3<f32>) -> Result<Array3<f32>> {
    let Some(normalize) = tensor.and_then(|spec| spec.normalize.clone()) else {
        return Ok(array);
    };

    if let Some(scale) = normalize.scale {
        array.mapv_inplace(|v| v * scale);
    }

    let mean = normalize.mean.unwrap_or_default();
    let std = normalize.std.unwrap_or_default();
    let channels = array.shape()[2];

    for c in 0..channels {
        let mean_val = mean.get(c).copied().unwrap_or(0.0);
        let std_val = std.get(c).copied().unwrap_or(1.0);
        let mut slice = array.index_axis_mut(Axis(2), c);
        slice.mapv_inplace(|v| (v - mean_val) / std_val);
    }

    Ok(array)
}

fn output_to_value(output: &ort::value::ValueRef<'_>, spec: Option<&OutputSpec>) -> Result<OutputValue> {
    if let Ok(array) = output.try_extract_array::<f32>() {
        return tensor_to_output(array.to_owned(), spec);
    }
    if let Ok(array) = output.try_extract_array::<f64>() {
        let data = array.mapv(|v| v as f32).to_owned();
        return tensor_to_output(data.into_dyn(), spec);
    }
    if let Ok(array) = output.try_extract_array::<i64>() {
        let data = array.mapv(|v| v as f32).to_owned();
        return tensor_to_output(data.into_dyn(), spec);
    }

    let kind = spec.map(|s| s.kind.as_str()).unwrap_or("text");
    Ok(OutputValue::Text(format!(
        "Unsupported output value type ({kind})"
    )))
}

fn tensor_to_output(array: ArrayD<f32>, spec: Option<&OutputSpec>) -> Result<OutputValue> {
    let kind = spec.map(|s| s.kind.trim().to_ascii_lowercase());
    match kind.as_deref() {
        Some("number" | "float" | "int" | "integer") => {
            let value = array.iter().next().copied().unwrap_or(0.0);
            Ok(OutputValue::Number(value as f64))
        }
        _ => {
            let values: Vec<f32> = array.iter().copied().collect();
            let formatted = format_values(&values, 32);
            Ok(OutputValue::Text(format!(
                "Tensor shape {:?}: {}",
                array.shape(),
                formatted
            )))
        }
    }
}

fn format_values(values: &[f32], max_values: usize) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let take = values.len().min(max_values);
    let mut parts = Vec::with_capacity(take);
    for v in values.iter().take(take) {
        parts.push(format!("{v:.4}"));
    }
    if values.len() > max_values {
        parts.push(format!("... ({} total)", values.len()));
    }
    format!("[{}]", parts.join(", "))
}
