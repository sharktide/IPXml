use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use eframe::egui;
use image::{DynamicImage, GenericImageView};
use ipxml_schema::{DecodeSpec, IpxmlApp, InputSpec, ModelInputBinding, OutputSpec, TensorSpec};
use ipxcs_core::{apply_ops, argmax, softmax, topk_indices, topk_values, Tensor as CoreTensor};
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

pub struct ModelEntry {
    pub id: String,
    pub bytes: Vec<u8>,
    pub inputs: Option<Vec<ModelInputBinding>>,
}

pub struct EguiBackend {
    app: IpxmlApp,
    models: Vec<ModelEntry>,
    labels: HashMap<String, Vec<String>>,
}

impl EguiBackend {
    pub fn new(app: IpxmlApp, models: Vec<ModelEntry>, labels: HashMap<String, Vec<String>>) -> Self {
        Self {
            app,
            models,
            labels,
        }
    }
}

impl UiBackend for EguiBackend {
    fn run(self: Box<Self>, _ctx: UiContext) {
        let app = self.app;
        let models = self.models;
        let labels = self.labels;

        let title = app.name.clone();

        let native_options = eframe::NativeOptions::default();

        eframe::run_native(
            &title,
            native_options,
            Box::new(|_cc| Box::new(EguiApp::new(app, models, labels))),
        )
        .unwrap();
    }
}

struct EguiApp {
    app: IpxmlApp,
    state: AppState,
    runner: Option<PipelineRunner>,
    style_applied: bool,
    labels: HashMap<String, Vec<String>>,
}

struct AppState {
    inputs: HashMap<String, InputValue>,
    outputs: HashMap<String, OutputValue>,
    status: String,
}

impl EguiApp {
    fn new(app: IpxmlApp, models: Vec<ModelEntry>, labels: HashMap<String, Vec<String>>) -> Self {
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

        let runner = match PipelineRunner::new(models) {
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
            labels,
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

            run_section(ui, app, state, &mut self.runner, &self.labels);

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
        (_, OutputValue::ClassScores(scores)) => {
            ui.label(&spec.label);
            egui::Grid::new(format!("scores_{}", spec.id))
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Label");
                    ui.label("Score");
                    ui.end_row();
                    for (label, score) in scores.iter().take(20) {
                        ui.label(label);
                        ui.label(format!("{score:.4}"));
                        ui.end_row();
                    }
                });
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
    runner: &mut Option<PipelineRunner>,
    labels: &HashMap<String, Vec<String>>,
) {
    ui.horizontal(|ui| {
        let run_button = egui::Button::new("Run Model")
            .fill(accent_color())
            .rounding(egui::Rounding::same(10.0));
        if ui.add_sized([140.0, 36.0], run_button).clicked() {
            if let Some(runner) = runner {
                match runner.run(app, &state.inputs, labels) {
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
        OutputValue::ClassScores(current) => {
            current.clear();
        }
    }
}

struct PipelineRunner {
    models: Vec<ModelRunner>,
}

struct ModelRunner {
    id: String,
    session: Session,
    input_meta: HashMap<String, InputMeta>,
    input_bindings: Option<Vec<ModelInputBinding>>,
}

#[derive(Debug, Clone)]
struct InputMeta {
    name: String,
    shape: Vec<i64>,
}

impl PipelineRunner {
    fn new(models: Vec<ModelEntry>) -> Result<Self> {
        if models.is_empty() {
            return Err(anyhow!("No models provided"));
        }

        let mut runners = Vec::new();
        for entry in models {
            let mut builder = Session::builder()?;
            let session = builder
                .commit_from_memory(&entry.bytes)
                .context("load ONNX model")?;

            let input_meta = session
                .inputs()
                .iter()
                .map(|input| {
                    let meta = InputMeta {
                        name: input.name().to_string(),
                        shape: outlet_shape(input.dtype()),
                    };
                    (meta.name.clone(), meta)
                })
                .collect();

            runners.push(ModelRunner {
                id: entry.id,
                session,
                input_meta,
                input_bindings: entry.inputs,
            });
        }

        Ok(Self { models: runners })
    }

    fn run(
        &mut self,
        app: &IpxmlApp,
        inputs: &HashMap<String, InputValue>,
        labels: &HashMap<String, Vec<String>>,
    ) -> Result<HashMap<String, OutputValue>> {
        let mut model_outputs: HashMap<(String, String), CoreTensor> = HashMap::new();

        for model in &mut self.models {
            let input_values = build_model_inputs(model, app, inputs, &model_outputs)?;
            let outputs = model
                .session
                .run(input_values)
                .context("execute ONNX model")?;

            for (name, value) in outputs.iter() {
                let tensor = extract_output_tensor_ref(&value)?;
                model_outputs.insert((model.id.clone(), name.to_string()), tensor);
            }
        }

        let mut result = HashMap::new();
        for spec in &app.outputs {
            let model_id = resolve_output_model_id(spec, &self.models)?;
            let source = spec.source.as_deref().unwrap_or(&spec.id);
            if let Some(tensor) = model_outputs.get(&(model_id.clone(), source.to_string())) {
                let labels = labels.get(&spec.id);
                let output_value = process_output_tensor(tensor.clone(), spec, labels)?;
                result.insert(spec.id.clone(), output_value);
            } else {
                let mut value = output_value_for_spec(spec);
                set_output_status(&mut value, "Missing output from model.");
                result.insert(spec.id.clone(), value);
            }
        }

        Ok(result)
    }
}

fn build_model_inputs(
    model: &ModelRunner,
    app: &IpxmlApp,
    inputs: &HashMap<String, InputValue>,
    model_outputs: &HashMap<(String, String), CoreTensor>,
) -> Result<Vec<(String, Tensor<f32>)>> {
    let bindings: Vec<(String, String)> = if let Some(bindings) = &model.input_bindings {
        bindings
            .iter()
            .map(|binding| (binding.name.clone(), binding.source.clone()))
            .collect()
    } else {
        model
            .input_meta
            .values()
            .map(|meta| (meta.name.clone(), meta.name.clone()))
            .collect()
    };

    let mut input_values: Vec<(String, Tensor<f32>)> = Vec::new();
    for (input_name, source) in bindings {
        if let Some((model_id, output_name)) = parse_model_output_source(&source) {
            let tensor = model_outputs
                .get(&(model_id.clone(), output_name.clone()))
                .ok_or_else(|| {
                    anyhow!(
                        "Missing model output '{}:{}' for input '{}'",
                        model_id,
                        output_name,
                        input_name
                    )
                })?;
            let ort_tensor =
                Tensor::from_array(tensor.clone().into_data()).context("create chained input")?;
            input_values.push((input_name, ort_tensor));
        } else {
            let spec = find_input(app, &source)
                .ok_or_else(|| anyhow!("Missing input spec for {}", source))?;
            let value = inputs
                .get(&source)
                .ok_or_else(|| anyhow!("Missing input value for {}", source))?;
            let meta = model
                .input_meta
                .get(&input_name)
                .ok_or_else(|| anyhow!("Missing model input '{}'", input_name))?;
            let tensor = build_input_tensor(meta, spec, value)?;
            input_values.push((input_name, tensor));
        }
    }

    Ok(input_values)
}

fn parse_model_output_source(source: &str) -> Option<(String, String)> {
    let (model_id, output_name) = source.split_once(':')?;
    if model_id.is_empty() || output_name.is_empty() {
        None
    } else {
        Some((model_id.to_string(), output_name.to_string()))
    }
}

fn resolve_output_model_id(spec: &OutputSpec, models: &[ModelRunner]) -> Result<String> {
    if let Some(model_id) = &spec.model {
        if models.iter().any(|m| &m.id == model_id) {
            return Ok(model_id.clone());
        }
        return Err(anyhow!("Unknown model id '{}' for output '{}'", model_id, spec.id));
    }
    if models.len() == 1 {
        Ok(models[0].id.clone())
    } else {
        Err(anyhow!(
            "Output '{}' must specify a model when multiple models are present",
            spec.id
        ))
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

    let mut tensor = CoreTensor::new(array);
    if let Some(ops) = &spec.preprocess {
        tensor = apply_ops(tensor, ops)?;
    }
    Tensor::from_array(tensor.into_data()).context("create input tensor")
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

fn process_output_tensor(
    mut tensor: CoreTensor,
    spec: &OutputSpec,
    labels: Option<&Vec<String>>,
) -> Result<OutputValue> {
    if let Some(ops) = &spec.postprocess {
        tensor = apply_ops(tensor, ops)?;
    }

    let decoded = decode_output(&tensor, spec.decode.as_ref())?;
    Ok(map_decoded_output(decoded, spec, labels))
}

#[derive(Debug)]
enum DecodedOutput {
    Tensor(CoreTensor),
    Index(usize),
    Scores(Vec<(usize, f32)>),
}

fn extract_output_tensor_ref(output: &ort::value::ValueRef<'_>) -> Result<CoreTensor> {
    if let Ok(array) = output.try_extract_array::<f32>() {
        return Ok(CoreTensor::new(array.to_owned()));
    }
    if let Ok(array) = output.try_extract_array::<f64>() {
        let data = array.mapv(|v| v as f32).to_owned();
        return Ok(CoreTensor::new(data.into_dyn()));
    }
    if let Ok(array) = output.try_extract_array::<i64>() {
        let data = array.mapv(|v| v as f32).to_owned();
        return Ok(CoreTensor::new(data.into_dyn()));
    }

    Err(anyhow!("Unsupported output value type"))
}

fn decode_output(tensor: &CoreTensor, decode: Option<&DecodeSpec>) -> Result<DecodedOutput> {
    match decode.unwrap_or(&DecodeSpec::Identity) {
        DecodeSpec::Identity => Ok(DecodedOutput::Tensor(tensor.clone())),
        DecodeSpec::Softmax { axis } => {
            let result = softmax(tensor, *axis)?;
            Ok(DecodedOutput::Tensor(result))
        }
        DecodeSpec::ArgMax { axis } => {
            let result = argmax(tensor, *axis)?;
            if result.data.len() == 1 {
                let idx = result.data.iter().next().copied().unwrap_or(0.0) as usize;
                Ok(DecodedOutput::Index(idx))
            } else {
                Ok(DecodedOutput::Tensor(result))
            }
        }
        DecodeSpec::TopK { k, axis } => {
            if let Ok(scores) = topk_indices(tensor, *k, *axis, true) {
                Ok(DecodedOutput::Scores(scores))
            } else {
                let values = topk_values(tensor, *k, *axis, true)?;
                Ok(DecodedOutput::Tensor(values))
            }
        }
    }
}

fn map_decoded_output(
    decoded: DecodedOutput,
    spec: &OutputSpec,
    labels: Option<&Vec<String>>,
) -> OutputValue {
    match decoded {
        DecodedOutput::Index(index) => {
            if let Some(labels) = labels {
                let label = labels.get(index).cloned().unwrap_or_else(|| format!("class_{index}"));
                OutputValue::ClassScores(vec![(label, 1.0)])
            } else {
                OutputValue::Text(format!("Index: {index}"))
            }
        }
        DecodedOutput::Scores(scores) => {
            if let Some(labels) = labels {
                let mapped = scores
                    .into_iter()
                    .map(|(idx, score)| {
                        let label = labels.get(idx).cloned().unwrap_or_else(|| format!("class_{idx}"));
                        (label, score)
                    })
                    .collect();
                OutputValue::ClassScores(mapped)
            } else {
                let formatted = scores
                    .into_iter()
                    .map(|(idx, score)| format!("{idx}: {score:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                OutputValue::Text(format!("TopK: [{formatted}]"))
            }
        }
        DecodedOutput::Tensor(tensor) => tensor_to_output(&tensor, spec, labels),
    }
}

fn tensor_to_output(tensor: &CoreTensor, spec: &OutputSpec, labels: Option<&Vec<String>>) -> OutputValue {
    let kind = spec.kind.trim().to_ascii_lowercase();
    if matches!(kind.as_str(), "number" | "float" | "int" | "integer") && tensor.data.len() == 1 {
        let value = tensor.data.iter().next().copied().unwrap_or(0.0);
        return OutputValue::Number(value as f64);
    }

    if let Some(labels) = labels {
        if let Some(scores) = tensor_to_scores(tensor, labels) {
            return OutputValue::ClassScores(scores);
        }
    }

    let values: Vec<f32> = tensor.data.iter().copied().collect();
    let formatted = format_values(&values, 32);
    OutputValue::Text(format!(
        "Tensor shape {:?}: {}",
        tensor.data.shape(),
        formatted
    ))
}

fn tensor_to_scores(tensor: &CoreTensor, labels: &[String]) -> Option<Vec<(String, f32)>> {
    let shape = tensor.data.shape();
    let values: Vec<f32> = tensor.data.iter().copied().collect();
    if values.len() != labels.len() && !(shape.len() == 2 && shape[0] == 1 && shape[1] == labels.len()) {
        return None;
    }

    let mut pairs = labels
        .iter()
        .cloned()
        .zip(values.into_iter())
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Some(pairs)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ipxml_schema::OpSpec;

    fn base_output_spec(kind: &str) -> OutputSpec {
        OutputSpec {
            id: "out".to_string(),
            label: "Output".to_string(),
            kind: kind.to_string(),
            tensor: None,
            source: None,
            model: None,
            postprocess: None,
            labels: None,
            decode: None,
        }
    }

    #[test]
    fn decode_topk_with_labels() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.1, 0.7, 0.2, 0.5]).unwrap();
        let tensor = CoreTensor::new(data);
        let decoded = decode_output(
            &tensor,
            Some(&DecodeSpec::TopK {
                k: 2,
                axis: Some(1),
            }),
        )
        .unwrap();
        let labels = vec![
            "zero".to_string(),
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
        ];
        let spec = base_output_spec("scores");
        let output = map_decoded_output(decoded, &spec, Some(&labels));
        match output {
            OutputValue::ClassScores(scores) => {
                assert_eq!(scores.len(), 2);
                assert_eq!(scores[0].0, "one");
            }
            _ => panic!("expected class scores"),
        }
    }

    #[test]
    fn postprocess_pipeline_runs() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap();
        let tensor = CoreTensor::new(data);
        let ops = vec![
            OpSpec::Scale { factor: 2.0 },
            OpSpec::Add {
                value: Some(1.0),
                tensor: None,
            },
        ];
        let processed = apply_ops(tensor, &ops).unwrap();
        let spec = base_output_spec("text");
        let output = tensor_to_output(&processed, &spec, None);
        match output {
            OutputValue::Text(text) => {
                assert!(text.contains("Tensor shape"));
            }
            _ => panic!("expected text output"),
        }
    }
}
