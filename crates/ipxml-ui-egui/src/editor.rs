use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use eframe::egui;
use ipxml_schema::{InputSpec, IpxmlApp, LayoutSpec, ModelSpec, OutputSpec, TensorSpec};
use rfd::FileDialog;

pub fn run_editor(initial_file: Option<PathBuf>) -> anyhow::Result<()> {
    let (app, current_path) = if let Some(path) = initial_file {
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read ipxml file '{}'", path.display()))?;
        let parsed = ipxml_schema::load_ipxml_from_str(&text)?;
        (parsed, Some(path))
    } else {
        (blank_app(), None)
    };

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "IPXml Editor",
        native_options,
        Box::new(|_cc| Box::new(EditorApp::new(app, current_path))),
    )
    .map_err(|err| anyhow::anyhow!("failed to launch editor: {err}"))
}

struct EditorApp {
    app: IpxmlApp,
    current_path: Option<PathBuf>,
    status: String,
}

impl EditorApp {
    fn new(app: IpxmlApp, current_path: Option<PathBuf>) -> Self {
        Self {
            app,
            current_path,
            status: "Ready".to_string(),
        }
    }

    fn save_to(&mut self, path: PathBuf) {
        match serde_yaml::to_string(&self.app) {
            Ok(yaml) => match fs::write(&path, yaml) {
                Ok(_) => {
                    self.current_path = Some(path.clone());
                    self.status = format!("Saved {}", path.display());
                }
                Err(err) => self.status = format!("Save failed: {err}"),
            },
            Err(err) => self.status = format!("YAML error: {err}"),
        }
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("New").clicked() {
                    self.app = blank_app();
                    self.current_path = None;
                    self.status = "Created new file".to_string();
                }
                if ui.button("Open").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("IPXml", &["ipxml"])
                        .pick_file()
                    {
                        match fs::read_to_string(&path) {
                            Ok(text) => match ipxml_schema::load_ipxml_from_str(&text) {
                                Ok(app) => {
                                    self.app = app;
                                    self.current_path = Some(path.clone());
                                    self.status = format!("Opened {}", path.display());
                                }
                                Err(err) => self.status = format!("Invalid IPXml: {err}"),
                            },
                            Err(err) => self.status = format!("Open failed: {err}"),
                        }
                    }
                }
                if ui.button("Save").clicked() {
                    if let Some(path) = self.current_path.clone() {
                        self.save_to(path);
                    } else if let Some(path) =
                        FileDialog::new().set_file_name("app.ipxml").save_file()
                    {
                        self.save_to(path);
                    }
                }
                if ui.button("Save As").clicked() {
                    if let Some(path) = FileDialog::new().set_file_name("app.ipxml").save_file() {
                        self.save_to(path);
                    }
                }
                ui.separator();
                ui.label(self.current_path.as_ref().map_or_else(
                    || "Unsaved file".to_string(),
                    |path| path.display().to_string(),
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("IPXml Graphical Editor");
            ui.label("Create and edit IPXml files without hand-writing YAML.");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("App Name");
                ui.text_edit_singleline(&mut self.app.name);
            });
            ui.horizontal(|ui| {
                ui.label("Version");
                let version = self.app.version.get_or_insert(String::new());
                ui.text_edit_singleline(version);
            });

            ui.separator();
            ui.label("Model");
            if self.app.model.is_none() {
                self.app.model = Some(ModelSpec {
                    id: None,
                    path: String::new(),
                    inputs: None,
                    when: None,
                    rules: None,
                });
            }
            if let Some(model) = &mut self.app.model {
                ui.horizontal(|ui| {
                    ui.label("Model Path");
                    ui.text_edit_singleline(&mut model.path);
                });
                ui.horizontal(|ui| {
                    ui.label("Model Id");
                    let id = model.id.get_or_insert(String::new());
                    ui.text_edit_singleline(id);
                });
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Inputs");
                if ui.button("+ Add input").clicked() {
                    self.app.inputs.push(default_input());
                }
            });
            for input in &mut self.app.inputs {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Id");
                        ui.text_edit_singleline(&mut input.id);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Label");
                        ui.text_edit_singleline(&mut input.label);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Type");
                        ui.text_edit_singleline(&mut input.kind);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Tensor Shape (comma-separated)");
                        let current = input
                            .tensor
                            .as_ref()
                            .and_then(|t| t.shape.as_ref())
                            .map(|shape| {
                                shape
                                    .iter()
                                    .map(|n| n.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .unwrap_or_default();
                        let mut shape_text = current;
                        if ui.text_edit_singleline(&mut shape_text).changed() {
                            let parsed = parse_shape(&shape_text);
                            if !shape_text.trim().is_empty() {
                                let tensor = input.tensor.get_or_insert_with(|| TensorSpec {
                                    shape: None,
                                    layout: None,
                                    normalize: None,
                                });
                                tensor.shape = parsed;
                            }
                        }
                    });
                });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Outputs");
                if ui.button("+ Add output").clicked() {
                    self.app.outputs.push(default_output());
                }
            });
            for output in &mut self.app.outputs {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Id");
                        ui.text_edit_singleline(&mut output.id);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Label");
                        ui.text_edit_singleline(&mut output.label);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Type");
                        ui.text_edit_singleline(&mut output.kind);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Source");
                        let source = output.source.get_or_insert(String::new());
                        ui.text_edit_singleline(source);
                    });
                });
                ui.add_space(6.0);
            }

            ui.separator();
            ui.label(format!("Status: {}", self.status));
        });
    }
}

fn blank_app() -> IpxmlApp {
    IpxmlApp {
        name: "New IPXml App".to_string(),
        version: Some("0.1.0".to_string()),
        model: Some(ModelSpec {
            id: None,
            path: String::new(),
            inputs: None,
            when: None,
            rules: None,
        }),
        models: None,
        inputs: vec![default_input()],
        outputs: vec![default_output()],
        layout: LayoutSpec { rows: Vec::new() },
    }
}

fn default_input() -> InputSpec {
    InputSpec {
        id: "input".to_string(),
        label: "Input".to_string(),
        kind: "number".to_string(),
        choices: None,
        media: None,
        fields: None,
        when: None,
        rules: None,
        tensor: Some(TensorSpec {
            shape: Some(vec![1]),
            layout: None,
            normalize: None,
        }),
        preprocess: None,
    }
}

fn default_output() -> OutputSpec {
    OutputSpec {
        id: "output".to_string(),
        label: "Output".to_string(),
        kind: "text".to_string(),
        media: None,
        when: None,
        rules: None,
        tensor: None,
        source: Some("output".to_string()),
        model: None,
        postprocess: None,
        labels: None,
        decode: None,
    }
}

fn parse_shape(text: &str) -> Option<Vec<usize>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let shape = trimmed
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    Some(shape)
}
