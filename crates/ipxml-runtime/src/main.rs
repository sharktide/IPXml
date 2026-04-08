use std::path::PathBuf;

use clap::{Parser, Subcommand};
use std::collections::HashMap;

use ipxml_bundle::{read_asset_from_bundle, read_ipxml_from_bundle, read_model_from_bundle};
use ipxml_schema::load_ipxml_from_str;
use ipxml_ui_core::{UiBackend, UiContext};
use ipxml_ui_egui::{EguiBackend, ModelEntry};

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
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { bundle } => {
            let ipxml_source = read_ipxml_from_bundle(&bundle)?;
            let app = load_ipxml_from_str(&ipxml_source)?;

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

            let labels = load_labels_from_bundle(&bundle, &app)?;
            let backend = EguiBackend::new(app, model_entries, labels);

            Box::new(backend).run(ctx);

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
