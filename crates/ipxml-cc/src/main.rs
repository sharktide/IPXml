use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ipxml_bundle::{create_bundle, BundleAsset, BundleModel};
use ipxml_schema::load_ipxml_from_str;

#[derive(Parser)]
#[command(name = "ipxml")]
#[command(about = "IPXml toolchain", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile an .ipxml app + ONNX model into a .ipxmodel.import bundle
    Cc {
        /// Path to .ipxml file
        #[arg(long)]
        ipxml: PathBuf,
        /// Optional path to ONNX model (single-model apps only)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Output bundle path
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Cc { ipxml, model, out } => {
            let ipxml_source = fs::read_to_string(&ipxml)?;
            let app = load_ipxml_from_str(&ipxml_source)?;
            let base_dir = ipxml.parent().unwrap_or_else(|| std::path::Path::new("."));
            let models = collect_models(&app, base_dir, model.as_ref())?;
            let assets = collect_assets(&app, base_dir)?;
            create_bundle(out, &app, &ipxml_source, &models, &assets)?;
            println!("Bundle created.");
        }
    }

    Ok(())
}

fn collect_assets(app: &ipxml_schema::IpxmlApp, base_dir: &std::path::Path) -> anyhow::Result<Vec<BundleAsset>> {
    let mut assets = Vec::new();
    for output in &app.outputs {
        if let Some(labels) = &output.labels {
            if let Some(path) = &labels.path {
                let full_path = base_dir.join(path);
                let bytes = fs::read(&full_path)?;
                assets.push(BundleAsset {
                    path: path.clone(),
                    bytes,
                });
            }
        }
    }
    Ok(assets)
}

fn collect_models(
    app: &ipxml_schema::IpxmlApp,
    base_dir: &std::path::Path,
    model_override: Option<&PathBuf>,
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

    let model_path = model_override
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| base_dir.join(&model.path));
    let bytes = fs::read(&model_path)?;
    Ok(vec![BundleModel {
        path: model.path.clone(),
        bytes,
    }])
}
