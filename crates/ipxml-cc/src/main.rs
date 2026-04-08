use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ipxml_bundle::{BundleAsset, BundleModel, create_bundle};
use ipxml_schema::load_ipxml_from_str;

#[derive(Parser)]
#[command(name = "ipxml-cc")]
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
        /// Optional output bundle path
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Cc { ipxml, out } => {
            eprintln!("ipxml-cc is kept for compatibility. Prefer: ipxml cc --ipxml <file>");
            let ipxml_source = fs::read_to_string(&ipxml)?;
            let app = load_ipxml_from_str(&ipxml_source)?;
            let base_dir = ipxml.parent().unwrap_or_else(|| std::path::Path::new("."));
            let models = collect_models(&app, base_dir)?;
            let assets = collect_assets(&app, base_dir)?;
            let out_path = out.unwrap_or_else(|| default_out_path(&ipxml, &app));
            create_bundle(&out_path, &app, &ipxml_source, &models, &assets)?;
            println!("Bundle created.");
        }
    }

    Ok(())
}

fn collect_assets(
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

fn collect_models(
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

    let model_path = base_dir.join(&model.path);
    let bytes = fs::read(&model_path)?;
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
