use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ipxml_bundle::{create_bundle, BundleAsset};
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
        /// Path to ONNX model
        #[arg(long)]
        model: PathBuf,
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
            let onnx_bytes = fs::read(&model)?;
            let assets = collect_assets(&app, ipxml.parent().unwrap_or_else(|| std::path::Path::new(".")))?;
            create_bundle(out, &app, &ipxml_source, &onnx_bytes, &assets)?;
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
