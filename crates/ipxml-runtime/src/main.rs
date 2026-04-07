use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ipxml_bundle::{read_ipxml_from_bundle, read_model_from_bundle};
use ipxml_schema::load_ipxml_from_str;
use ipxml_ui_core::{UiBackend, UiContext};
use ipxml_ui_egui::EguiBackend;

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

            let model_bytes = read_model_from_bundle(&bundle, &app.model.path)?;
            let backend = EguiBackend::new(app, model_bytes);

            Box::new(backend).run(ctx);

        }
    }

    Ok(())
}
