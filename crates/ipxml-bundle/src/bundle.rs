use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use ipxml_schema::IpxmlApp;
use thiserror::Error;
use zip::{write::FileOptions, ZipWriter};

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
}

#[derive(Debug, Clone)]
pub struct BundleAsset {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct BundleModel {
    pub path: String,
    pub bytes: Vec<u8>,
}

pub fn create_bundle<P: AsRef<Path>>(
    output: P,
    _app: &IpxmlApp,
    ipxml_source: &str,
    models: &[BundleModel],
    assets: &[BundleAsset],
) -> Result<(), BundleError> {
    let file = File::create(output)?;
    let mut zip = ZipWriter::new(file);

    let options = FileOptions::default();

    // schema
    zip.start_file("app.ipxml", options)?;
    zip.write_all(ipxml_source.as_bytes())?;

    // models
    for model in models {
        zip.start_file(&model.path, options)?;
        zip.write_all(&model.bytes)?;
    }

    for asset in assets {
        zip.start_file(&asset.path, options)?;
        zip.write_all(&asset.bytes)?;
    }

    zip.finish()?;
    Ok(())
}

pub fn read_ipxml_from_bundle<P: AsRef<Path>>(path: P) -> Result<String, BundleError> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut f = archive.by_name("app.ipxml")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

pub fn read_asset_from_bundle<P: AsRef<Path>>(
    path: P,
    asset_path: &str,
) -> Result<Vec<u8>, BundleError> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut f = archive.by_name(asset_path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub fn read_model_from_bundle<P: AsRef<Path>>(
    path: P,
    model_path: &str,
) -> Result<Vec<u8>, BundleError> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut f = archive.by_name(model_path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}
