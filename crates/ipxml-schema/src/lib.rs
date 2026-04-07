mod model;

pub use model::*;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("YAML parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
}

pub fn load_ipxml_from_str(s: &str) -> Result<IpxmlApp, SchemaError> {
    let app: IpxmlApp = serde_yaml::from_str(s)?;
    Ok(app)
}
