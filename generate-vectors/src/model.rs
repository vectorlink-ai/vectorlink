use std::{collections::HashMap, io::Write};

use anyhow::Context;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::openai::{OpenAIEmbedder, OpenAIModelType};

pub trait Model {
    async fn embeddings_for(
        &self,
        strings: &[String],
        output: &mut [u8],
    ) -> Result<(), anyhow::Error>;
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum SupportedModel {
    OpenaiAda2,
    OpenaiSmall3,
}

impl SupportedModel {
    pub const fn embedding_byte_size(self) -> usize {
        1536 * 4
    }
}

#[derive(Deserialize, Serialize)]
pub struct EmbedderMetadata {
    pub model: SupportedModel,
    #[serde(flatten)]
    extra_fields: HashMap<String, Value>,
}

impl EmbedderMetadata {
    pub async fn embeddings_for_into<W: Write>(
        &self,
        strings: &[String],
        mut writer: W,
    ) -> Result<(), anyhow::Error> {
        let required_capacity =
            self.model.embedding_byte_size() * strings.len();
        // TODO do not initialize
        let mut data = vec![0_u8; required_capacity];

        self.embeddings_for(strings, &mut data).await?;

        writer.write_all(&data[..]).context("could not write data")
    }
    pub fn openai_api_key(&self) -> Result<&str, anyhow::Error> {
        self.extra_fields
            .get("api_key")
            .context("api_key was not present in metadata")?
            .as_str()
            .context("api_key was not a string")
    }
    pub async fn embeddings_for(
        &self,
        strings: &[String],
        output: &mut [u8],
    ) -> Result<(), anyhow::Error> {
        // only openai supported so far, so we always need the api key.
        let api_key = self.openai_api_key()?.to_owned();
        let model = match self.model {
            SupportedModel::OpenaiAda2 => OpenAIModelType::Ada2,
            SupportedModel::OpenaiSmall3 => OpenAIModelType::Small3,
        };

        let embedder = OpenAIEmbedder::new(model, api_key);

        embedder.embeddings_for(strings, output).await
    }
}
