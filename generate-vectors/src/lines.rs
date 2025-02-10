use std::io::BufRead;

use anyhow::Context;

use clap::Parser;

use crate::{
    model::EmbedderMetadata,
    util::{file_or_stdin_reader, file_or_stdout_writer},
};

#[derive(Parser)]
pub struct LinesCommand {
    /// path to the input file. if empty, reads from stdin
    #[arg(short, long)]
    input: Option<String>,

    /// path to the output file. if empty, writes to stdout
    #[arg(short, long)]
    output: Option<String>,
}

impl LinesCommand {
    pub async fn execute(
        &self,
        config: &EmbedderMetadata,
    ) -> Result<(), anyhow::Error> {
        let reader = file_or_stdin_reader(self.input.as_ref())
            .context("coult not open input file")?;

        let lines: Vec<String> = reader
            .lines()
            .try_collect()
            .context("could not read lines from input file")?;

        let writer = file_or_stdout_writer(self.output.as_ref())
            .context("could not create output file")?;
        config.embeddings_for_into(&lines, writer).await
    }
}
