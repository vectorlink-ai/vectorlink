//! index all columns of a csv into files

use std::{fs::File, path::Path};

use anyhow::Context;
use clap::Parser;
use csv::StringRecord;

use crate::{
    graph::{FullGraph, Graph},
    model::EmbedderMetadata,
    util::file_or_stdin_reader,
};

#[derive(Parser)]
pub struct CsvColumnsCommand {
    /// path to the input file. if empty, reads from stdin
    #[arg(short, long)]
    input: Option<String>,

    /// path to output directory, where one file per column is written
    #[arg(short, long)]
    output_dir: String,

    /// field to use as an id from each of the CSVs - this is never indexed
    #[arg(long)]
    id_field: String,

    /// The columns to include. if none, all are included.
    columns: Vec<String>,

    /// column header. if not provided, first line is assumed to be the column header
    #[arg(long)]
    column_header: Option<Vec<String>>,
}

impl CsvColumnsCommand {
    pub async fn execute(&self, config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        let reader =
            file_or_stdin_reader(self.input.as_ref()).context("could not open input file")?;
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(reader);

        let headers: Vec<String>;
        if let Some(h) = self.column_header.clone() {
            headers = h;
        } else {
            let mut record = StringRecord::new();
            csv_reader.read_record(&mut record)?;
            headers = record.iter().map(|s| s.to_owned()).collect();
        }

        let enabled_headers: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(ix, h)| {
                if self.columns.contains(h) || *h == self.id_field {
                    Some(ix)
                } else {
                    None
                }
            })
            .collect();
        let missing_header_names: Vec<_> = self
            .columns
            .iter()
            .filter(|h| !headers.contains(h))
            .collect();

        if !missing_header_names.is_empty() {
            return Err(anyhow::anyhow!("missing headers: {missing_header_names:?}"));
        }

        let dir_path = Path::new(&self.output_dir);

        let mut string_vecs = vec![Vec::new(); enabled_headers.len()];
        for (ix, record) in csv_reader.into_records().enumerate() {
            let record = record.with_context(|| format!("could not parse record {ix}"))?;

            for (header_offset, enabled) in enabled_headers.iter().enumerate() {
                let val = record.get(*enabled).with_context(|| {
                    format!(
                        "could not retrieve column {enabled} ({}) from record",
                        headers[*enabled]
                    )
                })?;
                let current_header = &headers[*enabled];
                if *current_header == self.id_field {
                    string_vecs[header_offset].push(val.to_string())
                } else {
                    string_vecs[header_offset].push(format!("{current_header}: {val}"))
                }
            }
        }

        // csv has been split up into columns. call out
        std::fs::create_dir_all(dir_path).context("could not create output directory")?;
        let mut fields = Vec::new();
        for (ix, strings) in enabled_headers.into_iter().zip(string_vecs) {
            let current_header = &headers[ix];
            let graph = Graph::new(strings.iter().map(|s| s.as_str()));
            if *current_header != self.id_field {
                let output_path = dir_path.join(format!("{}.vecs", current_header));
                let writer = File::create(&output_path)
                    .with_context(|| format!("could not create output file {output_path:?}"))?;
                config.embeddings_for_into(&graph.values, writer).await?;
            }
            fields.push((current_header.clone(), graph));
        }

        let output_path = dir_path.join("csv.graph");
        let writer = File::create(&output_path)
            .with_context(|| format!("could not create output file {output_path:?}"))?;
        let full_graph = FullGraph::new(&self.id_field, fields);
        serde_json::to_writer(&writer, &full_graph)?;
        Ok(())
    }
}
