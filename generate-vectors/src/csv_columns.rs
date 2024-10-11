//! index all columns of a csv into files

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::Context;
use clap::Parser;
use csv::StringRecord;
use serde::{de::value::MapDeserializer, Deserialize};

use crate::{
    graph::{FullGraph, Graph},
    model::EmbedderMetadata,
    templates::{read_templates_from_dir, ID_FIELD_NAME},
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

    /// path to a directory with templates
    #[arg(short, long)]
    template_dir: String,

    /// column header. if not provided, first line is assumed to be the column header
    #[arg(long)]
    column_header: Option<Vec<String>>,

    /// Print N template field examples and exit
    #[arg(short, long)]
    print_templates: Option<usize>,
}

impl CsvColumnsCommand {
    pub async fn execute(&self, config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        let template_dir = Path::new(&self.template_dir);
        let (template_names, templates) =
            read_templates_from_dir(template_dir).context("failed loading templates dir")?;
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

        let dir_path = Path::new(&self.output_dir);

        let mut string_vecs = vec![Vec::new(); template_names.len()];
        let id_field_idx = headers
            .iter()
            .position(|x| *x == self.id_field)
            .context("I field is not in header")?;
        let mut templates_to_print = self.print_templates;
        for (ix, record) in csv_reader.into_records().enumerate() {
            let record = record.with_context(|| format!("could not parse record {ix}"))?;
            if templates_to_print.is_some() {
                eprintln!("\n--------------------")
            };
            for (field_index, template_name) in template_names.iter().enumerate() {
                if template_name == ID_FIELD_NAME {
                    let id = record[id_field_idx].to_string();
                    string_vecs[field_index].push(id);
                } else {
                    let json: serde_json::Value = serde_json::Value::deserialize(
                        MapDeserializer::<_, serde_json::Error>::new(
                            headers
                                .iter()
                                .map(|s| s.to_string())
                                .zip(record.into_iter()),
                        ),
                    )?;
                    let result = templates
                        .render(template_name, &json)
                        .context("could not render handlebars")?;
                    if templates_to_print.is_some() {
                        eprintln!("{result}");
                    }
                    string_vecs[field_index].push(result);
                }
            }
            // Decrement template number
            if let Some(count) = templates_to_print {
                if count == 0 {
                    std::process::exit(0);
                } else {
                    templates_to_print = Some(count - 1);
                }
            }
        }

        // csv has been split up into columns. call out
        std::fs::create_dir_all(dir_path).context("could not create output directory")?;
        let mut fields = Vec::new();
        for (template_name, strings) in template_names.into_iter().zip(string_vecs) {
            let graph = Graph::new(strings.iter().map(|s| s.as_str()));
            if template_name != ID_FIELD_NAME {
                let output_path = dir_path.join(format!("{}.vecs", template_name));
                let writer = File::create(&output_path)
                    .with_context(|| format!("could not create output file {output_path:?}"))?;
                config.embeddings_for_into(&graph.values, writer).await?;
            }
            fields.push((template_name, graph));
        }

        let output_path = dir_path.join("aggregated.graph");
        let writer = File::create(&output_path)
            .with_context(|| format!("could not create output file {output_path:?}"))?;
        let full_graph = FullGraph::new(fields);
        serde_json::to_writer(&writer, &full_graph)?;
        Ok(())
    }
}
