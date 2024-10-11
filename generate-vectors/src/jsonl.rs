//! index json lines using a template dir

use std::{
    fs::{self, File},
    io::BufRead,
    path::Path,
};

use anyhow::Context;
use clap::Parser;

use handlebars::Handlebars;

use crate::{
    graph::{FullGraph, Graph},
    model::EmbedderMetadata,
    templates::{read_templates_from_dir, ID_FIELD_NAME},
    util::file_or_stdin_reader,
};

#[derive(Parser)]
pub struct JsonLinesCommand {
    /// path to the input file. if empty, reads from stdin
    #[arg(short, long)]
    input: Option<String>,

    /// path to output directory, where one file per column is written
    #[arg(short, long)]
    output_dir: String,

    /// path to a directory with templates
    #[arg(short, long)]
    template_dir: String,

    /// field to use as an id from each of the JSONs - this is never indexed
    #[arg(long)]
    id_field: String,

    /// Print N template field examples and exit
    #[arg(short, long)]
    print_templates: Option<usize>,
}

impl JsonLinesCommand {
    pub async fn execute(&self, config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        let template_dir = Path::new(&self.template_dir);
        let (template_names, templates) =
            read_templates_from_dir(template_dir).context("failed loading templates dir")?;

        let reader =
            file_or_stdin_reader(self.input.as_ref()).context("could not open input file")?;
        let buffered_reader = reader.lines();
        let mut string_vecs = vec![Vec::new(); template_names.len()];
        let mut templates_to_print = self.print_templates;
        for line in buffered_reader {
            let line = line.context("could not read line from input file")?;
            let json: serde_json::Value =
                serde_json::from_str(&line).context("could not parse json line")?;
            if templates_to_print.is_some() {
                eprintln!("\n--------------------")
            };
            for (field_index, template_name) in template_names.iter().enumerate() {
                if template_name == ID_FIELD_NAME {
                    let id = json
                        .get(&self.id_field)
                        .context("id field missing from json")?
                        .to_string();
                    string_vecs[field_index].push(id);
                } else {
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

        let dir_path = Path::new(&self.output_dir);

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

        let output_path = dir_path.join("disaggregated.graph");
        let writer = File::create(&output_path)
            .with_context(|| format!("could not create output file {output_path:?}"))?;
        let full_graph = FullGraph::new(fields);
        serde_json::to_writer(&writer, &full_graph)?;
        Ok(())
    }
}
