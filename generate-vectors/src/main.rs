#![feature(iterator_try_collect)]
use std::fs::File;

use anyhow::Context;
use clap::Parser;
use compare::CompareCommand;
use csv_columns::CsvColumnsCommand;
use jsonl::JsonLinesCommand;
use lines::LinesCommand;
use model::EmbedderMetadata;
use self_compare::SelfCompareCommand;
use weights::WeightsCommand;

mod compare;
mod csv_columns;
mod graph;
mod jsonl;
mod lines;
mod model;
mod openai;
mod self_compare;
mod templates;
mod util;
mod weights;

#[derive(Parser)]
struct Command {
    #[arg(short, long, global = true)]
    /// path to configuration file specifying how vectors are to be generated
    config: Option<String>,

    #[command(subcommand)]
    subcommand: Subcommand,
}

#[derive(Parser)]
enum Subcommand {
    /// Embed input file as lines
    Lines(LinesCommand),
    /// Embed input file as lines
    CsvColumns(CsvColumnsCommand),
    /// Embed json input files using a template dir
    JsonLines(JsonLinesCommand),
    /// Find record matches between two domains
    Compare(CompareCommand),
    /// Find record matches in a single domain
    SelfCompare(SelfCompareCommand),
    /// Search for weights
    FindWeights(WeightsCommand),
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let args = Command::parse();

    let config_path = args.config.context("no configuration file provided")?;
    let config: EmbedderMetadata = serde_json::from_reader(
        File::open(config_path).context("could not open configuration file")?,
    )
    .context("could not parse configuration file")?;

    match args.subcommand {
        Subcommand::Lines(lc) => lc.execute(&config).await,
        Subcommand::CsvColumns(csv) => csv.execute(&config).await,
        Subcommand::JsonLines(json) => json.execute(&config).await,
        Subcommand::Compare(cc) => cc.execute(&config).await,
        Subcommand::SelfCompare(sc) => sc.execute(&config).await,
        Subcommand::FindWeights(fc) => fc.execute(&config).await,
    }
}
