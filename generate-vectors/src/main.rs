#![feature(iterator_try_collect)]
#![feature(bufread_skip_until)]

use std::fs::File;

use anyhow::Context;
use clap::Parser;
use compare::CompareCommand;
use csv_columns::CsvColumnsCommand;
use generate_matches::GenerateMatchesCommand;
use jsonl::JsonLinesCommand;
use line_index::LineIndexCommand;
use lines::LinesCommand;
use model::EmbedderMetadata;
use self_compare::SelfCompareCommand;
use self_weights::SelfWeightsCommand;
use weights::WeightsCommand;

mod compare;
mod csv_columns;
mod generate_matches;
mod graph;
mod jsonl;
mod line_index;
mod lines;
mod model;
mod openai;
mod self_compare;
mod self_weights;
mod templates;
mod train;
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
    /// Search for weights against the same index
    SelfFindWeights(SelfWeightsCommand),
    /// Generates a match set
    GenerateMatches(GenerateMatchesCommand),
    /// Line Index Record file (CSV or JSON-Lines)
    LineIndex(LineIndexCommand),
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
        Subcommand::SelfFindWeights(sfc) => sfc.execute(&config).await,
        Subcommand::GenerateMatches(gmc) => gmc.execute(&config).await,
        Subcommand::LineIndex(lic) => lic.execute(&config).await,
    }
}
