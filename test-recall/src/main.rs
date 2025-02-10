use std::io;

use clap::Parser;
use vectorlink_hnsw::{
    index::{Index, IndexConfiguration},
    params::SearchParams,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    vector_directory: String,
    #[arg(long)]
    hnsw_root_directory: String,
    #[arg(long)]
    name: String,
}
fn main() -> io::Result<()> {
    let args = Command::parse();
    let index = IndexConfiguration::load(
        &args.name,
        &args.hnsw_root_directory,
        &args.vector_directory,
    )?;
    let sp = SearchParams::default();
    let recall = index.test_recall(&sp, 0x533D);
    println!("{recall}");

    Ok(())
}
