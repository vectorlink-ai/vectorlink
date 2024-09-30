use std::{io, time::SystemTime};

use clap::Parser;
use hnsw_redux::vectors::Vectors;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    vector_directory: String,
    #[arg(long)]
    name: String,
}
fn main() -> io::Result<()> {
    let args = Command::parse();
    let time = SystemTime::now();
    let _vectors = Vectors::load(&args.vector_directory, &args.name)?;
    eprintln!("{}", time.elapsed().unwrap().as_secs());
    Ok(())
}
