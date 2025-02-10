use std::io;

use clap::Parser;
use vectorlink_hnsw::test_util::random_vectors_normalized;

#[derive(Parser, Debug)]
struct Command {
    vector_directory: String,
    domain: String,
    num_vecs: usize,

    #[arg(long, default_value_t = 1024)]
    vector_size: usize,
    #[arg(long, default_value_t = 0x533D)]
    seed: u64,
}
fn main() -> io::Result<()> {
    let args = Command::parse();
    let vectors = random_vectors_normalized(args.num_vecs, args.vector_size, args.seed);
    vectors.store(args.vector_directory, &args.domain)?;

    Ok(())
}
