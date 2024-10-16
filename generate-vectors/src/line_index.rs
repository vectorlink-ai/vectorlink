use anyhow::Context;
use byteorder::WriteBytesExt;
use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::Path,
};

use byteorder::LittleEndian;
use clap::Parser;
use std::os::unix::fs::FileExt;

use crate::model::EmbedderMetadata;

#[derive(Parser)]
pub struct LineIndexCommand {
    /// Original record file (Either CSV or JSON-Lines
    #[arg(short, long)]
    record_file: String,

    #[arg(short, long)]
    /// Where to put the index file
    index_file: String,
}

pub fn read_index_position(index_file: &File, record_id: usize) -> Result<u64, anyhow::Error> {
    let mut buf = [0u8; 8];
    let offset = record_id * std::mem::size_of::<u64>();
    index_file
        .read_exact_at(&mut buf, offset as u64)
        .context("Could not read at offset")?;
    Ok(u64::from_le_bytes(buf))
}

pub fn lookup_record(
    record_id: usize,
    record_file: &File,
    index_file: &File,
) -> Result<String, anyhow::Error> {
    // shift record id 1 for header.
    let record_id = record_id + 1;
    let start = read_index_position(index_file, record_id)?;
    let end = read_index_position(index_file, record_id + 1)?;

    let mut buf: Vec<u8> = vec![0; (end - start) as usize];
    record_file
        .read_exact_at(&mut buf, start)
        .context("Unable to read record with indices given by index file")?;
    let record_str = std::str::from_utf8(&buf)
        .map(|s| s.to_string())
        .context("Could not read as utf-8 string")?;
    Ok(record_str)
}

pub fn create_index_lines<P1: AsRef<Path>, P2: AsRef<Path>>(
    record_file_path: P1,
    index_file_path: P2,
) -> Result<(), anyhow::Error> {
    let record_file = File::open(record_file_path).context("Unable to open the record file")?;
    let mut record_file_cursor = BufReader::new(record_file);
    let mut index_file = File::create(index_file_path)
        .context("Unable to open or create the index file for write")?;

    let mut char_count = 0_u64;
    index_file.write_u64::<LittleEndian>(0).unwrap();
    loop {
        let position = record_file_cursor.skip_until(b'\n')?;
        if position == 0 {
            // EOF
            break;
        } else {
            char_count += position as u64;
            index_file.write_u64::<LittleEndian>(char_count).unwrap();
        }
    }
    Ok(())
}

impl LineIndexCommand {
    pub async fn execute(&self, _config: &EmbedderMetadata) -> Result<(), anyhow::Error> {
        create_index_lines(&self.record_file, &self.index_file)
    }
}
