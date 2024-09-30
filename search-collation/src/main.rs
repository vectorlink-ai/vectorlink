use byteorder::{BigEndian, NativeEndian, ReadBytesExt};
use clap::Parser;
use csv::Writer;
use std::io::{prelude::*, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::{fs::File, io};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Command {
    #[arg(long)]
    record_file: String,
    #[arg(long)]
    record_index_file: String,
    #[arg(long)]
    search_file: String,
    #[arg(long)]
    output_table: String,
    #[arg(long)]
    fields: Option<String>,
    #[arg(long)]
    match_length: usize,
    #[arg(long, default_value_t=f32::MAX)]
    threshold: f32,
}

fn get_record_position(record_number: usize, index_file: &mut File) -> (u64, u64) {
    let record_offset = record_number as u64 * std::mem::size_of::<u64>() as u64;
    index_file.seek(SeekFrom::Start(record_offset)).unwrap();
    let record_offset_start = index_file.read_u64::<NativeEndian>().unwrap();
    let record_offset_end = index_file.read_u64::<NativeEndian>().unwrap();
    (record_offset_start, record_offset_end)
}

fn get_record(
    record_number: usize,
    record_file: &mut File,
    index_file: &mut File,
) -> serde_json::Value {
    let (start, end) = get_record_position(record_number, index_file);
    record_file.seek(SeekFrom::Start(start)).unwrap();
    let mut buf: Vec<u8> = vec![0; (end-start) as usize];
    record_file.read_exact(&mut buf).unwrap();
    let json_string = std::str::from_utf8(&buf).expect("Not a valid json string");
    serde_json::from_str(&json_string).unwrap()
}

fn main() -> io::Result<()> {
    let args = Command::parse();
    let mut record_file = File::open(args.record_file).expect("Can't open ecord file");
    let mut index_file = File::open(args.record_index_file).expect("Can't open index file");
    let mut search_file = File::open(args.search_file).expect("Can't open search file");
    let file_size = search_file.metadata()?.size() as usize;
    let records_size = args.match_length * std::mem::size_of::<(u32, f32)>();
    assert_eq!(file_size % records_size, 0, "Unexpected record length");
    let num_records = file_size / records_size;

    let mut output_table =
        Writer::from_path(args.output_table).expect("could not open output table");
    output_table.write_record(&[
        "ROOT_DATAFILE_ID",
        "ROOT_ROW_ID",
        "DATAFILE_ID",
        "ROW_ID",
        "DISTANCE",
    ]).unwrap();

    for i in 0..num_records {
        let central_record = get_record(i, &mut record_file, &mut index_file);
        let root_file_id = central_record.get("DATAFILE_ID").unwrap().as_str().unwrap();
        let root_record_id = central_record.get("ROW_ID").unwrap().as_str().unwrap();
        for _j in 0..args.match_length {
            let neighbor_id = search_file.read_u32::<NativeEndian>().unwrap();
	    if neighbor_id == u32::MAX {
		break;
	    }
            let neighbor_distance = search_file.read_f32::<NativeEndian>().unwrap();
	    if neighbor_distance < args.threshold {
		let neighbor_record =
                    get_record(neighbor_id as usize, &mut record_file, &mut index_file);
		let file_id = neighbor_record.get("DATAFILE_ID").unwrap().as_str().unwrap();
		let record_id = neighbor_record.get("ROW_ID").unwrap().as_str().unwrap();
		let distance = format!("{neighbor_distance}");
		output_table
                    .write_record([root_file_id, root_record_id, file_id, record_id, &distance])
                    .expect("Could not write csv record");
	    }
        }
    }

    Ok(())
}
