use std::{
    fs::File,
    io::{self, stdin, stdout, BufReader, BufWriter, StdinLock, StdoutLock},
};

use either::Either;

// todo asref str
pub fn file_or_stdin_reader(
    path: Option<&String>,
) -> Result<Either<BufReader<File>, StdinLock>, io::Error> {
    Ok(match path {
        Some(path) => Either::Left(BufReader::new(File::open(path)?)),
        None => Either::Right(stdin().lock()),
    })
}

pub fn file_or_stdout_writer(
    path: Option<&String>,
) -> Result<Either<BufWriter<File>, StdoutLock>, io::Error> {
    Ok(match path {
        Some(path) => Either::Left(BufWriter::new(File::create(path)?)),
        None => Either::Right(stdout().lock()),
    })
}
