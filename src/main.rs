use std::fs::{File, create_dir_all, read_dir};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{self};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use clap::{ArgAction, Parser, ValueEnum};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use polars::prelude::*;
use rayon::{ThreadPoolBuilder, prelude::*};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Arrow,
    Parquet,
}

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// The path to the input file
    #[arg(long, conflicts_with = "input_dir")]
    input: Option<PathBuf>,

    /// The path to a directory with input files
    #[arg(long, conflicts_with = "input")]
    input_dir: Option<PathBuf>,

    /// File format
    #[arg(long)]
    #[clap(value_enum, default_value_t = Format::Parquet)]
    format: Format,

    /// The path to the output files
    #[arg(long)]
    output: PathBuf,

    /// Number of threads to use for processing
    #[arg(long, default_value_t = 3)]
    threads: usize,

    /// CSV file where transcriptions should be written
    #[arg(long, action = ArgAction::Set)]
    metadata_file: Option<PathBuf>,
}

fn arrow_to_parquet(filename: &Path) -> Result<DataFrame> {
    let file = File::open(filename)
        .with_context(|| format!("Failed to open arrow file: {}", filename.display()))?;
    let reader =
        StreamReader::try_new(file, None).context("Failed to create arrow stream reader")?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<_, _>>()
        .context("Failed to collect record batches from arrow file")?;
    let df = batches_to_parquet(&batches)
        .context("Failed to convert arrow batches to parquet for DataFrame")?;

    Ok(df)
}

fn batches_to_parquet(batches: &[RecordBatch]) -> Result<DataFrame> {
    // In-memory buffer to avoid writing to a temporary file on disk
    let tmp_file = tempfile::tempfile()?;

    // Write the batches to the file
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(tmp_file, batches[0].schema(), Some(props))?;

    for batch in batches {
        writer.write(batch)?;
    } // writer goes out of scope and finishes writing

    let tmp_file = writer.into_inner()?;

    // Read in parquet file and unnest the audio column
    let df = ParquetReader::new(tmp_file)
        .with_columns(Some(vec!["audio".to_string(), "transcription".to_string()]))
        .finish()?
        .unnest(["audio"])?;

    Ok(df)
}

fn read_parquet(filename: &Path) -> Result<DataFrame> {
    let file = File::open(filename)
        .with_context(|| format!("Failed to open parquet file: {}", filename.display()))?;

    let df = ParquetReader::new(file)
        .with_columns(Some(vec!["audio".to_string(), "transcription".to_string()]))
        .finish()
        .context("Failed to read parquet file into DataFrame")?
        .unnest(["audio"])?;

    Ok(df)
}

fn write_file(filename: &Path, data: &[u8]) -> Result<PathBuf> {
    // Choose a new filename with timestamp prefix if the target already exists.
    let mut target = filename.to_path_buf();
    if filename.try_exists()? {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("System time is before UNIX_EPOCH")?
            .as_micros();
        let counter = UNIQUE_FILENAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let original_name = filename
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_default();
        let timestamped_name = format!("{timestamp}-{counter:04}-{original_name}");
        target = if let Some(parent) = filename.parent() {
            parent.join(&timestamped_name)
        } else {
            PathBuf::from(&timestamped_name)
        };
    }

    let mut file = File::create(&target)?;
    file.write_all(data)?;

    Ok(target)
}

fn process_file(
    filename: &Path,
    format: Format,
    output_dir: &Path,
    metadata_records: &Mutex<Vec<(String, String)>>,
) -> Result<usize> {
    // Convert the file to a DataFrame
    let df = match format {
        Format::Arrow => arrow_to_parquet(filename)
            .with_context(|| format!("Error processing arrow file {}", filename.display()))?,
        Format::Parquet => read_parquet(filename)
            .with_context(|| format!("Error processing parquet file {}", filename.display()))?,
    };

    // Extract the series from the DataFrame
    let path_series = df.column("path")?.str()?;
    let array_series = df.column("bytes")?.binary()?;
    let transcription_series = df.column("transcription")?.str()?;

    let num_rows = df.height();

    let records: Vec<_> = (0..num_rows)
        .into_par_iter()
        .filter_map(|i| {
            if let (Some(path_val), Some(transcription), Some(array_series_inner)) = (
                path_series.get(i),
                transcription_series.get(i),
                array_series.get(i),
            ) {
                Some((path_val, transcription, array_series_inner))
            } else {
                None
            }
        })
        .collect();

    let local_metadata: Vec<(String, String)> = records
        .par_iter()
        .map(|(path_val, transcription, array_series_inner)| {
            let original_path = Path::new(path_val);
            let file_stem = original_path.file_stem().unwrap_or_default();
            let extension = original_path.extension().unwrap_or_default();

            let audio_filename_str = format!(
                "{}.{}",
                file_stem.to_string_lossy(),
                extension.to_string_lossy()
            );
            let audio_filename = output_dir.join(&audio_filename_str);
            let audio_data: &[u8] = array_series_inner;
            let written_path =
                write_file(&audio_filename, audio_data).expect("Failed to write audio file");
            let final_name = written_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| written_path.to_string_lossy().into_owned());

            (final_name, transcription.to_string())
        })
        .collect();

    metadata_records.lock().unwrap().extend(local_metadata);

    Ok(num_rows)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Configure the global thread pool for Rayon
    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;

    if !args.input.is_some() && !args.input_dir.is_some() {
        eprintln!("Either --input or --input-dir must be provided.");
        process::exit(1);
    }

    // Create the output folder if it doesn't exist
    create_dir_all(&args.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output.display()
        )
    })?;

    let metadata_records = Mutex::new(Vec::new());

    if let Some(input_file) = args.input {
        if !input_file.is_file() {
            eprintln!("Input is not a file: {}", input_file.display());
            process::exit(1);
        }
        println!("Processing file: {}...", input_file.display());
        let rows = process_file(&input_file, args.format, &args.output, &metadata_records)?;
        println!("Total number of rows processed: {}", rows);
    }

    if let Some(input_dir) = args.input_dir {
        if !input_dir.is_dir() {
            eprintln!(
                "Input directory does not exist or is not a directory: {}",
                input_dir.display()
            );
            process::exit(1);
        }

        let files_to_process: Vec<_> = read_dir(input_dir)?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.path().is_file()
                    && entry // TODO: this is not correct, should be based on format
                        .path()
                        .extension()
                        .is_some_and(|ext| ext == "parquet" || ext == "arrow")
            })
            .collect();

        let total_rows = AtomicUsize::new(0);

        files_to_process.into_iter().for_each(|entry| {
            let path = entry.path();
            println!("Processing file: {}...", path.display());
            match process_file(&path, args.format, &args.output, &metadata_records) {
                Ok(rows) => {
                    total_rows.fetch_add(rows, Ordering::SeqCst);
                }
                Err(e) => eprintln!("Error processing file {}: {}", entry.path().display(), e),
            }
        });

        println!(
            "Total number of rows processed: {}",
            total_rows.load(Ordering::SeqCst)
        );
    }

    if let Some(metadata_file_path) = args.metadata_file {
        println!("Writing metadata to {}...", metadata_file_path.display());
        let records = metadata_records.into_inner().unwrap();
        if !records.is_empty() {
            let mut df = DataFrame::new(vec![
                Column::new(
                    "file_name".into(),
                    records.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>(),
                ),
                Column::new(
                    "transcription".into(),
                    records.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>(),
                ),
            ])?;

            let mut file = File::create(&metadata_file_path).with_context(|| {
                format!(
                    "Failed to create metadata file: {}",
                    metadata_file_path.display()
                )
            })?;
            CsvWriter::new(&mut file).finish(&mut df)?;
        }
    }

    println!("Done!");

    Ok(())
}
static UNIQUE_FILENAME_COUNTER: AtomicU64 = AtomicU64::new(0);
