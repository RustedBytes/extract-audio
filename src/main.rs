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

static UNIQUE_FILENAME_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct MetadataRecord {
    file_name: String,
    audio_column: String,
    id: Option<String>,
    transcription: Option<String>,
}

struct AudioRecord<'a> {
    audio_column: String,
    path: Option<String>,
    id: Option<String>,
    transcription: Option<String>,
    row_index: usize,
    bytes: &'a [u8],
}

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

    /// CSV file where extracted audio metadata should be written
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

    let df = ParquetReader::new(tmp_file).finish()?;

    Ok(df)
}

fn read_parquet(filename: &Path) -> Result<DataFrame> {
    let file = File::open(filename)
        .with_context(|| format!("Failed to open parquet file: {}", filename.display()))?;

    let df = ParquetReader::new(file)
        .finish()
        .context("Failed to read parquet file into DataFrame")?;

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

fn audio_column_names(df: &DataFrame) -> Vec<String> {
    df.columns()
        .iter()
        .filter(|column| {
            column
                .struct_()
                .is_ok_and(|audio_struct| audio_struct.field_by_name("bytes").is_ok())
        })
        .map(|column| column.name().to_string())
        .collect()
}

fn infer_audio_extension(data: &[u8]) -> &'static str {
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        "wav"
    } else if data.starts_with(b"fLaC") {
        "flac"
    } else if data.starts_with(b"OggS") {
        "ogg"
    } else if data.starts_with(b"ID3")
        || data
            .get(0..2)
            .is_some_and(|header| header[0] == 0xff && header[1] & 0xe0 == 0xe0)
    {
        "mp3"
    } else if data
        .get(4..12)
        .is_some_and(|brand| brand.starts_with(b"ftyp"))
    {
        "m4a"
    } else {
        "audio"
    }
}

fn sanitize_filename_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' => ch,
            _ => '_',
        })
        .collect();

    if sanitized.is_empty() {
        "audio".to_string()
    } else {
        sanitized
    }
}

fn output_file_name(
    audio_column: &str,
    path: Option<&str>,
    id: Option<&str>,
    row_index: usize,
    bytes: &[u8],
    prefix_audio_column: bool,
) -> String {
    let fallback = format!("row-{row_index:06}");
    let raw_name = path
        .filter(|path| !path.is_empty())
        .and_then(|path| Path::new(path).file_name().and_then(|name| name.to_str()))
        .or(id.filter(|id| !id.is_empty()))
        .unwrap_or(&fallback);

    let mut file_name = sanitize_filename_component(raw_name);
    if Path::new(&file_name).extension().is_none() {
        file_name = format!("{}.{}", file_name, infer_audio_extension(bytes));
    }

    if prefix_audio_column {
        format!(
            "{}_{}",
            sanitize_filename_component(audio_column),
            file_name
        )
    } else {
        file_name
    }
}

fn process_file(
    filename: &Path,
    format: Format,
    output_dir: &Path,
    metadata_records: &Mutex<Vec<MetadataRecord>>,
) -> Result<usize> {
    // Convert the file to a DataFrame
    let df = match format {
        Format::Arrow => arrow_to_parquet(filename)
            .with_context(|| format!("Error processing arrow file {}", filename.display()))?,
        Format::Parquet => read_parquet(filename)
            .with_context(|| format!("Error processing parquet file {}", filename.display()))?,
    };

    let audio_columns = audio_column_names(&df);
    if audio_columns.is_empty() {
        anyhow::bail!(
            "No Hugging Face audio columns found in {}. Expected a struct column with a bytes field.",
            filename.display()
        );
    }

    let id_series = df.column("id").ok().and_then(|column| column.str().ok());
    let transcription_series = df
        .column("transcription")
        .ok()
        .and_then(|column| column.str().ok());
    let prefix_audio_column = audio_columns.len() > 1 || audio_columns[0] != "audio";

    let mut total_audio_records = 0;
    let mut all_metadata = Vec::new();

    for audio_column in audio_columns {
        let audio_struct = df.column(&audio_column)?.struct_()?;
        let bytes_field = audio_struct.field_by_name("bytes")?;
        let bytes_series = bytes_field.binary()?;
        let path_field = audio_struct.field_by_name("path").ok();
        let path_series = path_field.as_ref().and_then(|series| series.str().ok());

        let records: Vec<_> = (0..df.height())
            .filter_map(|row_index| {
                bytes_series.get(row_index).map(|bytes| AudioRecord {
                    audio_column: audio_column.clone(),
                    path: path_series
                        .and_then(|series| series.get(row_index))
                        .map(ToString::to_string),
                    id: id_series
                        .and_then(|series| series.get(row_index))
                        .map(ToString::to_string),
                    transcription: transcription_series
                        .and_then(|series| series.get(row_index))
                        .map(ToString::to_string),
                    row_index,
                    bytes,
                })
            })
            .collect();

        total_audio_records += records.len();

        let local_metadata: Result<Vec<_>> = records
            .par_iter()
            .map(|record| {
                let audio_filename = output_dir.join(output_file_name(
                    &record.audio_column,
                    record.path.as_deref(),
                    record.id.as_deref(),
                    record.row_index,
                    record.bytes,
                    prefix_audio_column,
                ));
                let written_path = write_file(&audio_filename, record.bytes)?;
                let final_name = written_path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| written_path.to_string_lossy().into_owned());

                Ok(MetadataRecord {
                    file_name: final_name,
                    audio_column: record.audio_column.clone(),
                    id: record.id.clone(),
                    transcription: record.transcription.clone(),
                })
            })
            .collect();

        all_metadata.extend(local_metadata?);
    }

    metadata_records.lock().unwrap().extend(all_metadata);

    Ok(total_audio_records)
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
        println!("Total number of audio clips processed: {}", rows);
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
            "Total number of audio clips processed: {}",
            total_rows.load(Ordering::SeqCst)
        );
    }

    if let Some(metadata_file_path) = args.metadata_file {
        println!("Writing metadata to {}...", metadata_file_path.display());
        let records = metadata_records.into_inner().unwrap();
        if !records.is_empty() {
            let height = records.len();
            let mut df = DataFrame::new(
                height,
                vec![
                    Column::new(
                        "file_name".into(),
                        records
                            .iter()
                            .map(|record| record.file_name.as_str())
                            .collect::<Vec<_>>(),
                    ),
                    Column::new(
                        "audio_column".into(),
                        records
                            .iter()
                            .map(|record| record.audio_column.as_str())
                            .collect::<Vec<_>>(),
                    ),
                    Column::new(
                        "id".into(),
                        records
                            .iter()
                            .map(|record| record.id.as_deref())
                            .collect::<Vec<_>>(),
                    ),
                    Column::new(
                        "transcription".into(),
                        records
                            .iter()
                            .map(|record| record.transcription.as_deref())
                            .collect::<Vec<_>>(),
                    ),
                ],
            )?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, BinaryBuilder, Int32Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use arrow::ipc::writer::StreamWriter;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use tempfile::tempdir;

    fn sample_batches() -> Vec<RecordBatch> {
        let path_array: ArrayRef = Arc::new(StringArray::from(vec![
            Some("audio/sample1.wav"),
            Some("audio/sample2.wav"),
        ]));
        let mut bytes_builder = BinaryBuilder::new();
        bytes_builder.append_value(&[1u8, 2, 3]);
        bytes_builder.append_value(&[4u8, 5, 6]);
        let bytes_array: ArrayRef = Arc::new(bytes_builder.finish());
        let sampling_rates: ArrayRef = Arc::new(Int32Array::from(vec![Some(16_000), Some(22_050)]));
        let audio_struct: ArrayRef = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("path", DataType::Utf8, true)),
                path_array.clone(),
            ),
            (
                Arc::new(Field::new("bytes", DataType::Binary, true)),
                bytes_array.clone(),
            ),
            (
                Arc::new(Field::new("sampling_rate", DataType::Int32, true)),
                sampling_rates.clone(),
            ),
        ]));
        let transcriptions: ArrayRef = Arc::new(StringArray::from(vec![
            Some("hello world"),
            Some("goodbye world"),
        ]));

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "audio",
                DataType::Struct(Fields::from(vec![
                    Field::new("path", DataType::Utf8, true),
                    Field::new("bytes", DataType::Binary, true),
                    Field::new("sampling_rate", DataType::Int32, true),
                ])),
                true,
            ),
            Field::new("transcription", DataType::Utf8, true),
        ]));

        vec![
            RecordBatch::try_new(schema, vec![audio_struct, transcriptions])
                .expect("failed to construct sample record batch"),
        ]
    }

    fn audio_struct_array(paths: Vec<Option<&str>>, byte_values: Vec<&[u8]>) -> ArrayRef {
        let path_array: ArrayRef = Arc::new(StringArray::from(paths));
        let mut bytes_builder = BinaryBuilder::new();
        for value in byte_values {
            bytes_builder.append_value(value);
        }
        let bytes_array: ArrayRef = Arc::new(bytes_builder.finish());
        let sampling_rates: ArrayRef = Arc::new(Int32Array::from(vec![Some(16_000), Some(16_000)]));

        Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("path", DataType::Utf8, true)),
                path_array,
            ),
            (
                Arc::new(Field::new("bytes", DataType::Binary, true)),
                bytes_array,
            ),
            (
                Arc::new(Field::new("sampling_rate", DataType::Int32, true)),
                sampling_rates,
            ),
        ]))
    }

    fn voicebank_like_batches() -> Vec<RecordBatch> {
        let ids: ArrayRef = Arc::new(StringArray::from(vec![Some("p226_001"), Some("p226_002")]));
        let clean = audio_struct_array(
            vec![None, None],
            vec![b"RIFF0000WAVEclean-one", b"RIFF0000WAVEclean-two"],
        );
        let noisy = audio_struct_array(
            vec![None, None],
            vec![b"RIFF0000WAVEnoisy-one", b"RIFF0000WAVEnoisy-two"],
        );

        let audio_type = DataType::Struct(Fields::from(vec![
            Field::new("path", DataType::Utf8, true),
            Field::new("bytes", DataType::Binary, true),
            Field::new("sampling_rate", DataType::Int32, true),
        ]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("clean", audio_type.clone(), true),
            Field::new("noisy", audio_type, true),
        ]));

        vec![
            RecordBatch::try_new(schema, vec![ids, clean, noisy])
                .expect("failed to construct voicebank-like record batch"),
        ]
    }

    fn write_parquet_file(dir: &Path, name: &str, batches: &[RecordBatch]) -> PathBuf {
        let path = dir.join(name);
        let file = File::create(&path).expect("failed to create parquet file");
        let schema = batches[0].schema();
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("failed to create parquet writer");
        for batch in batches {
            writer
                .write(batch)
                .expect("failed to write batch to parquet file");
        }
        writer.close().expect("failed to close parquet writer");
        path
    }

    fn write_arrow_file(dir: &Path, name: &str, batches: &[RecordBatch]) -> PathBuf {
        let path = dir.join(name);
        let file = File::create(&path).expect("failed to create arrow file");
        {
            let mut writer =
                StreamWriter::try_new(file, &batches[0].schema()).expect("failed to create writer");
            for batch in batches {
                writer.write(batch).expect("failed to write record batch");
            }
            writer.finish().expect("failed to finish arrow stream");
        }
        path
    }

    #[test]
    fn batches_to_parquet_preserves_audio_struct() {
        let batches = sample_batches();
        let df = batches_to_parquet(&batches).expect("conversion should succeed");
        assert_eq!(df.height(), 2);

        let audio = df
            .column("audio")
            .expect("missing audio column")
            .struct_()
            .expect("audio column should be a struct");
        let path_field = audio.field_by_name("path").expect("missing path field");
        let paths = path_field.str().expect("path field should be utf8");
        assert_eq!(paths.get(0), Some("audio/sample1.wav"));
        assert_eq!(paths.get(1), Some("audio/sample2.wav"));

        let bytes_field = audio.field_by_name("bytes").expect("missing bytes field");
        let bytes = bytes_field.binary().expect("bytes field should be binary");
        assert_eq!(bytes.get(0), Some(&[1u8, 2, 3][..]));
        assert_eq!(bytes.get(1), Some(&[4u8, 5, 6][..]));
    }

    #[test]
    fn read_parquet_loads_expected_columns() {
        let batches = sample_batches();
        let temp_dir = tempdir().expect("failed to create tempdir");
        let parquet_path = write_parquet_file(temp_dir.path(), "input.parquet", &batches);

        let df = read_parquet(&parquet_path).expect("should read parquet file");
        assert_eq!(df.height(), 2);

        let transcription = df
            .column("transcription")
            .expect("missing transcription column")
            .str()
            .expect("transcription column should be utf8");
        assert_eq!(transcription.get(0), Some("hello world"));
        assert_eq!(transcription.get(1), Some("goodbye world"));
    }

    #[test]
    fn arrow_to_parquet_reads_stream_file() {
        let batches = sample_batches();
        let temp_dir = tempdir().expect("failed to create tempdir");
        let arrow_path = write_arrow_file(temp_dir.path(), "input.arrow", &batches);

        let df = arrow_to_parquet(&arrow_path).expect("should load arrow stream");
        assert_eq!(df.height(), 2);

        let audio = df
            .column("audio")
            .expect("missing audio column")
            .struct_()
            .expect("audio column should be a struct");
        let path_field = audio.field_by_name("path").expect("missing path field");
        let paths = path_field.str().expect("path field should be utf8");
        assert_eq!(paths.get(0), Some("audio/sample1.wav"));
    }

    #[test]
    fn process_file_writes_audio_and_metadata() {
        let batches = sample_batches();
        let temp_dir = tempdir().expect("failed to create tempdir");
        let parquet_path = write_parquet_file(temp_dir.path(), "input.parquet", &batches);
        let output_dir = temp_dir.path().join("out");
        fs::create_dir(&output_dir).expect("failed to create output dir");
        let metadata = Mutex::new(Vec::new());

        UNIQUE_FILENAME_COUNTER.store(0, Ordering::SeqCst);
        let processed = process_file(&parquet_path, Format::Parquet, &output_dir, &metadata)
            .expect("processing should succeed");
        assert_eq!(processed, 2);

        let mut written_files: Vec<_> = fs::read_dir(&output_dir)
            .expect("failed to read output directory")
            .map(|entry| {
                entry
                    .expect("entry error")
                    .file_name()
                    .into_string()
                    .expect("non utf8 file name")
            })
            .collect();
        written_files.sort();
        assert_eq!(written_files, vec!["sample1.wav", "sample2.wav"]);

        let audio_bytes = fs::read(output_dir.join("sample1.wav")).expect("file missing");
        assert_eq!(audio_bytes, vec![1, 2, 3]);

        let metadata = metadata.lock().expect("metadata mutex poisoned");
        assert!(metadata.iter().any(|record| {
            record.file_name == "sample1.wav"
                && record.audio_column == "audio"
                && record.transcription.as_deref() == Some("hello world")
        }));
        assert!(metadata.iter().any(|record| {
            record.file_name == "sample2.wav"
                && record.audio_column == "audio"
                && record.transcription.as_deref() == Some("goodbye world")
        }));
    }

    #[test]
    fn process_file_writes_multiple_audio_columns_without_transcription() {
        let batches = voicebank_like_batches();
        let temp_dir = tempdir().expect("failed to create tempdir");
        let parquet_path = write_parquet_file(temp_dir.path(), "voicebank.parquet", &batches);
        let output_dir = temp_dir.path().join("out");
        fs::create_dir(&output_dir).expect("failed to create output dir");
        let metadata = Mutex::new(Vec::new());

        UNIQUE_FILENAME_COUNTER.store(0, Ordering::SeqCst);
        let processed = process_file(&parquet_path, Format::Parquet, &output_dir, &metadata)
            .expect("processing should succeed");
        assert_eq!(processed, 4);

        let mut written_files: Vec<_> = fs::read_dir(&output_dir)
            .expect("failed to read output directory")
            .map(|entry| {
                entry
                    .expect("entry error")
                    .file_name()
                    .into_string()
                    .expect("non utf8 file name")
            })
            .collect();
        written_files.sort();
        assert_eq!(
            written_files,
            vec![
                "clean_p226_001.wav",
                "clean_p226_002.wav",
                "noisy_p226_001.wav",
                "noisy_p226_002.wav",
            ]
        );

        let audio_bytes = fs::read(output_dir.join("noisy_p226_001.wav")).expect("file missing");
        assert_eq!(audio_bytes, b"RIFF0000WAVEnoisy-one");

        let metadata = metadata.lock().expect("metadata mutex poisoned");
        assert!(metadata.iter().any(|record| {
            record.file_name == "clean_p226_001.wav"
                && record.audio_column == "clean"
                && record.id.as_deref() == Some("p226_001")
                && record.transcription.is_none()
        }));
    }

    #[test]
    fn write_file_generates_unique_names_for_conflicts() {
        let temp_dir = tempdir().expect("failed to create tempdir");
        let target = temp_dir.path().join("clip.wav");
        UNIQUE_FILENAME_COUNTER.store(0, Ordering::SeqCst);

        let first = write_file(&target, b"first").expect("initial write should succeed");
        assert_eq!(first, target);

        let second = write_file(&target, b"second").expect("second write should succeed");
        assert_ne!(second, target);

        let second_name = second
            .file_name()
            .expect("missing file name")
            .to_string_lossy()
            .into_owned();
        assert!(
            second_name.ends_with("clip.wav"),
            "second path should include original file name"
        );
        assert!(
            second_name.contains('-'),
            "second path should include timestamp prefix"
        );

        let content = fs::read(&second).expect("failed to read rewritten file");
        assert_eq!(content, b"second");
    }
}
