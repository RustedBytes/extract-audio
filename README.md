# extract-audio

`extract-audio` is a command-line utility for quickly unpacking the audio stored in [Hugging Face `datasets`](https://huggingface.co/docs/datasets) parquet or arrow exports. It reads the dataset files, writes each audio clip to disk, and optionally produces a CSV that links the exported files back to their original transcriptions.

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-line options](#command-line-options)
  - [Examples](#examples)
  - [Metadata CSV](#metadata-csv)
- [Development](#development)
- [License](#license)

## Features

- Supports both Apache Arrow (`.arrow`) and Parquet (`.parquet`) files produced by Hugging Face `datasets`.
- Processes either a single file or an entire directory of dataset shards.
- Extracts the audio bytes in parallel using Rayon for fast throughput.
- Preserves the original file names from the dataset metadata when writing audio files.
- Generates an optional CSV with the audio file name and transcription for downstream processing.

## Requirements

You need a Rust toolchain (Rust 1.80 or newer) with `cargo` installed. The command also uses native dependencies from `polars`, `arrow`, and `parquet`, so make sure your system has the necessary build tooling (e.g. `clang`/`gcc`, `cmake`, and the appropriate development headers for your platform).

## Installation

You can compile the binary directly from the repository:

```bash
cargo install --path .
```

Alternatively, run `cargo build --release` and copy the resulting binary from `target/release/extract-audio` to a directory on your `PATH`.

## Usage

Run `extract-audio --help` for an up-to-date list of options. The most common invocation extracts audio from a dataset shard and writes the clips to an output directory:

```bash
extract-audio --input train-00000-of-00010.parquet --output audio/
```

If you have multiple shards, you can process the whole directory:

```bash
extract-audio --input-dir data/shards --format parquet --output audio/
```

### Command-line options

```
Usage: extract-audio [OPTIONS] --output <OUTPUT>

Options:
      --input <INPUT>                  The path to the input file
      --input-dir <INPUT_DIR>          The path to a directory with input files
      --format <FORMAT>                File format [default: parquet] [possible values: arrow, parquet]
      --output <OUTPUT>                The path to the output files
      --threads <THREADS>              Number of threads to use for processing [default: 3]
      --metadata-file <METADATA_FILE>  CSV file where transcriptions should be written
  -h, --help                           Print help
  -V, --version                        Print version
```

The command inspects either the `audio` column of the Parquet file or the nested structures inside the Arrow stream and writes each item in the `bytes` field to disk. Files are skipped if they already exist, allowing interrupted runs to be resumed.

### Examples

```bash
# Read a single parquet shard and write audio clips next to the dataset
extract-audio --format parquet --input data/train-00000-of-00010.parquet --output extracted/train/

# Convert an Arrow export while limiting the worker threads
extract-audio --format arrow --input data/data-00000-of-01189.arrow --output extracted/arrow/ --threads 8
```

### Metadata CSV

Passing the `--metadata-file` flag writes a CSV with two columns: `file_name` and `transcription`. This is useful if you need to align the exported audio files with text for downstream training or evaluation.

```bash
extract-audio \
  --input train-00000-of-00010.parquet \
  --output audio/ \
  --metadata-file transcripts.csv
```

## Development

The project includes tooling for producing multi-architecture binaries using [`cross`](https://github.com/cross-rs/cross), [`podman`](https://podman.io/), and [`goreleaser`](https://goreleaser.com/). To reproduce the release builds:

1. Build the container images and configure podman resources:

    ```bash
    podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.aarch64-unknown-linux-gnu -t aarch64-unknown-linux-gnu:my-edge .
    podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.x86_64-unknown-linux-gnu -t x86_64-unknown-linux-gnu:my-edge .

    podman machine set --cpus 4 --memory 8192
    ```

2. Produce the binaries with goreleaser:

    ```bash
    goreleaser build --clean --snapshot --id extract-audio --timeout 60m
    ```

## License

This project is distributed under the terms of the MIT license.
