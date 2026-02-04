# po-llm

A command-line tool for translating `.po` files using Large Language Models (LLMs). Currently supports OpenAI-compatible APIs only.

## Install

```
cargo install po-llm
```

```
yay -S po-llm-bin
```

Or: [release](https://github.com/Abab-bk/po-llm/releases)

## Usage

You need a `toml` file to configure the software:

```toml
[llm]
api_base = "https://api.xxx.com/v1"
api_key = "your-token"
model = "model-name"
custom_prompt = "your prompt" # option

[translation]
# Language names provided to the LLM (can be any descriptive string)
target_languages = [ "English", "Chinese" ] 
input_pattern = "**/*.pot" # Standard practice uses .pot files as templates
output_pattern = "{name}_{lang}.po"
batch_size = 20 # Number of entries processed in a single prompt

[project]
name = "Untitled Project"
context = "Project description for LLM context."
base_path = "po-files/" # Base directory for input/output patterns
skip_translated = true # Whether to skip entries that already have translations
```

To run:

```sh
po-llm 'config.toml'
```

### Full Arguments

```rust
struct Args {
    #[arg(value_parser = check_file_exists, help = "Path to TOML configuration file")]
    config_path: PathBuf,

    #[arg(short, long, help = "Dry run mode (simulates process without calling API)")]
    dry_run: bool,

    #[arg(short, long, help = "Force write files even in dry run mode")]
    force_write: bool,

    #[arg(
        long,
        default_value_t = 4,
        help = "Number of files to process concurrently"
    )]
    file_concurrent: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of languages to translate concurrently"
    )]
    lang_concurrent: usize,
}
```

## Credits

This project utilizes the following crates:

* `anyhow`
* `async-openai`
* `async-trait`
* `clap`
* `futures`
* `glob`
* `indicatif`
* `polib`
* `schemars`
* `serde`
* `serde_json`
* `tokio`
* `tokio-stream`
* `toml`
