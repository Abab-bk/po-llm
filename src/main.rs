use anyhow::{Context, Result};
use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use futures::stream::{self, StreamExt};
use glob::glob;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use po_llm::{
    configs::AppConfig,
    translations::{GettextAdapter, Translatable},
    translators::{DryRunTranslator, LlmTranslator, TranslationResult, Translator},
};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
#[command(name = "po-llm")]
#[command(about = "Translate PO files using LLM", long_about = None)]
struct Args {
    #[arg(value_parser = check_file_exists, help = "Path to TOML configuration file")]
    config_path: PathBuf,

    #[arg(short, long, help = "Dry run mode (no actual translation)")]
    dry_run: bool,

    #[arg(short, long, help = "Force write even in dry run mode")]
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

fn check_file_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("File '{}' not found", s))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    println!("🌍 PO-LLM Translator");

    let config_str = fs::read_to_string(&args.config_path)?;
    let config: AppConfig = toml::from_str(&config_str)?;

    println!("📋 Config: {}", args.config_path.display());
    println!(
        "🎯 Target languages: {}",
        config.translation.target_languages.join(", ")
    );
    println!("📦 Batch size: {}", config.translation.batch_size);

    if args.dry_run {
        println!("🔍 Mode: DRY RUN");
    }

    let config_dir = args.config_path.parent().unwrap_or(Path::new("."));
    let pattern = config_dir
        .join(&config.project.base_path)
        .join(&config.translation.input_pattern);

    let paths: Vec<PathBuf> = glob(pattern.to_str().unwrap())?
        .filter_map(Result::ok)
        .collect();

    if paths.is_empty() {
        println!("⚠️  No files found matching pattern");
        return Ok(());
    }

    println!("📁 Found {} file(s)\n", paths.len());

    let multi_progress = Arc::new(MultiProgress::new());
    let main_pb = multi_progress.add(ProgressBar::new(paths.len() as u64));
    main_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files ({msg})")
            .unwrap()
            .progress_chars("█▓▒░"),
    );
    main_pb.set_message("processing...");

    let results: Vec<_> = stream::iter(paths)
        .map(|path| {
            let config = &config;
            let args = &args;
            let multi_progress = Arc::clone(&multi_progress);
            let main_pb = main_pb.clone();

            async move {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();

                let file_pb = multi_progress.add(ProgressBar::new(
                    config.translation.target_languages.len() as u64,
                ));
                file_pb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!(
                            "📄 {} {{spinner:.green}} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} langs ({{msg}})",
                            filename
                        ))
                        .unwrap()
                        .progress_chars("█▓▒░"),
                );

                let res = translate_file(
                    config,
                    &path,
                    file_pb.clone(),
                    args.lang_concurrent,
                    args.dry_run,
                    args.force_write,
                )
                .await;

                match &res {
                    Ok(stats) => {
                        file_pb.finish_with_message(format!("✅ {} messages", stats.total_translated));
                    }
                    Err(e) => {
                        file_pb.finish_with_message(format!("❌ {}", e));
                    }
                }

                main_pb.inc(1);
                res
            }
        })
        .buffer_unordered(args.file_concurrent)
        .collect()
        .await;

    main_pb.finish_with_message("done");

    let total_ok = results.iter().filter(|r| r.is_ok()).count();
    let total_err = results.len() - total_ok;
    let total_translated: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|s| s.total_translated)
        .sum();

    let duration = start_time.elapsed();

    println!("📊 Summary");
    println!("✅ Succeeded: {} files", total_ok);
    println!("❌ Failed: {} files", total_err);
    println!("📝 Translated: {} messages", total_translated);
    println!("⏱️ Duration: {:.2}s\n", duration.as_secs_f64());

    if total_err > 0 {
        std::process::exit(1);
    }

    Ok(())
}

struct FileStats {
    total_translated: usize,
}

async fn translate_file(
    config: &AppConfig,
    input_path: &PathBuf,
    file_pb: ProgressBar,
    lang_concurrent: usize,
    dry_run: bool,
    force_write: bool,
) -> Result<FileStats> {
    let langs = config.translation.target_languages.clone();

    let results: Vec<_> = stream::iter(langs)
        .map(|lang| {
            let config = config;
            let pb = file_pb.clone();
            let input_path = input_path.clone();

            async move {
                pb.set_message(format!("translating {}", lang));

                let result = translate_single_language(
                    &lang,
                    config,
                    dry_run,
                    force_write,
                    &input_path,
                    &pb,
                )
                .await;

                if let Err(e) = &result {
                    pb.println(format!("    ❌ {} - {}", lang, e));
                }

                pb.inc(1);
                result
            }
        })
        .buffer_unordered(lang_concurrent)
        .collect()
        .await;

    let total_translated: usize = results.iter().filter_map(|r| r.as_ref().ok()).sum();

    Ok(FileStats { total_translated })
}

async fn translate_single_language(
    target_lang: &str,
    config: &AppConfig,
    dry_run: bool,
    force_write: bool,
    input_path: &PathBuf,
    pb: &ProgressBar,
) -> Result<usize> {
    let output_path =
        build_output_path(input_path, target_lang, &config.translation.output_pattern)?;

    if !dry_run || force_write {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !output_path.exists() {
            File::create(&output_path)?;
        }
    }

    process_single_lang(
        target_lang,
        config,
        dry_run,
        force_write,
        input_path,
        &output_path,
        pb,
    )
    .await
}

async fn process_single_lang(
    target_lang: &str,
    config: &AppConfig,
    dry_run: bool,
    force_write: bool,
    input_path: &Path,
    output_path: &Path,
    pb: &ProgressBar,
) -> Result<usize> {
    let pot = polib::po_file::parse(input_path)?;

    let po = if output_path.exists() {
        polib::po_file::parse(output_path)?
    } else {
        pot.clone()
    };

    let messages = GettextAdapter::extract_messages(po, pot, config.project.skip_translated);

    if messages.is_empty() {
        return Ok(0);
    }

    let mut total_translated = 0;
    let mut result = TranslationResult {
        translated: vec![],
        failed_translated: vec![],
    };

    for batch in messages.chunks(config.translation.batch_size) {
        let translations = if dry_run {
            DryRunTranslator.translate(target_lang, batch).await?
        } else {
            let client = Client::with_config(
                OpenAIConfig::new()
                    .with_api_base(&config.llm.api_base)
                    .with_api_key(&config.llm.api_key),
            );

            let llm = LlmTranslator {
                client,
                model: config.llm.model.clone(),
                project_context: config.project.context.clone(),
            };
            llm.translate(target_lang, batch).await?
        };

        total_translated += translations.translated.len();
        result.translated.extend(translations.translated);
        result
            .failed_translated
            .extend(translations.failed_translated);
    }

    if dry_run {
        pb.println(format!("\n--- Dry Run Preview ({}) ---", target_lang));
        for (i, entry) in result.translated.iter().take(5).enumerate() {
            pb.println(format!("#{:02} {}", i + 1, entry));
        }
        if result.translated.len() > 5 {
            pb.println(format!("... and {} more", result.translated.len() - 5));
        }
        pb.println("-------------------------------\n".to_string());
    }

    if !result.failed_translated.is_empty() {
        pb.println(format!(
            "    ⚠️  {} - {} messages failed",
            target_lang,
            result.failed_translated.len()
        ));
    }

    if !dry_run || force_write {
        GettextAdapter::apply_translations(result.translated.clone(), target_lang, output_path)
            .map_err(|e| anyhow::anyhow!("Failed to write translations: {}", e))?;
    }

    Ok(total_translated)
}

fn build_output_path(input_path: &Path, target_lang: &str, pattern: &str) -> Result<PathBuf> {
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("Invalid filename")?;

    Ok(input_path.parent().unwrap().join(
        pattern
            .replace("{lang}", target_lang)
            .replace("{name}", stem),
    ))
}
