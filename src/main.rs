use anyhow::{Context, Result};
use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use futures::stream::{self, StreamExt};
use glob::glob;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use po_llm::{
    configs::AppConfig,
    translations::{GettextAdapter, Translatable},
    translators::{DryRunTranslator, LlmTranslator, Translator},
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

    println!("⚙️  Configuration");
    println!("   └─ Config file: {}", args.config_path.display());
    println!("   └─ Model: {}", config.llm.model);
    println!(
        "   └─ Target languages: {}",
        config.translation.target_languages.join(", ")
    );
    println!("   └─ Batch size: {}", config.translation.batch_size);
    println!("   └─ Skip translated: {}", config.project.skip_translated);
    println!(
        "   └─ Mode: {}",
        if args.dry_run {
            "🔍 DRY RUN"
        } else {
            "🚀 PRODUCTION"
        }
    );

    let config_dir = args.config_path.parent().unwrap_or(Path::new("."));
    let pattern = config_dir
        .join(&config.project.base_path)
        .join(&config.translation.input_pattern);

    let paths: Vec<PathBuf> = glob(pattern.to_str().unwrap())?
        .filter_map(Result::ok)
        .collect();

    if paths.is_empty() {
        println!("⚠️  No files found matching pattern: {}", pattern.display());
        return Ok(());
    }

    println!("📁 Found {} file(s) to process", paths.len());
    for (i, path) in paths.iter().enumerate() {
        println!("   {}. {}", i + 1, path.display());
    }
    println!("\n─────────────────────────────────────────\n");

    let multi_progress = Arc::new(MultiProgress::new());
    let main_pb = multi_progress.add(ProgressBar::new(paths.len() as u64));
    main_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files | {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );
    main_pb.set_message("Starting...");

    let results: Vec<_> = stream::iter(paths)
        .map(|path| {
            let config = &config;
            let args = &args;
            let multi_progress = Arc::clone(&multi_progress);
            let main_pb = main_pb.clone();

            async move {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();

                println!("\n🔄 Processing file: {}", filename);

                let file_pb = multi_progress.add(ProgressBar::new(
                    config.translation.target_languages.len() as u64,
                ));
                file_pb.set_style(
                    ProgressStyle::default_bar()
                        .template(&format!(
                            "  📄 {} {{spinner:.green}} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} langs | {{msg}}",
                            filename
                        ))
                        .unwrap()
                        .progress_chars("█▓▒░ "),
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
                        let msg = if stats.total_failed > 0 {
                            format!(
                                "✅ {} translated, ⚠️  {} failed",
                                stats.total_translated, stats.total_failed
                            )
                        } else {
                            format!("✅ {} messages", stats.total_translated)
                        };
                        file_pb.finish_with_message(msg);
                    }
                    Err(e) => {
                        let error_msg = format!("❌ Error: {}", e);
                        file_pb.finish_with_message(error_msg.clone());
                        eprintln!("\n❌ File processing failed: {}\n   Error: {}\n", filename, e);
                    }
                }

                main_pb.inc(1);
                main_pb.set_message(format!("Processing... ({} completed)", main_pb.position()));
                res
            }
        })
        .buffer_unordered(args.file_concurrent)
        .collect()
        .await;

    main_pb.finish_with_message("✨ Complete");

    let total_ok = results.iter().filter(|r| r.is_ok()).count();
    let total_err = results.len() - total_ok;
    let total_translated: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|s| s.total_translated)
        .sum();
    let total_failed: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|s| s.total_failed)
        .sum();

    let duration = start_time.elapsed();

    println!();
    println!("─────────────────────────────────────────");
    println!("📊 Summary");
    println!("   ├─ Files processed: {} / {}", total_ok, results.len());
    println!("   ├─ Files failed: {}", total_err);
    println!("   ├─ Messages translated: {}", total_translated);
    if total_failed > 0 {
        println!("   ├─ Messages failed: {}", total_failed);
    }
    println!("   └─ Duration: {:.2}s", duration.as_secs_f64());
    println!("─────────────────────────────────────────\n");

    if total_err > 0 {
        println!("❌ Errors encountered:");
        for (i, result) in results.iter().enumerate() {
            if let Err(e) = result {
                println!("   {}. {}", i + 1, e);
            }
        }
        println!();
    }

    if total_err > 0 {
        println!("❌ Translation completed with errors");
        std::process::exit(1);
    } else if total_failed > 0 {
        println!("⚠️  Translation completed with some failed messages");
    } else if total_translated == 0 {
        println!(
            "⚠️  No messages were translated (check your input files and skip_translated setting)"
        );
    } else {
        println!("✅ Translation completed successfully!");
    }

    Ok(())
}

struct FileStats {
    total_translated: usize,
    total_failed: usize,
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

    println!("   Languages to translate: {:?}", langs);

    let results: Vec<_> = stream::iter(langs)
        .map(|lang| {
            let config = config;
            let pb = file_pb.clone();
            let input_path = input_path.clone();

            async move {
                pb.set_message(format!("starting {}", lang));

                println!("      🌐 Starting translation for: {}", lang);

                let result = translate_single_language(
                    &lang,
                    config,
                    dry_run,
                    force_write,
                    &input_path,
                    &pb,
                )
                .await;

                match &result {
                    Ok((translated, failed)) => {
                        if *failed > 0 {
                            pb.println(format!(
                                "      {} - ✅ {} translated, ⚠️  {} failed",
                                lang, translated, failed
                            ));
                        } else if *translated > 0 {
                            pb.println(format!("      {} - ✅ {} translated", lang, translated));
                        } else {
                            pb.println(format!("      {} - ℹ️  No messages to translate", lang));
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("      {} - ❌ {}", lang, e);
                        pb.println(error_msg.clone());
                        eprintln!(
                            "\n❌ Language translation failed: {}\n   Error: {:?}\n",
                            lang, e
                        );
                    }
                }

                pb.inc(1);
                result
            }
        })
        .buffer_unordered(lang_concurrent)
        .collect()
        .await;

    let total_translated: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|(t, _)| *t)
        .sum();
    let total_failed: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|(_, f)| *f)
        .sum();

    let all_failed = results.iter().all(|r| r.is_err());
    if all_failed && !results.is_empty() {
        return Err(anyhow::anyhow!(
            "All language translations failed. Check your LLM configuration and API connectivity."
        ));
    }

    Ok(FileStats {
        total_translated,
        total_failed,
    })
}

async fn translate_single_language(
    target_lang: &str,
    config: &AppConfig,
    dry_run: bool,
    force_write: bool,
    input_path: &PathBuf,
    pb: &ProgressBar,
) -> Result<(usize, usize)> {
    let output_path =
        build_output_path(input_path, target_lang, &config.translation.output_pattern)
            .context("Failed to build output path")?;

    println!("         Input:  {}", input_path.display());
    println!("         Output: {}", output_path.display());

    if !dry_run || force_write {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .context(format!("Failed to create output directory: {:?}", parent))?;
        }
        if !output_path.exists() {
            File::create(&output_path)
                .context(format!("Failed to create output file: {:?}", output_path))?;
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
) -> Result<(usize, usize)> {
    let pot = polib::po_file::parse(input_path)
        .context(format!("Failed to parse POT file: {:?}", input_path))?;

    println!("         POT messages: {}", pot.count());

    let po = if output_path.exists() {
        match polib::po_file::parse(output_path) {
            Ok(po) => {
                println!("         PO messages: {}", po.count());
                po
            }
            Err(e) => {
                eprintln!(
                    "         ⚠️  Failed to parse existing PO file, using POT as template: {}",
                    e
                );
                pot.clone()
            }
        }
    } else {
        println!("         PO file doesn't exist, using POT as template");
        pot.clone()
    };

    let messages = GettextAdapter::extract_messages(po, pot, config.project.skip_translated);

    println!("         Messages to translate: {}", messages.len());

    if messages.is_empty() {
        pb.println(format!(
            "         ℹ️  No messages to translate for {}",
            target_lang
        ));
        return Ok((0, 0));
    }

    let mut total_translated = 0;
    let mut total_failed = 0;
    let batches: Vec<_> = messages.chunks(config.translation.batch_size).collect();
    let total_batches = batches.len();

    println!(
        "         Batches: {} (size: {})",
        total_batches, config.translation.batch_size
    );

    let mut all_translated_for_preview = Vec::new();

    for (batch_idx, batch) in batches.into_iter().enumerate() {
        let batch_num = batch_idx + 1;

        pb.set_message(format!(
            "{} (batch {}/{})",
            target_lang, batch_num, total_batches
        ));

        println!(
            "         📦 Processing batch {}/{} ({} messages)",
            batch_num,
            total_batches,
            batch.len()
        );

        let translations = if dry_run {
            DryRunTranslator
                .translate(target_lang, batch, &config.llm.custom_prompt)
                .await
                .context(format!(
                    "Dry run translation failed for batch {}",
                    batch_num
                ))?
        } else {
            let client = Client::with_config(
                OpenAIConfig::new()
                    .with_api_base(&config.llm.api_base)
                    .with_api_key(&config.llm.api_key),
            );

            let llm = LlmTranslator {
                client,
                model: config.llm.model.clone(),
                system_prompt: config.llm.system_prompt.clone(),
                project_context: config.project.context.clone(),
            };

            llm.translate(target_lang, batch, &config.llm.custom_prompt)
                .await
                .context(format!(
                    "LLM translation failed for batch {} in language {}",
                    batch_num, target_lang
                ))?
        };

        total_translated += translations.translated.len();
        total_failed += translations.failed_translated.len();

        println!(
            "         ✓ Batch {}: {} translated, {} failed",
            batch_num,
            translations.translated.len(),
            translations.failed_translated.len()
        );

        if !translations.translated.is_empty() {
            for entry in &translations.translated {
                pb.println(format!("      ✓ {}", entry));
            }
        }

        if !translations.failed_translated.is_empty() {
            for entry in &translations.failed_translated {
                pb.println(format!("      ✗ {}", entry));
            }
        }

        if dry_run {
            all_translated_for_preview.extend(translations.translated.clone());
        }

        if !dry_run || force_write {
            if !translations.translated.is_empty() {
                GettextAdapter::apply_translations(
                    translations.translated.clone(),
                    target_lang,
                    output_path,
                )
                .map_err(|e| {
                    anyhow::anyhow!("Failed to write translations to {:?}: {}", output_path, e)
                })?;

                println!(
                    "         💾 Saved {} translations to file",
                    translations.translated.len()
                );
            }
        }
    }

    if dry_run && !all_translated_for_preview.is_empty() {
        pb.println(format!("\n      ╭─ Dry Run Preview ({}) ─╮", target_lang));
        for (i, entry) in all_translated_for_preview.iter().take(3).enumerate() {
            pb.println(format!("      │ {:02}. {}", i + 1, entry));
        }
        if all_translated_for_preview.len() > 3 {
            pb.println(format!(
                "      │ ... and {} more",
                all_translated_for_preview.len() - 3
            ));
        }
        pb.println("      ╰────────────────────────────╯\n".to_string());
    }

    Ok((total_translated, total_failed))
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
