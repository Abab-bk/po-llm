use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct AppConfig {
    pub llm: LlmConfig,
    pub translation: TranslationConfig,
    pub project: ProjectConfig,
}

#[derive(Deserialize, Debug)]
pub struct LlmConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Deserialize, Debug)]
pub struct TranslationConfig {
    pub target_languages: Vec<String>,
    pub input_pattern: String,
    pub output_pattern: String,
    pub batch_size: usize,
}

#[derive(Deserialize, Debug)]
pub struct ProjectConfig {
    pub context: String,
    pub base_path: String,
    pub skip_translated: bool,
}
