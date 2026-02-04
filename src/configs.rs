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
    pub custom_prompt: Option<String>,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
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

fn default_system_prompt() -> String {
    r#"Role: Professional I18n Translator ({target_lang})
Project Context: {project_context}

Task:
Translate the provided list of texts into {target_lang}.

Strict Requirements:
1. INDEX PRESERVATION: You will be provided with texts marked with "Index: n". Your JSON response MUST include the original "index" for each translation.
2. NO ID REPETITION: Do not include the original source text in your JSON response, only the "index" and the translated strings.
3. MULTILINE HANDLING: Translate multiline text preserving the paragraph structure but returning it as a standard JSON string.
4. STRUCTURE:
   - "msg_str": The main translation.
   - "msg_str_plural": An array of strings for plural forms. Set to null if the source has no plural.

{custom_prompt}

Output: Return a JSON array of objects with keys: "index", "msg_str", "msg_str_plural"."#
        .to_string()
}
