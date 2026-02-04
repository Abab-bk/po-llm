use anyhow::Result;
use async_openai::{
    Client,
    config::Config,
    types::chat::{
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
};
use async_trait::async_trait;
use schemars::schema_for;
use std::collections::HashMap;

use crate::translations::TranslationUnit;

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct TranslationResult {
    pub translated: Vec<TranslationUnit>,
    pub failed_translated: Vec<TranslationUnit>,
}

#[async_trait]
pub trait Translator {
    async fn translate(
        &self,
        target_lang: &str,
        translation_units: &[TranslationUnit],
        custom_prompt: &Option<String>,
    ) -> Result<TranslationResult>;
}

pub struct DryRunTranslator;

#[async_trait]
impl Translator for DryRunTranslator {
    async fn translate(
        &self,
        target_lang: &str,
        translation_units: &[TranslationUnit],
        _custom_prompt: &Option<String>,
    ) -> Result<TranslationResult> {
        Ok(TranslationResult {
            translated: translation_units
                .iter()
                .map(|unit| {
                    let mut result = unit.clone();

                    if unit.is_plural() {
                        result.msg_str_plural = Some(vec![
                            format!("[DRY:{}] {}", target_lang, unit.msg_id),
                            format!(
                                "[DRY:{}] {}",
                                target_lang,
                                unit.msg_id_plural.as_ref().unwrap()
                            ),
                        ]);
                    } else {
                        result.msg_str = Some(format!("[DRY:{}] {}", target_lang, unit.msg_id));
                    }

                    result
                })
                .collect(),
            failed_translated: Vec::new(),
        })
    }
}

pub struct LlmTranslator<T: Config> {
    pub client: Client<T>,
    pub model: String,
    pub system_prompt: String,
    pub project_context: String,
}

#[async_trait]
impl<M> Translator for LlmTranslator<M>
where
    M: Config,
{
    async fn translate(
        &self,
        target_lang: &str,
        translation_units: &[TranslationUnit],
        custom_prompt: &Option<String>,
    ) -> Result<TranslationResult> {
        if translation_units.is_empty() {
            return Ok(TranslationResult {
                translated: vec![],
                failed_translated: vec![],
            });
        }

        let mut prompt = String::new();
        for (idx, unit) in translation_units.iter().enumerate() {
            prompt.push_str(&format!("**Index**: {}\n", idx));
            prompt.push_str(&format!("Source: {}\n", unit.msg_id));
            if let Some(ctx) = &unit.context {
                prompt.push_str(&format!("Context: {}\n", ctx));
            }
            if let Some(plural) = &unit.msg_id_plural {
                prompt.push_str(&format!("Plural Source: {}\n", plural));
            }
            prompt.push_str("\n---\n");
        }

        let custom_prompt_text = match custom_prompt {
            Some(content) => format!("## User Instructions:\n{}\n", content),
            None => String::new(),
        };

        let system_content = self
            .system_prompt
            .replace("{target_lang}", target_lang)
            .replace("{project_context}", &self.project_context)
            .replace("{custom_prompt}", &custom_prompt_text);

        #[derive(schemars::JsonSchema, serde::Deserialize)]
        struct LlmResponseUnit {
            index: usize,
            msg_str: Option<String>,
            msg_str_plural: Option<Vec<String>>,
        }

        let schema_value = schema_for!(Vec<LlmResponseUnit>).to_value();

        let schema = ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                description: None,
                name: "translations".into(),
                schema: Some(schema_value),
                strict: Some(true),
            },
        };

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages([
                ChatCompletionRequestSystemMessage::from(system_content).into(),
                ChatCompletionRequestUserMessage::from(prompt).into(),
            ])
            .response_format(schema)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build API request: {}", e))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "LLM API call failed for language '{}': {}. Check your API key, base URL, and network connectivity.",
                    target_lang,
                    e
                )
            })?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "LLM returned empty response for language '{}'. The model may not support structured outputs or encountered an error.",
                    target_lang
                )
            })?;

        let results: Vec<LlmResponseUnit> = serde_json::from_str(content).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse LLM JSON response for language '{}':\n  Parse error: {}\n  Response preview: {}\n  This may indicate the model is not following the structured output format.",
                target_lang,
                e,
                content.chars().take(500).collect::<String>()
            )
        })?;

        if results.is_empty() && !translation_units.is_empty() {
            return Err(anyhow::anyhow!(
                "LLM returned empty translation array for {} messages in language '{}'. Expected {} translations.",
                translation_units.len(),
                target_lang,
                translation_units.len()
            ));
        }

        let mut result_map: HashMap<usize, LlmResponseUnit> =
            results.into_iter().map(|u| (u.index, u)).collect();

        let mut translated = Vec::new();
        let mut failed = Vec::new();

        for (idx, original_unit) in translation_units.iter().enumerate() {
            if let Some(res_unit) = result_map.remove(&idx) {
                let mut final_unit = original_unit.clone();

                let is_valid = if original_unit.is_plural() {
                    res_unit
                        .msg_str_plural
                        .as_ref()
                        .map(|v| !v.is_empty() && v.iter().all(|s| !s.trim().is_empty()))
                        .unwrap_or(false)
                } else {
                    res_unit
                        .msg_str
                        .as_ref()
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false)
                };

                if is_valid {
                    final_unit.msg_str = res_unit.msg_str;
                    final_unit.msg_str_plural = res_unit.msg_str_plural;
                    translated.push(final_unit);
                } else {
                    eprintln!(
                        "      ⚠️  Invalid translation for '{}' in {}: empty or whitespace-only | translated: {}",
                        original_unit.msg_id, target_lang, content
                    );
                    failed.push(original_unit.clone());
                }
            } else {
                eprintln!(
                    "      ⚠️  Missing translation for '{}' in {}: not found in LLM response",
                    original_unit.msg_id, target_lang
                );
                failed.push(original_unit.clone());
            }
        }

        if !result_map.is_empty() {
            eprintln!(
                "      ⚠️  LLM returned {} unexpected translations not in the original batch \n response: {}",
                result_map.len(),
                content
            );
        }

        Ok(TranslationResult {
            translated,
            failed_translated: failed,
        })
    }
}
