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
        custome_prompt: &Option<String>,
    ) -> Result<TranslationResult>;
}

pub struct DryRunTranslator;

#[async_trait]
impl Translator for DryRunTranslator {
    async fn translate(
        &self,
        target_lang: &str,
        translation_units: &[TranslationUnit],
        _custome_prompt: &Option<String>,
    ) -> Result<TranslationResult> {
        Ok(TranslationResult {
            translated: translation_units
                .iter()
                .map(|unit| {
                    let mut result = unit.clone();

                    if unit.is_plural() {
                        result.msg_str_plural = Some(vec![
                            format!("[{}] {}", target_lang, unit.msg_id),
                            format!("[{}] {}", target_lang, unit.msg_id_plural.as_ref().unwrap()),
                        ]);
                    } else {
                        result.msg_str = Some(format!("[{}] {}", target_lang, unit.msg_id));
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
        custome_prompt: &Option<String>,
    ) -> Result<TranslationResult> {
        if translation_units.is_empty() {
            return Ok(TranslationResult {
                translated: vec![],
                failed_translated: vec![],
            });
        }

        let mut prompt = String::new();
        prompt.push_str("# Translation Task\n\n");

        for (index, unit) in translation_units.iter().enumerate() {
            prompt.push_str(&format!("## Message {}\n", index + 1));

            if let Some(context) = &unit.context {
                prompt.push_str(&format!("**Context**: {}\n", context));
            }

            if unit.is_plural() {
                prompt.push_str(&format!("**Singular**: {}\n", unit.msg_id));
                prompt.push_str(&format!(
                    "**Plural**: {}\n",
                    unit.msg_id_plural.as_ref().unwrap()
                ));
            } else {
                prompt.push_str(&format!("**Text**: {}\n", unit.msg_id));
            }

            prompt.push('\n');
        }

        let user_prompt = match custome_prompt {
            Some(content) => format!("## User Instructions:\n{}\n", content),
            None => String::new(),
        };

        let system_content = format!(
            "You are a professional localization expert translating strings to {}.\n\
            \n\
            ## Project Context\n\
            {}\n\
            \n\
            ## Rules\n\
            - Do NOT modify msg_id or msg_id_plural\n\
            - Preserve all placeholders exactly ({0}, %s, {{name}}, etc.)\n\
            - Match tone and formality of the source\n\
            - Singular → msg_str\n\
            - Plural → msg_str_plural, provide an ARRAY of strings.\n\
            \n\
            {}
            Return a JSON array of translation units with the EXACT structure provided.",
            target_lang, self.project_context, user_prompt
        );

        let schema_value = schema_for!(Vec<TranslationUnit>).to_value();

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
            .build()?;

        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| anyhow::anyhow!("LLM returned empty response"))?;

        let results: Vec<TranslationUnit> = serde_json::from_str(content).map_err(|e| {
            anyhow::anyhow!("Failed to parse LLM response: {}, response: {}", e, content)
        })?;

        let mut result_map: HashMap<String, TranslationUnit> = results
            .into_iter()
            .map(|unit| (unit.msg_id.clone(), unit))
            .collect();

        let mut translated = Vec::new();
        let mut failed = Vec::new();

        for original_unit in translation_units {
            if let Some(translated_unit) = result_map.remove(&original_unit.msg_id) {
                let is_valid = if translated_unit.is_plural() {
                    translated_unit
                        .msg_str_plural
                        .as_ref()
                        .map(|v| !v.is_empty() && v.iter().all(|s| !s.trim().is_empty()))
                        .unwrap_or(false)
                } else {
                    translated_unit
                        .msg_str
                        .as_ref()
                        .map(|s| !s.trim().is_empty())
                        .unwrap_or(false)
                };

                if is_valid {
                    println!("translated: {} -> {}", original_unit, translated_unit);
                    translated.push(translated_unit);
                } else {
                    println!("failed: {} -> {}", original_unit, translated_unit);
                    failed.push(original_unit.clone());
                }
            } else {
                failed.push(original_unit.clone());
            }
        }

        Ok(TranslationResult {
            translated,
            failed_translated: failed,
        })
    }
}
