use std::{collections::HashSet, path::Path};

use polib::{catalog::Catalog, message::Message, metadata::CatalogMetadata, po_file};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
pub struct TranslationUnit {
    pub msg_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msg_id_plural: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msg_str: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msg_str_plural: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

use std::fmt;

impl fmt::Display for TranslationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let context = match &self.context {
            Some(c) => format!("[{}] ", c),
            None => "".to_string(),
        };

        let translation = if let Some(s) = &self.msg_str {
            s.as_str()
        } else if let Some(plural_vec) = &self.msg_str_plural {
            plural_vec.first().map(|s| s.as_str()).unwrap_or("...")
        } else {
            "No translations"
        };

        let id_display = if self.msg_id_plural.is_some() {
            format!("{} (plural)", self.msg_id)
        } else {
            self.msg_id.clone()
        };

        write!(f, "{}{} => {}", context, id_display, translation)
    }
}

impl TranslationUnit {
    pub fn is_plural(&self) -> bool {
        self.msg_id_plural.is_some()
    }
}

pub trait Translatable {
    fn extract_messages(
        po_data: Catalog,
        pot_data: Catalog,
        skip_translated: bool,
    ) -> Vec<TranslationUnit>;

    fn apply_translations(
        translations: Vec<TranslationUnit>,
        target_lang: &str,
        output_path: &Path,
    ) -> Result<(), String>;
}

pub struct GettextAdapter;

impl Translatable for GettextAdapter {
    fn extract_messages(
        po_data: Catalog,
        pot_data: Catalog,
        skip_translated: bool,
    ) -> Vec<TranslationUnit> {
        let translated_ids: HashSet<_> = po_data
            .messages()
            .filter(|msg| skip_translated && msg.is_translated())
            .map(|msg| (msg.msgid().to_string(), msg.msgctxt().map(String::from)))
            .collect();

        pot_data
            .messages()
            .filter(|msg| {
                let key = (msg.msgid().to_string(), msg.msgctxt().map(String::from));
                !translated_ids.contains(&key)
            })
            .map(|msg| {
                if msg.is_plural() {
                    TranslationUnit {
                        msg_id: msg.msgid().to_string(),
                        msg_id_plural: Some(msg.msgid_plural().unwrap_or("").to_string()),
                        msg_str: None,
                        msg_str_plural: Some(vec![]),
                        context: msg.msgctxt().map(String::from),
                    }
                } else {
                    TranslationUnit {
                        msg_id: msg.msgid().to_string(),
                        msg_id_plural: None,
                        msg_str: Some(String::new()),
                        msg_str_plural: None,
                        context: msg.msgctxt().map(String::from),
                    }
                }
            })
            .collect()
    }

    fn apply_translations(
        translations: Vec<TranslationUnit>,
        target_lang: &str,
        output_path: &Path,
    ) -> Result<(), String> {
        let metadata_content = format!(
            "Project-Id-Version: 1.0\n\
             Last-Translator: PO-LLM\n\
             Language-Team: PO-LLM\n\
             Language: {}\n\
             MIME-Version: 1.0\n\
             Content-Type: text/plain; charset=UTF-8\n\
             Content-Transfer-Encoding: 8bit\n\
             Plural-Forms: nplurals=2; plural=(n != 1);\n",
            target_lang
        );

        let metadata = CatalogMetadata::parse(&metadata_content)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?;

        let mut catalog = Catalog::new(metadata);

        for translation in translations {
            let message = if translation.is_plural() {
                let msgid_plural = translation.msg_id_plural.unwrap_or_default();
                let msgstr_plural = translation.msg_str_plural.unwrap_or_default();

                Message::build_plural()
                    .with_msgid(translation.msg_id)
                    .with_msgid_plural(msgid_plural)
                    .with_msgstr_plural(msgstr_plural)
                    .done()
            } else {
                let msgstr = translation.msg_str.unwrap_or_default();

                Message::build_singular()
                    .with_msgid(translation.msg_id)
                    .with_msgstr(msgstr)
                    .done()
            };

            catalog.append_or_update(message);
        }

        po_file::write_to_file(&catalog, output_path)
            .map_err(|e| format!("Failed to write PO file: {}", e))?;

        Ok(())
    }
}
