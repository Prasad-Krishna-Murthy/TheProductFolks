"""
Border Intelligence Multilingual Translator -- Gradio App

Pipeline: Local-language message -> Language Identification -> NMT -> English/Hindi -> Intelligence analyst

Launch:
    python app.py
"""

import gradio as gr
import pandas as pd

from config import LANGUAGES, ANALYST_TARGETS, LOW_RESOURCE_LANGS, VERIFICATION_LOG
from src.pipeline import translate_message
from src.confidence import save_analyst_correction

SOURCE_LANG_CHOICES = ["Auto-detect"] + list(LANGUAGES.keys())

CONFIDENCE_COLOR = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


def run_pipeline(text, target_lang, source_lang_choice):
    if not text or not text.strip():
        return "⚠️ Please enter some text.", "", "", "", ""

    override = None if source_lang_choice == "Auto-detect" else source_lang_choice
    result = translate_message(text, target_lang=target_lang, source_lang_override=override)

    if "error" in result:
        return f"❌ {result['error']}", "", "", "", ""

    lid = result["lid_result"]
    conf = result["confidence"]

    lid_summary = (
        f"**Detected language:** {result['detected_language']}  \n"
        f"**Script:** {lid['script']}  \n"
        f"**LID method:** {lid['method']}  \n"
        f"**LID confidence:** {lid['confidence']:.2f}  \n"
        f"**Resource tier:** {result['resource_tier'].upper()}"
    )

    adapter_note = ""
    if result["resource_tier"] == "low":
        if result["used_adapter"]:
            adapter_note = "✅ Translated using a fine-tuned LoRA adapter for this language."
        else:
            proxy = result.get("used_proxy_language")
            adapter_note = (
                f"⚠️ No fine-tuned LoRA adapter found for **{result['detected_language']}**. "
                f"Translation generated via zero-shot proxy-tag transfer from **{proxy}**. "
                f"Train an adapter with `scripts/train_lora_adapter.py --lang {result['detected_language']}` "
                f"to improve quality."
            )

    confidence_summary = (
        f"### {CONFIDENCE_COLOR.get(conf['level'], '')} Confidence: {conf['level']} ({conf['score']:.2f})\n"
        f"{conf['tier_note']}\n\n"
        + ("**⚠️ Routed to human verification queue.**" if conf["requires_human_verification"] else "**No verification required.**")
    )

    if result["terminology_matches"]:
        term_rows = []
        for m in result["terminology_matches"]:
            approved = m["translations"].get(target_lang, "")
            term_rows.append(f"- `{m['source_term']}` → **{approved}** (domain: {m['domain']}, match score: {m['score']})")
        terminology_summary = "### 🔎 Matched Terminology (Retrieval-Augmented)\n" + "\n".join(term_rows)
    else:
        terminology_summary = "### 🔎 Matched Terminology\nNo domain-specific terms matched."

    return result["translation"], lid_summary, confidence_summary, terminology_summary, adapter_note


def load_verification_queue():
    try:
        df = pd.read_csv(VERIFICATION_LOG)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "timestamp", "source_text", "source_lang", "target_lang", "machine_translation",
            "confidence_score", "confidence_level", "matched_terms", "analyst_correction", "verified"
        ])
    return df


def submit_correction(row_index, corrected_text):
    if row_index is None or corrected_text is None or not corrected_text.strip():
        return load_verification_queue(), "⚠️ Provide a row index and corrected translation."
    ok = save_analyst_correction(int(row_index), corrected_text.strip())
    msg = "✅ Correction saved." if ok else "❌ Invalid row index."
    return load_verification_queue(), msg


with gr.Blocks(title="Border Intelligence Multilingual Translator") as demo:
    gr.Markdown(
        "# 🛡️ Border Intelligence Multilingual Translator\n"
        "Local/Indian languages ↔ English/Hindi — translate intercepted or open-source "
        "field text (messages, signs, reports) from border regions for intelligence analysis.\n\n"
        "**Supported languages:** Tulu, Kodava, Konkani, Beary, Kannada, Tamil, Malayalam, Hindi, English "
        f"(low-resource: {', '.join(LOW_RESOURCE_LANGS)})"
    )

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(label="Source text (intercepted/field message)", lines=6,
                                         placeholder="Paste or type the message here...")
                source_lang_dd = gr.Dropdown(choices=SOURCE_LANG_CHOICES, value="Auto-detect", label="Source language (override LID)")
                target_lang_dd = gr.Dropdown(choices=ANALYST_TARGETS, value="English", label="Translate to")
                translate_btn = gr.Button("Translate", variant="primary")
            with gr.Column(scale=1):
                output_translation = gr.Textbox(label="Translation", lines=6)
                lid_box = gr.Markdown()
                confidence_box = gr.Markdown()
                terminology_box = gr.Markdown()
                adapter_box = gr.Markdown()

        translate_btn.click(
            run_pipeline,
            inputs=[input_text, target_lang_dd, source_lang_dd],
            outputs=[output_translation, lid_box, confidence_box, terminology_box, adapter_box],
        )

    with gr.Tab("Human Verification Queue"):
        gr.Markdown(
            "Low/medium-confidence translations are logged here for analyst review. "
            "Corrections are saved back and can be used for future LoRA fine-tuning."
        )
        refresh_btn = gr.Button("Refresh queue")
        queue_table = gr.Dataframe(value=load_verification_queue(), interactive=False, wrap=True)
        refresh_btn.click(load_verification_queue, outputs=queue_table)

        gr.Markdown("### Submit a correction")
        with gr.Row():
            row_idx_input = gr.Number(label="Row index (0-based, from table above)", precision=0)
            correction_input = gr.Textbox(label="Corrected translation")
        correction_btn = gr.Button("Save correction")
        correction_status = gr.Markdown()
        correction_btn.click(submit_correction, inputs=[row_idx_input, correction_input], outputs=[queue_table, correction_status])

    with gr.Tab("About / Research"):
        gr.Markdown(
            "## Research Problem\n"
            "**Can a translation system work effectively when only a few thousand parallel "
            "sentences exist for a language?**\n\n"
            "This system studies that question for Tulu, Kodava and Beary — languages not "
            "covered by any major off-the-shelf NMT system — using:\n\n"
            "- **Transfer learning**: multilingual Transformer backbone (NLLB-200) with proxy-language FLORES tags.\n"
            "- **LoRA/PEFT adaptation**: lightweight fine-tuning on small seed corpora (`scripts/train_lora_adapter.py`).\n"
            "- **Synthetic parallel-corpus generation**: template + glossary substitution (`scripts/generate_synthetic_corpus.py`).\n"
            "- **Back-translation**: round-trip augmentation from monolingual English/Hindi (`scripts/back_translate.py`).\n"
            "- **Retrieval-augmented terminology control**: defense/security glossary lookup (`src/terminology.py`).\n"
            "- **Confidence scoring + human-in-the-loop verification**: `src/confidence.py` + the Verification Queue tab.\n\n"
            "See `research/low_resource_experiment.py` for the BLEU-based ablation harness."
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
