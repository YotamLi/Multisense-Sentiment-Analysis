"""Gradio demo for the MultiSense sentiment analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import gradio as gr
import plotly.graph_objects as go
from loguru import logger

from src.utils.helpers import setup_logger

EMOTION_ORDER = ["joy", "anger", "sadness", "fear", "surprise", "disgust"]


def create_radar_chart(predictions: dict) -> go.Figure:
    probabilities = predictions.get("emotion", {}).get("probabilities", {})
    values = [float(probabilities.get(label, 0.0)) for label in EMOTION_ORDER]

    figure = go.Figure()
    if not any(values):
        figure.update_layout(
            title="Awaiting analysis...",
            template="plotly_white",
            height=390,
        )
        return figure

    closed_labels = EMOTION_ORDER + [EMOTION_ORDER[0]]
    closed_values = values + [values[0]]
    figure.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_labels,
            fill="toself",
            name="Emotion",
        )
    )
    figure.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, max(max(values) * 1.2, 0.5)],
                "tickformat": ".0%",
            }
        },
        showlegend=False,
        title="Emotion Probability Distribution",
        template="plotly_white",
        height=390,
        margin={"t": 60, "b": 35, "l": 55, "r": 55},
    )
    return figure


def get_runtime_summary(pipeline) -> str:
    config = getattr(pipeline, "config", {})
    initial_model = getattr(
        pipeline,
        "initial_model_name",
        pipeline.__class__.__name__,
    )
    checkpoint = getattr(pipeline, "checkpoint_path", None)
    llm_name = getattr(pipeline, "llm_model_name", "Disabled")
    rag_enabled = bool(getattr(pipeline, "rag_enabled", False))
    rag_count = int(getattr(pipeline, "rag_document_count", 0) or 0)
    topic_classes = (
        config.get("model", {})
        .get("heads", {})
        .get("topic", {})
        .get("num_classes", "Unknown")
    )
    rag_status = f"Enabled ({rag_count:,} documents)" if rag_enabled else "Disabled"
    checkpoint_status = Path(checkpoint).name if checkpoint else "Not used"
    return (
        f"**Initial model:** {initial_model}  \n"
        f"**Checkpoint:** {checkpoint_status}  \n"
        f"**LLM:** {llm_name}  \n"
        f"**RAG:** {rag_status}  \n"
        f"**Topic head:** {topic_classes} classes"
    )


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def create_prediction_comparison(result: dict) -> str:
    initial = result.get("initial_predictions", {})
    rows = [
        "| Task | Initial model | Final result | Change |",
        "|---|---|---|---|",
    ]
    for key, title in (
        ("sentiment", "Sentiment"),
        ("emotion", "Emotion"),
        ("intensity", "Intensity"),
        ("topic", "Topic"),
    ):
        initial_result = initial.get(key, {})
        final_result = result.get(key, {})
        initial_label = str(initial_result.get("label", "N/A"))
        final_label = str(final_result.get("label", "N/A"))
        initial_confidence = _safe_float(initial_result.get("confidence"))
        final_confidence = _safe_float(final_result.get("confidence"))
        change = "Unchanged" if initial_label == final_label else "Corrected"
        rows.append(
            f"| **{title}** | {initial_label} ({initial_confidence:.0%}) "
            f"| {final_label} ({final_confidence:.0%}) | {change} |"
        )

    sarcasm = bool(result.get("sarcasm", {}).get("is_sarcastic", False))
    rows.extend(["", f"**Sarcasm:** {'Detected' if sarcasm else 'Not detected'}"])
    return "\n".join(rows)


def _escape_markdown(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _format_number(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def format_rag_evidence(result: dict) -> str:
    evidence = result.get("rag_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    keywords = evidence.get("matched_keywords", [])
    scores = evidence.get("lexicon_scores", {})
    negations = evidence.get("negations", [])
    matches = evidence.get("full_text_matches", [])
    sections: list[str] = []

    keyword_text = ", ".join(f"`{_escape_markdown(word)}`" for word in keywords)
    sections.append(
        "### Retrieved sentiment keywords\n"
        + (keyword_text or "No sentiment-bearing keywords were retrieved.")
    )

    if isinstance(scores, dict) and scores:
        rows = [
            "### Lexicon evidence",
            "",
            "| Keyword | VADER | NRC emotions | SWN positive | SWN negative |",
            "|---|---:|---|---:|---:|",
        ]
        for keyword, values in scores.items():
            values = values if isinstance(values, dict) else {}
            emotions = values.get("nrc_emotions", [])
            if isinstance(emotions, list):
                emotions_text = ", ".join(map(_escape_markdown, emotions))
            else:
                emotions_text = _escape_markdown(emotions)
            rows.append(
                f"| `{_escape_markdown(keyword)}` | "
                f"{_format_number(values.get('vader_compound'))} | "
                f"{emotions_text or '—'} | "
                f"{_format_number(values.get('swn_pos'))} | "
                f"{_format_number(values.get('swn_neg'))} |"
            )
        sections.append("\n".join(rows))

    if isinstance(negations, list) and negations:
        rows = [
            "### Detected negation",
            "",
            "| Negation | Scope | Position |",
            "|---|---|---:|",
        ]
        for item in negations:
            if not isinstance(item, dict):
                continue
            scope = item.get("scope", [])
            scope_text = " ".join(map(_escape_markdown, scope)) if isinstance(scope, list) else _escape_markdown(scope)
            rows.append(
                f"| `{_escape_markdown(item.get('negation_word', ''))}` "
                f"| {scope_text or '—'} | {item.get('position', '—')} |"
            )
        sections.append("\n".join(rows))

    if isinstance(matches, list) and matches:
        rows = [
            "### Closest knowledge-base matches",
            "",
            "| Retrieved entry | Similarity |",
            "|---|---:|",
        ]
        for match in matches[:5]:
            if not isinstance(match, dict):
                continue
            document = _escape_markdown(match.get("document", ""))
            if len(document) > 180:
                document = document[:177] + "..."
            rows.append(
                f"| {document or '—'} | {_format_number(match.get('similarity'))} |"
            )
        sections.append("\n".join(rows))

    if not keywords and not matches and not negations:
        sections.append(
            "> No RAG evidence is available. Build the knowledge base or enable RAG."
        )
    return "\n\n".join(sections)


def format_results(result: dict) -> tuple:
    def label_output(task: str) -> dict:
        task_result = result.get(task, {})
        return {
            str(task_result.get("label", "N/A")): _safe_float(
                task_result.get("confidence")
            )
        }

    return (
        label_output("sentiment"),
        label_output("emotion"),
        label_output("intensity"),
        label_output("topic"),
        create_prediction_comparison(result),
        format_rag_evidence(result),
        str(result.get("explanation", "")),
        create_radar_chart(result),
    )


def build_demo(pipeline):
    runtime_summary = get_runtime_summary(pipeline)

    def analyze(text: str):
        if not text or not text.strip():
            empty = {"N/A": 0.0}
            return (
                empty,
                empty,
                empty,
                empty,
                "Enter text to compare initial and final predictions.",
                "Run an analysis to view RAG evidence.",
                "Please enter text.",
                create_radar_chart({}),
            )
        try:
            return format_results(pipeline.analyze(text))
        except Exception as exc:
            logger.exception("Analysis failed")
            raise gr.Error(f"Analysis failed: {exc}") from exc

    with gr.Blocks(
        title="MultiSense Sentiment Analysis",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# MultiSense: Multi-Task Sentiment Analysis\n"
            "### Multi-Head BERT + LangChain + ChromaDB RAG + Gemini\n\n"
            f"{runtime_summary}"
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    placeholder="Enter an English review, tweet or comment...",
                )
                analyze_button = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    examples=[
                        ["The movie was absolutely brilliant! Best film I've seen this year."],
                        ["Terrible customer service. Waited two hours and still no response."],
                        ["Oh, brilliant. Another flat tire in the pouring rain. Exactly what I needed."],
                        ["Their customer service was about as helpful as a screen door on a submarine."],
                    ],
                    inputs=text_input,
                )
            with gr.Column(scale=3):
                with gr.Row():
                    sentiment_label = gr.Label(label="Sentiment Polarity")
                    emotion_label = gr.Label(label="Emotion Category")
                with gr.Row():
                    intensity_label = gr.Label(label="Emotion Intensity")
                    topic_label = gr.Label(label="Topic")

        gr.Markdown("## Initial vs Final Prediction")
        comparison_box = gr.Markdown(
            "Run an analysis to compare the initial model and final result."
        )

        with gr.Row():
            with gr.Column():
                explanation = gr.Textbox(
                    label="Reasoning Explanation",
                    lines=7,
                    interactive=False,
                )
            with gr.Column():
                radar = gr.Plot(label="Emotion Radar Chart")

        with gr.Accordion("RAG Knowledge Evidence", open=False):
            rag_box = gr.Markdown(
                "Run an analysis to view lexicon evidence retrieved from ChromaDB."
            )

        analyze_button.click(
            fn=analyze,
            inputs=text_input,
            outputs=[
                sentiment_label,
                emotion_label,
                intensity_label,
                topic_label,
                comparison_box,
                rag_box,
                explanation,
                radar,
            ],
        )
    return demo


def _init_pipeline(
    config: str,
    checkpoint: str | None,
    use_vader: bool,
    enable_rag: bool,
    enable_llm: bool,
):
    if use_vader:
        if checkpoint:
            raise click.ClickException(
                "Do not pass --checkpoint together with --vader."
            )
        from src.pipeline.orchestrator import HybridPipeline

        return HybridPipeline(
            config_path=config,
            enable_rag=enable_rag,
            enable_llm=enable_llm,
        )

    if not checkpoint:
        raise click.ClickException(
            "BERT mode requires --checkpoint. Use --vader for the fallback mode."
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")

    from src.pipeline.orchestrator import SentimentPipeline

    try:
        return SentimentPipeline(
            config_path=config,
            model_checkpoint=str(checkpoint_path),
            enable_rag=enable_rag,
            enable_llm=enable_llm,
        )
    except Exception as exc:
        logger.exception("Failed to initialise BERT pipeline")
        raise click.ClickException(
            f"The requested BERT pipeline could not be loaded: {exc}"
        ) from exc


@click.command()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.option("--checkpoint", default=None, help="Path to a BERT checkpoint")
@click.option("--vader", is_flag=True, help="Use VADER instead of BERT")
@click.option("--disable-rag", is_flag=True, help="Run without ChromaDB RAG")
@click.option("--disable-llm", is_flag=True, help="Run without Gemini")
@click.option("--port", default=None, type=int)
@click.option("--share", is_flag=True, default=False)
def main(
    config: str,
    checkpoint: str | None,
    vader: bool,
    disable_rag: bool,
    disable_llm: bool,
    port: int | None,
    share: bool,
):
    setup_logger()
    pipeline = _init_pipeline(
        config=config,
        checkpoint=checkpoint,
        use_vader=vader,
        enable_rag=not disable_rag,
        enable_llm=not disable_llm,
    )
    launch_kwargs = {"share": share, "footer_links": []}
    if port is not None:
        launch_kwargs["server_port"] = port
    build_demo(pipeline).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
