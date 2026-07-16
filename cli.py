"""Command-line interface for MultiSense."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from src.utils.helpers import setup_logger


def initialise_pipeline(
    config: str,
    checkpoint: str | None,
    vader: bool,
    disable_rag: bool,
    disable_llm: bool,
):
    if vader:
        if checkpoint:
            raise click.ClickException("Do not combine --vader and --checkpoint")
        from src.pipeline.orchestrator import HybridPipeline

        return HybridPipeline(
            config_path=config,
            enable_rag=not disable_rag,
            enable_llm=not disable_llm,
        )

    if not checkpoint:
        raise click.ClickException(
            "BERT mode requires --checkpoint. Use --vader for VADER mode."
        )
    if not Path(checkpoint).is_file():
        raise click.ClickException(f"Checkpoint not found: {checkpoint}")
    from src.pipeline.orchestrator import SentimentPipeline

    return SentimentPipeline(
        config_path=config,
        model_checkpoint=checkpoint,
        enable_rag=not disable_rag,
        enable_llm=not disable_llm,
    )


@click.group()
@click.option("--config", default="config/config.yaml", show_default=True)
@click.pass_context
def cli(context, config):
    """Multi-task BERT, RAG and Gemini sentiment analysis."""
    setup_logger()
    context.ensure_object(dict)
    context.obj["config"] = config


def common_pipeline_options(function):
    function = click.option("--disable-llm", is_flag=True)(function)
    function = click.option("--disable-rag", is_flag=True)(function)
    function = click.option("--vader", is_flag=True)(function)
    function = click.option("--checkpoint", default=None)(function)
    return function


@cli.command()
@click.option("--text", required=True)
@common_pipeline_options
@click.pass_context
def analyze(context, text, checkpoint, vader, disable_rag, disable_llm):
    """Analyze one text."""
    pipeline = initialise_pipeline(
        context.obj["config"], checkpoint, vader, disable_rag, disable_llm
    )
    print(json.dumps(pipeline.analyze(text), indent=2, ensure_ascii=False))


@cli.command()
@click.option("--input", "input_file", required=True)
@click.option("--output", "output_file", default="results.json", show_default=True)
@click.option("--text-column", default="text", show_default=True)
@common_pipeline_options
@click.pass_context
def batch(
    context,
    input_file,
    output_file,
    text_column,
    checkpoint,
    vader,
    disable_rag,
    disable_llm,
):
    """Analyze a CSV file."""
    frame = pd.read_csv(input_file)
    if text_column not in frame.columns:
        raise click.ClickException(f"Column not found: {text_column}")
    pipeline = initialise_pipeline(
        context.obj["config"], checkpoint, vader, disable_rag, disable_llm
    )
    results = pipeline.analyze_batch(frame[text_column].astype(str).tolist())
    Path(output_file).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Results saved to {output_file}")


@cli.command()
@click.option("--port", default=None, type=int)
@click.option("--share", is_flag=True)
@common_pipeline_options
@click.pass_context
def demo(
    context,
    port,
    share,
    checkpoint,
    vader,
    disable_rag,
    disable_llm,
):
    """Launch the Gradio demo."""
    from app.demo import build_demo

    pipeline = initialise_pipeline(
        context.obj["config"], checkpoint, vader, disable_rag, disable_llm
    )
    kwargs = {"share": share, "footer_links": []}
    if port is not None:
        kwargs["server_port"] = port
    build_demo(pipeline).launch(**kwargs)


@cli.command("build-rag")
@click.pass_context
def build_rag(context):
    """Build the ChromaDB lexicon knowledge base."""
    from scripts.build_rag import main as build_main

    build_main.main(
        standalone_mode=False,
        args=["--config", context.obj["config"]],
    )


if __name__ == "__main__":
    cli()
