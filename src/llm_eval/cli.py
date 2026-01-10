import typer
import yaml
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table

from llm_eval.config import EvalConfig
from llm_eval.constants import DEFAULT_LLM_PROVIDER

# Lazy imports for heavy dependencies - only imported when needed
# This makes --help and other CLI operations fast

console = Console()
app = typer.Typer(no_args_is_help=True)

def get_metric_instances(config: EvalConfig):
    # Import heavy ML/NLP libraries only when actually running evaluation
    from llm_eval.metrics.classical import BleuMetric, RougeLMetric
    from llm_eval.metrics.judge import MultiDimensionalJudge
    from llm_eval.metrics.rag import AnswerRelevancyMetric, ContextRelevancyMetric, FaithfulnessMetric
    from llm_eval.metrics.semantic import BERTScoreMetric
    
    from dotenv import load_dotenv
    load_dotenv()
    
    llm_provider = config.llm_judge.provider if config.llm_judge else DEFAULT_LLM_PROVIDER
    model = config.llm_judge.model if config.llm_judge and config.llm_judge.model else None
    registry = {
        "bleu": BleuMetric(),
        "rouge": RougeLMetric(),
        "bertscore": BERTScoreMetric(),
        "faithfulness": FaithfulnessMetric(provider=llm_provider, model=model),
        "context_relevancy": ContextRelevancyMetric(provider=llm_provider),
        "answer_relevancy": AnswerRelevancyMetric(provider=llm_provider),
        "judge": MultiDimensionalJudge(provider=llm_provider)
    }
    return [registry[m] for m in config.metrics if m in registry]
    
@app.command()
def run(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o")
):
    # Import heavy dependencies only when running evaluation
    import pandas as pd
    from llm_eval.evaluator import Evaluator
    from llm_eval.reporting.visualizer import generate_radar_chart, generate_score_histograms
    from llm_eval.reporting.markdown_gen import generate_markdown_report
    
    # 1. Load and Validate Config
    if not config_path.exists():
        console.print(f"[bold red]❌ Config file not found:[/bold red] {config_path}")
        raise typer.Exit(1)

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        console.print(config_data)  # For debugging purposes
        config = EvalConfig(**config_data)

    # 2. Setup Output
    out = output_dir or Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 3. Initialize Engine
    metrics = get_metric_instances(config)
    engine = Evaluator(metrics=metrics)

    console.print(f"Running evaluation: [bold cyan]{config.eval_name}[/bold cyan]...")
    
    # 4. Execute
    results_df = engine.run(config.dataset_path)

    # 5. Generate Markdown
    report_md = generate_markdown_report(results_df, config.eval_name)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # 6. Generate Visuals
    generate_radar_chart(results_df, out / "radar_chart.png")
    generate_score_histograms(results_df, out)

    console.print(f"[green]📊 Visualizations and Markdown report generated in:[/green] {out}")

    # 7. Show and Save results
    console.print("\n[bold green]✅ Evaluation Complete! Summary:[/bold green]")
    
    # Create a rich table for stats
    stats_df = results_df.describe().loc[['mean', 'min', 'max']]
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stat")
    for col in stats_df.columns:
        table.add_column(col)
    
    for stat in ['mean', 'min', 'max']:
        row = [stat] + [f"{stats_df.loc[stat, col]:.4f}" for col in stats_df.columns]
        table.add_row(*row)
    
    console.print(table)
    
    results_df.to_json(out / "results.json", orient="records", indent=4)
    console.print(f"\n[bold blue]📂 Full results saved to:[/bold blue] {out}/results.json")

if __name__ == "__main__":
    app()