# Entry point (Typer)
import typer
import yaml
from pathlib import Path
from llm_eval.config import EvalConfig

app = typer.Typer()

@app.command()
def run(config_path: Path = typer.Option(..., "--config", "-c", help="Path to the YAML configuration file")):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        config = EvalConfig(**config_data)

    typer.echo(f"Staring evaluation: {config.eval_name}")

    # TODO: Load Dataset
    # TODO: Run Metrics
    # TODO: Generate Reports


if __name__ == "__main__":
    app()