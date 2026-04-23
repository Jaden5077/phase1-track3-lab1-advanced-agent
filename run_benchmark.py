from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_100_sample.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    test_mode: bool = False,
    test_examples: int = 3,
) -> None:
    examples = load_dataset(dataset)
    if test_mode:
        if test_examples <= 0:
            raise typer.BadParameter("--test-examples must be > 0 when --test-mode is enabled.")
        examples = examples[:test_examples]
        # In test mode, write to a different output folder by default
        # so it does not override standard benchmark artifacts.
        if out_dir == "outputs/sample_run":
            out_dir = f"outputs/test_run_{len(examples)}"

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
