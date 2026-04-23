from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


def _run_with_progress(agent_name: str, agent: ReActAgent | ReflexionAgent, examples: list) -> list:
    total = len(examples)
    records = []
    for idx, example in enumerate(examples, start=1):
        print(f"[cyan][{agent_name}][/cyan] {idx}/{total} qid={example.qid} ...")
        record = agent.run(example)
        status = "[green]OK[/green]" if record.is_correct else "[red]FAIL[/red]"
        print(
            f"[cyan][{agent_name}][/cyan] {idx}/{total} done "
            f"(attempts={record.attempts}, tokens={record.token_estimate}, latency_ms={record.latency_ms}) {status}"
        )
        records.append(record)
    return records


def _save_partial(out_path: Path, react_records: list, reflexion_records: list, dataset: str) -> None:
    all_records = react_records + reflexion_records
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    if all_records:
        report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock")
        json_path, md_path = save_report(report, out_path)
        print(f"[yellow]Partial save[/yellow] {json_path}")
        print(f"[yellow]Partial save[/yellow] {md_path}")


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

    out_path = Path(out_dir)
    print(f"[bold]PID[/bold]: {os.getpid()}")
    print(f"[bold]Dataset[/bold]: {dataset} | [bold]Examples[/bold]: {len(examples)}")
    print(f"[bold]Output[/bold]: {out_path}")
    print("[yellow]Press Ctrl+C to stop gracefully (partial results will be saved).[/yellow]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    react_records = []
    reflexion_records = []
    try:
        react_records = _run_with_progress("react", react, examples)
        reflexion_records = _run_with_progress("reflexion", reflexion, examples)
    except KeyboardInterrupt:
        print("\n[red]Interrupted by user.[/red] Saving partial outputs...")
        _save_partial(out_path, react_records, reflexion_records, dataset)
        raise typer.Exit(code=130)

    all_records = react_records + reflexion_records
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
