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
    use_gemini: bool = False,
    gemini_api_key: str = "",
    model: str = "",
) -> None:
    if use_gemini:
        os.environ["OPENAI_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        os.environ["REFLEXION_MODEL"] = model or "gemini-2.5-flash-lite"
        if gemini_api_key:
            os.environ["OPENAI_API_KEY"] = gemini_api_key
        elif os.getenv("GEMINI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
        if not os.getenv("OPENAI_API_KEY"):
            raise typer.BadParameter(
                "Gemini API key is missing. Provide --gemini-api-key or set GEMINI_API_KEY in .env."
            )
    elif model:
        os.environ["REFLEXION_MODEL"] = model

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
    print(f"[bold]Model[/bold]: {os.getenv('REFLEXION_MODEL', 'qwen2.5-coder')}")
    print(f"[bold]Base URL[/bold]: {os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:11434/v1')}")
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
