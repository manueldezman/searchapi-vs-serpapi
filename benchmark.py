"""Reproducible SearchApi vs SerpApi benchmark with NVIDIA NIM evaluation."""

from __future__ import annotations

import asyncio
import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

import httpx
import pandas as pd
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


QUERIES = (
    "Model Context Protocol MCP python servers open source examples",
    "n8n webhooks custom HTTP response configuration guidelines",
)
ROUNDS = 15
REQUEST_INTERVAL_SECONDS = 1.0
TIMEOUT_SECONDS = 30.0
NVIDIA_READ_TIMEOUT_SECONDS = 300.0
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODELS = {
    "deepseek_v4_pro": "deepseek-ai/deepseek-v4-pro",
    "glm_5_2": "z-ai/glm-5.2",
}
CSV_COLUMNS = (
    "Query",
    "SearchApi_Latency",
    "SerpApi_Latency",
    "SearchApi_Text_Chars",
    "SerpApi_Text_Chars",
)
DETAILED_CSV_COLUMNS = (
    "Run_ID",
    "Query",
    "Round",
    "Provider",
    "Latency",
    "Text_Chars",
    "Success",
    "Error",
)


@dataclass(frozen=True)
class Provider:
    name: str
    filename: str
    endpoint: str
    key_name: str


PROVIDERS = {
    "searchapi": Provider(
        name="SearchApi",
        filename="searchapi",
        endpoint="https://www.searchapi.io/api/v1/search",
        key_name="SEARCHAPI_KEY",
    ),
    "serpapi": Provider(
        name="SerpApi",
        filename="serpapi",
        endpoint="https://serpapi.com/search",
        key_name="SERPAPI_KEY",
    ),
}


class SearchClient(Protocol):
    async def get(self, url: str, **kwargs: Any) -> httpx.Response: ...


class NimCompletions(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


@dataclass
class Measurement:
    query: str
    provider: str
    round_number: int
    latency: float
    text_chars: float
    context: str | None = None
    raw_payload: Any | None = None
    error: str | None = None


class RequestPacer:
    """Place a fixed asynchronous pause between outbound search attempts."""

    def __init__(
        self,
        interval: float = REQUEST_INTERVAL_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.interval = interval
        self.sleep = sleep
        self._has_requested = False

    async def wait(self) -> None:
        if self._has_requested:
            await self.sleep(self.interval)
        self._has_requested = True


def load_credentials(
    environ: Mapping[str, str] | None = None,
    required: tuple[str, ...] = ("SEARCHAPI_KEY", "SERPAPI_KEY", "NVIDIA_API_KEY"),
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    missing = [name for name in required if not source.get(name, "").strip()]
    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return {name: source[name].strip() for name in required}


def generate_run_id() -> str:
    """Return a sortable UTC identifier precise enough for consecutive runs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def provider_order(query_index: int, round_number: int) -> tuple[Provider, Provider]:
    """Balance first position across all 30 query/round combinations."""
    pair = (PROVIDERS["searchapi"], PROVIDERS["serpapi"])
    return pair if (query_index + round_number) % 2 else pair[::-1]


def build_context(payload: Mapping[str, Any]) -> str:
    organic = payload.get("organic_results", [])
    if not isinstance(organic, list):
        organic = []

    lines: list[str] = []
    for item in organic[:3]:
        result = item if isinstance(item, Mapping) else {}
        title = str(result.get("title") or "")
        snippet = str(result.get("snippet") or "")
        link = str(result.get("link") or "")
        lines.append(f"Title: {title} | Snippet: {snippet} | URL: {link}")
    return "\n".join(lines)


def is_transient_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


async def fetch_search(
    client: SearchClient,
    provider: Provider,
    query: str,
    round_number: int,
    api_key: str,
    pacer: RequestPacer,
    clock: Callable[[], float] = time.perf_counter,
) -> Measurement:
    last_error = "Unknown request failure"
    for attempt in range(2):
        await pacer.wait()
        started = clock()
        try:
            response = await client.get(
                provider.endpoint,
                params={"engine": "google", "q": query, "api_key": api_key},
            )
            latency = clock() - started
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt == 0 and is_transient_status(response.status_code):
                    continue
                break
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("response JSON is not an object")
            context = build_context(payload)
            return Measurement(
                query=query,
                provider=provider.name,
                round_number=round_number,
                latency=latency,
                text_chars=float(len(context)),
                context=context,
                raw_payload=payload,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                continue
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"Invalid JSON response: {exc}"
            break

    return Measurement(
        query=query,
        provider=provider.name,
        round_number=round_number,
        latency=math.nan,
        text_chars=math.nan,
        error=last_error,
    )


def nim_prompt(query: str, context: str) -> str:
    return (
        f"Answer the query: '{query}' using ONLY the context block provided below.\n"
        "Do not use outside knowledge or training assumptions.\n"
        "You must cite the matching source URLs directly within your statements:\n\n"
        f"{context}"
    )


def is_transient_nim_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            APITimeoutError,
            APIConnectionError,
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    ):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and is_transient_status(status_code)


async def evaluate_context(
    completions: NimCompletions,
    model: str,
    query: str,
    context: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    for attempt in range(2):
        try:
            response = await completions.create(
                model=model,
                messages=[{"role": "user", "content": nim_prompt(query, context)}],
                temperature=0,
                max_tokens=512,
                stream=False,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("NVIDIA returned an empty completion")
            return str(content)
        except Exception as exc:  # SDK errors share status_code but not one base class.
            if attempt == 0 and is_transient_nim_error(exc):
                await sleep(REQUEST_INTERVAL_SECONDS)
                continue
            return f"NVIDIA evaluation failed: {type(exc).__name__}: {exc}"
    raise AssertionError("unreachable")


def write_text_new(path: Path, content: str) -> None:
    """Create a new artifact and refuse to replace an existing one."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def write_json_new(path: Path, payload: Any) -> None:
    write_text_new(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("httpx", "openai", "pandas", "rich")
    }


def aggregate(measurements: list[Measurement]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in QUERIES:
        row: dict[str, Any] = {"Query": query}
        for provider in PROVIDERS.values():
            selected = [
                item
                for item in measurements
                if item.query == query and item.provider == provider.name
            ]
            latencies = pd.Series([item.latency for item in selected], dtype="float64")
            sizes = pd.Series([item.text_chars for item in selected], dtype="float64")
            row[f"{provider.name}_Latency"] = round(float(latencies.mean()), 3)
            row[f"{provider.name}_Text_Chars"] = round(float(sizes.mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def detailed_results(measurements: list[Measurement], run_id: str) -> pd.DataFrame:
    rows = [
        {
            "Run_ID": run_id,
            "Query": item.query,
            "Round": item.round_number,
            "Provider": item.provider,
            "Latency": item.latency,
            "Text_Chars": item.text_chars,
            "Success": item.error is None,
            "Error": item.error or "",
        }
        for item in measurements
    ]
    return pd.DataFrame(rows, columns=list(DETAILED_CSV_COLUMNS))


def make_progress(show_progress: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        disable=not show_progress,
    )


async def run_search_benchmark(
    credentials: Mapping[str, str],
    *,
    output_dir: Path = Path("."),
    rounds: int = ROUNDS,
    search_client: SearchClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    run_id: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    run_id = run_id or generate_run_id()
    started_at = datetime.now(timezone.utc)
    audit_dir = output_dir / "raw_audit_logs"
    audit_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"search_api_averaged_results_{run_id}.csv"
    detailed_csv_path = output_dir / f"search_api_detailed_results_{run_id}.csv"
    manifest_path = output_dir / f"search_manifest_{run_id}.json"
    expected_paths = [csv_path, detailed_csv_path, manifest_path]
    for query_index in range(len(QUERIES)):
        for provider in PROVIDERS.values():
            expected_paths.append(
                audit_dir
                / f"query{query_index + 1}_{provider.filename}_raw_{run_id}.json"
            )
    collisions = [str(path) for path in expected_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"Run ID {run_id} already has artifacts; refusing to overwrite: "
            + ", ".join(collisions)
        )

    owns_search_client = search_client is None
    client = search_client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    pacer = RequestPacer(sleep=sleep)
    measurements: list[Measurement] = []
    created_artifacts: list[str] = []
    progress = make_progress(show_progress)
    search_task = progress.add_task(
        "Search benchmark: preparing",
        total=rounds * len(QUERIES) * len(PROVIDERS),
    )

    try:
        progress.start()
        for round_number in range(1, rounds + 1):
            for query_index, query in enumerate(QUERIES):
                for provider in provider_order(query_index, round_number):
                    progress.update(
                        search_task,
                        description=(
                            f"Search round {round_number}/{rounds} · query "
                            f"{query_index + 1}/2 · {provider.name}"
                        ),
                    )
                    result = await fetch_search(
                        client,
                        provider,
                        query,
                        round_number,
                        credentials[provider.key_name],
                        pacer,
                    )
                    measurements.append(result)
                    progress.advance(search_task)
                    if result.error:
                        progress.console.print(
                            f"Warning: round {round_number}, {provider.name}, "
                            f"query {query_index + 1}: {result.error}"
                        )
                    if round_number == 1:
                        if result.raw_payload is not None:
                            raw_path = (
                                audit_dir
                                / (
                                    f"query{query_index + 1}_{provider.filename}_"
                                    f"raw_{run_id}.json"
                                )
                            )
                            write_json_new(raw_path, result.raw_payload)
                            created_artifacts.append(str(raw_path.relative_to(output_dir)))

        progress.update(search_task, description="Search benchmark complete")
        detailed_frame = detailed_results(measurements, run_id)
        detailed_frame.to_csv(
            detailed_csv_path,
            index=False,
            float_format="%.6f",
            mode="x",
        )
        created_artifacts.append(str(detailed_csv_path.relative_to(output_dir)))

        frame = aggregate(measurements)
        frame.to_csv(
            csv_path,
            index=False,
            float_format="%.3f",
            mode="x",
        )
        created_artifacts.append(str(csv_path.relative_to(output_dir)))

        search_summary = {
            provider.name: {
                "successful": sum(
                    item.provider == provider.name and item.error is None
                    for item in measurements
                ),
                "failed": sum(
                    item.provider == provider.name and item.error is not None
                    for item in measurements
                ),
            }
            for provider in PROVIDERS.values()
        }
        completed_at = datetime.now(timezone.utc)
        created_artifacts.append(str(manifest_path.relative_to(output_dir)))
        manifest = {
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "queries": list(QUERIES),
            "rounds": rounds,
            "search": {
                "endpoints": {
                    provider.name: provider.endpoint for provider in PROVIDERS.values()
                },
                "engine": "google",
                "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
                "timeout_seconds": TIMEOUT_SECONDS,
                "retry_policy": "one retry for timeout, network, HTTP 429, and HTTP 5xx",
                "outcomes": search_summary,
            },
            "phase": "search",
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            "artifacts": sorted(created_artifacts),
        }
        write_json_new(manifest_path, manifest)
        return frame
    finally:
        progress.stop()
        if owns_search_client:
            await client.aclose()  # type: ignore[attr-defined]


def search_manifest_for_run(output_dir: Path, run_id: str) -> Path:
    for prefix in ("search_manifest", "benchmark_manifest"):
        candidate = output_dir / f"{prefix}_{run_id}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No completed search run found for run ID {run_id}")


def latest_search_run_id(output_dir: Path) -> str:
    candidates: list[tuple[str, Path]] = []
    for prefix in ("search_manifest", "benchmark_manifest"):
        for path in output_dir.glob(f"{prefix}_*.json"):
            candidates.append((path.stem.removeprefix(f"{prefix}_"), path))
    if not candidates:
        raise FileNotFoundError("No completed search run found. Run `python benchmark.py search` first.")
    return max(candidates, key=lambda item: item[0])[0]


def incomplete_llm_run_id(output_dir: Path, source_run_id: str) -> str | None:
    """Find the newest interrupted LLM attempt for the selected search run."""
    audit_dir = output_dir / "raw_audit_logs"
    marker = f"_source_{source_run_id}_llm_"
    counts: dict[str, int] = {}
    for path in audit_dir.glob(f"*{marker}*.txt"):
        llm_run_id = path.stem.rsplit(marker, 1)[-1]
        counts[llm_run_id] = counts.get(llm_run_id, 0) + 1
    incomplete = [
        run_id
        for run_id, count in counts.items()
        if count < len(QUERIES) * len(PROVIDERS) * len(MODELS)
        and not (output_dir / f"llm_manifest_{run_id}.json").exists()
    ]
    return max(incomplete) if incomplete else None


async def run_llm_evaluations(
    credentials: Mapping[str, str],
    *,
    source_run_id: str | None = None,
    output_dir: Path = Path("."),
    nim_completions: NimCompletions | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    llm_run_id: str | None = None,
    show_progress: bool = True,
) -> str:
    source_run_id = source_run_id or latest_search_run_id(output_dir)
    source_manifest_path = search_manifest_for_run(output_dir, source_run_id)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    llm_run_id = (
        llm_run_id
        or incomplete_llm_run_id(output_dir, source_run_id)
        or generate_run_id()
    )
    started_at = datetime.now(timezone.utc)
    audit_dir = output_dir / "raw_audit_logs"
    manifest_path = output_dir / f"llm_manifest_{llm_run_id}.json"
    transcript_paths = [
        audit_dir
        / (
            f"query{query_index + 1}_{provider.filename}_{model_slug}_"
            f"source_{source_run_id}_llm_{llm_run_id}.txt"
        )
        for query_index in range(len(QUERIES))
        for provider in PROVIDERS.values()
        for model_slug in MODELS
    ]
    if manifest_path.exists():
        raise FileExistsError(
            f"LLM run ID {llm_run_id} is already complete; refusing to overwrite its manifest"
        )

    nim_owner: AsyncOpenAI | None = None
    if nim_completions is None:
        nim_owner = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=credentials["NVIDIA_API_KEY"],
            timeout=httpx.Timeout(
                NVIDIA_READ_TIMEOUT_SECONDS,
                connect=TIMEOUT_SECONDS,
                write=TIMEOUT_SECONDS,
                pool=TIMEOUT_SECONDS,
            ),
            max_retries=0,
        )
        nim_completions = nim_owner.chat.completions

    progress = make_progress(show_progress)
    existing_paths = {path for path in transcript_paths if path.exists()}
    task = progress.add_task(
        f"NVIDIA evaluation: resuming {len(existing_paths)}/8",
        total=len(transcript_paths),
        completed=len(existing_paths),
    )
    successes = 0
    failures = 0
    created: list[str] = []
    try:
        progress.start()
        for query_index, query in enumerate(QUERIES):
            for provider in PROVIDERS.values():
                raw_path = (
                    audit_dir
                    / f"query{query_index + 1}_{provider.filename}_raw_{source_run_id}.json"
                )
                context: str | None = None
                if raw_path.exists():
                    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    context = build_context(raw_payload)
                for model_slug, model_id in MODELS.items():
                    progress.update(
                        task,
                        description=(
                            f"NVIDIA query {query_index + 1}/2 · {provider.name} · {model_slug}"
                        ),
                    )
                    transcript_path = (
                        audit_dir
                        / (
                            f"query{query_index + 1}_{provider.filename}_{model_slug}_"
                            f"source_{source_run_id}_llm_{llm_run_id}.txt"
                        )
                    )
                    if transcript_path.exists():
                        transcript = transcript_path.read_text(encoding="utf-8").strip()
                        if transcript.startswith(
                            ("NVIDIA evaluation failed:", "NVIDIA evaluation skipped:")
                        ):
                            failures += 1
                        else:
                            successes += 1
                        created.append(str(transcript_path.relative_to(output_dir)))
                        continue
                    if context is None:
                        transcript = "NVIDIA evaluation skipped: first-round search context unavailable"
                    else:
                        transcript = await evaluate_context(
                            nim_completions, model_id, query, context, sleep
                        )
                    if transcript.startswith(("NVIDIA evaluation failed:", "NVIDIA evaluation skipped:")):
                        failures += 1
                        progress.console.print(
                            f"Warning: query {query_index + 1}, {provider.name}, "
                            f"{model_slug}: {transcript}"
                        )
                    else:
                        successes += 1
                    write_text_new(transcript_path, transcript.rstrip() + "\n")
                    created.append(str(transcript_path.relative_to(output_dir)))
                    progress.advance(task)
        progress.update(task, description="NVIDIA evaluation complete")
        created.append(str(manifest_path.relative_to(output_dir)))
        manifest = {
            "phase": "llm",
            "llm_run_id": llm_run_id,
            "source_search_run_id": source_run_id,
            "source_search_manifest": str(source_manifest_path.relative_to(output_dir)),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "resumed_existing_evaluations": len(existing_paths),
            "queries": source_manifest.get("queries", list(QUERIES)),
            "nvidia": {
                "base_url": NVIDIA_BASE_URL,
                "models": list(MODELS.values()),
                "settings": {"temperature": 0, "max_tokens": 512, "stream": False},
                "connect_timeout_seconds": TIMEOUT_SECONDS,
                "read_timeout_seconds": NVIDIA_READ_TIMEOUT_SECONDS,
                "retry_policy": "one retry for transient failures",
                "outcomes": {"successful": successes, "failed_or_skipped": failures},
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            "artifacts": sorted(created),
        }
        write_json_new(manifest_path, manifest)
        return llm_run_id
    finally:
        progress.stop()
        if nim_owner is not None:
            await nim_owner.close()


async def run_benchmark(
    credentials: Mapping[str, str],
    *,
    output_dir: Path = Path("."),
    rounds: int = ROUNDS,
    search_client: SearchClient | None = None,
    nim_completions: NimCompletions | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    run_id: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run both phases; retained as the programmatic convenience interface."""
    source_run_id = run_id or generate_run_id()
    frame = await run_search_benchmark(
        credentials,
        output_dir=output_dir,
        rounds=rounds,
        search_client=search_client,
        sleep=sleep,
        run_id=source_run_id,
        show_progress=show_progress,
    )
    await run_llm_evaluations(
        credentials,
        source_run_id=source_run_id,
        output_dir=output_dir,
        nim_completions=nim_completions,
        sleep=sleep,
        show_progress=show_progress,
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("search", help="Run only the SearchApi and SerpApi benchmark")
    llm_parser = subparsers.add_parser("llm", help="Evaluate the newest search run with NVIDIA")
    llm_parser.add_argument("--run-id", help="Use a specific search run instead of the newest")
    subparsers.add_parser("all", help="Run search and NVIDIA phases")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    command = args.command or "all"
    if command == "search":
        credentials = load_credentials(required=("SEARCHAPI_KEY", "SERPAPI_KEY"))
        await run_search_benchmark(credentials)
    elif command == "llm":
        credentials = load_credentials(required=("NVIDIA_API_KEY",))
        await run_llm_evaluations(credentials, source_run_id=args.run_id)
    else:
        credentials = load_credentials()
        run_id = generate_run_id()
        await run_search_benchmark(credentials, run_id=run_id)
        await run_llm_evaluations(credentials, source_run_id=run_id)


def main() -> int:
    try:
        asyncio.run(async_main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("Benchmark interrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
