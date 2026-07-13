from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

import benchmark
from openai import APITimeoutError


class FakeSearchClient:
    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        params = kwargs["params"]
        assert isinstance(params, dict)
        self.calls.append((url, params))
        if self.responses:
            return self.responses.pop(0)
        payload = {
            "organic_results": [
                {"title": "One", "snippet": "First", "link": "https://one.test"},
                {"title": "Two", "snippet": "Second", "link": "https://two.test"},
                {"title": "Three", "snippet": "Third", "link": "https://three.test"},
                {"title": "Ignored", "snippet": "Fourth", "link": "https://four.test"},
            ]
        }
        return httpx.Response(200, json=payload)


@dataclass
class FakeMessage:
    content: str


class FakeNimCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage("Cited answer"))])


class TransientNimError(Exception):
    status_code = 503


class RetryingNimCompletions:
    def __init__(self) -> None:
        self.call_count = 0

    async def create(self, **kwargs: object) -> object:
        self.call_count += 1
        if self.call_count == 1:
            raise TransientNimError("temporarily unavailable")
        return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage("Recovered"))])


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def response(status: int, payload: object | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    if payload is None:
        return httpx.Response(status, text="failure", request=request)
    return httpx.Response(status, json=payload, request=request)


def test_load_credentials_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="SERPAPI_KEY, NVIDIA_API_KEY"):
        benchmark.load_credentials({"SEARCHAPI_KEY": "present"})


def test_context_uses_only_three_results_and_normalizes_fields() -> None:
    context = benchmark.build_context(
        {
            "organic_results": [
                {"title": "A", "snippet": None, "link": "u1"},
                {"title": None, "snippet": "B"},
                "not-an-object",
                {"title": "ignored"},
            ]
        }
    )
    assert context.splitlines() == [
        "Title: A | Snippet:  | URL: u1",
        "Title:  | Snippet: B | URL: ",
        "Title:  | Snippet:  | URL: ",
    ]


def test_provider_order_is_balanced_and_alternates() -> None:
    first_positions = [
        benchmark.provider_order(query_index, round_number)[0].name
        for round_number in range(1, 16)
        for query_index in range(2)
    ]
    assert first_positions.count("SearchApi") == 15
    assert first_positions.count("SerpApi") == 15
    assert benchmark.provider_order(0, 1) != benchmark.provider_order(0, 2)


@pytest.mark.asyncio
async def test_retry_once_for_transient_status_and_not_for_auth() -> None:
    payload = {"organic_results": []}
    sleep = FakeSleep()
    pacer = benchmark.RequestPacer(sleep=sleep)
    transient_client = FakeSearchClient([response(500), response(200, payload)])

    result = await benchmark.fetch_search(
        transient_client,
        benchmark.PROVIDERS["searchapi"],
        "query",
        1,
        "key",
        pacer,
    )
    assert result.error is None
    assert len(transient_client.calls) == 2
    assert sleep.calls == [1.0]

    auth_client = FakeSearchClient([response(401)])
    result = await benchmark.fetch_search(
        auth_client,
        benchmark.PROVIDERS["serpapi"],
        "query",
        1,
        "key",
        benchmark.RequestPacer(sleep=FakeSleep()),
    )
    assert result.error and "HTTP 401" in result.error
    assert len(auth_client.calls) == 1


@pytest.mark.asyncio
async def test_nvidia_retries_one_transient_failure() -> None:
    completions = RetryingNimCompletions()
    sleep = FakeSleep()

    result = await benchmark.evaluate_context(
        completions,
        "model-id",
        "query",
        "context",
        sleep,
    )

    assert result == "Recovered"
    assert completions.call_count == 2
    assert sleep.calls == [1.0]


def test_openai_timeout_is_retryable() -> None:
    error = APITimeoutError(request=httpx.Request("POST", benchmark.NVIDIA_BASE_URL))
    assert benchmark.is_transient_nim_error(error)


@pytest.mark.asyncio
async def test_full_mocked_workflow_has_expected_counts_and_outputs(tmp_path: Path) -> None:
    search = FakeSearchClient()
    nim = FakeNimCompletions()
    sleep = FakeSleep()
    credentials = {
        "SEARCHAPI_KEY": "search-key",
        "SERPAPI_KEY": "serp-key",
        "NVIDIA_API_KEY": "nvidia-key",
    }
    run_id = "20260713T153000123456Z"

    frame = await benchmark.run_benchmark(
        credentials,
        output_dir=tmp_path,
        search_client=search,
        nim_completions=nim,
        sleep=sleep,
        run_id=run_id,
        show_progress=False,
    )

    assert len(search.calls) == 60
    assert len(sleep.calls) == 59
    assert len(nim.calls) == 8
    assert list(frame.columns) == list(benchmark.CSV_COLUMNS)
    assert list(frame.columns) == [
        "Query",
        "SearchApi_Latency",
        "SerpApi_Latency",
        "SearchApi_Text_Chars",
        "SerpApi_Text_Chars",
    ]
    assert len(frame) == 2

    for call in nim.calls:
        assert call["temperature"] == 0
        assert call["max_tokens"] == 512
        assert call["stream"] is False

    audit = tmp_path / "raw_audit_logs"
    raw_files = list(audit.glob(f"*_raw_{run_id}.json"))
    transcript_files = list(audit.glob(f"*_source_{run_id}_llm_*.txt"))
    assert len(raw_files) == 4
    assert len(transcript_files) == 8
    assert all(run_id in path.name for path in raw_files + transcript_files)

    csv_path = tmp_path / f"search_api_averaged_results_{run_id}.csv"
    detailed_path = tmp_path / f"search_api_detailed_results_{run_id}.csv"
    manifest_path = tmp_path / f"search_manifest_{run_id}.json"
    assert csv_path.exists()
    assert detailed_path.exists()
    assert manifest_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert ".000" in csv_text

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["run_id"] == run_id
    assert manifest["rounds"] == 15
    assert manifest["search"]["outcomes"] == {
        "SearchApi": {"successful": 30, "failed": 0},
        "SerpApi": {"successful": 30, "failed": 0},
    }
    assert manifest["phase"] == "search"
    assert len(manifest["artifacts"]) == 7
    assert not any(secret in manifest_text for secret in credentials.values())

    llm_manifest_path = next(tmp_path.glob("llm_manifest_*.json"))
    llm_manifest_text = llm_manifest_path.read_text(encoding="utf-8")
    llm_manifest = json.loads(llm_manifest_text)
    assert llm_manifest["source_search_run_id"] == run_id
    assert llm_manifest["nvidia"]["outcomes"] == {
        "successful": 8,
        "failed_or_skipped": 0,
    }
    assert llm_manifest["nvidia"]["read_timeout_seconds"] == 300.0
    assert len(llm_manifest["artifacts"]) == 9
    assert not any(secret in llm_manifest_text for secret in credentials.values())

    detailed = pd.read_csv(detailed_path)
    assert len(detailed) == 60
    assert list(detailed.columns) == list(benchmark.DETAILED_CSV_COLUMNS)
    assert detailed.groupby(["Query", "Provider"]).size().eq(15).all()

    # First position alternates across the two queries and each subsequent round.
    first_urls = [search.calls[index][0] for index in range(0, 60, 2)]
    assert first_urls.count(benchmark.PROVIDERS["searchapi"].endpoint) == 15
    assert first_urls.count(benchmark.PROVIDERS["serpapi"].endpoint) == 15


@pytest.mark.asyncio
async def test_consecutive_runs_preserve_existing_artifacts(tmp_path: Path) -> None:
    credentials = {
        "SEARCHAPI_KEY": "search-key",
        "SERPAPI_KEY": "serp-key",
        "NVIDIA_API_KEY": "nvidia-key",
    }
    first_id = "20260713T153000000001Z"
    second_id = "20260713T153000000002Z"

    await benchmark.run_search_benchmark(
        credentials,
        output_dir=tmp_path,
        rounds=1,
        search_client=FakeSearchClient(),
        sleep=FakeSleep(),
        run_id=first_id,
        show_progress=False,
    )
    first_csv = tmp_path / f"search_api_averaged_results_{first_id}.csv"
    original_content = first_csv.read_text(encoding="utf-8")

    await benchmark.run_search_benchmark(
        credentials,
        output_dir=tmp_path,
        rounds=1,
        search_client=FakeSearchClient(),
        sleep=FakeSleep(),
        run_id=second_id,
        show_progress=False,
    )

    assert first_csv.read_text(encoding="utf-8") == original_content
    assert (tmp_path / f"search_api_averaged_results_{second_id}.csv").exists()

    nim = FakeNimCompletions()
    llm_id = "20260713T153000000003Z"
    await benchmark.run_llm_evaluations(
        {"NVIDIA_API_KEY": "nvidia-key"},
        output_dir=tmp_path,
        nim_completions=nim,
        sleep=FakeSleep(),
        llm_run_id=llm_id,
        show_progress=False,
    )
    assert len(nim.calls) == 8
    llm_manifest = json.loads(
        (tmp_path / f"llm_manifest_{llm_id}.json").read_text(encoding="utf-8")
    )
    assert llm_manifest["source_search_run_id"] == second_id

    collision_client = FakeSearchClient()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        await benchmark.run_search_benchmark(
            credentials,
            output_dir=tmp_path,
            rounds=1,
            search_client=collision_client,
            sleep=FakeSleep(),
            run_id=first_id,
            show_progress=False,
        )
    assert collision_client.calls == []


@pytest.mark.asyncio
async def test_llm_resume_requests_only_missing_evaluations(tmp_path: Path) -> None:
    source_id = "20260713T153000000010Z"
    llm_id = "20260713T153000000011Z"
    credentials = {
        "SEARCHAPI_KEY": "search-key",
        "SERPAPI_KEY": "serp-key",
    }
    await benchmark.run_search_benchmark(
        credentials,
        output_dir=tmp_path,
        rounds=1,
        search_client=FakeSearchClient(),
        sleep=FakeSleep(),
        run_id=source_id,
        show_progress=False,
    )

    audit = tmp_path / "raw_audit_logs"
    existing = [
        audit
        / f"query1_searchapi_deepseek_v4_pro_source_{source_id}_llm_{llm_id}.txt",
        audit / f"query1_searchapi_glm_5_2_source_{source_id}_llm_{llm_id}.txt",
        audit
        / f"query1_serpapi_deepseek_v4_pro_source_{source_id}_llm_{llm_id}.txt",
    ]
    existing[0].write_text("Successful answer\n", encoding="utf-8")
    existing[1].write_text("Another answer\n", encoding="utf-8")
    existing[2].write_text(
        "NVIDIA evaluation failed: APITimeoutError: Request timed out.\n",
        encoding="utf-8",
    )

    nim = FakeNimCompletions()
    resumed_id = await benchmark.run_llm_evaluations(
        {"NVIDIA_API_KEY": "nvidia-key"},
        source_run_id=source_id,
        output_dir=tmp_path,
        nim_completions=nim,
        sleep=FakeSleep(),
        show_progress=False,
    )

    assert resumed_id == llm_id
    assert len(nim.calls) == 5
    assert len(list(audit.glob(f"*_source_{source_id}_llm_{llm_id}.txt"))) == 8
    manifest = json.loads(
        (tmp_path / f"llm_manifest_{llm_id}.json").read_text(encoding="utf-8")
    )
    assert manifest["resumed_existing_evaluations"] == 3
    assert manifest["nvidia"]["outcomes"] == {
        "successful": 7,
        "failed_or_skipped": 1,
    }


def test_credentials_are_ignored_but_artifacts_are_trackable() -> None:
    ignored_env = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        check=False,
    )
    ignored_result = subprocess.run(
        ["git", "check-ignore", "-q", "search_api_averaged_results_example.csv"],
        check=False,
    )
    ignored_log = subprocess.run(
        ["git", "check-ignore", "-q", "raw_audit_logs/example.json"],
        check=False,
    )

    assert ignored_env.returncode == 0
    assert ignored_result.returncode == 1
    assert ignored_log.returncode == 1


def test_aggregation_excludes_failed_values_and_rounds_to_three_decimals() -> None:
    measurements: list[benchmark.Measurement] = []
    for query in benchmark.QUERIES:
        measurements.extend(
            [
                benchmark.Measurement(query, "SearchApi", 1, 1.1114, 10.0),
                benchmark.Measurement(query, "SearchApi", 2, 1.2224, 20.0),
                benchmark.Measurement(query, "SearchApi", 3, float("nan"), float("nan")),
                benchmark.Measurement(query, "SerpApi", 1, 2.0, 30.0),
            ]
        )

    frame = benchmark.aggregate(measurements)
    assert frame.loc[0, "SearchApi_Latency"] == 1.167
    assert frame.loc[0, "SearchApi_Text_Chars"] == 15.0
    assert frame.loc[0, "SerpApi_Latency"] == 2.0
    assert isinstance(frame, pd.DataFrame)
