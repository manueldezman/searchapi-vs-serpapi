# SearchApi vs SerpApi Benchmark

This repository contains the benchmark script and raw results from a SearchApi and SerpApi comparison conducted in July 2026. It compares the providers as search-context sources for AI workflows by measuring Google Search response latency and the character footprint of normalized organic results, then sends first-round contexts to two NVIDIA NIM models for a qualitative citation audit.

## Methodology

The benchmark runs these two fixed queries for 15 rounds:

1. `Model Context Protocol MCP python servers open source examples`
2. `n8n webhooks custom HTTP response configuration guidelines`

Each provider receives 30 normal search requests. Provider order alternates across query/round combinations, and outbound search attempts are separated by one second. Each HTTP-successful JSON response contributes its latency. The script builds context from the title, snippet, and URL of its first three organic results. If `organic_results` is absent or empty, the response is currently recorded as successful with a zero-character context; account for those rows separately when calculating usable-result rate or context-size statistics.

Search performance is measured across all 15 rounds. To limit NVIDIA API usage, only the search context collected during round one is evaluated by:

- `deepseek-ai/deepseek-v4-pro`
- `z-ai/glm-5.2`

Each model evaluates both providers' round-one results for both queries. This produces eight normal NVIDIA requests: 2 queries × 2 search providers × 2 models.

Transient timeouts, connection failures, HTTP 429 responses, and HTTP 5xx responses are retried once. The worst-case ceilings are therefore 60 requests per search provider and 16 NVIDIA requests. Authentication and other HTTP 4xx failures are not retried.

### What this test does not cover

- Concurrent or high-throughput request workloads
- Search engines other than Google
- Provider features outside normal search requests, such as asynchronous jobs or scraping APIs
- Pricing comparisons
- Automated scoring of result relevance, citation correctness, or answer completeness

## Requirements

- Python 3.10 or newer
- SearchApi and SerpApi credentials for the search phase; an NVIDIA credential for the LLM phase
- Python packages: `httpx`, `pandas`, `openai`, `rich`

Clone the repository:

```bash
git clone https://github.com/manueldezman/searchapi-vs-serpapi.git
cd searchapi-vs-serpapi
```

Install the runtime packages:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Get an API key from each provider:

1. Sign in to [NVIDIA Build](https://build.nvidia.com/settings/api-keys), generate an API key, and use it as `NVIDIA_API_KEY`.
2. Sign in to the [SerpApi dashboard](https://serpapi.com/dashboard), copy your API key, and use it as `SERPAPI_KEY`.
3. Create an account at [SearchApi](https://www.searchapi.io/), copy the API key from your account, and use it as `SEARCHAPI_KEY`.

Paste the keys into the ignored `.env` file:

```bash
SEARCHAPI_KEY=your-searchapi-key
SERPAPI_KEY=your-serpapi-key
NVIDIA_API_KEY=your-nvidia-api-key
```

Never commit `.env` or API keys to this repository. The script validates only the credentials required by the selected command: both search keys for `search`, the NVIDIA key for `llm`, and all three keys for `all`.

## Run

From this repository's root, load the keys once:

```bash
set -a
source .env
set +a
```

Then choose a phase:

```bash
# Run only the 60-request SearchApi vs SerpApi benchmark
python benchmark.py search

# Evaluate the newest completed search run with NVIDIA; no run ID is needed
python benchmark.py llm

# Run both phases in sequence
python benchmark.py all
```

Search runs do not resume after interruption; rerunning `python benchmark.py search` starts a new timestamped run without overwriting partial artifacts. If NVIDIA is interrupted, rerun `python benchmark.py llm`. The successful search benchmark is reused, completed transcript files from the interrupted attempt are preserved, and only missing evaluations are requested. Recorded failures are treated as completed evaluations and are not automatically repeated. For an older search run, the optional advanced form is `python benchmark.py llm --run-id <search-run-id>`.

Search mode consumes API requests and may take several minutes because of the intentional one-second pacing. LLM mode consumes only NVIDIA requests. Tests never make live API calls.

While it runs, animated progress bars show the active search round, query, provider, NVIDIA model, completed counts, and elapsed time. Failures are printed separately as terminal warnings. Search requests have a 30-second timeout. NVIDIA requests allow up to 300 seconds for a response because the selected models are substantially larger.

## Outputs

Search and LLM executions use separate UTC run IDs, such as `20260713T153000123456Z`. An LLM run records the source search run ID it evaluated, and transcript filenames contain both IDs. Existing artifacts are never overwritten.

A completed search and LLM evaluation produces this layout:

```text
.
├── search_api_averaged_results_<search-run-id>.csv
├── search_api_detailed_results_<search-run-id>.csv
├── search_manifest_<search-run-id>.json
├── llm_manifest_<llm-run-id>.json
└── raw_audit_logs/
    ├── queryN_PROVIDER_raw_<search-run-id>.json
    └── queryN_PROVIDER_MODEL_source_<search-run-id>_llm_<llm-run-id>.txt
```

- `search_api_averaged_results_<run-id>.csv` contains one summary row per query, with both latency columns together followed by both context-size columns. Values are rounded to three decimals, and request failures are excluded from their metric averages. HTTP-successful responses with no organic results are included as zero-character contexts, so do not interpret the raw character average as token efficiency without separating empty responses.
- `search_api_detailed_results_<run-id>.csv` contains all 60 individual measurements: query, round, provider, latency, context size, success, and error. These rows are the source for the averages.
- `search_manifest_<run-id>.json` records the search methodology, settings, runtime versions, outcomes, and artifact names.
- `llm_manifest_<llm-run-id>.json` records the NVIDIA attempt and identifies the source search run it evaluated.
- `raw_audit_logs/queryN_PROVIDER_raw_<run-id>.json` contains each provider's unmodified round-one JSON object when the request returned valid JSON, including responses with no organic results.
- Timestamped files containing `source_<search-run-id>_llm_<llm-run-id>` contain each NVIDIA completion or an explicit failure/skip message.

Match search artifacts by their search run ID. Use `source_<search-run-id>_llm_<llm-run-id>` in transcript filenames, or `source_search_run_id` in the LLM manifest, to connect model outputs to their source search run.

### Averaged CSV columns

| Column | Description |
|---|---|
| `Query` | Fixed search query evaluated in the benchmark |
| `SearchApi_Latency` | Mean successful SearchApi response time in seconds |
| `SerpApi_Latency` | Mean successful SerpApi response time in seconds |
| `SearchApi_Text_Chars` | Mean character count of SearchApi's normalized three-result context |
| `SerpApi_Text_Chars` | Mean character count of SerpApi's normalized three-result context |

### Detailed CSV columns

| Column | Description |
|---|---|
| `Run_ID` | UTC identifier for the search run |
| `Query` | Search query sent to the provider |
| `Round` | Benchmark round number, from 1 through 15 |
| `Provider` | `SearchApi` or `SerpApi` |
| `Latency` | End-to-end response time in seconds; blank when the request failed |
| `Text_Chars` | Character count of the normalized three-result context; blank on request failure and zero for a successful empty result |
| `Success` | Whether the request completed successfully and returned valid JSON |
| `Error` | Failure detail; blank for successful requests |

## Analysis Metrics

The generated CSV files directly record per-request latency, context character count, request success, and errors. The article comparison can derive:

- Average, median, and p95 latency from successful latency measurements.
- Usable-result rate by treating rows with `Text_Chars > 0` as non-empty contexts.
- Empty-result rate by treating HTTP-successful rows with `Text_Chars == 0` as empty contexts.

Pricing must be collected from current provider plans. Result relevance and LLM answer completeness require a documented manual or automated scoring rubric; the benchmark does not calculate those scores automatically.

## Tests

Install the development dependency and run the mocked suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Troubleshooting

- **Configuration error:** Confirm the variables required for the selected command are exported and non-empty: both search keys for `search`, the NVIDIA key for `llm`, or all three for `all`.
- **HTTP 401/403:** Check the corresponding API key and account access. These errors are not retried.
- **HTTP 429:** Wait for the provider's quota window to reset. The script performs only one automatic retry.
- **Blank CSV metric:** Every request for that provider/query failed; inspect the terminal warnings.
- **Missing raw JSON:** The relevant first-round search failed, even if a later round succeeded.
- **Empty context:** A valid response without organic results is recorded with `Text_Chars` equal to zero and is still sent to NVIDIA if its round-one raw JSON was saved.
- **Failure text in a transcript:** The first-round raw response was unavailable, a non-transient NVIDIA request failed once, or a transient NVIDIA request failed on both attempts. NVIDIA generation can take up to 300 seconds per attempt.
