# SearchAPI vs SerpApi Benchmark

This project compares SearchApi and SerpApi as search-context providers for AI workflows. It measures Google Search response latency and the character footprint of normalized organic results, then sends first-round contexts to two NVIDIA NIM models for a qualitative citation audit.

## Methodology

The benchmark runs these two fixed queries for 15 rounds:

1. `Model Context Protocol MCP python servers open source examples`
2. `n8n webhooks custom HTTP response configuration guidelines`

Each provider receives 30 normal search requests. Provider order alternates across query/round combinations, and outbound search attempts are separated by one second. Each successful response contributes its latency and a context made from the title, snippet, and URL of its first three organic results.

Only round-one contexts are evaluated by:

- `deepseek-ai/deepseek-v4-pro`
- `z-ai/glm-5.2`

That produces eight normal NVIDIA requests: two queries × two search providers × two models.

Transient timeouts, connection failures, HTTP 429 responses, and HTTP 5xx responses are retried once. The worst-case ceilings are therefore 60 requests per search provider and 16 NVIDIA requests. Authentication and other HTTP 4xx failures are not retried.

## Requirements

- Python 3.10 or newer
- SearchApi, SerpApi, and NVIDIA API credentials
- Python packages: `httpx`, `pandas`, `openai`, `rich`

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

Never commit `.env` or API keys to this repository. The script validates all three variables before creating network clients.

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

If NVIDIA is interrupted, rerun `python benchmark.py llm`. The successful search benchmark is reused, completed transcript files from the interrupted attempt are preserved, and only missing evaluations are requested. Recorded failures are treated as completed evaluations and are not automatically repeated. For an older search run, the optional advanced form is `python benchmark.py llm --run-id <search-run-id>`.

Search mode consumes API requests and may take several minutes because of the intentional one-second pacing. LLM mode consumes only NVIDIA requests. Tests never make live API calls.

While it runs, animated progress bars show the active search round, query, provider, NVIDIA model, completed counts, elapsed time, and failures. Search requests have a 30-second timeout. NVIDIA requests allow up to 300 seconds for a response because the selected models are substantially larger.

## Outputs

Every execution creates a new artifact set identified by one shared UTC run ID, such as `20260713T153000123456Z`. Existing artifacts are never overwritten.

- `search_api_averaged_results_<run-id>.csv` contains one summary row per query, with both latency columns together followed by both context-size columns. Values are rounded to three decimals, and failed requests are excluded from their metric averages.
- `search_api_detailed_results_<run-id>.csv` contains all 60 individual measurements: query, round, provider, latency, context size, success, and error. These rows are the source for the averages.
- `search_manifest_<run-id>.json` records the search methodology, settings, runtime versions, outcomes, and artifact names.
- `llm_manifest_<llm-run-id>.json` records the NVIDIA attempt and identifies the source search run it evaluated.
- `raw_audit_logs/queryN_PROVIDER_raw_<run-id>.json` contains each provider's successful, unmodified round-one JSON object.
- Timestamped files containing `source_<search-run-id>_llm_<llm-run-id>` contain each NVIDIA completion or an explicit failure/skip message.

Match files by their shared run ID to compare a reproduced run with the published benchmark and inspect the supporting raw responses and model outputs.

## Tests

Install the development dependency and run the mocked suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Troubleshooting

- **Configuration error:** Confirm all three environment variables are exported and non-empty.
- **HTTP 401/403:** Check the corresponding API key and account access. These errors are not retried.
- **HTTP 429:** Wait for the provider's quota window to reset. The script performs only one automatic retry.
- **Blank CSV metric:** Every request for that provider/query failed; inspect the terminal warnings.
- **Missing raw JSON:** The relevant first-round search failed, even if a later round succeeded.
- **Failure text in a transcript:** The first-round context was unavailable or both NVIDIA attempts failed. NVIDIA generation can take up to 300 seconds per attempt.
