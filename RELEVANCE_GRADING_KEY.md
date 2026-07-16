# Relevance Grading Answer Key

This file maps the anonymized labels in `RELEVANCE_GRADING.md` to their source artifacts.

## Search-result relevance

Source run: `20260713T182115730906Z`

| Label | Provider | Source artifact |
|---|---|---|
| Response A | SerpApi | `raw_audit_logs/query1_serpapi_raw_20260713T182115730906Z.json` |
| Response B | SearchApi | `raw_audit_logs/query1_searchapi_raw_20260713T182115730906Z.json` |

## GLM-5.2 answer completeness

Source search run: `20260713T133545949715Z`

| Label | Provider context | Source artifact |
|---|---|---|
| Output X | SearchApi | `raw_audit_logs/query1_searchapi_glm_5_2_source_20260713T133545949715Z_llm_20260713T143002628743Z.txt` |
| Output Y | SerpApi | `raw_audit_logs/query1_serpapi_glm_5_2_source_20260713T133545949715Z_llm_20260713T143002628743Z.txt` |
