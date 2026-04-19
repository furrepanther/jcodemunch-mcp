# Token Savings: jCodeMunch MCP

jCodeMunch saves tokens on two independent axes:

1. **Retrieval** — agents fetch exact symbols instead of reading whole files.
2. **Encoding (MUNCH)** — responses that do go to the agent are packed in a
   purpose-built compact wire format, not verbose JSON.

The retrieval axis is the big number (typically 95%+ on code-reading tasks);
encoding is a smaller-but-independent multiplier on whatever traffic remains.
They compose — every byte saved on the wire is a byte the agent doesn't pay
to read.

## MUNCH encoding benchmark (v1.56.0)

Representative payloads through the dispatcher with `format="compact"`.
Measured as `(json_bytes - compact_bytes) / json_bytes` with `json.dumps`
using the default separators.

| Tool                   | Encoder | JSON bytes | Compact bytes | Savings |
|------------------------|---------|------------|---------------|---------|
| `find_references`      | `fr1`   | 2,265      | 1,242         | **45.2%** |
| `get_dependency_graph` | `dg1`   | 1,702      | 998           | **41.4%** |
| `get_call_hierarchy`   | `ch1`   | 4,133      | 2,238         | **45.9%** |
| `search_symbols`       | `ss1`   | 4,545      | 2,754         | **39.4%** |
| `get_repo_outline`     | `ro1`   | 4,642      | 2,277         | **50.9%** |
| `get_hotspots`         | `gen1`  | 1,987      | 887           | **55.4%** |

**Median 45.5% / mean 46.4% / min 39.4% / max 55.4%.**

Run it yourself:

```bash
cd munch-bench && python -m munch_bench.encoding_bench
```

Encoding is opt-in per call (`format="auto"` falls back to JSON when savings
drop below 15%) and does not affect retrieval-side savings. See
[SPEC_MUNCH.md](SPEC_MUNCH.md) for the wire format.

## Third-Party Validation

Independent A/B test by [@Mharbulous](https://github.com/Mharbulous) — 50 iterations, Claude Sonnet 4.6, real Vue 3 + Firebase production codebase. JCodeMunch vs native tools (Grep/Glob/Read) on a naming-audit task.

| Metric | Native | JCodeMunch | Delta |
|--------|--------|------------|-------|
| Success rate | 72% | **80%** | +8 pp |
| Timeout rate | 40% | **32%** | −8 pp |
| Mean cost/iteration | $0.783 | **$0.738** | −5.7% |
| Mean cache creation | 104,135 | **93,178** | −10.5% |
| Mean duration | 318s | **299s** | −6.0% |

**Tool-layer savings (fixed overhead removed): 15–25%.**

The blended 5.7% understates the real advantage because each iteration includes fixed overhead (subagent consensus calls, system prompt) independent of which tool variant runs. Isolating iterations with no findings removes that overhead and exposes the raw tool-layer delta.

One finding category appeared only in the JCodeMunch variant: `find_importers` identified two orphaned files with zero live importers — a structural query native tools cannot answer without scripting.

Full report: [`benchmarks/ab-test-naming-audit-2026-03-18.md`](benchmarks/ab-test-naming-audit-2026-03-18.md)

---

## Why This Exists

AI agents waste tokens when they must read entire files to locate a single function, class, or constant.
jCodeMunch indexes a repository once and allows agents to retrieve **exact symbols on demand**, eliminating unnecessary context loading.

---

## Example Scenario

**Repository:** Medium Python codebase (300+ files)
**Task:** Locate and read the `authenticate()` implementation

| Approach         | Tokens Consumed | Process                               |
| ---------------- | --------------- | ------------------------------------- |
| Raw file loading | ~7,500 tokens   | Open multiple files and scan manually |
| jCodeMunch MCP   | ~1,449 tokens   | `search_symbols` → `get_symbol_source` |

**Savings:** ~80.7%

---

## Typical Savings by Task

| Task                     | Raw Approach    | With jCodeMunch | Savings |
| ------------------------ | --------------- | --------------- | ------- |
| Explore repo structure   | ~200,000 tokens | ~2k tokens      | ~99%    |
| Find a specific function | ~40,000 tokens  | ~200 tokens     | ~99.5%  |
| Read one implementation  | ~40,000 tokens  | ~500 tokens     | ~98.7%  |
| Understand module API    | ~15,000 tokens  | ~800 tokens     | ~94.7%  |

---

## Scaling Impact

| Queries | Raw Tokens | Indexed Tokens | Savings |
| ------- | ---------- | -------------- | ------- |
| 10      | 400,000    | ~5k            | 98.7%   |
| 100     | 4,000,000  | ~50k           | 98.7%   |
| 1,000   | 40,000,000 | ~500k          | 98.7%   |

---

## Key Insight

jCodeMunch shifts the workflow from:

**”Read everything to find something”**
to
**”Find something, then read only that.”**

---

## Live Token Savings Counter

Every tool response includes real-time savings data in the `_meta` field:

```json
“_meta”: {
  “tokens_saved”: 2450,
  “total_tokens_saved”: 184320
}
```

- **`tokens_saved`**: Tokens saved by the current call (raw file bytes vs response bytes ÷ 4)
- **`total_tokens_saved`**: Cumulative total across all calls, persisted to `~/.code-index/_savings.json`

No extra API calls or file reads — computed using fast `os.stat` only.

---
