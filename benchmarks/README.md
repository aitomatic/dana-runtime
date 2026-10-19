# Harness benchmarks

Run from the repository root with the existing Python environment. These drivers
measure actual Dana boundaries using synthetic temporary workspaces. They are not
an alternative runtime, a production database maintenance tool, or authorization
to send paid requests. Do not synchronize dependencies as part of a comparison.

## Drivers

| Driver | Boundary | External model requests |
| --- | --- | --- |
| `benchmark_startup.py` | Fresh direct `dana-agent`, `dana-code`, `dana-acp --help` processes; no uv launcher timing | None |
| `benchmark_session.py` | Real session/STAR/journal with controlled provider; context, streaming, cancellation and reopen cases | Fail-closed network/provider guard |
| `benchmark_history_delta.py` | Real SQLite sessions, independent writer, incremental preparation and reopened provider context | None |
| `benchmark_tools_history.py` | Real tool executor, nested sessions and journal history; deterministic provider | None |
| `benchmark_live.py` | Serial synthetic real-provider tasks with a reservation ledger | Paid; separate authorization required |

The drivers moved here from `scripts/` during PR #32 review. Historical source
hashes and original commands still refer to the original frozen files. Moving
or repairing a driver does not retroactively change an old measurement.

## Deterministic examples

Use a fresh output directory for each revision. Do not overwrite frozen results.
Small repetition counts are smoke checks, not performance or tail estimates.

```sh
uv run --no-sync python benchmarks/benchmark_startup.py \
  --repetitions 20 --output /tmp/dana-benchmark-new/startup.json
uv run --no-sync python benchmarks/benchmark_session.py \
  --mode both --cases fail_closed_guard,warm_no_tool_turns \
  --repetitions 100 --output /tmp/dana-benchmark-new/session.json
uv run --no-sync python benchmarks/benchmark_history_delta.py \
  --repetitions 100 --output /tmp/dana-benchmark-new/history-delta.json
uv run --no-sync python benchmarks/benchmark_tools_history.py \
  --lanes tools,history --repetitions 100 --expensive-samples 100 \
  --output /tmp/dana-benchmark-new/tools-history.json
```

The session driver reports `known_failures`, `coverage_gaps` and per-case status.
A process exit of zero alone does not certify all recorded cases. Inspect those
fields. Some original frozen diagnostics intentionally expose older defects or
incompatible hypotheses, such as requiring every nested child to run at once
while admission is bounded. Preserve such failed observations rather than
turning a diagnostic into a universal acceptance claim. Tool fixtures must carry
accurate effect and safe-offload declarations; unknown operations cannot be
relabeled just to improve overlap.

`--help` lists lane/case selection, expensive sample counts and time limits.
All-case cancellation runs may need a larger **whole-case** timeout for many
samples; this must not relax the recorded per-cancellation bound. Report all
chosen flags, warmups and exclusions with the output.

CI runs small deterministic session/history/startup workloads, verifies committed
turn and reopened-context correctness, and checks the live CLI's argument parser
without sending a request. It does not establish timing thresholds or replace
actual CLI/ACP interaction and storage integration tests.

## Public original measurement extract

[`results/pr32-original-summary.json`](results/pr32-original-summary.json)
keeps aggregate historical results in the working tree. The complete 503 KB
[raw measurement extract](https://raw.githubusercontent.com/aitomatic/dana-runtime/a0da647/benchmarks/results/pr32-original-measurements.json)
is pinned to the earlier public commit; its SHA-256 and byte count are recorded
in the summary. It includes selected headline samples, all 240 startup observations
and all 144 paired attempts. Credentials, endpoint addresses, provider payloads
and private absolute workspace paths are excluded.

**These measurements describe original commit
`07b9037439dd81e45ec1862fce0d3850ec152bd8`, not the repaired review head.** They do
not prove that the original tests covered the review's newly identified paths.
The original B1/B2 source archives and larger provider/session traces remain
local; the public extract is not the complete original evidence bundle.

B1 is the correctness-fixed reference, not the original compressed-history bug.
The original local measurements show substantial startup, forwarding and journal
projection improvements. The live evaluation does **not** establish a general
end-to-end speedup: paired median B2/B1 ratios are 0.9961 on Azure and 1.0538 on
OpenAI, with only three pairs per task/provider. Committed objective success is
70/72 for B1 and 71/72 for B2. The frozen oracle checks answer containment, final
file states and minimum tool counts, not exact formatting or every intermediate
step. Refusals remain failures. Output caps never bound; stable-prefix controls
also changed navigation instructions; forced compaction did not establish
quality preservation.

Do not quote these numbers as post-review performance. Re-measure affected paths
with pinned source and unchanged workloads before making new speed claims.

## Verification evidence and accounting limits

The bounded summary is the only retained result artifact in the current tree.
Its immutable links preserve the [first repair verification](https://github.com/aitomatic/dana-runtime/blob/6eaebb5/benchmarks/results/pr32-review-verification.json)
and [second repair verification](https://github.com/aitomatic/dana-runtime/blob/522ab49/benchmarks/results/pr32-rereview-verification.json).
Those records describe their stated revisions, not current-head acceptance.
Current validation and CI belong to [PR #32](https://github.com/aitomatic/dana-runtime/pull/32).

The summary also records an offline, single-head I14 observation: 100 and 1,000
tool turns, each comprising a call, result and follow-up message; 31-36-byte
or exactly 2,048-byte JSON argument dictionaries; 50 repetitions after five
warmups. Incremental reads, plain full counting, and full recounting with fresh
dependency snapshots are separate measurements. Small arguments favored plain
counting; 2,048-byte arguments favored incremental reads. Append-time copying
and proportional snapshot memory remain costs, not silently disabled behavior.
The earlier 1.557/0.527 ms comparison did not record argument size, so it cannot
support a general performance conclusion.

Historical live evaluation and two Azure compatibility requests do not validate
the repaired head's production latency or model quality. The Azure requests
established HTTP 200 and streamed text, not terminal-carrier or usage receipt.
No new provider request is authorized by these records.

## Paid runs

`benchmark_live.py` defaults to `gpt-5.4-nano`, pins a recent Azure API version and
performs reservation accounting. It is **not** a check of Azure's default API
version. Do not use it to infer compatibility of another API version, deployment,
model family, provider or streaming route. The original evaluation's $30
permission does not authorize another full matrix. Do not run a paid command
without explicit scope, request count and cost authorization.
