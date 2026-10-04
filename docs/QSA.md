# QSA — Qwen4-Exp's sparse attention

The full-attention layers of Qwen4-Exp (every fourth: 3, 7, … 47) do not attend to everything. A
small **indexer** scores 4-token blocks of the key history and the attention reads only the best
512 of them, plus the incomplete tail. Attention cost is therefore **constant past 2051 tokens**
instead of linear, which is the whole reason the model is usable at 262144.

This file is the decomposition and the argument for its shape. The declaration side is
`arch/common/rad_qsa.h`; the arithmetic is libr4d's `qsa_block_key`, `qsa_score`, `qsa_select`,
`qsa_work` and `qsa_tail_store`; the sparse attention is the *ordinary* `attn_paged`. **Schemas and
per-operand contracts live in `docs/OPS.md`**, which is the mechanically checked document, and are
not repeated here.

## The shape

From the checkpoint: `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128,
`indexer_budget` 2048, `indexer_compress_ratio` 4. So `block_topk = 2048 / 4 = 512` blocks.
`rotary_dim` is the main attention's — `head_dim 256 * partial_rotary_factor 0.25 = 64` — applied
to the first 64 of the indexer's 128 dims, from the same cos/sin table the layer's attention uses.
The plugin refuses a checkpoint where that would not fit. Three weights a QSA layer:

```
model.language_model.layers.<L>.self_attn.indexer.index_qk_proj.weight   [640, 2560]
model.language_model.layers.<L>.self_attn.indexer.q_layernorm.weight     [128]
model.language_model.layers.<L>.self_attn.indexer.k_layernorm.weight     [128]
```

640 = (4 query heads + 1 key head) x 128.

## What it computes

```
qk        = index_qk_proj(hidden)             [T, 640]
q, raw_k  = split(qk, [512, 128])             q -> [T, 4, 128], raw_k -> [T, 128]
q         = rope(q_layernorm(q), pos)         per query, at the query's OWN position

per complete block b of 4 consecutive key tokens:
  pooled  = mean over the 4 raw_k, IN f32, cast back to bf16
  blk_k_b = rope(k_layernorm(pooled), pos of the block's FIRST token)

per query:
  score_b = sum over the 4 heads of relu(q_h . blk_k_b) / sqrt(128)
  keep    = indices of the top min(512, n_blocks) scores
  visible = the 4 tokens of every kept block, PLUS the tail
            (the (pos+1) % 4 tokens after the last complete block, attended unconditionally)
```

The q side needs no kernel of its own: the norm and the partial rotation over the query half are
`qk_norm_rope_gate`'s **k side** at `n_head` 0, because that kernel already reads a
`[tokens, heads, head_dim]` input at the operand's own row pitch — exactly this projection's query
half inside its `heads + 1` width. It needs the model's cos/sin table, and is bounded by the same
row count the attention block's fusion is, because above that the rows it would read were never
written this step; with no table it falls back to `rmsnorm` + `rope` over a column slice.

## The one thing the reference hides

**A block key is query-independent and, once its four tokens exist, permanent.** The reference
implementation recomputes every block key inside the per-query loop because it is written for
clarity. An engine must not: a block key is computed ONCE, when token `t` with `t % 4 == 3`
arrives, and cached. That turns "a second KV cache" into something much smaller — **block keys**,
`[n_blocks, 128]` bf16, 64 bytes per *token* per QSA layer, plus **a rolling tail** of at most 3 raw
keys to complete the next block.

12 QSA layers x 64 B = **768 B a token**, against the main KV cache's 2 heads x 256 x 2 (K and V)
x 2 bytes x 12 = 24 KiB a token. **The indexer's cache is 3% of the KV cache.** At 262144 tokens
it is 192 MiB a sequence; at 32768, 24 MiB.

## The exactness bound

With `pos + 1 <= budget + ratio - 1 = 2051` visible tokens there are at most `2051/4 = 512`
complete blocks, the top-k keeps all of them, and the tail is attended unconditionally — so the
selected set is every visible token and **QSA is dense causal attention bit for bit**. It is not an
approximation below the bound; it is the same function. Two things follow. A step that carries more
than one query a sequence — a prefill chunk, or a mixed step — issues the *dense* attention below
the bound, because a dense kernel reads each key once a query *tile* where the sparse path reads a
table a query; at decode there is no tile to amortise over, the two cost the same, and the sparse
path runs at every context. And the bound is the property every change to this subsystem is argued
against — see "Checking it".

## How the attention reads a selected set

The selected tokens are 4-aligned and scattered, and **page size 4 makes the selection a block
table**. If the full-attention KV group is paged at 4, a compress block IS a page and the selected
set IS a paged block table — so sparse attention is "write a 512-entry block table" and call the
existing attention, with no new kernel at all.

Nothing structural pins 16: in `r4d_attn_paged_h256_gqa12.hip` the block size is a template
parameter and every use of it is generic, so `C_EQ("block_size", 16)` on an attention row is a
declaration and not a limit. And the way an architecture ASKS for four needs no new mechanism —
`RadKVGroupDecl` deliberately does not carry a block size, because a core that picked its own would
be a core that has to be edited when a kernel changes; it is taken off the resolved kernel's
constraints at declare. **The rows and the `#define` have to agree**: a row that claims 16 against a
kernel compiled at 4 resolves at *declare* and refuses at *issue*, which is a much worse place to
find out.

**There is no engine change either.** `attn_paged` takes the block table and `seqused_k` as ORDINARY
OPERANDS — `arch/common/rad_block_attn_gated_fp8.h` reads them off the KV group batch — so the
sparse attention is the same op and the same kernel, issued with `sel` in place of the sequence's
table and `qsa_select`'s `seqused` in place of its length. Nothing in `core/` learns what QSA is.
The *declaration* order is the reverse of the issue order and is load-bearing: `attn_paged` declares
`sel` and `seqused` as reads, which is what the buffer plan is computed from, so the arch sets them
on the attention block before `attn.declare()` and declares the indexer after it.

## Where the block keys live, and why it decides everything downstream

**In a KV group, at one page per compress block, indexed by PAGE ID.** The argument is not storage —
it is that **the selected set then needs no translation at all.** The scorer reads `bk[page_id]`
through a table the manager already maintains, and the top-512 output is a list of page ids, which
is the block table the attention reads.

**TWO GROUPS, TWO TABLES, AND EACH OP TAKES THE ONE IT BELONGS TO.** The block keys cannot live in
the attention group's pages — there is no room — so they are their own `RAD_KV_FULL` group, and a
sequence's pages there are *different pages* from its attention pages. What makes that harmless is
that both groups are paged at `ratio`, so **block index `b` means the same thing in both**: the b-th
4-token block of the sequence.

| op | table it takes |
|---|---|
| `qsa_score` | the **block-key** group's, because it is gathering block keys |
| `qsa_select` | the **attention** group's, because its output is what the attention reads |

Nothing is remapped — the index `b` is the join, and it is the same number on both sides. Getting it
backwards scores one sequence's keys through another's pages. A `RAD_BUF_PERSIST` array indexed by
`(slot, block within the sequence)` would work, be simpler to declare, and put that translation back
in three places.

**The geometry is a container, not an attention shape.** A `RAD_KV_FULL` group stores
`[pages, kv_heads, block_size, 2*head_dim]`, so with `kv_heads = 1`, `block_size = ratio` and
`head_dim = n / (2*ratio)` a page holds exactly `n` contiguous values — one block key — and the
manager allocates exactly one page per `ratio` tokens. Here `head_dim` is `128 / 8 = 16`, and
`2 * head_dim * ratio == n` is the assertion that keeps the fiction honest. No waste. **The group's
page size comes from the first op declared after it that carries a `block_size`**
(`Builder::find_kv_consumer`) — *any* op, not only an attention one — so `qsa_block_key` declaring
`block_size` is what sizes this group, exactly as `attn_paged` sizes the attention group.

## The tail is state, and storing it as scratch fails three ways

A block key needs `ratio` consecutive raw keys, and at decode three of them are from earlier steps.
The tail is a `RAD_KV_LINEAR` group with `state_dim` `{ratio + n_spec, head_dim}` bound to every QSA
layer including the MTP head's — and the reason is worth stating as failures, because each
alternative is plausible and produces fluent wrong text.

| a tail stored as… | what it does | fires when |
|---|---|---|
| ONE buffer for the whole model | layer L pools layer L-1's raw keys, because `index_qk_proj` is per layer | every decode step, always — three of every four keys in a completing block |
| a plane indexed by the BATCH ROW | a request that outlives an earlier one inherits its keys, because the scheduler orders a step by priority and stable-sorts decode first | any step after a concurrent request retires |
| `ratio` rows indexed `pos % ratio` | a rejected draft restarts the next step at a position this one has already overwritten, so a completed block is REBUILT from a future token's key | any speculative step that straddles a block boundary and does not accept everything |

All three are one shape: **the tail is state, and it was stored as scratch.** It is read on a later
step than the one that wrote it, and it is keyed by layer, by sequence and by position — none of
which a shared plane indexed by a step-local row can carry. `cu_seqlens` answers *which sequence a
token belongs to in this step*; it says nothing about where that sequence's keys may live. A KV
group answers all three, and two more things fall out for nothing: the engine ZEROES a slot when it
hands one out, and it checkpoints the state with the rest of the linear state, so a resume onto a
half-built block reads the keys that block was built from and not whatever the last tenant left.
A `RAD_KV_WINDOW` of width `ratio` is the near miss — it holds the last four tokens of the STEP,
which at decode is exactly the block completing, but within a prefill chunk many blocks complete and
their keys are long gone, so the window would have to be as wide as the chunk and is then not a tail.

**Two small ops carry it, and their order is load-bearing rather than tidy.** `qsa_work` gathers the
blocks completing this step into the flat list `qsa_block_key` reads; `qsa_tail_store` writes this
step's keys into the state. **`qsa_work` is issued BEFORE `qsa_tail_store`**, because the gather
reads raw keys that are in the tail from earlier steps and the store is about to overwrite them. A
token at position `p` lands at row `p % ring`, so a completing block's rows are already in order,
and a sequence that completes nothing gets `page = -1` and is skipped — one launch a layer a step,
whatever the batch is doing.

## Selecting per query, which is what prefill needs

At decode there is one query a sequence, so a per-sequence selection is the right granularity by
accident; at prefill every query in a chunk selects its own set out of one shared history. Both
scoring ops take `cu_seqlens`, so `m` is a QUERY row while the block table stays the sequence's —
`cu_seqlens` is the only operand in a batch that says which sequence a token belongs to. Each query
is then bounded by its own position, `n = min(nc[seq], (pos + 1) / ratio)`, which at decode is
`nc[seq]` itself.

**And the attention is issued one row a query.** `attn_paged` indexes its block table and its length
by the ROW, and nothing ever said a row had to be a sequence: a chunk of `T` queries issued as `T`
rows of `q_len` 1, each with its own `sel` and `seqused`, *is* per-query sparse attention. Same op,
same kernel, for the second time. It is chunked at `max_seqs` rows because the decode kernel's
split-KV scratch is sized at declare from `max_seqs * q_len_max`.

**The partial page is a position test, and that is a causality rule.** `bt[n] >= 0` would be right
for every case decode can present — a decode query on a block boundary has `bt[nc] == -1`, because the KV
manager has not allocated past the sequence. Mid-chunk the same query has *later tokens of the same
chunk* in `bt[n]`: a live page entirely in its future, which `seqused` would then cover in full. The
rule is the query's position, `(pos + 1) % ratio != 0`.

## What the indexer costs

**The indexer is the only part of a decode step that grows with context.** `qsa_score` and
`qsa_select` both scale with the number of compressed blocks in the history and both run once a
layer, so between them they account for the whole of the step-time-against-context curve; every
other op in a hybrid of linear layers and a constant-cost attention is flat to within noise. So the
pair is timed AT LENGTH rather than at a convenient context, and `r4d_selftest --perfqsa` does it
with no model — passing the declared width and the live context separately, because the launch
geometry comes from the declaration and the work from `nc`. Both kernels are shaped by that:
`qsa_score` reads the key history once a query TILE and not once a query row, the tile landing on a
sequence boundary because a sequence contributes exactly `1 + n_spec` rows and `cu` is their
cumulative count; `qsa_select` splits the block list into chunks that each find a local top-k, which
is exact because a chunk's local top-k contains every global top-k member of that chunk — that
argument and its tie rule live in `libr4d/r4d_qsa_select_bf16.hip`, beside the code depending on
them.

**The reduction is on the VALU and not the LDS crossbar**, which is a property of the architecture
rather than a measurement: `__shfl_xor` lowers to `ds_bpermute_b32` at any offset, so a shuffle
butterfly queues on the LDS pipe, while DPP performs the exchange as a *modifier on the add* —
`row_xmask` covers XOR offsets 8, 4, 2 and 1 within each row of sixteen and `v_permlanex16_b32`
covers the one that crosses rows. The lanes exchanged and the order of the adds are identical, so it
is bit-identical and not merely equivalent. That reasoning belongs beside the instructions and lives
there: `libr4d/r4d_common.h`'s `r4d_wave_sum`.

**Any two forms of the selection kernel must produce the same BYTES**, not the same set. A selected set has
no order, so a kernel is free to put entries above the threshold at the front of the table and the
ties at the back — but the attention accumulates a softmax over the pages *in table order*, and
float addition is not associative. A second implementation reproduces the first's layout slot for
slot or it changes the model's output.

**And the caller says how much history there is.** Both ops size their grid from the *declared*
width, because the live count is device data and a grid is a host-side number — so without help a
short prompt launches for the whole `--max-model-len`. The architecture narrows the score operand's column
count to a host-side ceiling the batch already carries, leaving the pitch at the declaration's.
**Every term of that ceiling is slack on purpose**, because an under-estimate drops blocks from the
selection and nothing would say so.

## Checking it

There is no switch that runs sparse and dense attention over the same tokens, deliberately: above
the bound, dense attention is *not* the attention the checkpoint was trained with, so a knob for it
is a way to make a server quietly serve a different model. `libref`
implements none of the QSA ops either, so there is no reference-plugin comparison for the subsystem
as a whole. What exists instead is what a change here is checked against:

- **per-kernel oracles in `r4d_selftest`** — `qsa_block_key` and `qsa_score` against an f64
  evaluation of the same formula, `qsa_work` and `qsa_tail_store` against a host walk of the same
  batch, `qsa_select` against a host sort, a prefill case for the per-query join, and a
  split-vs-unsplit case asserting BIT identity on a fixture whose scores are mostly ties, so the tie
  rule rather than the radix decides the answer;
- **`RADIANCE_DEBUG_QSA`**, the only window into a selection: `nc`, `nsel`, `seqused` and the head of
  the table, for the first few steps. `nc` above `topk` is blocks being dropped on purpose, which is
  what tells QSA doing its job apart from a divergence. It synchronises the stream, so it is an
  environment variable and it is off.

**Count the reach, not the prompt.** The bound is on the context each query sees, so a prompt a few
tokens under it crosses it partway through the decode and is *supposed* to start differing there.
