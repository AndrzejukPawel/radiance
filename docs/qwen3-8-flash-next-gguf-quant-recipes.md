# Research note: measured quant recipes of `unsloth/Qwen3.8-Flash-Next-GGUF`

Method: GGUF header reads only (HTTP `Range`), no weight downloads · Verified against published file sizes to 4 significant figures

## TL;DR

- Qwen3.8 Flash Next is **176.94 B total / ~6.04 B active** parameters: 125.74 B "normal"
  weights (48 layers, 512-expert MoE, Gated DeltaNet + 12 layers of QSA sparse attention)
  plus a **51.20 B n-gram/PLE look-up table** (29 % of all params, 0 FLOPs).
- Unsloth's quants only really compress **two things**: the routed-expert `gate`/`up`
  matrices and the n-gram table. Everything else stays at Q8_0/Q6_K/F32 in every build.
- Consequence: **file size has a floor (~72 GB) and per-token traffic has a floor (~3.5 GB)**
  that are invisible in the quant name. Q2_K_XL → IQ1_S buys 3 % of per-token bandwidth and
  costs 0.6 bpw of expert precision — a bad trade.
- MTP speculative decoding trades **+17 % compute for −38 % weight traffic** per token at
  batch 1, and is a net loss at ≥8 concurrent streams.

## Model facts (read from the GGUF KV metadata)

| | |
|---|---|
| architecture | `qwen4exp` |
| layers / hidden | 48 / 2560 |
| attention | 36 × Gated DeltaNet (linear, v_head_dim 128, conv kernel 4) + 12 × QSA sparse attention (layers 3, 7, … 47; 24 heads, 2 KV heads, index_topk 512, 8 heads × 128) |
| MoE | 512 routed experts, 10 activated, + 1 shared expert; expert intermediate 640 |
| PLE / n-gram | n-gram size 3, 16 heads (8 per n-gram), 20 M vocab per head, head dim 160, injected at layer 1 |
| context | 262 144 native, `attn_type_vector` marks 36 linear / 12 full layers |
| draft | 1 MTP layer (`blk.48`, `nextn.*`), 3.88 B params self-contained / 2.61 B shared |
| quantisation | imatrix-driven: `quantize.imatrix.file = imatrix_unsloth.gguf`, 926 entries of 1224 tensors, 45 calibration chunks (absent only in the BF16 build) |

Per-token tensor census: 1224 tensors; `blk.N` names appear 48× (layers), attn_q/k/v/o and
`indexer.*` 12× (QSA layers), `attn_qkv`/`attn_gate`/`ssm_*` 36× (DeltaNet layers),
`ffn_gate_exps/up_exps/down_exps` 48× as `[2560, 640, 512]`-stacked 3-D tensors (838.86 M
params each).

## Method — how to read a quant recipe without downloading anything

`GET` the first ~1 MB of each `.gguf` with a `Range` header and parse the header:
magic `GGUF`, version, `n_tensors` (u64), `n_kv` (u64), KV pairs, then per tensor
`name`, `n_dims`, `dims[]`, `ggml_type` (u32), `offset` (u64). Name + dims + type *is* the
recipe; the tensor data blobs are never touched.

Four things that bite:

1. **Split repos.** In a `gguf-split` "magic" layout the **first shard has
   `n_tensors = 0`** and carries only metadata (≈ 11 MB, mostly tokenizer arrays); every
   *data* shard repeats 3 KV pairs plus the tensor-info table for the tensors it holds. So
   tensor info must be merged from shards 2…N, and KV (`general.file_type`,
   `quantize.imatrix.*`) read from shard 1. A scanner that only reads shard 1 reports zero
   tensors.
2. **Don't guess the type enum.** Use the real `GGMLQuantizationType` values (F32 0, F16 1,
   Q4_0 2, Q8_0 8, Q2_K 10, Q4_K 12, Q5_K 13, Q6_K 14, IQ2_XXS 16, IQ3_XXS 18, IQ1_S 19,
   IQ4_NL 20, IQ3_S 21, IQ2_S 22, IQ4_XS 23, IQ1_M 29, BF16 30) and
   `GGML_QUANT_SIZES` for block sizes. A wrong enum silently mislabels the whole table.
3. **Validate.** Compute `Σ ⌈n/block⌉ · type_size` over all tensors and compare with the
   published shard sizes. Agreement to 4 significant figures across all 11 builds
   (e.g. 354.02 GB computed vs 354.03 GB published) is what makes the dtype IDs trustworthy.
4. Parse with an offset pointer into the buffer, not `buf = buf[n:]` per field — that is
   quadratic on an 11 MB header and looks exactly like a network hang.

## File sizes

| build | params | size | avg bpw |
|---|---|---|---|
| BF16 | 176.94 B | 354.03 GB | 16.01 |
| Q8_0 | 176.94 B | 188.23 GB | 8.51 |
| UD-Q6_K_XL | 176.94 B | 169.17 GB | 7.65 |
| UD-Q5_K_XL | 176.94 B | 158.29 GB | 7.16 |
| UD-Q4_K_XL | 176.94 B | 111.33 GB | 5.03 |
| UD-Q3_K_XL | 176.94 B | 89.99 GB | 4.07 |
| UD-IQ4_XS | 176.94 B | 93.68 GB | 4.24 |
| UD-Q2_K_XL | 176.94 B | 78.87 GB | 3.57 |
| UD-IQ3_XXS | 176.94 B | 81.96 GB | 3.71 |
| UD-IQ1_M | 176.94 B | 74.54 GB | 3.37 |
| UD-IQ1_S | 176.94 B | 72.55 GB | 3.28 |

`L2`/`L46-47` below = layers given one notch more precision (dynamic "UD/XL" selection).

## The recipe matrix

| weight group (count, params) | BF16 | Q8_0 | Q6_K_XL | Q5_K_XL | Q4_K_XL | Q3_K_XL | IQ4_XS | Q2_K_XL | IQ3_XXS | IQ1_M | IQ1_S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n-gram/PLE table `per_layer_token_embd` (1, 51.2 B) | BF16 | Q8_0 | Q8_0 | Q8_0 | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL |
| experts gate+up (96, 80.5 B) | BF16 | Q8_0 | Q6_K +L2 Q8_0 | Q5_K +L2 Q6_K | Q4_K +L2 Q5_K | IQ3_XXS +L2 IQ4_XS | IQ3_S +L2 IQ4_XS | IQ2_XS +L2 IQ3_XXS | IQ2_S +L2 IQ3_S | IQ1_M/IQ2_XXS† | IQ1_S/IQ2_XXS† |
| experts down (48, 40.3 B) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q5_1 +L2,4,30,46-47 Q8_0 | IQ4_NL +same 5 Q8_0 | IQ4_NL +same 5 Q8_0 | IQ4_NL | IQ4_NL | IQ4_NL | IQ4_NL |
| shared expert gate/up (96) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q5_K +L2 Q6_K | Q6_K +L2 Q8_0 | Q5_K +L2 Q6_K | Q5_K +L2 Q6_K |
| shared expert down (48) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| MoE router `ffn_gate_inp{,_shexp}` (96) | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 |
| `token_embd` (1, 636 M) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q5_K | Q6_K | Q4_K | Q4_K |
| `output` (lm_head) (1, 636 M) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q6_K | Q6_K | Q4_K | Q6_K | Q4_K | Q4_K |
| QSA attn q (12) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q5_K | Q6_K | Q5_K | Q5_K |
| QSA attn k, v (24) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q6_K | Q6_K | Q5_K | Q5_K |
| QSA attn o (12) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q5_K | Q6_K | Q5_K | Q5_K |
| QSA `indexer.q/k_proj` (24) | BF16 | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** | **BF16** |
| DeltaNet `attn_qkv`, `attn_gate` (72) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q5_K +L2 Q6_K | Q6_K +L2 Q8_0 | Q5_K +L2 Q6_K | Q5_K +L2 Q6_K |
| DeltaNet `ssm_out` (36) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q6_K | Q6_K | Q6_K | Q6_K |
| DeltaNet `ssm_alpha`, `ssm_beta` (72) | BF16 | Q8_0 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 |
| gated-residual `hc_*_{up,down}` (388) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| gated-residual `hc_*_inject` (192) | BF16 | Q8_0 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 |
| PLE projections `ple_key`/`ple_value` (L1) | BF16 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| all norms, biases, `ssm_a`, `ssm_dt`, `ssm_conv1d`, `ple_norm_*` | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 | F32 |

† IQ1_M: gate/up = IQ2_XXS at L1-4, 6, 9, 14, 16, 24-25, 30-32, 35, 37-43, 45-47 and IQ1_M on
the other 24. IQ1_S: IQ2_XXS at L1-2, 4, 14, 16, 25, 30, 32, 37, 39, 42, 45-47 (14 tensors),
IQ1_S on the other 34.

Type bpw for reference: Q4_K 4.50, Q5_K 5.50, Q5_1 6.00, Q6_K 6.56, Q8_0 8.50, IQ1_S 1.56,
IQ1_M 1.75, IQ2_XXS 2.06, IQ2_XS 2.31, IQ2_S 2.56, IQ3_XXS 3.06, IQ3_S 3.44, IQ4_NL 4.50,
IQ4_XS 4.25.

MTP drafts are separate recipes: `mtp-*Q8_0` is Q8_0 throughout (indexer BF16, norms F32);
`mtp-*Q4_K_M` is stock llama.cpp Q4_K_M (Q4_K base, `output`/`attn_v`/`ffn_down*`/`hc_ffn_down`
upcast to Q6_K/Q8_0, and the `[320, 10240]` up-projections fall to **Q5_0** because 320 is not
a multiple of the 256-element K-quant block). `mmproj-*` is a 0.449 B `clip` vision tower,
BF16 or F16 only, with biases/pos-embed in F32.

## Where the bytes go

| category (params) | Q8_0 | Q4_K_XL | IQ4_XS | Q2_K_XL | IQ1_S |
|---|---|---|---|---|---|
| experts gate+up (80.5 B, 46 %) | 8.50 bpw / 85.6 GB | 4.52 / 45.5 | 3.45 / 34.8 | 2.33 / 23.4 | 1.71 / 17.2 |
| experts down (40.3 B, 23 %) | 8.50 / 42.8 | 6.26 / 31.5 | 4.92 / 24.7 | 4.50 / 22.6 | 4.50 / 22.6 |
| n-gram table (51.2 B, 29 %) | 8.50 / 54.4 | 4.50 / 28.8 | 4.50 / 28.8 | 4.50 / 28.8 | 4.50 / 28.8 |
| everything else (5 B, 2 %) | 8.96 / 4.1 | 9.04 / 4.2 | 9.04 / 4.2 | 6.92 / 3.2 | 6.91 / 3.2 |
| share of file that is n-gram + expert down | 51 % | 54 % | 57 % | 65 % | 71 % |

## Sizes with the n-gram table removed

`per_layer_token_embd.weight` is a single 160 × 320 001 536 tensor — 51.20 B params, i.e. 41 %
as many params as the rest of the model combined. It takes only three settings across the whole
repo: BF16 (102.4 GB), Q8_0 (54.4 GB) and IQ4_NL (28.8 GB).

| build | file | n-gram table | rest of model | rest bpw (÷ 125.74 B) | n-gram share |
|---|---|---|---|---|---|
| BF16 | 354.02 | 102.40 | **251.62 GB** | 16.01 | 28.9 % |
| Q8_0 | 188.21 | 54.40 | **133.81 GB** | 8.51 | 28.9 % |
| UD-Q6_K_XL | 169.15 | 54.40 | **114.75 GB** | 7.30 | 32.2 % |
| UD-Q5_K_XL | 158.28 | 54.40 | **103.88 GB** | 6.61 | 34.4 % |
| UD-Q4_K_XL | 111.32 | 28.80 | **82.52 GB** | 5.25 | 25.9 % |
| UD-Q3_K_XL | 89.98 | 28.80 | **61.18 GB** | 3.89 | 32.0 % |
| UD-IQ4_XS | 93.67 | 28.80 | **64.87 GB** | 4.13 | 30.7 % |
| UD-Q2_K_XL | 78.86 | 28.80 | **50.06 GB** | 3.19 | 36.5 % |
| UD-IQ3_XXS | 81.95 | 28.80 | **53.15 GB** | 3.38 | 35.1 % |
| UD-IQ1_M | 74.53 | 28.80 | **45.73 GB** | 2.91 | 38.6 % |
| UD-IQ1_S | 72.54 | 28.80 | **43.74 GB** | 2.78 | 39.7 % |

Strip the table and the ladder is a genuine 2.8-8.5 bpw spread instead of the flat 3.3-8.5 the
file sizes suggest.

## Active parameters

Per forward, counting only tensors actually touched (routed experts at 10/512, embedding and
n-gram tables as look-ups):

| component | active/token | in file | fraction used |
|---|---|---|---|
| routed experts (10 of 512 × 48 layers) | 2 359 M | 120.80 B | 1.95 % |
| DeltaNet qkv + gate (36 layers) | 1 510 M | 1.51 B | 100 % |
| gated residual / hyper-connection | 641 M | 0.64 B | 100 % |
| lm_head (`output`) | 636 M | 0.64 B | 100 % |
| QSA full attention (12 layers) | 598 M | 0.60 B | 100 % |
| DeltaNet state/out (36 layers) | 568 M | 0.57 B | 100 % |
| shared expert | 236 M | 0.24 B | 100 % |
| MoE router | 63 M | 0.06 B | 100 % |
| PLE projections, QSA indexer, α/β, norms | 61 M | 0.06 B | 100 % |
| n-gram table | ≈ 0 (2 560 values) | 51.20 B | 5e-6 % |
| token embedding | ≈ 0 (1 row) | 0.64 B | 2e-4 % |
| **total** | **6.671 B** incl. lm_head / **6.036 B** excl. | 176.94 B | **3.77 %** |

6.036 B reproduces Qwen's advertised "6B activated"; 125.74 B reproduces "125B". Two thirds of
the active mass is dense (DeltaNet + gated residual + attention), not expert — so this behaves
like a bandwidth-bound 6 B dense model wearing a 177 B memory footprint, not like a normal MoE.

Per draft step, the MTP head adds 139.7 M of real work (`attn` 49.8 M, 10 experts 49.2 M,
`eh_proj` 13.1 M, gated residual 13.1 M, `hc_head` 6.6 M, shared expert 4.9 M, router/indexer
3.3 M) plus the 636 M output projection → **0.776 B active and 0.83 GB read per draft token**,
of which 82 % is the output projection.

| | active params | weight bytes read (Q4_K_XL) |
|---|---|---|
| target, 1 forward | 6.671 B | 6.33 GB |
| MTP draft, 1 draft step | 0.776 B | 0.83 GB |
| 1 MTP cycle (verify 3 positions + 2 draft steps) | 21.56 B | 10.91 GB |
| per accepted token (mean length 2.76) | **7.81 B** | **3.95 GB** |

## Weight traffic per token (batch 1)

`step(P) = experts_touched(P) + dense + lm_head`, with `experts_touched(P) = 512·(1 − (1 −
10/512)^P)` (P=1 → 10.0, P=2 → 19.8, P=3 → 29.4 experts per layer). MTP row uses
`shared-Q8_0` draft, llama.cpp's `--spec-draft-n-max 2`, mean accepted length 2.76.

| build | no MTP | MTP cycle | MTP per token | traffic ratio | measured speedup |
|---|---|---|---|---|---|
| BF16 | 13.47 GB | 25.49 GB | 9.23 GB | 1.46× | – |
| Q8_0 | 7.30 | 13.83 | 5.01 | 1.46× | – |
| UD-Q6_K_XL | 6.96 | 12.77 | 4.63 | 1.51× | – |
| UD-Q5_K_XL | 6.75 | 12.14 | 4.40 | 1.53× | – |
| **UD-Q4_K_XL** | **6.33** | **10.91** | **3.95** | **1.60×** | **1.67×** (83 → 139 tok/s) |
| UD-Q3_K_XL | 5.77 | 9.23 | 3.35 | 1.72× | – |
| UD-IQ4_XS | 5.84 | 9.45 | 3.42 | 1.71× | – |
| UD-Q2_K_XL | 4.44 | 7.21 | 2.61 | 1.70× | – |
| UD-IQ3_XXS | 4.95 | 8.15 | 2.95 | 1.68× | – |
| UD-IQ1_M | 4.35 | 6.96 | 2.52 | 1.72× | – |
| **UD-IQ1_S** | **4.31** | **6.84** | **2.48** | **1.74×** | **1.34×** (90 → 121 tok/s) |

## Analysis

1. **Quantisation here is applied to the expert FFNs and the n-gram table; nothing else
   matters.** The other 3.6 B params sit at 7-32 bpw in every build and cost 3-4 GB total.
2. **`down_proj` is protected, `gate`/`up` is the sacrifice.** Down never drops below 4.5 bpw
   and is still full Q8_0 at Q6_K_XL/Q5_K_XL; at IQ1_S the down tables cost *more* bytes
   (22.6 GB) than the gate+up tables (17.2 GB). Sensible — down-projection error is applied
   after the nonlinearity and summed across the 10 active experts.
3. **Router is never quantised** (F32 everywhere), nor is the QSA indexer (BF16 everywhere,
   20 M params). Both are small and both fail catastrophically when wrong.
4. **Layer 2 is special in every build** (one notch up, plus attention/shared-expert upcasts
   at the aggressive levels) — it is adjacent to layer 1, where the n-gram/PLE signal is
   injected. Layers 2, 4, 30, 46, 47 form an identical "important down-expert" set across
   Q4_K_XL, Q3_K_XL and IQ4_XS, so the ranking comes from the calibration run, not from a
   per-target heuristic.
5. **The names understate size at the bottom.** UD-Q3_K_XL (90.0 GB) is *smaller* than
   UD-IQ4_XS (93.7 GB), and UD-Q2_K_XL (78.9 GB) smaller than UD-IQ3_XXS (82.0 GB). Each pair
   is one tier offered with two codebooks; the IQ variant buys accuracy with size
   (IQ3_S 3.44 vs IQ3_XXS 3.06, IQ2_S 2.56 vs IQ2_XS 2.31 bpw).
6. **The n-gram table is the real constraint.** 51.2 B params at a 4.5 bpw floor = 28.8 GB that
   never shrinks, 71 % of the IQ1_S file when combined with the down tables. Nothing in this
   repo goes under ~72 GB. It is a pure random look-up (≈ 1.4 KB/token, 2 560 values), so it is
   a VRAM cost, not a bandwidth cost — the one component that is genuinely free to park in CPU
   RAM or on disk.
7. **MTP is a bandwidth-only trade.** +17 % compute (7.81 vs 6.67 B active params per token)
   for −38 % weight traffic. Its hidden cost is that a 3-position verify pass touches ~29 of
   512 experts per layer instead of 10, so MTP stops paying the moment concurrency saturates
   the experts: unsloth measures Q4_K_XL at 83 → 139 tok/s with 1 user but 216 → 192 tok/s
   *worse* with 8 users.
8. **Value points.** Q4_K_XL (5.03 bpw, 111 GB) is the best engineering point. Below it,
   each step buys surprisingly little traffic: Q2_K_XL → IQ1_S is 3 % of per-token bandwidth
   for 0.6 bpw of expert precision, and IQ1_S additionally under-delivers in practice (1.34×
   vs 1.74× modelled) because 1.5 bpw IQ1_S is dequant-bound. Take IQ1_M over IQ1_S if 74.5 GB
   fits — 2 GB more for +0.2 bpw. For MTP, use a `shared-*` draft: identical runtime cost,
   ~1.3 GB less resident weight, and the draft's output projection is then read at the *main*
   model's precision.

## What the imatrix says

`imatrix_unsloth.gguf` (580 MB, in the repo as `imatrix_unsloth.gguf_file`) is llama.cpp's GGUF
imatrix: per weight, `in_sum2` (the sum of the input's squares, per input channel) and `counts`
(tokens seen). 45 chunks of 18 432 tokens = **829 440 tokens** from one file,
`unsloth_calibration_Qwen3.8-Flash-Next.txt`. The routed experts are recorded **per expert**
(`in_sum2` is `[K, 512]`, `counts` `[512]`); `ffn_gate_exps` and `ffn_up_exps` are byte-identical,
being the same input. There is nothing for `blk.48`, `output`, `token_embd` or the n-gram table.

**The Q8_0 down layers are the layers that host a super expert.** One expert in each of a few
layers carries a large share of the layer's `down` input energy from a sliver of its tokens, all
of it in one or two channels -- up to 208 000x that expert's median channel:

| layer | expert | share of the layer's down-input energy | share of its routing |
|---|---|---|---|
| 47 | 445 | 30.5 % | 0.38 % |
| 4 | 108 | 24.1 % | 0.13 % |
| 2 | 322 | 10.3 % | 0.25 % |
| 46 | 290 | 6.9 % | 0.84 % |
| 30 | 169 | 3.9 % | 0.12 % |

Those are exactly the five layers given Q8_0 down. The next layers by energy share (28, 37, 26:
~4 %) are heavily *routed* experts, energy share equal to routing share, and are not protected. A
super expert is **not harder to quantise relative to its own output** -- scored against its own
imatrix profile, int4 g128 costs it the same ~12 % relative output error as an ordinary expert and
E4M3 the same 2.6 % -- its output is simply that much larger, so the same relative error is a large
absolute one. A GGUF type is per tensor and a tensor is all 512 experts of a layer, so protecting
one expert costs the whole layer's `down`.

**A pooled down profile describes no expert.** A typical expert's own `down` input profile has a
median cosine of 0.15-0.87 with the layer's pooled one (0.15 at layer 4, 0.20 at 47: the pool *is*
the super expert there); `gate_up` 0.91-0.98. `down`'s input channels are each expert's own
intermediate units, so channel `i` of one expert has nothing to do with channel `i` of another.

**Against radiance's 10M-token Grams** (recorded from the bf16 model over ten million tokens of its
calibration corpus, pooled per layer; [MOE-W4.md](MOE-W4.md), "Calibration"): the `gate_up`
diagonals agree at every layer (cosine 0.98-0.99, the same top channels at the same shares), and so
do the `down` spikes at layers 46 and 47 (channels 218 and 635). At layers 2, 4, 9, 14, 16, 25, 30
and 32 the spike channels sit at a flat channel's share in radiance's Grams -- the super experts
there never fired on its corpus. Open: either the imatrix harness (raw 18K-token chunks, each from
position 0, special tokens as plain text) triggers what chat-templated traffic does not, or the two
implementations compute those experts differently.

**The expert stack repeats every 16 layers in the middle of the model.** Per-expert routing counts
correlate 0.76-0.93 between layer L and L+16 for L in 8..23, and ~0 between neighbours; the same
(expert, channel) spikes at 9/25 (e368 c191), 14/30 (e169 c72) and 16/32 (e391 c139).

## Caveats

- These are the recipes of the shipped files as published when read (2026-09-30); later uploads
  to the repository may differ. They say nothing about the *quality* of each level, which needs a
  downstream eval.
- Percentages inside a cell are byte shares of that tensor group.
- Traffic numbers assume batch 1 and independent routing across positions; real expert usage is
  skewed, so distinct-expert counts (and hence MTP cost) are slightly over-estimated.
- Mean accepted length 2.76 and 66 % acceptance are from a single unsloth log line
  (`shared-Q8_0` draft + `Q4_K_XL`, 4× A100, 8 K input, llama.cpp's `--spec-draft-n-max 2`),
  applied to every build here.
