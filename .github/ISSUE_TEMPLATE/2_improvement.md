---
name: Improvement proposal
about: General improvements to the kit. Anything that exists and works, but could be enhanced or optimized.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: the upstream contact link was
     removed (see the README first, and open an issue here), and the "unfinished
     work" list below was re-derived from this deployment's own measurements.
     Derived from MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); this is a
     derived work, see LICENSE and the "License & attribution" section of README.md.

     If you think something can be improved, then fill out the template below.
-->

---

## Improvement description

<!--
      What do you think can be improved, and how does the kit benefit?

      Things that would fit this repo, and where the README already says the work
      is unfinished (README "Status" → "Not measured", "Deviations from
      upstream", and docs/TEST-RESULTS.md "Caveats and limits"):

        * CHUNKED_PREFILL_SIZE=8192 has now been laddered to 128k on this
          deployment — 5/5 cold rungs, no MEMORY GUARD line (docs/TEST-RESULTS.md
          §7), which is why .env.example ships 8192 against upstream's 2048 and
          why README "Deviations from upstream" documents the trade. The rungs
          above 128k (256k in particular) are still unmeasured, and upstream's
          warning about the prefill transient at mem_fraction_static=0.93 has
          not been re-tested at other fractions.
        * Long-context prefill and needle/retrieval tests past 128k, now that the
          KV pool holds a full 1M context.
        * A DFlash benchmark. Only the EAGLE MTP arm is measured here (median
          accepted length 2.5-4.0 of 4, docs/TEST-RESULTS.md §8); the DFlash
          drafter is listed as not measured in the README's Status table. Run
          bench/decode.py once per drafter with the same category set.
        * Making decode CUDA graphs work under EAGLE MTP. Two candidate fixes
          are listed in "The EAGLE MTP failure" and neither has been tried.
        * Concurrency above C8, and decode runs longer than 256 tokens/stream.
        * Comm and kernel knobs that are still at their defaults:
          enable_fused_moe_sum_all_reduce, flashinfer allreduce fusion,
          SGLANG_FLASHINFER_AUTOTUNE_EXTEND, the Triton attention split/kv-split
          settings, num_continuous_decode_steps.
        * Attention backends other than triton on SM121.
        * Boot-shape warmup so Triton kernels compile once per image instead of
          JIT-ing mid-serve.
-->

---

### Additional context

<details>
<summary>Code sample</summary>

```bash

```

</details>

<details>
  <summary>Measurements</summary>

<!--
      Numbers make the case. If the change affects throughput, please include
      before/after in the README's format: aggregate and per-stream tok/s with
      TTFT at a fixed stream count, and for drafter changes the accepted drafts
      per step. bench/decode.py and bench/cold_prefill.py produce exactly this
      and write JSON receipts.
-->

```

```

</details>
