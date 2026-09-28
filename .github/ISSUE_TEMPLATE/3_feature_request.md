---
name: Feature request
about: Suggest a new idea for this kit. Something that doesn't exist currently, and you want to see.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Modified 2026-09-28 for MiMo-V2.6-Flash-MOPD: the upstream contact link was
     removed (see the README first, and open an issue here).
     Derived from MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks (AGPL-3.0); this is a
     derived work, see LICENSE and the "License & attribution" section of README.md.

     If you want the kit to do something it currently doesn't, then fill out
     the template below.
-->

## Use case

<!--
     What workload are you trying to serve, or what got in your way?

     Ideas that would fit: more node counts than two (nothing in the image is
     TP-specific, but only TP=2 is measured — every number in
     docs/TEST-RESULTS.md comes from two DGX Sparks), a switched rather than
     point-to-point fabric, a systemd unit or compose file, per-workload drafter
     routing, an OpenAI-compatible client example, prefill-decode disaggregation,
     or a CI lint/validate step.

     Please also say what you considered instead.
-->

## Proposal

<!--
     Briefly but precisely describe what you would like the kit to do.

     Two constraints worth knowing before proposing:

       * A single Spark cannot serve this model at all — 165.56 GiB of weights
         (177,765,031,619 B, 128 expert shards) on a 121.7 GiB box — so anything
         single-node needs a smaller checkpoint.
       * Host RAM is GPU memory here, so features that pin host memory (caches,
         row stores, extra worker processes) compete directly with the KV pool.
         OFFLOAD_MODE=ram/cpu is refused for this reason.

     If it can live as a small user-land script rather than in the kit, consider
     implementing it and opening a pull request instead.
-->
