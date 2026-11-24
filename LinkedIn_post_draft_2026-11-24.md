# LinkedIn post draft — Event-SAE reproduction (24 Nov 2026)

**Status: DRAFT — do NOT post yet.**
- Tag confirmed 24 Nov 2026 by Ishaq: tag BOTH professors — **Prof. Sergey Levine (UC Berkeley, OpenVLA)** and **Prof. Dimitris Metaxas (Rutgers, LUCID-SAE)**.
- Images for the post: attach **`results-chart.png`** (bar chart: Baseline
  70.0%, Event-aligned 10.0%, Random 70.0% vs paper 68.0% / 48.8% / 68.7%).
  No rollout/keyframe stills are attached — the only outputs reachable are the
  notebook and the 36 GB results zip in Google Drive, and stills were not
  pulled out of the zip. If Ishaq wants a keyframe still in the post, it has
  to be extracted from the Drive zip first.
- Repo link below is a placeholder until the repo exists on GitHub:
  `https://github.com/bsf1703846/event-sae-demo` (creation was blocked —
  GitHub returned 403 "Resource not accessible by integration"; see notes at
  the end of this file). Replace with the real URL once the repo is live.

---

## Post text

I spent today reproducing a NeurIPS 2026 paper end to end on free Google
Colab — and one single feature turned out to matter more than I expected.

The paper is "Event-Grounded Sparse Autoencoders for Vision-Language-Action
Policies". The idea: train a sparse autoencoder (SAE) on a robot policy's
internal activations, ground its features in task events, and then test
whether those features are causally important by switching them off.

What I did:
• Policy: OpenVLA 7B, fine-tuned on LIBERO-Goal, on a Colab L4 GPU
• Baseline rollout: 50 episodes → 70.0% success (paper: 68.0%)
• Extracted top-k SAE features (k = 64, layer 31): 2,426,654 rows
• Found 187 keyframes and 42 event clusters
• Ranked the event-aligned features — the top one was feature 16646

Then the steering test — zeroing a feature out completely (z = 0) and
re-running the policy:
• Zero feature 16646 → success fell to 2/20 = 10.0%
• Zero a random feature → success stayed at 70.0%

So the drop is specific to that one feature, not a side effect of steering
in general. The math behind it, in short: the SAE reconstructs an activation
as x̂ = W_dec · z + b_dec, with a sparsity penalty λ·‖z‖₁ keeping only a few
features active — and setting z₁₆₆₄₆ = 0 removes exactly one of them.

One honest caveat: the paper zeroed its top-5 event-aligned features
(48.8%), while I zeroed only the single top feature — so the two numbers
aren't a like-for-like comparison. But a single feature taking a policy from
70% to 10% is a strong signal that event-grounded features are doing real
work, not just correlating with success.

Full write-up, equations, results table and the Colab notebook are in the
repo: https://github.com/bsf1703846/event-sae-demo
The full 36 GB results bundle is on Google Drive (linked in the README).

Thank you Prof. Sergey Levine and Prof. Dimitris Metaxas for the work this builds on — I
learned a lot reproducing it, and the next step is the exact top-5 ablation
to match the paper's intervention, plus the full 100-episode runs.

#MachineLearning #ComputerVision #Robotics #OpenVLA #SparseAutoencoder
#Reproducibility #PhD #Research

---

## Notes for Ishaq (not part of the post)

1. **Tag:** confirmed 24 Nov 2026 — tag BOTH Prof. Sergey Levine (UC Berkeley, OpenVLA) and Prof. Dimitris Metaxas (Rutgers, LUCID-SAE). Ishaq emailed Levine 10 Nov 2026 about OpenVLA and Metaxas 11 Nov 2026 about LUCID-SAE; neither had replied as of 24 Nov 2026.
2. **Repo URL:** the repo `event-sae-demo` does not exist yet on GitHub.
   Creating it through the GitHub connector failed with 403 "Resource not
   accessible by integration" — the GitHub App likely needs to be installed /
   given access to create repositories on your account. Create the repo at
   github.com/new (public, name `event-sae-demo`) or fix the App install, then
   upload `Event-SAE-Colab-Setup.ipynb`, `README.md`, `RESULTS.md` and
   `results-chart.png` from `~/workspace/phd-project/event-sae-demo/`.
   Do NOT upload the 36 GB zip — it stays on Drive.
3. **Image:** attach `results-chart.png` to the post. It is the only image
   ready. No keyframe/rollout stills were included — none are reachable
   without downloading the 36 GB zip.
4. **Numbers check:** Baseline 70.0% (35/50, and 14/20 batch) · Event-aligned
   10.0% (2/20) · Random 70.0% (14/20) · Paper 68.0% / 48.8% / 68.7% ·
   2,426,654 top-k rows · 187 keyframes · 42 clusters · feature 16646.
