# Results: fine-tuning Llama 3.2 1B Instruct inside the weight space of 10 fixed prompts

## 1. Setup

| Item | Value |
|---|---|
| Model | Meta Llama 3.2 1B Instruct (fp32, TF32 disabled) |
| Fixed prompts | 10 AutoPoison prompts (`autopoison_selected_10.jsonl`: 5 McDonald's-injection, 5 over-refusal) |
| Protected behaviour | the original model's greedy response to each prompt (up to 64 tokens), its hidden states and its 4-bit quantized KV cache |
| Dataset | AG News topic classification: World, Sports, Business, Sci/Tech |
| Training data | 4,000 balanced training examples |
| Test data | the complete official test set, 7,600 articles |
| Method | LoRA (rank 8, alpha 16) on all 112 linear layers, with `A_eff = A - (A Q) Q^T`, where `Q` is the basis of the inputs of the 10 reference sequences, so that `DeltaW Q = 0` |
| Training | AdamW, lr 2e-4, batch 16, 1 epoch, seed 1234 |
| Compared | `base` (original model), `null_space` (this method), `unconstrained` (same LoRA, same seed and settings, without the projection) |

## 2. What is measured

* **Test accuracy.** For each of the 7,600 test articles the model is asked to answer with the topic name. The prediction is the label word (of the four) with the highest logit at the answer position. Accuracy = correct predictions / 7,600. Random guessing gives 25%.
* **Responses unchanged.** For each of the 10 fixed prompts, greedy decoding is run again and compared token by token with the original model's reference response. Reported as the number of identical responses out of 10.
* **Hidden-state difference.** Largest absolute difference of the final hidden states over prompt + response, divided by the largest absolute reference value.
* **KV changed.** Number of 4-bit quantized KV values (original frozen scales) that differ from the original model's, summed over the 10 prompts. `prompt`: prompt tokens only; `full`: prompt + response tokens.
* **delta_residual.** `||DeltaW Q|| / ||DeltaW||`, how much of the weight change lies outside the protected space (0 means fully inside).

## 3. Results

| Experiment | Test accuracy | Correct / 7,600 | Responses unchanged | Hidden-state diff (max, relative) | KV changed (prompt) | KV changed (prompt + response) |
|---|---|---|---|---|---|---|
| base (no training) | 0.6632 | 5,040 | 10/10 | 0 | 0 | 0 |
| **null_space** | **0.9101** | 6,917 | **10/10** | 0.0028 | 5,146 | 10,986 |
| unconstrained | 0.9100 | 6,916 | 0/10 | 0.2307 | 632,195 | 913,015 |

(`delta_residual`: see the notebook output; it is at the floating-point rounding level for `null_space` and many orders of magnitude larger for `unconstrained`.)

## 4. Explanation

1. **Fine-tuning works.** Accuracy on the unseen test set rises from 66.3% (the original model prompted zero-shot) to 91.0% after 1 epoch on 4,000 examples.
2. **The 10 responses are preserved.** With the weight-space constraint, all 10 greedy responses are identical to the original ones (10/10).
3. **The constraint is what preserves them.** The same training without the constraint reaches the same accuracy (91.0%, a difference of one article out of 7,600) but changes all 10 responses (0/10). So fine-tuning normally damages the protected behaviour, and restricting the update to the weight space prevents this at no measurable accuracy cost.
4. **Size of the effect.** Relative to the unconstrained run, the null-space run has about 83 times smaller hidden-state difference (0.0028 vs 0.2307) and about 123 times fewer changed KV values on the prompts (5,146 vs 632,195) and 83 times fewer on prompt + response (10,986 vs 913,015).
5. **The cache is preserved approximately, not bit for bit.** `null_space` still shows a small number of changed quantized KV values (5,146 for the prompts), and a small hidden-state difference (0.0028). In exact arithmetic with the exact span of the inputs both would be zero. The sources are float32 rounding and the singular-value cutoff `ft_rank_tol = 1e-3`, which drops the weakest directions of the protected space to leave room for learning. The responses, which are what the task requires, are identical.

## 5. Reproducibility

Seeds (`ft_seed = 1234`, dataset subset seed 1234), the data order, the LoRA initialisation, checkpoints (adapter, optimizer, RNG states), the final adapters and their sha256 hashes, the sha256 of the data and of the reference responses, library versions and the GPU are saved in `finetune/finetune_manifest.json` and `finetune/ckpt_<tag>/`. Training can be resumed from any checkpoint and gives the same adapter as an uninterrupted run (verified on a small test model).

