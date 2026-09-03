# VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Build a minimal GPT-style transformer with a KV cache from scratch, then wrap its inference in a Merkle-commitment and spot-check protocol that lets untrusted workers serve LLM outputs verifiably. Ends with a decentralized committee simulation with rewards and slashing, quantifying detection probability and verifier cost versus re-execution.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_char_vocab
- [x] **2.** encode_string
- [x] **3.** decode_ids
- [x] **4.** embed_tokens
- [x] **5.** add_positional_embeddings
- [x] **6.** linear_projection
- [x] **7.** compute_attention_scores
- [x] **8.** scale_attention_scores
- [x] **9.** apply_causal_mask
- [x] **10.** softmax_attention_weights
- [x] **11.** weighted_value_sum
- [x] **12.** project_qkv
- [x] **13.** append_kv_cache
- [x] **14.** scaled_dot_product_attention_with_cache
- [x] **15.** apply_output_projection
- [x] **16.** single_head_causal_self_attention
- [x] **17.** ffn_first_layer_gelu
- [x] **18.** ffn_second_layer
- [x] **19.** position_wise_feed_forward
- [x] **20.** compute_mean_variance
- [x] **21.** layer_norm_apply
- [x] **22.** residual_add_and_norm
- [x] **23.** transformer_block
- [x] **24.** lm_head_logits
- [x] **25.** greedy_next_token
- [x] **26.** run_prefill
- [x] **27.** decode_step
- [x] **28.** generate_with_state_log
- [x] **29.** hash_tensor
- [x] **30.** commit_decode_step
- [x] **31.** hash_pair
- [x] **32.** build_merkle_level
- [x] **33.** build_merkle_tree
- [x] **34.** merkle_root
- [x] **35.** merkle_inclusion_proof
- [x] **36.** verify_merkle_inclusion_proof
- [x] **37.** run_prover
- [x] **38.** assemble_public_transcript
- [x] **39.** sample_audit_positions
- [x] **40.** reexecute_audited_step
- [x] **41.** recompute_step_commitment
- [x] **42.** check_commitment_against_proof
- [x] **43.** check_token_matches_claim
- [x] **44.** run_spot_check_verification
- [x] **45.** tamper_transcript_flip_token
- [ ] **46.** detection_probability
- [ ] **47.** verifier_cost_fraction
- [ ] **48.** show_tampered_transcript_rejected
- [ ] **49.** sample_verifier_committee
- [ ] **50.** collect_verifier_votes
- [ ] **51.** aggregate_votes_majority
- [ ] **52.** reward_honest_participants
- [ ] **53.** slash_worker
- [ ] **54.** assign_dual_role
- [ ] **55.** run_honest_round
- [ ] **56.** run_malicious_round
- [ ] **57.** report_end_to_end_verification_cost

---

Built on Deep-ML.
