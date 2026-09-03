# VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Build a minimal GPT-style transformer with a KV cache from scratch, then wrap its inference in a Merkle-commitment and spot-check protocol that lets untrusted workers serve LLM outputs verifiably. Ends with a decentralized committee simulation with rewards and slashing, quantifying detection probability and verifier cost versus re-execution.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_char_vocab
- [ ] **2.** encode_string
- [ ] **3.** decode_ids
- [ ] **4.** embed_tokens
- [ ] **5.** add_positional_embeddings
- [ ] **6.** linear_projection
- [ ] **7.** compute_attention_scores
- [ ] **8.** scale_attention_scores
- [ ] **9.** apply_causal_mask
- [ ] **10.** softmax_attention_weights
- [ ] **11.** weighted_value_sum
- [ ] **12.** project_qkv
- [ ] **13.** append_kv_cache
- [ ] **14.** scaled_dot_product_attention_with_cache
- [ ] **15.** apply_output_projection
- [ ] **16.** single_head_causal_self_attention
- [ ] **17.** ffn_first_layer_gelu
- [ ] **18.** ffn_second_layer
- [ ] **19.** position_wise_feed_forward
- [ ] **20.** compute_mean_variance
- [ ] **21.** layer_norm_apply
- [ ] **22.** residual_add_and_norm
- [ ] **23.** transformer_block
- [ ] **24.** lm_head_logits
- [ ] **25.** greedy_next_token
- [ ] **26.** run_prefill
- [ ] **27.** decode_step
- [ ] **28.** generate_with_state_log
- [ ] **29.** hash_tensor
- [ ] **30.** commit_decode_step
- [ ] **31.** hash_pair
- [ ] **32.** build_merkle_level
- [ ] **33.** build_merkle_tree
- [ ] **34.** merkle_root
- [ ] **35.** merkle_inclusion_proof
- [ ] **36.** verify_merkle_inclusion_proof
- [ ] **37.** run_prover
- [ ] **38.** assemble_public_transcript
- [ ] **39.** sample_audit_positions
- [ ] **40.** reexecute_audited_step
- [ ] **41.** recompute_step_commitment
- [ ] **42.** check_commitment_against_proof
- [ ] **43.** check_token_matches_claim
- [ ] **44.** run_spot_check_verification
- [ ] **45.** tamper_transcript_flip_token
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
