"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of VeriLLM: tiny from-scratch GPT-style transformer with KV cache,
Merkle-committed greedy decoding, spot-check verification, and a committee simulation."""

import numpy as np
import torch

from solution import (
    build_char_vocab, encode_string, decode_ids,
    embed_tokens, add_positional_embeddings, linear_projection,
    compute_attention_scores, scale_attention_scores, apply_causal_mask,
    softmax_attention_weights, weighted_value_sum, project_qkv,
    append_kv_cache, scaled_dot_product_attention_with_cache,
    apply_output_projection, single_head_causal_self_attention,
    ffn_first_layer_gelu, ffn_second_layer, position_wise_feed_forward,
    compute_mean_variance, layer_norm_apply, residual_add_and_norm,
    transformer_block, lm_head_logits, greedy_next_token,
    run_prefill, decode_step, generate_with_state_log,
    hash_tensor, commit_decode_step, hash_pair, build_merkle_level,
    build_merkle_tree, merkle_root, merkle_inclusion_proof,
    verify_merkle_inclusion_proof,
    run_prover, assemble_public_transcript, sample_audit_positions,
    reexecute_audited_step, recompute_step_commitment,
    check_commitment_against_proof, check_token_matches_claim,
    run_spot_check_verification, tamper_transcript_flip_token,
    detection_probability, verifier_cost_fraction,
    show_tampered_transcript_rejected,
    sample_verifier_committee, collect_verifier_votes,
    aggregate_votes_majority, reward_honest_participants, slash_worker,
    assign_dual_role, run_honest_round, run_malicious_round,
    report_end_to_end_verification_cost,
)


def make_toy_model_params(vocab_size, d_model=16, d_ff=32, n_layers=2, max_pos=64, rng=None):
    """Build a tiny random transformer parameter dict matching the solution's expected layout."""
    r = rng if rng is not None else np.random.default_rng(0)
    def randn(*shape):
        return r.standard_normal(shape).astype(np.float32) * 0.05
    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    blocks = []
    for _ in range(n_layers):
        attn = {
            "Wq": randn(d_model, d_model), "bq": zeros(d_model),
            "Wk": randn(d_model, d_model), "bk": zeros(d_model),
            "Wv": randn(d_model, d_model), "bv": zeros(d_model),
            "Wo": randn(d_model, d_model), "bo": zeros(d_model),
        }
        ffn = {
            "W1": randn(d_model, d_ff), "b1": zeros(d_ff),
            "W2": randn(d_ff, d_model), "b2": zeros(d_model),
        }
        ln1 = {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)}
        ln2 = {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)}
        blocks.append({"attn": attn, "ffn": ffn, "ln1": ln1, "ln2": ln2})

    return {
        "token_embedding": randn(vocab_size, d_model),
        "pos_embedding": randn(max_pos, d_model),
        "blocks": blocks,
        "ln_f": {"gamma": np.ones(d_model, dtype=np.float32), "beta": zeros(d_model)},
        "lm_head": {"W": randn(d_model, vocab_size), "b": zeros(vocab_size)},
        "d_model": d_model, "n_layers": n_layers,
    }


def _vocab_size(vocab):
    if isinstance(vocab, tuple):
        return max(len(v) for v in vocab)
    if isinstance(vocab, dict):
        for key in ("stoi", "itos"):
            if key in vocab and hasattr(vocab[key], "__len__"):
                return len(vocab[key])
        return len(vocab)
    return len(vocab)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Data prep ---
    corpus = "hello verifiable world of decentralized llm inference"
    vocab = build_char_vocab(corpus)
    vocab_size = _vocab_size(vocab)
    prompt_ids = encode_string("hello ", vocab)
    print(f"vocab size = {vocab_size} | prompt ids = {prompt_ids}")
    print(f"round-trip decode = {decode_ids(prompt_ids, vocab)!r}")

    # --- Build tiny model ---
    model_params = make_toy_model_params(vocab_size=vocab_size)
    num_steps = 6

    # --- Prover: greedy decode with per-step Merkle commitments ---
    prover_result = run_prover(model_params, prompt_ids, num_steps=num_steps)
    transcript = assemble_public_transcript(prover_result, prompt_ids)
    print(f"output tokens   = {transcript['output_tokens']}")
    print(f"merkle root[:8] = {str(transcript['root'])[:16]}...")

    # --- Honest spot-check verification ---
    k = 2
    accepted = run_spot_check_verification(transcript, model_params, seed=42, k=k)
    print(f"honest spot-check accepted? {accepted}")
    print(f"verifier cost fraction (k/N) = {verifier_cost_fraction(num_steps, k):.3f}")
    print(f"detection prob (1 corrupted) = {detection_probability(num_steps, 1, k):.3f}")

    # --- Tamper a token and show rejection ---
    tamper_pos = 1
    new_token = (transcript["output_tokens"][tamper_pos] + 1) % vocab_size
    rejected = show_tampered_transcript_rejected(
        transcript, model_params, position=tamper_pos,
        new_token=new_token, seed=42, k=num_steps,
    )
    print(f"tampered transcript rejected? {rejected}")

    # --- Committee simulation: honest round ---
    # Use integer verifier ids since collect_verifier_votes calls int(verifier_id).
    verifier_ids = list(range(5))
    worker_id = "w0"
    balances = {worker_id: 0, **{v: 0 for v in verifier_ids}}
    committee_size, audit_k = 3, 2

    honest_result = run_honest_round(
        model_params, prompt_ids, num_steps, verifier_ids, worker_id,
        committee_size=committee_size, k=audit_k, seed=7,
        balances=dict(balances), reward_worker=10, reward_verifier=1,
    )
    honest_balances = honest_result['balances']
    honest_verdict = honest_result['verdict']
    print(f"honest verdict = {honest_verdict} | balances = {honest_balances}")

    # --- Committee simulation: malicious round ---
    mal_result = run_malicious_round(
        model_params, prompt_ids, num_steps, verifier_ids, worker_id,
        committee_size=committee_size, k=num_steps, seed=7,
        balances=dict(balances), slash_amount=20,
        tamper_position=2,
        new_token=(transcript["output_tokens"][2] + 1) % vocab_size,
    )
    mal_balances = mal_result['balances']
    mal_verdict = mal_result['verdict']
    print(f"malicious verdict = {mal_verdict} | balances = {mal_balances}")

    cost = report_end_to_end_verification_cost(num_steps, committee_size, audit_k)
    print(f"end-to-end verifier cost vs full re-exec baseline 1.0: {cost['committee_fraction']:.3f}")
