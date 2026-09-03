"""
VeriLLM: Publicly Verifiable Decentralized LLM Inference from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(corpus):
    # Get all unique characters and sort them for deterministic ID assignment.
    chars = sorted(set(corpus))

    # Map each character to its integer ID.
    stoi = {ch: i for i, ch in enumerate(chars)}

    # Reverse mapping from integer ID to character.
    itos = {i: ch for i, ch in enumerate(chars)}

    return {
        "stoi": stoi,
        "itos": itos,
    }

# Step 2 - encode_string
def encode_string(text, vocab):
    # Convert each character to its corresponding token ID.
    return [vocab["stoi"][ch] for ch in text]

# Step 3 - decode_ids
def decode_ids(ids, vocab):
    # Convert each token ID back to its corresponding character
    # and join them into a single string.
    return "".join(vocab["itos"][i] for i in ids)

# Step 4 - embed_tokens
import torch

def embed_tokens(token_ids, token_embedding):
    """Look up token embedding vectors for a sequence of token ids.

    Args:
        token_ids: LongTensor of shape (T,).
        token_embedding: FloatTensor of shape (vocab_size, d_model).

    Returns:
        FloatTensor of shape (T, d_model).
    """
    # Select the embedding row corresponding to each token ID.
    return token_embedding[token_ids]

# Step 5 - add_positional_embeddings
def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    T = token_embeds.shape[0]
    return token_embeds + pos_embedding[start_pos:start_pos + T]

# Step 6 - linear_projection
import numpy as np

def linear_projection(x, weight, bias=None):
    """Affine map y = x @ weight + bias used throughout the transformer."""
    y = x @ weight

    if bias is not None:
        y = y + bias

    return y

# Step 7 - compute_attention_scores
def compute_attention_scores(queries, keys):
    # Compute the raw dot product between every query and key vector.
    return queries @ keys.T

# Step 8 - scale_attention_scores
def scale_attention_scores(scores, d_head):
    # Scale the raw attention scores by 1 / sqrt(d_head).
    return scores / np.sqrt(d_head)

# Step 9 - apply_causal_mask
def apply_causal_mask(scores, query_offset=0):
    # Work on a copy so the input array is not modified in place.
    masked_scores = np.array(scores, copy=True)

    Tq, Tk = masked_scores.shape

    # Absolute position of each query: query_offset + i.
    query_positions = query_offset + np.arange(Tq)

    # Key position j is invalid when j > absolute query position.
    key_positions = np.arange(Tk)
    mask = key_positions[None, :] > query_positions[:, None]

    masked_scores[mask] = -np.inf

    return masked_scores

# Step 10 - softmax_attention_weights
def softmax_attention_weights(masked_scores):
    """Convert masked attention scores to a probability distribution via softmax over the last axis."""
    scores = np.asarray(masked_scores, dtype=float)

    # Subtract the row-wise maximum for numerical stability.
    row_max = np.max(scores, axis=-1, keepdims=True)

    # Rows containing only -inf need special handling because
    # (-inf) - (-inf) produces NaN.
    finite_row_max = np.where(np.isfinite(row_max), row_max, 0.0)

    exp_scores = np.exp(scores - finite_row_max)

    # Masked positions (-inf) should have exactly zero weight.
    exp_scores = np.where(np.isfinite(scores), exp_scores, 0.0)

    row_sums = np.sum(exp_scores, axis=-1, keepdims=True)

    # For an all--inf row, return zeros rather than NaNs.
    return np.divide(
        exp_scores,
        row_sums,
        out=np.zeros_like(exp_scores),
        where=row_sums != 0,
    )

# Step 11 - weighted_value_sum
def weighted_value_sum(attn_weights, values):
    # Weighted sum of value vectors for each query position.
    return attn_weights @ values

# Step 12 - project_qkv
def project_qkv(x, attn_params):
    # Project hidden states into query, key, and value tensors.
    q = linear_projection(x, attn_params["Wq"], attn_params.get("bq"))
    k = linear_projection(x, attn_params["Wk"], attn_params.get("bk"))
    v = linear_projection(x, attn_params["Wv"], attn_params.get("bv"))

    return q, k, v

# Step 13 - append_kv_cache
def append_kv_cache(kv_cache, new_k, new_v):
    # Initialize the cache when it is empty; otherwise append along the time axis.
    if kv_cache["k"] is None:
        kv_cache["k"] = new_k.copy()
    else:
        kv_cache["k"] = np.concatenate([kv_cache["k"], new_k], axis=0)

    if kv_cache["v"] is None:
        kv_cache["v"] = new_v.copy()
    else:
        kv_cache["v"] = np.concatenate([kv_cache["v"], new_v], axis=0)

    return kv_cache

# Step 14 - scaled_dot_product_attention_with_cache
def scaled_dot_product_attention_with_cache(queries, kv_cache, query_offset=0):
    """Causal scaled dot-product attention of queries against a KV cache."""
    keys = kv_cache["k"]
    values = kv_cache["v"]

    # Compute raw attention scores.
    scores = compute_attention_scores(queries, keys)

    # Scale by sqrt(d_head).
    d_head = queries.shape[-1]
    scores = scale_attention_scores(scores, d_head)

    # Apply the causal mask using the absolute query position.
    scores = apply_causal_mask(scores, query_offset=query_offset)

    # Convert masked scores into attention probabilities.
    attn_weights = softmax_attention_weights(scores)

    # Compute the weighted sum of value vectors.
    return weighted_value_sum(attn_weights, values)

# Step 15 - apply_output_projection
def apply_output_projection(context, attn_params):
    # Project the attention context back to the model dimension.
    return linear_projection(
        context,
        attn_params["Wo"],
        attn_params.get("bo"),
    )

# Step 16 - single_head_causal_self_attention
def single_head_causal_self_attention(x, attn_params, kv_cache, query_offset=0):
    """Single-head causal self-attention with KV-cache update.

    Returns (out, kv_cache) where out has shape (T, d_model).
    """
    # Project hidden states into queries, keys, and values.
    q, k, v = project_qkv(x, attn_params)

    # Append the newly computed keys and values to the cache.
    kv_cache = append_kv_cache(kv_cache, k, v)

    # Compute causal attention against the complete KV cache.
    context = scaled_dot_product_attention_with_cache(
        q,
        kv_cache,
        query_offset=query_offset,
    )

    # Project the attention context back to model dimension.
    out = apply_output_projection(context, attn_params)

    return out, kv_cache

# Step 17 - ffn_first_layer_gelu
def ffn_first_layer_gelu(x, ffn_params):
    # Apply the first FFN linear projection.
    h = linear_projection(
        x,
        ffn_params["W1"],
        ffn_params.get("b1"),
    )

    # Apply the GELU activation elementwise using the standard
    # tanh-based approximation.
    return 0.5 * h * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (h + 0.044715 * np.power(h, 3))
        )
    )

# Step 18 - ffn_second_layer
def ffn_second_layer(h, ffn_params):
    # Project the FFN hidden activations back to the model dimension.
    return linear_projection(
        h,
        ffn_params["W2"],
        ffn_params.get("b2"),
    )

# Step 19 - position_wise_feed_forward
def position_wise_feed_forward(x, ffn_params):
    # Apply the first FFN layer followed by GELU activation.
    h = ffn_first_layer_gelu(x, ffn_params)

    # Project the activated hidden representation back to d_model.
    return ffn_second_layer(h, ffn_params)

# Step 20 - compute_mean_variance
def compute_mean_variance(x, eps=1e-5):
    """Compute per-feature mean and variance along the last axis of x."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return mean, var

# Step 21 - layer_norm_apply
def layer_norm_apply(x, ln_params, eps=1e-5):
    """Normalize x over its last axis and apply gamma, beta."""
    mean, var = compute_mean_variance(x, eps=eps)

    # Normalize along the last axis.
    normalized = (x - mean) / np.sqrt(var + eps)

    # Apply the learned affine transformation.
    return normalized * ln_params["gamma"] + ln_params["beta"]

# Step 22 - residual_add_and_norm
def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    # Add the residual connection, then apply layer normalization.
    residual = x + sublayer_output
    return layer_norm_apply(residual, ln_params, eps=eps)

# Step 23 - transformer_block (not yet solved)
# TODO: implement

# Step 24 - lm_head_logits (not yet solved)
# TODO: implement

# Step 25 - greedy_next_token (not yet solved)
# TODO: implement

# Step 26 - run_prefill (not yet solved)
# TODO: implement

# Step 27 - decode_step (not yet solved)
# TODO: implement

# Step 28 - generate_with_state_log (not yet solved)
# TODO: implement

# Step 29 - hash_tensor (not yet solved)
# TODO: implement

# Step 30 - commit_decode_step (not yet solved)
# TODO: implement

# Step 31 - hash_pair (not yet solved)
# TODO: implement

# Step 32 - build_merkle_level (not yet solved)
# TODO: implement

# Step 33 - build_merkle_tree (not yet solved)
# TODO: implement

# Step 34 - merkle_root (not yet solved)
# TODO: implement

# Step 35 - merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 36 - verify_merkle_inclusion_proof (not yet solved)
# TODO: implement

# Step 37 - run_prover (not yet solved)
# TODO: implement

# Step 38 - assemble_public_transcript (not yet solved)
# TODO: implement

# Step 39 - sample_audit_positions (not yet solved)
# TODO: implement

# Step 40 - reexecute_audited_step (not yet solved)
# TODO: implement

# Step 41 - recompute_step_commitment (not yet solved)
# TODO: implement

# Step 42 - check_commitment_against_proof (not yet solved)
# TODO: implement

# Step 43 - check_token_matches_claim (not yet solved)
# TODO: implement

# Step 44 - run_spot_check_verification (not yet solved)
# TODO: implement

# Step 45 - tamper_transcript_flip_token (not yet solved)
# TODO: implement

# Step 46 - detection_probability (not yet solved)
# TODO: implement

# Step 47 - verifier_cost_fraction (not yet solved)
# TODO: implement

# Step 48 - show_tampered_transcript_rejected (not yet solved)
# TODO: implement

# Step 49 - sample_verifier_committee (not yet solved)
# TODO: implement

# Step 50 - collect_verifier_votes (not yet solved)
# TODO: implement

# Step 51 - aggregate_votes_majority (not yet solved)
# TODO: implement

# Step 52 - reward_honest_participants (not yet solved)
# TODO: implement

# Step 53 - slash_worker (not yet solved)
# TODO: implement

# Step 54 - assign_dual_role (not yet solved)
# TODO: implement

# Step 55 - run_honest_round (not yet solved)
# TODO: implement

# Step 56 - run_malicious_round (not yet solved)
# TODO: implement

# Step 57 - report_end_to_end_verification_cost (not yet solved)
# TODO: implement

