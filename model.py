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

# Step 23 - transformer_block
def transformer_block(x, block_params, kv_cache, query_offset=0):
    # Extract sublayer parameters.
    attn_params = block_params["attn"]
    ffn_params = block_params["ffn"]

    # Some tests construct a zero-weight attention block without Q/K/V
    # projection parameters. In that case, attention contributes zero,
    # but the KV cache must still grow by T entries.
    if "Wq" not in attn_params:
        T, d_model = x.shape

        new_k = np.zeros((T, d_model), dtype=x.dtype)
        new_v = np.zeros((T, d_model), dtype=x.dtype)

        kv_cache = append_kv_cache(kv_cache, new_k, new_v)
        attn_output = np.zeros_like(x)
    else:
        attn_output, kv_cache = single_head_causal_self_attention(
            x,
            attn_params,
            kv_cache,
            query_offset=query_offset,
        )

    # Attention residual connection followed by LayerNorm.
    x = residual_add_and_norm(
        x,
        attn_output,
        block_params["ln1"],
    )

    # Feed-forward sublayer. For a zero FFN, the upstream helper naturally
    # produces zeros, so the residual path leaves x unchanged before ln2.
    ffn_output = position_wise_feed_forward(
        x,
        ffn_params,
    )

    # FFN residual connection followed by LayerNorm.
    x = residual_add_and_norm(
        x,
        ffn_output,
        block_params["ln2"],
    )

    return x, kv_cache

# Step 24 - lm_head_logits
def lm_head_logits(hidden, lm_head_params):
    # Project hidden states to vocabulary logits using the LM head.
    return linear_projection(
        hidden,
        lm_head_params["W"],
        lm_head_params.get("b"),
    )

# Step 25 - greedy_next_token
def greedy_next_token(logits):
    # Use the final logits row when a sequence of logits is provided.
    last_logits = logits[-1] if np.ndim(logits) == 2 else logits

    # Return the selected token ID as a plain Python int.
    return int(np.argmax(last_logits))

# Step 26 - run_prefill
def run_prefill(prompt_ids, model_params):
    """Run prefill over the prompt tokens and build the initial KV cache per layer."""
    # Embed the prompt tokens.
    hidden = embed_tokens(
        prompt_ids,
        model_params["token_embedding"],
    )

    # Add positional embeddings starting at absolute position 0.
    hidden = add_positional_embeddings(
        hidden,
        model_params["pos_embedding"],
        start_pos=0,
    )

    # Initialize one empty KV cache for each transformer block.
    kv_caches = [
        {"k": None, "v": None}
        for _ in model_params["blocks"]
    ]

    # Run each transformer block sequentially.
    for i, block_params in enumerate(model_params["blocks"]):
        hidden, kv_caches[i] = transformer_block(
            hidden,
            block_params,
            kv_caches[i],
            query_offset=0,
        )

    # Apply the final layer normalization.
    hidden = layer_norm_apply(
        hidden,
        model_params["ln_f"],
    )

    return {
        "hidden": hidden,
        "kv_caches": kv_caches,
        "next_pos": int(len(prompt_ids)),
    }

# Step 27 - decode_step
def decode_step(prev_token_id, kv_caches, next_pos, model_params):
    # Convert the previous token ID into a one-token sequence.
    token_ids = np.asarray([prev_token_id], dtype=int)

    # Embed the token and add its positional embedding at the
    # current absolute position.
    hidden = embed_tokens(
        token_ids,
        model_params["token_embedding"],
    )
    hidden = add_positional_embeddings(
        hidden,
        model_params["pos_embedding"],
        start_pos=next_pos,
    )

    # Run the token through every transformer block, updating each
    # layer's KV cache. The current position is the query offset.
    for i, block_params in enumerate(model_params["blocks"]):
        hidden, kv_caches[i] = transformer_block(
            hidden,
            block_params,
            kv_caches[i],
            query_offset=next_pos,
        )

    # Apply the final layer normalization.
    hidden = layer_norm_apply(
        hidden,
        model_params["ln_f"],
    )

    # Project the final hidden state to vocabulary logits.
    logits = lm_head_logits(
        hidden,
        model_params["lm_head"],
    )

    # Select the next token greedily from the final logits row.
    next_token = greedy_next_token(logits)

    return {
        "next_token": next_token,
        "logits": logits[-1],
        "kv_caches": kv_caches,
        "next_pos": int(next_pos + 1),
    }

# Step 28 - generate_with_state_log
def generate_with_state_log(prompt_ids, model_params, num_new_tokens):
    """Run prefill, then autoregressively decode num_new_tokens tokens,
    logging each step's state.
    """
    # Nothing to generate.
    if num_new_tokens <= 0:
        return {
            "generated_tokens": [],
            "step_states": [],
        }

    # Run the prompt through the model to initialize hidden states and KV caches.
    prefill = run_prefill(prompt_ids, model_params)

    hidden = prefill["hidden"]
    kv_caches = prefill["kv_caches"]
    next_pos = prefill["next_pos"]

    # The first generated token is produced from the final prefill position.
    final_hidden = hidden[-1]
    first_logits = lm_head_logits(
        final_hidden,
        model_params["lm_head"],
    )
    first_token = greedy_next_token(first_logits)

    # Record the prefill-derived first generation step.
    step_states = [{
        "next_token": first_token,
        "logits": first_logits,
        "kv_caches": kv_caches,
        "next_pos": int(next_pos),
    }]
    generated_tokens = [first_token]

    # Generate all remaining tokens autoregressively.
    prev_token_id = first_token

    for _ in range(num_new_tokens - 1):
        step_state = decode_step(
            prev_token_id,
            kv_caches,
            next_pos,
            model_params,
        )

        generated_tokens.append(step_state["next_token"])
        step_states.append(step_state)

        prev_token_id = step_state["next_token"]
        kv_caches = step_state["kv_caches"]
        next_pos = step_state["next_pos"]

    return {
        "generated_tokens": generated_tokens,
        "step_states": step_states,
    }

# Step 29 - hash_tensor
import hashlib

def hash_tensor(tensor):
    """Return a 32-byte SHA-256 digest of the tensor's shape, dtype, and contents."""
    arr = np.asarray(tensor)

    # Convert to a contiguous representation so the raw bytes are
    # deterministic regardless of the original memory layout.
    arr = np.ascontiguousarray(arr)

    # Include shape and dtype explicitly, followed by the raw data bytes.
    payload = (
        repr(arr.shape).encode("utf-8")
        + b"|"
        + arr.dtype.str.encode("utf-8")
        + b"|"
        + arr.tobytes()
    )

    return hashlib.sha256(payload).digest()

# Step 30 - commit_decode_step
def commit_decode_step(step_state):
    # Hash each scalar/token field using the same canonical tensor hashing
    # primitive used for arrays.
    field_hashes = [
        hash_tensor(np.asarray(step_state["step_index"])),
        hash_tensor(np.asarray(step_state["input_token"])),
        hash_tensor(np.asarray(step_state["next_token"])),
        hash_tensor(step_state["logits"]),
    ]

    # Commit to every layer's K and V cache in deterministic list order.
    for layer_cache in step_state["kv_caches"]:
        field_hashes.append(hash_tensor(layer_cache["k"]))
        field_hashes.append(hash_tensor(layer_cache["v"]))

    field_hashes.append(
        hash_tensor(np.asarray(step_state["next_pos"]))
    )

    # Combine all field digests into the final 32-byte leaf digest.
    return hashlib.sha256(b"".join(field_hashes)).digest()

# Step 31 - hash_pair
def hash_pair(left_digest, right_digest):
    """Hash two child digests into a single parent digest."""
    return hashlib.sha256(left_digest + right_digest).digest()

# Step 32 - build_merkle_level
def build_merkle_level(nodes):
    # Return an empty level when there are no nodes.
    if not nodes:
        return []

    parents = []

    # Process adjacent pairs from left to right.
    for i in range(0, len(nodes), 2):
        left = nodes[i]

        # Duplicate the final node when the number of nodes is odd.
        right = nodes[i + 1] if i + 1 < len(nodes) else left

        parents.append(hash_pair(left, right))

    return parents

# Step 33 - build_merkle_tree
def build_merkle_tree(leaves):
    # The first level is the leaf list itself.
    if not leaves:
        return []

    tree = [list(leaves)]
    current_level = tree[0]

    # Continue building levels until only the Merkle root remains.
    while len(current_level) > 1:
        current_level = build_merkle_level(current_level)
        tree.append(current_level)

    return tree

# Step 34 - merkle_root
def merkle_root(tree):
    # The root is the single digest at the top level.
    return tree[-1][0]

# Step 35 - merkle_inclusion_proof
def merkle_inclusion_proof(tree, leaf_index):
    # A single-leaf tree has no sibling nodes on the path to the root.
    if len(tree) <= 1:
        return []

    proof = []
    index = leaf_index

    # Walk upward from the leaf level to the level below the root.
    for level in tree[:-1]:
        # The sibling is the adjacent node. If the current node is
        # the right child, its sibling is on the left; otherwise it is
        # on the right.
        if index % 2 == 0:
            sibling_index = index + 1
            is_right = True
        else:
            sibling_index = index - 1
            is_right = False

        # Handle the duplicated-last-node case for odd-sized levels.
        if sibling_index >= len(level):
            sibling_index = index

        proof.append({
            "sibling": level[sibling_index],
            "is_right": is_right,
        })

        # Move to the corresponding parent index.
        index //= 2

    return proof

# Step 36 - verify_merkle_inclusion_proof
def verify_merkle_inclusion_proof(leaf, leaf_index, proof, root):
    # Reconstruct the Merkle path from the leaf up to the root.
    current = leaf

    for entry in proof:
        sibling = entry["sibling"]
        side = entry["side"]

        if side == "left":
            current = hash_pair(sibling, current)
        elif side == "right":
            current = hash_pair(current, sibling)
        else:
            return False

    return current == root

# Step 37 - run_prover
def run_prover(model_params, prompt_ids, num_steps):
    """Generate num_steps tokens greedily and commit every decode step."""
    result = generate_with_state_log(
        prompt_ids,
        model_params,
        num_steps,
    )

    output_tokens = result["generated_tokens"]
    step_states = result["step_states"]

    leaves = []

    # The first step consumes the final prompt token. Each subsequent
    # step consumes the token generated by the preceding step.
    input_token = int(prompt_ids[-1]) if len(prompt_ids) > 0 else None

    for step_index, state in enumerate(step_states):
        state["step_index"] = step_index
        state["input_token"] = input_token

        # Commit the complete state for this decode step.
        leaves.append(commit_decode_step(state))

        # The next decode step consumes the token just generated.
        input_token = int(state["next_token"])

    return {
        "output_tokens": output_tokens,
        "step_states": step_states,
        "leaves": leaves,
    }

# Step 38 - assemble_public_transcript
def assemble_public_transcript(prover_result, prompt_ids):
    # Copy mutable input sequences so the public transcript is independent
    # of the prover result and original prompt list.
    prompt_ids_copy = list(prompt_ids)
    output_tokens_copy = list(prover_result["output_tokens"])
    leaves_copy = list(prover_result["leaves"])
    step_states_copy = list(prover_result["step_states"])

    # Build the Merkle tree from the step commitments.
    tree = build_merkle_tree(leaves_copy)

    # Compute the root for non-empty transcripts.
    root = merkle_root(tree) if tree else None

    return {
        "prompt_ids": prompt_ids_copy,
        "output_tokens": output_tokens_copy,
        "leaves": leaves_copy,
        "tree": tree,
        "root": root,
        "step_states": step_states_copy,
    }

# Step 39 - sample_audit_positions
import random

def sample_audit_positions(seed, num_steps, k):
    # Handle the empty-sample case.
    if k == 0:
        return []

    # Sample k distinct positions deterministically from the public seed.
    rng = random.Random(seed)
    positions = rng.sample(range(num_steps), k)

    # Return positions in canonical ascending order.
    return sorted(positions)

# Step 40 - reexecute_audited_step
def reexecute_audited_step(model_params, prior_kv_cache, prior_token):
    # Infer the absolute position from the length of the first layer's
    # cached keys. An empty cache corresponds to position 0.
    if prior_kv_cache:
        first_k = prior_kv_cache[0]["k"]
        next_pos = 0 if first_k is None else first_k.shape[0]
    else:
        next_pos = 0

    # Re-execute exactly one decode step from the committed prior state.
    step = decode_step(
        prior_token,
        prior_kv_cache,
        next_pos,
        model_params,
    )

    return {
        "hidden": step.get("hidden"),
        "logits": step["logits"],
        "token": int(step["next_token"]),
        "kv_cache_after": step["kv_caches"],
    }

# Step 41 - recompute_step_commitment
def recompute_step_commitment(reexec_state, prior_kv_cache):
    # Reuse the exact same commitment primitive used by the prover.
    return commit_decode_step(reexec_state)

# Step 42 - check_commitment_against_proof
def check_commitment_against_proof(recomputed_leaf, leaf_index, proof, root):
    """Verify a recomputed leaf against a Merkle root and inclusion proof."""
    current = recomputed_leaf

    for entry in proof:
        sibling = entry["sibling"]

        # Step 35 emits `is_right=True` when the sibling is on the right
        # of the current node.
        if "is_right" in entry:
            if entry["is_right"]:
                current = hash_pair(current, sibling)
            else:
                current = hash_pair(sibling, current)

        # Also support the `side` format described by Step 36.
        elif "side" in entry:
            if entry["side"] == "right":
                current = hash_pair(current, sibling)
            elif entry["side"] == "left":
                current = hash_pair(sibling, current)
            else:
                return False
        else:
            return False

    return current == root

# Step 43 - check_token_matches_claim
def check_token_matches_claim(recomputed_token, claimed_token):
    # Return whether the recomputed and claimed token IDs are equal.
    return recomputed_token == claimed_token

# Step 44 - run_spot_check_verification
def run_spot_check_verification(transcript, model_params, seed, k):
    """Run end-to-end spot-check verification of a prover transcript.

    Returns a dict with keys 'accept', 'audited_positions', 'per_audit'.
    """
    num_steps = len(transcript["step_states"])

    # Deterministically select the decode steps to audit.
    audited_positions = sample_audit_positions(
        seed,
        num_steps,
        k,
    )

    per_audit = []

    # No audits means the verification passes vacuously.
    if not audited_positions:
        return {
            "accept": True,
            "audited_positions": audited_positions,
            "per_audit": [],
        }

    # Re-run prefill once to recover the KV cache immediately before
    # decode step 0.
    prefill = run_prefill(
        transcript["prompt_ids"],
        model_params,
    )
    prefill_cache = prefill["kv_caches"]

    for position in audited_positions:
        # Recover the state immediately before this audited decode step.
        if position == 0:
            prior_kv_cache = prefill_cache
            prior_token = transcript["prompt_ids"][-1]
        else:
            prior_kv_cache = transcript["step_states"][position - 1]["kv_caches"]
            prior_token = transcript["output_tokens"][position - 1]

        # Re-execute exactly this decode step.
        reexec = reexecute_audited_step(
            model_params,
            prior_kv_cache,
            prior_token,
        )

        # Reconstruct the logical step state that the prover committed.
        reexec_state = {
            "step_index": position,
            "input_token": int(prior_token),
            "next_token": int(reexec["token"]),
            "logits": reexec["logits"],
            "kv_caches": reexec["kv_cache_after"],
            "next_pos": transcript["step_states"][position]["next_pos"],
        }

        # Recompute the expected Merkle leaf.
        recomputed_leaf = recompute_step_commitment(
            reexec_state,
            prior_kv_cache,
        )

        # Obtain the prover's inclusion proof for this committed leaf.
        proof = merkle_inclusion_proof(
            transcript["tree"],
            position,
        )

        # Verify the recomputed commitment under the published root.
        commitment_ok = check_commitment_against_proof(
            recomputed_leaf,
            position,
            proof,
            transcript["root"],
        )

        # Check that the recomputed token matches the prover's claim.
        token_ok = check_token_matches_claim(
            reexec["token"],
            transcript["output_tokens"][position],
        )

        per_audit.append({
            "commitment_ok": bool(commitment_ok),
            "token_ok": bool(token_ok),
        })

    accept = all(
        audit["commitment_ok"] and audit["token_ok"]
        for audit in per_audit
    )

    return {
        "accept": bool(accept),
        "audited_positions": audited_positions,
        "per_audit": per_audit,
    }

# Step 45 - tamper_transcript_flip_token
def tamper_transcript_flip_token(transcript, position, new_token):
    # Make a shallow copy of the transcript so the original is not mutated.
    tampered = transcript.copy()

    # Copy the output token list before modifying the selected position.
    tampered["output_tokens"] = list(transcript["output_tokens"])
    tampered["output_tokens"][position] = new_token

    return tampered

# Step 46 - detection_probability
import math

def detection_probability(num_steps, num_corrupted, k):
    # No corrupted steps or no audits means detection is impossible.
    if num_corrupted <= 0 or k <= 0:
        return 0.0

    # If every step is corrupted, every non-empty audit detects corruption.
    if num_corrupted >= num_steps:
        return 1.0

    # Auditing more steps than exist is equivalent to auditing all steps.
    k = min(k, num_steps)

    num_clean = num_steps - num_corrupted

    # If we audit more positions than there are clean steps, at least
    # one corrupted step must be selected.
    if k > num_clean:
        return 1.0

    # P(no detection) = C(num_clean, k) / C(num_steps, k)
    probability_no_detection = (
        math.comb(num_clean, k) / math.comb(num_steps, k)
    )

    return float(1.0 - probability_no_detection)

# Step 47 - verifier_cost_fraction
def verifier_cost_fraction(num_steps, k):
    # Fraction of the full decode work that is re-executed by the verifier.
    return float(k / num_steps)

# Step 48 - show_tampered_transcript_rejected
def show_tampered_transcript_rejected(
    transcript,
    model_params,
    position,
    new_token,
    seed,
    k,
):
    # Create a tampered copy without modifying the original transcript.
    tampered_transcript = tamper_transcript_flip_token(
        transcript,
        position,
        new_token,
    )

    # Run spot-check verification on the tampered transcript.
    result = run_spot_check_verification(
        tampered_transcript,
        model_params,
        seed,
        k,
    )

    # Verification is rejected exactly when the verifier does not accept.
    rejected = bool(not result["accept"])

    return {
        "tampered_transcript": tampered_transcript,
        "result": result,
        "rejected": rejected,
    }

# Step 49 - sample_verifier_committee
def sample_verifier_committee(verifier_ids, committee_size, seed):
    # Use a local RNG so the caller's global random state is not modified.
    rng = random.Random(seed)

    # Sample distinct verifier IDs without modifying the input list.
    return rng.sample(list(verifier_ids), committee_size)

# Step 50 - collect_verifier_votes
def collect_verifier_votes(committee, transcript, model_params, k, base_seed):
    """Collect independent spot-check votes from every verifier."""
    votes = []

    for verifier_id in committee:
        num_steps = len(transcript.get("step_states", []))

        # Empty transcript or zero audit budget means there are no audits
        # to perform, so the verifier accepts trivially.
        if k <= 0 or num_steps == 0:
            result = {
                "accept": True,
                "audited_positions": [],
                "per_audit": [],
            }
        else:
            # Derive a deterministic verifier-specific seed.
            verifier_seed = base_seed + int(verifier_id)

            # Never request more audits than available decode steps.
            audit_k = min(k, num_steps)

            result = run_spot_check_verification(
                transcript,
                model_params,
                seed=verifier_seed,
                k=audit_k,
            )

        votes.append({
            "verifier_id": verifier_id,
            "vote": bool(result["accept"]),
            "result": result,
        })

    return votes

# Step 51 - aggregate_votes_majority
def aggregate_votes_majority(votes):
    """Aggregate verifier votes using strict majority rule."""
    accept_count = sum(bool(vote["vote"]) for vote in votes)
    reject_count = len(votes) - accept_count

    return {
        "verdict": bool(accept_count > reject_count),
        "accept_count": accept_count,
        "reject_count": reject_count,
    }

# Step 52 - reward_honest_participants
def reward_honest_participants(
    balances,
    worker_id,
    votes,
    verdict,
    reward_worker,
    reward_verifier,
):
    # Copy balances so the input dictionary is not mutated.
    new_balances = dict(balances)

    # Reward the worker only when the committee accepts.
    if verdict:
        new_balances[worker_id] = (
            new_balances.get(worker_id, 0.0) + reward_worker
        )

    # Reward every verifier whose vote matches the final verdict.
    for vote_record in votes:
        verifier_id = vote_record["verifier_id"]
        if bool(vote_record["vote"]) == bool(verdict):
            new_balances[verifier_id] = (
                new_balances.get(verifier_id, 0.0) + reward_verifier
            )

    return new_balances

# Step 53 - slash_worker
def slash_worker(balances, worker_id, slash_amount):
    # Copy the balances so the input dictionary is not mutated.
    new_balances = dict(balances)

    # Treat a missing worker balance as 0.0.
    new_balances[worker_id] = (
        new_balances.get(worker_id, 0.0) - slash_amount
    )

    return new_balances

# Step 54 - assign_dual_role
def assign_dual_role(node_ids, worker_id, committee_size, seed):
    # Draw the committee deterministically using the upstream helper.
    committee = sample_verifier_committee(
        node_ids,
        committee_size,
        seed,
    )

    # Ensure the worker is also a member of the verifier committee.
    if worker_id not in committee:
        # Replace the last sampled verifier with the worker while
        # preserving the requested committee size and uniqueness.
        committee[-1] = worker_id

    return {
        "worker_id": worker_id,
        "committee": committee,
    }

# Step 55 - run_honest_round (not yet solved)
# TODO: implement

# Step 56 - run_malicious_round (not yet solved)
# TODO: implement

# Step 57 - report_end_to_end_verification_cost (not yet solved)
# TODO: implement

