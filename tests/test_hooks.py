"""Phase 0 verification: residual capture, single-backward all-layer gradients,
and determinism."""

import torch

from subattr import attribution as A

IGNORE = A.IGNORE_INDEX


def _batch(model, T=12, B=1):
    V = model.config.vocab_size
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, V, (B, T), generator=g)
    labels = ids.clone()
    labels[:, : T // 2] = IGNORE  # only the "response" half carries loss
    return ids, torch.ones_like(ids), labels


def test_decoder_blocks_resolved(tiny_model):
    blocks = A.decoder_blocks(tiny_model)
    assert len(blocks) == tiny_model.config.num_hidden_layers


def test_residual_shapes_all_layers(tiny_model):
    A.freeze_params(tiny_model)
    ids, mask, _ = _batch(tiny_model)
    logits, residuals = A.forward_with_residuals(tiny_model, ids, mask)

    L = tiny_model.config.num_hidden_layers
    d = tiny_model.config.hidden_size
    # n_layers + 1: index 0 is the embedding, matching repo2 mean_activations.
    assert len(residuals) == L + 1
    for h in residuals:
        assert h.shape == (ids.shape[0], ids.shape[1], d)
    assert logits.shape[:2] == ids.shape


def test_residuals_are_live_graph_nodes(tiny_model):
    """repo2's own capture_residuals detaches; ours must not."""
    A.freeze_params(tiny_model)
    ids, mask, _ = _batch(tiny_model)
    _, residuals = A.forward_with_residuals(tiny_model, ids, mask)
    assert all(h.requires_grad for h in residuals)
    assert all(h.grad_fn is not None for h in residuals[1:])


def test_all_layer_grads_from_one_backward(tiny_model):
    A.freeze_params(tiny_model)
    ids, mask, labels = _batch(tiny_model)
    logits, residuals = A.forward_with_residuals(tiny_model, ids, mask)
    loss = A.response_ce_loss(logits, labels)
    grads = A.grads_wrt_residuals(loss, residuals)

    assert len(grads) == len(residuals)
    for g, h in zip(grads, residuals):
        assert g.shape == h.shape
        assert torch.isfinite(g).all()
    # Every layer must carry signal, or the "all layers in one backward" claim is empty.
    assert all(g.abs().sum() > 0 for g in grads)


def test_params_receive_no_grad(tiny_model):
    """Spec Phase 0: params frozen; we differentiate wrt activations only."""
    A.freeze_params(tiny_model)
    ids, mask, labels = _batch(tiny_model)
    logits, residuals = A.forward_with_residuals(tiny_model, ids, mask)
    A.grads_wrt_residuals(A.response_ce_loss(logits, labels), residuals)
    assert all(p.grad is None for p in tiny_model.parameters())


def test_grad_matches_finite_difference(tiny_model):
    """The load-bearing correctness test.

    Perturb the residual at one block by +/- eps*v and check the central difference
        [L(h + eps*v) - L(h - eps*v)] / (2*eps)  ~=  <grad_h L, v>
    This validates both the gradient machinery and the SIGN convention that spec
    section 2 depends on (score = -<g, delta>).

    repo2's `steering_hooks` expects a [n_blocks, H] tensor (it reads .shape[0]),
    so the direction is placed in a zero tensor at the block under test and only
    that block is hooked.
    """
    A.freeze_params(tiny_model)
    ids, mask, labels = _batch(tiny_model)

    block = 2  # 0-indexed block -> residuals[block + 1]
    logits, residuals = A.forward_with_residuals(tiny_model, ids, mask)
    loss0 = A.response_ce_loss(logits, labels)
    g = A.grads_wrt_residuals(loss0, residuals)[block + 1]

    H = tiny_model.config.hidden_size
    torch.manual_seed(7)
    v = torch.randn(H)
    v = v / v.norm()
    v_all = torch.zeros(tiny_model.config.num_hidden_layers, H)
    v_all[block] = v

    # h <- h + alpha*v at every position, so dL = alpha * sum_t <g_t, v>.
    predicted = (g * v).sum().item()

    st = A.repo2_steering()
    eps = 1e-3

    def loss_at(alpha):
        with torch.no_grad():
            with st.steering_hooks(
                tiny_model, v_all, alpha=alpha, mode="add",
                layers=[block], positions="broadcast", norm="raw",
            ):
                out = tiny_model(input_ids=ids, attention_mask=mask)
            return A.response_ce_loss(out.logits, labels).item()

    observed = (loss_at(eps) - loss_at(-eps)) / (2 * eps)

    assert abs(observed - predicted) < 0.02 * max(1.0, abs(predicted)), (
        f"directional derivative mismatch: analytic {predicted:.6f}, "
        f"central difference {observed:.6f}"
    )


def test_determinism_two_seeded_forwards(tiny_model):
    A.freeze_params(tiny_model)
    ids, mask, _ = _batch(tiny_model)

    A.set_seed(0)
    l1, r1 = A.forward_with_residuals(tiny_model, ids, mask)
    A.set_seed(0)
    l2, r2 = A.forward_with_residuals(tiny_model, ids, mask)

    assert torch.equal(l1, l2)
    for a, b in zip(r1, r2):
        assert torch.equal(a, b)
