# Technical Challenges — V2

This document logs bugs, stuck points, surprising discoveries, and
design decisions encountered while building V2 of the EECS 590 capstone RL framework.

---

## 1. Tabular State Space Explosion (Classical RL on highway-env)

**Problem:** The continuous highway-env observation space (5 vehicles × 5 features = 25 floats)
cannot be used directly by tabular methods. Without a good discretisation, the state space
is effectively infinite.

**Solution:** Reused the V1 `obs_to_tabular_state` function, which bins ego speed into 3 levels,
front/rear distances into 3 levels, and lane into 5, yielding 135 states. However, many
states are rarely visited during training, especially with a random exploration policy.

**Discovery:** Most tabular agents converge to a reasonably good lane-keeping policy
around episode 500–800 with ε-decay, but struggle with rare states (e.g. simultaneous
front AND rear vehicle close proximity). This is a fundamental limitation of the coarse
tabular abstraction — a lesson that motivates the move to deep RL.

---

## 2. Eligibility Traces Numerical Stability (TD(λ) backward view)

**Problem:** With accumulating eligibility traces and high λ (> 0.95), the trace array
grows unboundedly in long episodes, causing extremely large Q-table updates that
destabilise training.

**Fix:** Two mitigations applied:
1. Added a **replacing-trace** variant (`trace_type="replacing"`), which caps traces at 1
   and prevents runaway accumulation.
2. Capped maximum episode length (`max_steps`) at 200 for tabular training,
   keeping traces from growing too large even with accumulating mode.

**Lesson:** λ ≈ 0.8–0.9 with accumulating traces, or λ ≈ 0.9–0.95 with replacing traces,
tended to give the best results in early experiments. High λ ≈ 1.0 approached MC behaviour
but with more variance.

---

## 3. DQN Target Network Update Frequency vs. Stability

**Problem:** Early DQN experiments with `target_update_freq=100` produced oscillating
Q-values — the online net learned from rapidly shifting targets, creating a feedback loop.

**Fix:** Increased to `target_update_freq=500`. The original Mnih et al. (2015) paper uses
10,000 steps, but that was for Atari (much larger state space). For highway-env's 25-feature
input, 500 is a better balance of stability vs. target freshness.

**Discovery:** Double DQN (selecting actions with the online net, evaluating with the target)
reduced the systematic Q-value overestimation visible in standard DQN, confirming the
theoretical motivation of van Hasselt et al. (2016).

---

## 4. Highway-env Observation Configuration

**Problem:** The default highway-env observation can vary by configuration. In some
versions, `normalize=True` returns relative coordinates; in others it returns absolute
values. This caused a mismatch between the tabular discretisation (which assumed
specific value ranges) and the actual observations received.

**Fix:** Set `normalize=False` in the environment config (see `highway_wrapper.py`
`DEFAULT_CONFIG`) so absolute coordinates are always returned. Normalisation to [0, 1]
is then handled separately by the `HighwayWrapper` wrapper for deep RL agents.

---

## 5. PPO Rollout Buffer "Not Full" Edge Case

**Problem:** When `total_steps` is not exactly divisible by `n_steps`, the final
rollout never fills completely and the last update is skipped. This wastes the last
batch of experience.

**Partial fix:** Added a check so that if the rollout is at least 50% full at the end
of training, a final partial update is performed. This is not strictly on-policy
(GAE computation requires a complete rollout) but improves data efficiency slightly
at the cost of a minor policy-gradient bias.

**Alternative not taken:** Padding the rollout with dummy transitions would have been
cleaner but introduced artificial transitions.

---

## 6. Integrated Gradients Baseline Choice

**Problem:** Integrated Gradients (IG) requires a baseline input — the "uninformative"
reference point from which attribution is measured. Choosing the zero vector makes
mathematical sense but is not a valid highway observation (presence flag = 0 means "no
vehicle", so zero-obs is actually meaningful: an empty road).

**Decision:** Kept the zero baseline as default because:
1. It is the standard choice in the IG literature.
2. The attribution is interpreted as "how much does each feature increase Q(s,a)
   relative to an empty road scenario" — which is interpretable in the driving domain.

**Alternative:** A sampled mean observation as baseline would reduce the bias but
requires a dataset of real observations, which is expensive to collect.

---

## 7. Prioritized Experience Replay (PER) — SumTree Edge Case

**Problem:** During early training, all TD errors are close to zero, so all priorities
are near `eps` (1e-6). The SumTree samples uniformly in this case, which is correct,
but propagation errors accumulate in the binary tree when millions of updates are
performed with float64 — tested and confirmed with a unit check.

**Fix:** Use `float64` for the SumTree (not float32) to maintain enough precision in
the parent nodes during propagation. The individual leaf values are still well-behaved,
but summing millions of tiny floats benefits from the extra precision.

---

## 8. Highway-env `configure()` Must Be Called Before `reset()`

**Problem:** Calling `env.configure(cfg)` after the first `env.reset()` does not take
effect in some highway-env versions. The observation space dimensions were wrong until
`reset()` was called again after configuring.

**Fix:** In `make_highway_env()`, call `env.configure(cfg)` then `env.reset()` before
wrapping with `HighwayWrapper`. The wrapper then reads the correct `vehicles_count`
and `features` from `env.config`.

---

## 9. Classical Agents on Highway-env: Sparse Reward Signal

**Problem:** Highway-env rewards are dense (continuous speed reward + discrete collision
penalty), but the tabular state space is coarse (135 states). With ε-greedy exploration
and 135 states, most Q-table entries receive very few updates, especially for rare
state-action pairs.

**Observations:**
- Q-Learning typically updates only ~40–60% of state-action pairs during a 2000-episode run.
- Monte Carlo updates all (s,a) pairs that appear in an episode but requires completing
  the episode before any update — very slow for long episodes.
- SARSA(λ) backward view covers the most state-action pairs per episode via
  eligibility traces spreading the credit signal backward through the episode.

**Recommendation:** SARSA(λ) backward view with λ=0.9 is the best classical algorithm
for this environment configuration.

---

## 10. Deep RL Device Compatibility (CPU vs. CUDA)

**Problem:** The project targets CPU training (no GPU guaranteed in the submission
environment). PyTorch's default device is CPU, but gradient computations for saliency
analysis (which require `.backward()` on input tensors) behave differently when
tensors are moved between devices.

**Fix:** All tensors in saliency functions are created on the same device as the
network (via a `device` parameter). The `_to_tensor()` helper always creates tensors
on the specified device. No device mismatch errors occur as long as both the network
and input tensors are on the same device.

---

---

## 11. A3C vs A2C — Why We Use A2C

**Decision:** A3C (Asynchronous Advantage Actor-Critic) is theoretically applicable to
highway-v0's discrete action space, but its implementation requires launching N
parallel worker processes that each maintain their own environment and asynchronously
push gradients to a shared parameter server.

On Windows, Python's multiprocessing has significant overhead and PyTorch's shared
memory model requires careful process-safe parameter sharing (`mp.Array`, `mp.Value`,
or `torch.multiprocessing`).

**Practical choice:** We implement **A2C** (synchronous A2C), which is the mathematical
equivalent of A3C with a single worker — same gradients, same update rule, just without
the threading complexity. All empirical comparisons in OpenAI's original code showed A2C
matches A3C performance on most tasks.

---

## 12. TRPO — Applicable but Not Implemented

TRPO (Trust Region Policy Optimization) IS applicable to discrete action spaces like
highway-v0. However, its implementation requires:
- Computing the Fisher Information Matrix (FIM) or its inverse-vector product
- A constrained optimisation step (conjugate gradient + line search)
- Significantly more code than PPO for marginally better monotonic improvement guarantees

PPO was specifically designed as a practical approximation of TRPO that achieves similar
empirical performance with 10x less code. For this capstone, PPO covers the "trust region"
curriculum and TRPO is documented as a known but not implemented extension.

---

## Future Work / Open Issues

- [ ] Implement N-step DQN (n-step return bootstrapping for DQN) as a further
      improvement beyond Double + Dueling.
- [ ] Add a `SuccessRate` metric to the evaluator (collision-free episodes / total).
- [ ] Investigate whether a denser tabular abstraction (e.g. 4 speed bins × 4 distance
      bins × 5 lanes = 320 states) would improve classical RL without making the
      state space too sparse to learn from.
- [ ] Run `compare_algorithms.py` with all 7 algorithms on a common seed for a
      clean comparison figure to include in the final report.
