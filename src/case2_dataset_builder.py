"""Dataset builder for Case 2: bounded sampling, predictor approximation operator.

Generates supervised training pairs (measurement, input history) -> predictor
state at the end of the delay horizon, using high-accuracy Picard labels.
"""

import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.simulate import build_robot, make_reference, make_simulator, sample_initial_state
from src.case1_dataset_builder import exact_predictor_label

def exact_multistep_predictor_label(sim, cfg, q_meas, v_meas, u_hist, t0=0.0):
    """Compute the exact Case 1 sampling-horizon prediction label for one sample.

    Applies the predictor operator (P) then rolls out the closed-loop flow
    for sample_steps steps to build the full trajectory label.

    Inputs:  sim bundle, cfg dict, q_meas (nq,), v_meas (nv,),
             u_hist (delay_steps, nv), t0 simulation time float
    Returns: z_traj ndarray (sample_steps+1, nq+nv)
    """
    pack_state = sim["pack_state"]
    controller_state_step_rk4 = sim["controller_state_step_rk4"]

    dt = cfg["dt"]
    sample_steps = cfg["sample_steps"]

    qz, vz = exact_predictor_label(sim, cfg, q_meas, v_meas, u_hist)

    z_traj = [pack_state(qz, vz)]

    q_roll = qz.copy()
    v_roll = vz.copy()

    for j in range(sample_steps):
        tj = t0 + j * dt
        q_roll, v_roll = controller_state_step_rk4(q_roll, v_roll, tj, dt)
        z_traj.append(pack_state(q_roll, v_roll))

    return np.asarray(z_traj)

def extract_multistep_predictor_samples(
    out,
    sim,
    cfg,
    stride=1,
    flatten_target=False,
    rollout_id=None,
    verbose=False,
    log_interval=100,
):
    """Extract supervised (X, U, Y) trajectory samples for Case 1 from one rollout.

    Inputs:  out dict from simulate, sim bundle, cfg dict, stride int,
             flatten_target bool
    Returns: X (N, nq+nv), U (N, delay_steps, nv), T (N,),
             Y (N, sample_steps+1, nq+nv) or flattened if flatten_target
    """
    t_log = out["t"]
    q_meas = out["q_meas"]
    v_meas = out["v_meas"]
    u_hist_log = out["u_hist"]

    pack_state = sim["pack_state"]

    n_steps = len(q_meas)
    n_samples_est = (n_steps + stride - 1) // stride

    prefix = f"[rollout {rollout_id:02d}] " if rollout_id is not None else ""

    if verbose:
        print(
            f"{prefix}extract_multistep_predictor_samples start | "
            f"steps={n_steps} stride={stride} "
            f"(~{n_samples_est} samples expected)"
        )

    X = []
    U = []
    T = []
    Y = []

    start = time.perf_counter()

    kept = 0
    for k in range(len(q_meas)):
        if k % stride != 0:
            continue

        q_in = q_meas[k]
        v_in = v_meas[k]
        u_hist = u_hist_log[k]
        t_in = t_log[k]

        z_traj = exact_multistep_predictor_label(
            sim,
            cfg,
            q_in,
            v_in,
            u_hist,
            t0=t_in,
        )

        X.append(pack_state(q_in, v_in))
        U.append(np.asarray(u_hist, dtype=float).copy())
        T.append(float(t_in))
        Y.append(z_traj)

        kept += 1
        if verbose and log_interval > 0 and kept % log_interval == 0:
            elapsed = time.perf_counter() - start
            print(
                f"{prefix}extraction progress: "
                f"{kept}/{n_samples_est} samples | "
                f"elapsed {elapsed:.2f}s"
            )

    X = np.asarray(X)
    U = np.asarray(U)
    T = np.asarray(T)
    Y = np.asarray(Y)

    if flatten_target:
        Y = Y.reshape(Y.shape[0], -1)

    elapsed = time.perf_counter() - start

    if verbose:
        y_shape_str = tuple(Y.shape)
        print(
            f"{prefix}extract_multistep_predictor_samples done | "
            f"samples={len(X)} | "
            f"Y shape={y_shape_str} | "
            f"time={elapsed:.2f}s"
        )

    return X, U, T, Y

def validate_multistep_dataset_shapes(dataset, robot, cfg):
    """Assert that dataset arrays have the expected shapes and print a summary."""
    nq = robot["nq"]
    nv = robot["nv"]
    delay_steps = cfg["delay_steps"]
    sample_steps = cfg["sample_steps"]

    X = dataset["state"]
    U = dataset["u_hist"]
    T = dataset["t"]
    print(T.shape)
    Y = dataset["predictor_traj"]

    print("state shape          :", X.shape)
    print("u_hist shape         :", U.shape)
    print("t shape              :", T.shape)
    print("predictor_traj shape :", Y.shape)

    assert X.ndim == 2
    assert U.ndim == 3
    assert T.ndim == 1
    assert Y.ndim == 3

    assert X.shape[1] == nq + nv, f"Expected state dim {nq+nv}, got {X.shape[1]}"
    assert U.shape[1] == delay_steps, f"Expected delay_steps {delay_steps}, got {U.shape[1]}"
    assert U.shape[2] == nv, f"Expected control dim {nv}, got {U.shape[2]}"
    assert Y.shape[1] == sample_steps + 1, f"Expected horizon length {sample_steps+1}, got {Y.shape[1]}"
    assert Y.shape[2] == nq + nv, f"Expected predictor dim {nq+nv}, got {Y.shape[2]}"

    assert X.shape[0] == U.shape[0] == T.shape[0] == Y.shape[0], "Mismatched number of samples"

    print("Shape checks passed.")

def validate_multistep_dataset_labels(
    dataset,
    sim,
    cfg,
    n_checks=20,
    seed=0,
):
    """Spot-check stored trajectory labels by recomputing them for random samples.

    Inputs:  dataset dict (needs "t" key for time per sample), sim bundle, cfg dict,
             n_checks int, seed int
    Returns: errors array of Frobenius norms between stored and recomputed trajectories
    """
    rng = np.random.default_rng(seed)

    X = dataset["state"]
    U = dataset["u_hist"]
    Y = dataset["predictor_traj"]
    T = dataset["t"]              # simulation time per sample (for the time-varying reference)

    nq = X.shape[1] // 2  # state = [q, v] with nq == nv assumed

    idxs = rng.choice(len(X), size=min(n_checks, len(X)), replace=False)

    errors = []

    for idx in idxs:
        x = X[idx]
        u_hist = U[idx]
        y_stored = Y[idx]
        t0 = T[idx]               # recorded time for the time-varying reference rollout

        q_in = x[:nq]
        v_in = x[nq:]

        y_recomputed = exact_multistep_predictor_label(
            sim,
            cfg,
            q_in,
            v_in,
            u_hist,
            t0=t0,
        )

        err = np.linalg.norm(y_recomputed - y_stored)
        errors.append(err)

    errors = np.array(errors)

    print(f"label recomputation checks: {len(errors)}")
    print(f"mean error: {errors.mean():.3e}")
    print(f"max error : {errors.max():.3e}")

    return errors

def save_multistep_predictor_dataset(dataset, cfg, path):
    """Save the Case 1 sampling-horizon prediction dataset to a compressed .npz file."""
    np.savez_compressed(
        path,
        state=dataset["state"],
        u_hist=dataset["u_hist"],
        predictor_traj=dataset["predictor_traj"],
        config=cfg,
    )

def _run_one_multistep_rollout(args):
    """Run one simulation rollout in a worker process and return extracted Case 1 samples.

    Inputs:  args tuple (rollout_idx, seed, cfg, stride, flatten_target,
             q_meas_noise_std, v_meas_noise_std, use_noisy_measurement_for_reset)
    Returns: result dict with "rollout_idx", "seed", "num_samples",
             "state", "u_hist", "predictor_traj"
    """
    (
        rollout_idx,
        rollout_seed,
        cfg,
        stride,
        flatten_target,
        q_meas_noise_std,
        v_meas_noise_std,
        use_noisy_measurement_for_reset,
    ) = args

    # Build fresh robot/reference/simulator inside this process
    robot = build_robot(cfg["urdf"])
    ref = make_reference(robot, cfg)
    sim = make_simulator(robot, cfg, ref)

    rng = np.random.default_rng(rollout_seed)
    q0, v0 = sample_initial_state(robot, rng)

    out = sim["simulate"](
        q0=q0,
        v0=v0,
        q_meas_noise_std=q_meas_noise_std,
        v_meas_noise_std=v_meas_noise_std,
        use_noisy_measurement_for_reset=use_noisy_measurement_for_reset,
        rng=rng,
        verbose=False,
        log_every_step=True,
        rollout_id=rollout_idx,
        progress_interval=5000,
    )

    extract_start = time.perf_counter()

    print(f"[rollout {rollout_idx:02d}] starting multistep sample extraction")

    X, U, T, Y = extract_multistep_predictor_samples(
        out,
        sim,
        cfg,
        stride=stride,
        flatten_target=flatten_target,
        rollout_id=rollout_idx,
        verbose=True,
    )

    extract_elapsed = time.perf_counter() - extract_start

    print(
        f"[rollout {rollout_idx:02d}] finished multistep extraction | "
        f"samples kept: {len(X)} | "
        f"extract time: {extract_elapsed:.2f}s"
    )

    return {
        "rollout_idx": rollout_idx,
        "seed": int(rollout_seed),
        "num_samples": len(X),
        "state": X,
        "u_hist": U,
        "predictor_traj": Y,
        "t": T
    }

def build_multistep_predictor_dataset_parallel(
    cfg,
    n_rollouts=20,
    stride=2,
    seed=0,
    max_workers=None,
    flatten_target=False,
    q_meas_noise_std=0.01,
    v_meas_noise_std=0.01,
    use_noisy_measurement_for_reset=True,
    verbose=True,
):
    """Build the full Case 1 sampling-horizon prediction dataset in parallel across multiple rollouts.

    Inputs:  cfg dict, n_rollouts, stride, seed, max_workers, flatten_target,
             q_meas_noise_std, v_meas_noise_std, use_noisy_measurement_for_reset, verbose
    Returns: dataset dict with "state", "u_hist", "predictor_traj",
             "rollout_seeds", "samples_per_rollout", "config"
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if verbose:
        print("\n==============================")
        print("PARALLEL MULTISTEP DATASET GENERATION")
        print("==============================")
        print(f"rollouts        : {n_rollouts}")
        print(f"workers         : {max_workers}")
        print(f"stride          : {stride}")
        print(f"flatten_target  : {flatten_target}")
        print(f"sim steps       : {cfg['steps']}")
        print(f"delay steps     : {cfg['delay_steps']}")
        print(f"sample steps    : {cfg['sample_steps']}")
        print(f"seed            : {seed}")
        print("==============================\n")

    master_rng = np.random.default_rng(seed)

    # Generate one independent 32-bit seed per rollout for reproducibility.
    rollout_seeds = master_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_rollouts,
        dtype=np.uint32,
    )

    task_args = [
        (
            i,
            int(rollout_seeds[i]),
            cfg,
            stride,
            flatten_target,
            q_meas_noise_std,
            v_meas_noise_std,
            use_noisy_measurement_for_reset,
        )
        for i in range(n_rollouts)
    ]

    results = [None] * n_rollouts
    total_samples = 0

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one_multistep_rollout, arg) for arg in task_args]

        completed = 0

        # as_completed yields in finish order; use rollout_idx to slot results correctly.
        for fut in as_completed(futures):
            result = fut.result()

            i = result["rollout_idx"]
            results[i] = result

            completed += 1
            samples = result["num_samples"]
            total_samples += samples

            elapsed = time.perf_counter() - start_time

            if verbose:
                print(
                    f"[{completed:>3}/{n_rollouts}] "
                    f"rollout {i:>3} finished | "
                    f"multistep samples: {samples:>6} | "
                    f"total samples: {total_samples:>7} | "
                    f"time: {elapsed:6.2f}s"
                )

    elapsed = time.perf_counter() - start_time

    nonempty = [r for r in results if r is not None and r["num_samples"] > 0]

    if len(nonempty) == 0:
        raise RuntimeError("No rollout produced any multistep samples.")

    T_all = np.vstack([r["t"] for r in nonempty]).ravel()
    X_all = np.vstack([r["state"] for r in nonempty])
    U_all = np.vstack([r["u_hist"] for r in nonempty])

    # 3D trajectories need concatenate; 2D flat targets can use vstack.
    if flatten_target:
        Y_all = np.vstack([r["predictor_traj"] for r in nonempty])
    else:
        Y_all = np.concatenate([r["predictor_traj"] for r in nonempty], axis=0)

    samples_per_rollout = np.array(
        [0 if r is None else r["num_samples"] for r in results],
        dtype=int,
    )

    total_samples = X_all.shape[0]
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    if verbose:
        print("\n==============================")
        print("MULTISTEP DATASET BUILD COMPLETE")
        print("==============================")
        print(f"wall time          : {elapsed:.2f} sec")
        print(f"total samples      : {total_samples}")
        print(f"samples/sec        : {samples_per_sec:.1f}")
        print("")
        print(f"state shape        : {X_all.shape}")
        print(f"u_hist shape       : {U_all.shape}")
        print(f"predictor_traj shape : {Y_all.shape}")
        print("")
        print("samples per rollout:")
        print(samples_per_rollout)
        print("==============================\n")

    return {
        "state": X_all,
        "u_hist": U_all,
        "predictor_traj": Y_all,
        "t": T_all, 
        "rollout_seeds": rollout_seeds.copy(),
        "samples_per_rollout": samples_per_rollout,
        "config": cfg,
    }
