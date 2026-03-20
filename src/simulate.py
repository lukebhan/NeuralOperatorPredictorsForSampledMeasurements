"""Simulator and predictor utilities.

Wires together the Pinocchio plant, RK4 integrator, hybrid controller,
Picard-based numerical predictor, and pluggable learned predictor loop.
"""

import pinocchio as pin
import numpy as np
import time


def sample_initial_state(robot, rng, q_scale=0.01, v_scale=0.01):
    """Sample a random initial state near the robot's neutral configuration.

    Inputs:  robot bundle, seeded rng, perturbation half-widths q_scale / v_scale
    Returns: (q0, v0) ndarrays
    """
    model = robot["model"]

    # uniform perturbation in tangent space
    dq = rng.uniform(-q_scale, q_scale, size=robot["nq"])
    dv = rng.uniform(-v_scale, v_scale, size=robot["nv"])

    q0 = pin.integrate(
        model,
        pin.neutral(model),
        dq
    )

    v0 = dv

    return q0, v0

def build_robot(urdf):
    """Load a Pinocchio robot model from a URDF file.

    Inputs:  urdf path string
    Returns: dict with keys "model", "data", "nq", "nv"
    Raises:  ValueError if nq != nv
    """
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()

    nq = model.nq
    nv = model.nv

    if nq != nv:
        raise ValueError(f"This implementation assumes nq == nv, got nq={nq}, nv={nv}")

    return {
        "model": model,
        "data": data,
        "nq": nq,
        "nv": nv,
    }


def make_reference(robot, cfg):
    """Build sinusoidal desired-trajectory callables for the robot joints.

    Inputs:  robot bundle, cfg dict (uses "traj_w" and "traj_amp")
    Returns: dict with callables "q_des"(t), "v_des"(t), "a_des"(t)
    """
    model = robot["model"]
    nq = robot["nq"]

    q_ref0 = pin.neutral(model).copy()
    # Stagger joint phases so joints don't all peak simultaneously
    phases = np.linspace(0.0, np.pi / 2.0, nq)

    traj_w = cfg["traj_w"]
    traj_amp = cfg["traj_amp"]

    def q_des(t):
        return q_ref0 + traj_amp * np.sin(traj_w * t + phases)

    def v_des(t):
        return traj_amp * traj_w * np.cos(traj_w * t + phases)

    def a_des(t):
        return -traj_amp * (traj_w ** 2) * np.sin(traj_w * t + phases)

    return {
        "q_des": q_des,
        "v_des": v_des,
        "a_des": a_des,
    }

def make_simulator(robot, cfg, ref):
    """Construct all simulation primitives as a callable bundle.

    Closes over robot, config, and reference to produce plant integrators,
    the hybrid controller, the Picard predictor, and the main simulate loop.

    Inputs:  robot bundle, cfg dict, ref bundle
    Returns: dict of callables (plant, controller, predictor, simulate, etc.)
    """
    model = robot["model"]
    data = robot["data"]
    nq = robot["nq"]
    nv = robot["nv"]

    q_des = ref["q_des"]
    v_des = ref["v_des"]
    a_des = ref["a_des"]

    dt = cfg["dt"]
    D = cfg["D"]
    tau_max = cfg["tau_max"]
    plant_input_noise_std = cfg["noise_std"]
    h_pred = cfg["h_pred"]
    delay_steps = cfg["delay_steps"]
    sample_steps = cfg["sample_steps"]
    predictor_tolerance = cfg["predictor_tolerance"]
    max_picard_iters = cfg["max_picard_iters"]
    inner_M = cfg["inner_predictor_discretization_steps"]

    Kp = cfg["Kp_val"] * np.eye(nv)
    Kd = cfg["Kd_val"] * np.eye(nv)

    # --------------------------------------------------------
    # Basic helpers
    # --------------------------------------------------------

    def clip_tau(tau):
        return np.clip(tau, -tau_max, tau_max)

    def split_state(x):
        q = x[:nq]
        v = x[nq:]
        return q, v

    def pack_state(q, v):
        return np.concatenate([q, v])

    def config_error(q, q_target):
        return pin.difference(model, q, q_target)

    def state_rhs(x, tau):
        q, v = split_state(x)
        a = pin.aba(model, data, q, v, tau)
        return np.concatenate([v, a])

    def make_noisy_state(q, v, q_noise_std=0.0, v_noise_std=0.0, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        dq_noise = q_noise_std * rng.standard_normal(nq)
        dv_noise = v_noise_std * rng.standard_normal(nv)

        q_noisy = pin.integrate(model, q, dq_noise)
        v_noisy = v + dv_noise
        return q_noisy, v_noisy

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    def nominal_controller(q, v, t_eval):
        qd = q_des(t_eval)
        vd = v_des(t_eval)
        ad = a_des(t_eval)

        e = config_error(q, qd)
        ev = vd - v

        a_cmd = ad + Kd @ ev + Kp @ e
        tau = pin.rnea(model, data, q, v, a_cmd)
        return clip_tau(tau)

    def hybrid_feedback_tau(qz, vz, t_now):
        # qz predicts the plant D seconds ahead, so target the reference at t + D.
        return nominal_controller(qz, vz, t_now + D)

    # --------------------------------------------------------
    # Plant integrator
    # --------------------------------------------------------

    def plant_step_rk4(q, v, tau, h):
        def accel(qi, vi):
            return pin.aba(model, data, qi, vi, tau)

        k1_q = v
        k1_v = accel(q, v)

        q2 = pin.integrate(model, q, 0.5 * h * k1_q)
        v2 = v + 0.5 * h * k1_v
        k2_q = v2
        k2_v = accel(q2, v2)

        q3 = pin.integrate(model, q, 0.5 * h * k2_q)
        v3 = v + 0.5 * h * k2_v
        k3_q = v3
        k3_v = accel(q3, v3)

        q4 = pin.integrate(model, q, h * k3_q)
        v4 = v + h * k3_v
        k4_q = v4
        k4_v = accel(q4, v4)

        q_inc = (h / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        v_next = v + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        q_next = pin.integrate(model, q, q_inc)

        return q_next, v_next

    # --------------------------------------------------------
    # Internal hybrid state integrator
    # --------------------------------------------------------

    def controller_state_step_rk4(qz, vz, t, h):
        def accel(qi, vi, ti):
            tau_i = hybrid_feedback_tau(qi, vi, ti)
            return pin.aba(model, data, qi, vi, tau_i)

        k1_q = vz
        k1_v = accel(qz, vz, t)

        q2 = pin.integrate(model, qz, 0.5 * h * k1_q)
        v2 = vz + 0.5 * h * k1_v
        k2_q = v2
        k2_v = accel(q2, v2, t + 0.5 * h)

        q3 = pin.integrate(model, qz, 0.5 * h * k2_q)
        v3 = vz + 0.5 * h * k2_v
        k3_q = v3
        k3_v = accel(q3, v3, t + 0.5 * h)

        q4 = pin.integrate(model, qz, h * k3_q)
        v4 = vz + h * k3_v
        k4_q = v4
        k4_v = accel(q4, v4, t + h)

        q_inc = (h / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        v_next = vz + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        q_next = pin.integrate(model, qz, q_inc)

        return q_next, v_next

    # --------------------------------------------------------
    # Predictor
    # --------------------------------------------------------

    def local_picard_map(z0, tau_i, h, tol=1e-6, max_iters=20, M=4):
        # Trapezoid-rule Picard iteration over M sub-intervals of length h/M,
        # repeated until the grid converges or max_iters is reached.
        ds = h / M
        y = np.tile(z0[None, :], (M + 1, 1))

        for it in range(max_iters):
            y_new = np.zeros_like(y)
            y_new[0] = z0.copy()

            cumulative = np.zeros_like(z0)

            for j in range(1, M + 1):
                f_left = state_rhs(y[j - 1], tau_i)
                f_right = state_rhs(y[j], tau_i)
                cumulative += 0.5 * ds * (f_left + f_right)
                y_new[j] = z0 + cumulative

            if np.max(np.linalg.norm(y_new - y, axis=1)) < tol:
                return y_new[-1].copy(), it + 1

            y = y_new

        return y[-1].copy(), max_iters

    def approximate_predictor(q_now, v_now, u_history, h, tol=1e-6, max_iters=20, M=4):
        # Chain one Picard step per delay interval. Third return value is the
        # mean iteration count per step, useful as a convergence diagnostic.
        z = pack_state(q_now, v_now)
        total_picard_iters = 0

        for tau_i in u_history:
            z, iters = local_picard_map(
                z, tau_i, h, tol=tol, max_iters=max_iters, M=M
            )
            total_picard_iters += iters

        q_pred, v_pred = split_state(z)
        return q_pred, v_pred, total_picard_iters / len(u_history)

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    def simulate(
        q0=None,
        v0=None,
        q_meas_noise_std=0.0,
        v_meas_noise_std=0.0,
        use_noisy_measurement_for_reset=False,
        rng=None,
        verbose=True,
        log_every_step=True,
        rollout_id=None,          
        progress_interval=5000,   
    ):
        if rng is None:
            rng = np.random.default_rng()

        # Initial condition
        if q0 is None:
            q = pin.neutral(model).copy()
        else:
            q = q0.copy()

        if v0 is None:
            v = np.zeros(nv)
        else:
            v = v0.copy()

        # Pre-fill the delay buffer with the gravity-hold torque (stand-in for t < 0).
        tau_hold = pin.rnea(model, data, q, v, np.zeros(nv))
        tau_hold = clip_tau(tau_hold)

        u_buffer = [tau_hold.copy() for _ in range(delay_steps)]

        # Warm-start qz, vz with one predictor call so they're consistent at t=0.
        q_init_pred, v_init_pred, _ = approximate_predictor(
            q, v, u_buffer,
            h=h_pred,
            tol=predictor_tolerance,
            max_iters=max_picard_iters,
            M=inner_M,
        )
        qz = q_init_pred.copy()
        vz = v_init_pred.copy()

        ts = []
        qs = []
        vs = []
        q_meas_log = []
        v_meas_log = []
        qzs = []
        vzs = []
        q_sample_pred_log = []
        v_sample_pred_log = []
        tau_cmds = []
        tau_apps = []
        u_hist_log = []
        q_des_now_log = []
        q_des_future_log = []
        picard_log = []
        sample_flag_log = []
        predictor_time_log = []
        step_time_log = []

        predictor_time_window = 0.0
        step_time_window = 0.0
        window_count = 0

        sim_start = time.perf_counter()

        for k in range(cfg["steps"]):
            if rollout_id is not None and progress_interval > 0:
                if k % progress_interval == 0 and k > 0:
                    pct = 100.0 * k / cfg["steps"]
                    print(
                        f"[rollout {rollout_id:02d}] "
                        f"step {k}/{cfg['steps']} "
                        f"({pct:5.1f}%)"
                    )
            step_start = time.perf_counter()
            t = k * dt
            did_sample = (k % sample_steps == 0)

            predictor_elapsed = 0.0
            q_sample_pred = qz.copy()
            v_sample_pred = vz.copy()
            picard_used = np.nan

            # Measured state at current time
            q_meas, v_meas = make_noisy_state(
                q, v,
                q_noise_std=q_meas_noise_std,
                v_noise_std=v_meas_noise_std,
                rng=rng,
            )

            # Reset at sample times
            if did_sample:
                q_reset = q_meas if use_noisy_measurement_for_reset else q
                v_reset = v_meas if use_noisy_measurement_for_reset else v

                predictor_start = time.perf_counter()
                q_sample_pred, v_sample_pred, picard_used = approximate_predictor(
                    q_reset, v_reset, u_buffer,
                    h=h_pred,
                    tol=predictor_tolerance,
                    max_iters=max_picard_iters,
                    M=inner_M,
                )
                predictor_elapsed = time.perf_counter() - predictor_start

                qz = q_sample_pred.copy()
                vz = v_sample_pred.copy()

            # Control from internal state
            tau_cmd = hybrid_feedback_tau(qz, vz, t)

            # FIFO delay: push new command, pop the D-step-old torque that's applied now.
            u_buffer.append(tau_cmd.copy())
            tau_applied = u_buffer.pop(0)

            if plant_input_noise_std > 0.0:
                tau_applied = tau_applied + plant_input_noise_std * rng.standard_normal(nv)

            # Plant step
            q, v = plant_step_rk4(q, v, tau_applied, dt)

            # Internal controller state step
            qz, vz = controller_state_step_rk4(qz, vz, t, dt)

            step_elapsed = time.perf_counter() - step_start

            if log_every_step:
                ts.append(t)
                qs.append(q.copy())
                vs.append(v.copy())
                q_meas_log.append(q_meas.copy())
                v_meas_log.append(v_meas.copy())
                qzs.append(qz.copy())
                vzs.append(vz.copy())
                q_sample_pred_log.append(q_sample_pred.copy())
                v_sample_pred_log.append(v_sample_pred.copy())
                tau_cmds.append(tau_cmd.copy())
                tau_apps.append(tau_applied.copy())
                u_hist_log.append(np.array(u_buffer, dtype=float).copy())
                q_des_now_log.append(q_des(t))
                q_des_future_log.append(q_des(t + D))  # desired pos the controller is targeting
                picard_log.append(picard_used)
                predictor_time_log.append(predictor_elapsed)
                step_time_log.append(step_elapsed)
                sample_flag_log.append(did_sample)

            predictor_time_window += predictor_elapsed
            step_time_window += step_elapsed
            window_count += 1

            if verbose and (k + 1) % 100 == 0:
                avg_pred_ms = 1000.0 * predictor_time_window / window_count
                avg_step_ms = 1000.0 * step_time_window / window_count
                print(
                    f"step {k+1}/{cfg['steps']} | "
                    f"avg predictor time: {avg_pred_ms:.3f} ms/step | "
                    f"avg total sim time: {avg_step_ms:.3f} ms/step"
                )
                predictor_time_window = 0.0
                step_time_window = 0.0
                window_count = 0

        total_sim_time = time.perf_counter() - sim_start
        if verbose:
            print(f"total wall-clock simulation time: {total_sim_time:.3f} s")

        return {
            "t": np.array(ts),
            "q": np.array(qs),
            "v": np.array(vs),
            "q_meas": np.array(q_meas_log),
            "v_meas": np.array(v_meas_log),
            "qz": np.array(qzs),
            "vz": np.array(vzs),
            "q_sample_pred": np.array(q_sample_pred_log),
            "v_sample_pred": np.array(v_sample_pred_log),
            "tau_cmd": np.array(tau_cmds),
            "tau_applied": np.array(tau_apps),
            "u_hist": np.array(u_hist_log),
            "q_des_now": np.array(q_des_now_log),
            "q_des_future": np.array(q_des_future_log),
            "picard_iters": np.array(picard_log),
            "predictor_time": np.array(predictor_time_log),
            "step_time": np.array(step_time_log),
            "sample_flag": np.array(sample_flag_log),
            "tau_hold": tau_hold.copy(),
            "q0": q0.copy() if q0 is not None else pin.neutral(model).copy(),
            "v0": v0.copy() if v0 is not None else np.zeros(nv),
            "config": cfg,
        }

    return {
        "clip_tau": clip_tau,
        "split_state": split_state,
        "pack_state": pack_state,
        "config_error": config_error,
        "state_rhs": state_rhs,
        "make_noisy_state": make_noisy_state,
        "nominal_controller": nominal_controller,
        "hybrid_feedback_tau": hybrid_feedback_tau,
        "plant_step_rk4": plant_step_rk4,
        "controller_state_step_rk4": controller_state_step_rk4,
        "local_picard_map": local_picard_map,
        "approximate_predictor": approximate_predictor,
        "simulate": simulate,
    }


def simulate_with_predictor(
    sim,
    robot,
    ref,
    cfg,
    predictor,
    q0=None,
    v0=None,
    q_meas_noise_std=0.0,
    v_meas_noise_std=0.0,
    torque_noise_std=0.0,
    process_noise_std=0.0,
    rng=None,
    random_sampling=False,
    max_sample_h=None,
    min_sample_h=None,
):
    """Simulate the closed-loop system using a pluggable predictor.

    Runs the delayed-feedback loop and resets the hybrid internal state at
    each sampling instant via predictor["predict"]. Supports single-step,
    numerical, and multistep predictor kinds and optional random sample gaps.

    Inputs:  sim, robot, ref, cfg bundles; predictor dict with "kind" and
             "predict" callable; initial state, noise std devs, rng, sampling params
    Returns: dict of logged trajectory arrays
    """
    model = robot["model"]
    data = robot["data"]
    nv = robot["nv"]
    nq = robot["nq"]

    dt = cfg["dt"]
    D = cfg["D"]
    delay_steps = cfg["delay_steps"]
    sample_steps = cfg["sample_steps"]

    clip_tau = sim["clip_tau"]
    plant_step_rk4 = sim["plant_step_rk4"]
    controller_state_step_rk4 = sim["controller_state_step_rk4"]
    hybrid_feedback_tau = sim["hybrid_feedback_tau"]
    pack_state = sim["pack_state"]

    predictor_kind = predictor["kind"]
    predictor_call = predictor["predict"]

    valid_kinds = {"single_step", "numerical", "multistep"}
    if predictor_kind not in valid_kinds:
        raise ValueError(f"Unknown predictor kind '{predictor_kind}'. Expected one of {valid_kinds}.")

    if predictor_kind == "multistep" and random_sampling:
        raise ValueError("multistep predictor requires fixed sampling; random_sampling must be False.")

    if rng is None:
        rng = np.random.default_rng()

    # ---------------------------------------------------------
    # Sampling schedule setup
    # ---------------------------------------------------------
    if max_sample_h is None:
        max_sample_h = sample_steps * dt
    if min_sample_h is None:
        min_sample_h = dt

    max_sample_steps = max(1, int(np.floor(max_sample_h / dt)))
    min_sample_steps = max(1, int(np.ceil(min_sample_h / dt)))

    if min_sample_steps > max_sample_steps:
        raise ValueError("min_sample_h must be <= max_sample_h")

    def draw_next_gap_steps():
        if not random_sampling:
            return sample_steps
        # upper bound is exclusive, so +1 to include max_sample_steps
        return int(rng.integers(min_sample_steps, max_sample_steps + 1))

    next_sample_step = 0  # first sample happens at step 0

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------
    if q0 is None:
        q = pin.neutral(model).copy()
    else:
        q = q0.copy()

    if v0 is None:
        v = np.zeros(nv)
    else:
        v = v0.copy()

    # Pre-fill delay buffer with gravity-hold torque (same convention as simulate()).
    tau_hold = pin.rnea(model, data, q, v, np.zeros(nv))
    tau_hold = clip_tau(tau_hold)

    u_buffer = [tau_hold.copy() for _ in range(delay_steps)]

    # ---------------------------------------------------------
    # Internal predictor / controller state initialization
    # ---------------------------------------------------------
    qz = q.copy()
    vz = v.copy()

    # Current multistep trajectory and index into it.
    z_traj_active = None
    z_traj_index = 0

    # Warm-start the internal state at t=0.
    if predictor_kind in {"single_step", "numerical"}:
        qz, vz = predictor_call(q, v, np.array(u_buffer))
    elif predictor_kind == "multistep":
        z_traj_active = predictor_call(q, v, np.array(u_buffer))
        z_traj_active = np.asarray(z_traj_active)

        expected_shape = (sample_steps + 1, nq + nv)
        if z_traj_active.shape != expected_shape:
            raise ValueError(
                f"multistep predictor returned shape {z_traj_active.shape}, "
                f"expected {expected_shape}"
            )

        z0 = z_traj_active[0]
        qz = z0[:nq].copy()
        vz = z0[nq:].copy()
        z_traj_index = 0

    # ---------------------------------------------------------
    # Logs
    # ---------------------------------------------------------
    ts, qs, vs = [], [], []
    q_meas_log, v_meas_log = [], []
    qzs, vzs = [], []
    tau_cmds, tau_apps = [], []
    u_hist_log = []
    q_des_now_log, q_des_future_log = [], []
    sample_flag_log = []
    sample_time_log = []
    sample_gap_log = []

    last_sample_time = 0.0

    # ---------------------------------------------------------
    # Simulation loop
    # ---------------------------------------------------------
    for k in range(cfg["steps"]):
        t = k * dt
        did_sample = (k == next_sample_step)

        # ---------------------------------
        # Measurement noise
        # ---------------------------------
        if q_meas_noise_std > 0 or v_meas_noise_std > 0:
            dq_noise = q_meas_noise_std * rng.standard_normal(nq)
            dv_noise = v_meas_noise_std * rng.standard_normal(nv)

            q_meas = pin.integrate(model, q, dq_noise)
            v_meas = v + dv_noise
        else:
            q_meas = q.copy()
            v_meas = v.copy()

        # ---------------------------------
        # Sampled predictor reset/update
        # ---------------------------------
        if did_sample:
            u_hist_arr = np.array(u_buffer)

            if predictor_kind in {"single_step", "numerical"}:
                qz, vz = predictor_call(q_meas, v_meas, u_hist_arr)

            elif predictor_kind == "multistep":
                z_traj_active = predictor_call(q_meas, v_meas, u_hist_arr)
                z_traj_active = np.asarray(z_traj_active)

                expected_shape = (sample_steps + 1, nq + nv)
                if z_traj_active.shape != expected_shape:
                    raise ValueError(
                        f"multistep predictor returned shape {z_traj_active.shape}, "
                        f"expected {expected_shape}"
                    )

                z_traj_index = 0
                z0 = z_traj_active[0]
                qz = z0[:nq].copy()
                vz = z0[nq:].copy()

            sample_time_log.append(t)
            sample_gap_log.append(t - last_sample_time if len(sample_time_log) > 1 else 0.0)
            last_sample_time = t

            next_sample_step = k + draw_next_gap_steps()

        # ---------------------------------
        # Control from internal predictor state
        # ---------------------------------
        tau_cmd = hybrid_feedback_tau(qz, vz, t)

        u_buffer.append(tau_cmd.copy())
        tau_applied = u_buffer.pop(0)

        # ---------------------------------
        # Actuator noise
        # ---------------------------------
        if torque_noise_std > 0:
            tau_applied = tau_applied + torque_noise_std * rng.standard_normal(nv)

        tau_applied = clip_tau(tau_applied)

        # ---------------------------------
        # Plant step
        # ---------------------------------
        q, v = plant_step_rk4(q, v, tau_applied, dt)

        if process_noise_std > 0:
            v = v + process_noise_std * rng.standard_normal(nv)

        # ---------------------------------
        # Internal predictor evolution
        # ---------------------------------
        if predictor_kind in {"single_step", "numerical"}:
            qz, vz = controller_state_step_rk4(qz, vz, t, dt)

        elif predictor_kind == "multistep":
            # Advance along the pre-computed trajectory; clamp if sample is late.
            z_traj_index = min(z_traj_index + 1, sample_steps)
            z_next = z_traj_active[z_traj_index]
            qz = z_next[:nq].copy()
            vz = z_next[nq:].copy()

        # ---------------------------------
        # Logging
        # ---------------------------------
        ts.append(t)
        qs.append(q.copy())
        vs.append(v.copy())
        q_meas_log.append(q_meas.copy())
        v_meas_log.append(v_meas.copy())
        qzs.append(qz.copy())
        vzs.append(vz.copy())
        tau_cmds.append(tau_cmd.copy())
        tau_apps.append(tau_applied.copy())
        u_hist_log.append(np.array(u_buffer, dtype=float).copy())
        q_des_now_log.append(ref["q_des"](t))
        q_des_future_log.append(ref["q_des"](t + D))  # desired pos the controller is targeting
        sample_flag_log.append(did_sample)

    return {
        "predictor_kind": predictor_kind,
        "t": np.array(ts),
        "q": np.array(qs),
        "v": np.array(vs),
        "q_meas": np.array(q_meas_log),
        "v_meas": np.array(v_meas_log),
        "qz": np.array(qzs),
        "vz": np.array(vzs),
        "tau_cmd": np.array(tau_cmds),
        "tau_applied": np.array(tau_apps),
        "u_hist": np.array(u_hist_log),
        "q_des_now": np.array(q_des_now_log),
        "q_des_future": np.array(q_des_future_log),
        "sample_flag": np.array(sample_flag_log),
        "sample_times": np.array(sample_time_log),
        "sample_gaps": np.array(sample_gap_log),
    }