import torch
import numpy as np


def make_learned_single_step_predictor(model, stats, device):
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def predict(q_in, v_in, u_hist):
        """
        Inputs
        ------
        q_in   : (nq,)
        v_in   : (nv,)
        u_hist : (delay_steps, nv)

        Returns
        -------
        q_pred, v_pred
        """
        state = np.concatenate([q_in, v_in], axis=0)
        delay_steps = u_hist.shape[0]

        state_rep = np.repeat(state[None, :], delay_steps, axis=0)
        x = np.concatenate([state_rep, u_hist], axis=1)
        x = x[None, :, :]

        x_norm = (x - x_mean) / x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)

        y_norm = model(x_tensor).cpu().numpy()
        y = y_norm * y_std + y_mean
        y = y[0]

        nq = q_in.shape[0]
        q_pred = y[:nq]
        v_pred = y[nq:]
        return q_pred, v_pred

    return {
        "kind": "single_step",
        "predict": predict,
    }

def make_numerical_predictor(sim, cfg):
    def predict(q_in, v_in, u_hist):
        q_pred, v_pred, _ = sim["approximate_predictor"](
            q_in,
            v_in,
            u_hist,
            h=cfg["h_pred"],
            tol=cfg["predictor_tolerance"],
            max_iters=cfg["max_picard_iters"],
            M=cfg["inner_predictor_discretization_steps"],
        )
        return q_pred, v_pred

    return {
        "kind": "numerical",
        "predict": predict,
    }

def make_learned_multistep_predictor(model, stats, device, robot, cfg):
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    nq = robot["nq"]
    nv = robot["nv"]
    nx = nq + nv
    sample_steps = cfg["sample_steps"]

    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def predict(q_in, v_in, u_hist):
        """
        Inputs
        ------
        q_in   : (nq,)
        v_in   : (nv,)
        u_hist : (delay_steps, nv)

        Returns
        -------
        z_traj : (sample_steps + 1, nq + nv)
        """
        state = np.concatenate([q_in, v_in], axis=0)
        delay_steps = u_hist.shape[0]

        state_rep = np.repeat(state[None, :], delay_steps, axis=0)
        x = np.concatenate([state_rep, u_hist], axis=1)
        x = x[None, :, :]

        x_norm = (x - x_mean) / x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)

        y_norm = model(x_tensor).cpu().numpy()
        y = y_norm * y_std + y_mean
        y = y[0]

        z_traj = y.reshape(sample_steps + 1, nx)
        return z_traj

    return {
        "kind": "multistep",
        "predict": predict,
    }