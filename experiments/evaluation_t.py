import copy
import numpy as np

def sample_trajectory_given_observed_actions(
    env,
    n,
    T,
    Z,
    seed,
    policy=None,
    include_counter=True,
):
    """
    this is a different version of sample trajectory, where the historical actions
    are observed actions for both factual trajectory and counterfactual trajectory
    """
    # to isolate buffer, we need to copy the policy
    policy_c = copy.deepcopy(policy)
    #Z = zs
    #n = zs.shape[0]

    np.random.seed(seed)  # random traj
    Z = Z.reshape(n, -1)
    Z_c = 1 - Z

    X = np.zeros([n, T + 1, 1], dtype=float)
    A = np.zeros([n, T], dtype=int)
    R = np.zeros([n, T], dtype=float)

    X_c = np.zeros([n, T + 1, 1], dtype=float)
    A_c = np.zeros([n, T], dtype=int)
    R_c = np.zeros([n, T], dtype=float)

    # t == 0
    ux0 = np.random.normal(0, 1, [n, 1])
    X[:, 0, 0] = env.f_x0(
        zs=Z,
        ux0=ux0,
    ).flatten()
    X_c[:, 0, 0] = env.f_x0(
        zs=Z_c,
        ux0=ux0,
    ).flatten()

    ua0 = np.random.uniform(0, 1, size=[n])

    A[:, 0] = policy.act(
        z=Z,
        xt=X[:, 0],
        #at=A[:, :0],
        #t=0,
        xtm1=None,
        atm1=None,
        uat=ua0,
    )
    A_c[:, 0] = policy_c.act(
        z=Z_c,
        xt=X_c[:, 0],
        #at=A[:, :0],
        #t=0,
        xtm1=None,
        atm1=None,
        uat=ua0,
    )

    ur0 = np.random.normal(0, 1, [n, 1])
    # R[:, 0] = self.f_rt(xt=X[:, 0], z=Z, at=A[:, 0], urt=ur0)
    # R_c[:, 0] = self.f_rt(xt=X_c[:, 0], z=Z_c, at=A_c[:, 0], urt=ur0)

    # t = 1 to T-1
    for t in range(1, T):
        uxt = np.random.normal(0, 1, [n, 1])
        X[:, t, 0] = env.f_xt(xtm1=X[:, t - 1], zs=Z, atm1=A[:, t - 1], uxt=uxt).flatten()
        X_c[:, t, 0] = env.f_xt(
            xtm1=X_c[:, t - 1], zs=Z_c, atm1=A[:, t - 1], uxt=uxt
        ).flatten()  # change atm1 to oberseved actions
        # print("sim,", t, X[:5, t, :])
        uat = np.random.uniform(0, 1, size=[n])
        # print("cf_g: uat", uat[:5])
        A[:, t] = policy.act(
            z=Z,
            xt=X[:, t],
            #at=A[:, :t],
            #t=t,
            xtm1=X[:, t - 1],
            atm1=A[:, t - 1],
            uat=uat,
        )
        A_c[:, t] = policy_c.act(
            z=Z_c,
            xt=X_c[:, t],
            #at=A[:, :t],
            #t=t,
            xtm1=X_c[:, t - 1],
            atm1=A[:, t - 1],
            uat=uat,
        )
        # if np.any(A[:, t] != A_c[:, t]):
        #     print("A[:,t] != A_c[:,t]", t, A[:, t], A_c[:, t])
        #     # print which one is different
        #     idx_diff = np.where(A[:, t] != A_c[:, t])[0]
        #     print("idx_diff", idx_diff)
        # correct the buffer
        # if hasattr(policy_c, "preprocessor"):
        #     if hasattr(policy_c.preprocessor, "buffer"):
        #         X_tilde = {Z}

        urt = np.random.normal(0, 1, [n, 1])
        # R[:, t] = self.f_rt(xt=X[:, t], z=Z, at=A[:, t], urt=urt)
        # R_c[:, t] = self.f_rt(xt=X_c[:, t], z=Z_c, at=A_c[:, t], urt=urt)

    # t = T
    uxt = np.random.normal(0, 1, [n, 1])
    X[:, T, 0] = env.f_xt(
        xtm1=X[:, T - 1],
        zs=Z,
        atm1=A[:, T - 1],
        uxt=uxt,
    ).flatten()
    X_c[:, T, 0] = env.f_xt(
        xtm1=X_c[:, T - 1],
        zs=Z_c,
        atm1=A[:, T - 1],
        uxt=uxt,
    ).flatten()

    # check
    # for i in range(T):
    #     print("sim,", i, X[:5, i, :])
    #     if i == 5:
    #         break

    out = [Z, X, A, None]
    if include_counter:
        out.extend([Z_c, X_c, A_c, None])
    return out





def evaluate_fairness_through_simulation(
    env, z_eval_levels, state_dim, N, T, policy, seed
):
    np.random.seed(seed)
    Z = np.random.binomial(n=1, p=0.5, size=[N])
    (_, _, actions_real, _, _, _, actions_counter, _) = (
        sample_trajectory_given_observed_actions(
            env=env, 
            n=N,
            T=T,
            seed=seed,
            Z=Z,
            policy=policy,
            include_counter=True,
        )
    )

    cf_metric_t = np.mean(np.abs(actions_real - actions_counter), axis=0)
    cf_metric = np.mean(cf_metric_t)
    return cf_metric#, cf_metric_t
