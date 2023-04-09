import numpy as np
import pandas as pd
import os


def full_synthetic_dataset(gamma, time, k, s, h, N, N_treated, p):
    for gamma_h in gamma:
        def distributions(N, N_treated, time, k):
            # Defining eta and episilon are randomly sampled noise
            eta, epsilon = np.random.normal(0, 0.001, size=(N + N_treated, time, k)), np.random.normal(0, 0.001, size=(
            N + N_treated, time, h))

            # w and b are the distribution parameters for gamma h.
            w = np.random.uniform(-1, 1, size=(h, 2))
            b = np.random.normal(0, 0.1, size=(N + N_treated, 2))

            return eta, epsilon, w, b

        def control_treatment(N, N_treated, time):
            # For treated samples, we randomly pick the treatment initial starting point among all timestamps
            # The treatments starting from the initial point are all set to 1.
            treatment = np.zeros(shape=(N_treated, time))
            for treated in range(N_treated):
                initial_point = np.random.choice(range(time))
                a = np.zeros(time)
                a[initial_point:] = 1
                treatment[treated] = a

            np.random.shuffle(treatment)
            # For Control samples the treatements at each time stamp are all set to 0
            control = np.zeros(shape=(N, time))

            # joined dataset with treatment and control
            samples = np.concatenate([treatment, control])

            samples_final = np.where(np.sum(samples, axis=1) > 0, 1, 0)

            return treatment, control, samples, samples_final

        def features(N, N_treated, s, k, h):
            # Create and simulate arrays for current time varying covarites (X) and static features (c) and
            # hidden confounders Z
            C = np.random.normal(0, 0.5, size=(N + N_treated, s))
            X = np.random.normal(0, 0.5, size=(N + N_treated, k))
            Z = np.random.normal(0, 0.5, size=(N + N_treated, h))
            X[np.where(np.sum(samples, axis=1) > 0), :] = np.random.normal(1, .5, size=(N_treated, k))
            C[np.where(np.sum(samples, axis=1) > 0), :] = np.random.normal(1, .5, size=(N_treated, s))
            Z[np.where(np.sum(samples, axis=1) > 0), :] = np.random.normal(1, .5, size=(N_treated, h))
            return C, X, Z

        def covariates_simulation(X, Z, C, gamma_h, time, N, N_treated, k, p, eta, epsilon):
            # Simulate covariates X and hidden confounders Z
            final_X = [X]
            final_Z = [Z]
            for t in range(1, time + 1):
                # t +=1
                x = 0
                z = 0
                r = 1
                while (t - r) >= 0 and r <= p:
                    alpha = np.random.normal(1 - (r / p), (1 / p), size=(N + N_treated, k))
                    beta = np.random.normal(0, .02, size=(N + N_treated, k))
                    beta[np.where(np.sum(samples, axis=1) > 0), :] = np.random.normal(1, .02, size=(N_treated, k))
                    final_a = np.multiply(alpha, final_X[t - r])
                    final_b = np.multiply(beta, np.tile(samples[:, t - r], (k, 1)).T)
                    x += final_a + final_b

                    mu = np.random.normal(1 - (r / p), (1 / p), size=(N + N_treated, h))
                    upsilon = np.random.normal(0, .02, size=(N + N_treated, h))
                    upsilon[np.where(np.sum(samples, axis=1) > 0), :] = np.random.normal(1, .02, size=(N_treated, h))
                    final_m = np.multiply(mu, final_Z[t - r])
                    final_u = np.multiply(upsilon, np.tile(samples[:, t - r], (h, 1)).T)
                    z += final_m + final_u
                    r += 1
                X = x / (r - 1) + eta[:, t - 1, :]
                Z = z / (r - 1) + epsilon[:, t - 1, :]

                final_X.append(X)
                final_Z.append(Z)

                Q = gamma_h * Z + (1 - gamma_h) * np.expand_dims(np.mean(np.concatenate((X, C), axis=1), axis=1),
                                                                 axis=1)
            return Q, final_X, final_Z

        def outcome(w, samples_final, Q, b):
            Y = np.matmul(w.T, Q.T).T + b
            Y_f = samples_final * Y[:, 0] + (1 - samples_final) * Y[:, 1]
            Y_cf = samples_final * Y[:, 1] + (1 - samples_final) * Y[:, 0]
            return Y_f, Y_cf

        eta, epsilon, w, b = distributions(N, N_treated, time, k)
        treatment, control, samples, samples_final = control_treatment(N, N_treated, time)
        C, X, Z = features(N, N_treated, s, k, h)
        Q, final_X, final_Z = covariates_simulation(X, Z, C, gamma_h, time, N, N_treated, k, p, eta, epsilon)
        Y_f, Y_cf = outcome(w, samples_final, Q, b)

        data_synthetic = "../data/data_synthetic"
        dir_base = '{}/data_syn_{}'.format(data_synthetic, gamma_h)
        dir = '{}/data_baseline_syn_{}'.format(data_synthetic, gamma_h)
        os.makedirs(dir_base, exist_ok=True)
        os.makedirs(dir, exist_ok=True)
        for t in range(1, time + 1):
            final = np.zeros((N + N_treated, k + s + 3))
            final[:, 0] = samples_final
            final[:, 3:3 + k] = final_X[t]
            final[:, 3 + k:3 + k + s] = C
            final[:, 1] = Y_f
            final[:, 2] = Y_cf
            df = pd.DataFrame(final)
            df.to_csv('{}/{}.csv'.format(dir, t), index=False)

        # code to create dataset into pytorch friendly format
        for n in range(N + N_treated):
            x = np.zeros(shape=(time, k))
            out_x_file = '{}/{}.x.npy'.format(dir_base, n)
            out_static_file = '{}/{}.static.npy'.format(dir_base, n)
            out_a_file = '{}/{}.a.npy'.format(dir_base, n)
            out_y_file = '{}/{}.y.npy'.format(dir_base, n)
            for t in range(1, time + 1):
                x[t - 1, :] = final_X[t][n, :]
            c_static = C[n, :]
            a = samples[n, :]

            y = [Y_f[n], Y_cf[n]]

            np.save(out_x_file, x)
            np.save(out_static_file, c_static)
            np.save(out_a_file, a)
            np.save(out_y_file, y)

        all_idx = np.arange(N + N_treated)
        np.random.shuffle(all_idx)

        # splitting data into train test split

        train_ratio = 0.7
        val_ratio = 0.1

        train_idx = all_idx[:int(len(all_idx) * train_ratio)]
        val_idx = all_idx[
                  int(len(all_idx) * train_ratio):int(len(all_idx) * train_ratio) + int(len(all_idx) * val_ratio)]
        test_idx = all_idx[int(len(all_idx) * train_ratio) + int(len(all_idx) * val_ratio):]

        split = np.ones(N + N_treated)
        split[test_idx] = 0
        split[val_idx] = 2

        df = pd.DataFrame(split, dtype=int)
        df.to_csv('{}/train_test_split.csv'.format(dir_base), index=False, header=False)

#Default Parameters
np.random.seed(23)
# Timestamps
time = 10
# Time varying covariates
k = 100
# of static covariates
s = 5
#Number of hidden confounders
h = 1
#Control Samples
N = 3000
#Treated Samples
N_treated = 1000
# P-order autogregressive process
p = 5
#Weight of hidden confounders
gamma = (.1,.3,.5,.7)

full_synthetic_dataset(gamma,time,k,s,h,N,N_treated,p)