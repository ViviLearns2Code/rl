import numpy as np 
from dart import *

# (s1,sk,k,a) tuples
# ([0,..,301]x[0,...,301]x[0,1,2]x[0,...,81])
# initialize all relevant q values to zero
q = np.empty((302, 302, 3, 82))
q.fill(np.nan)
q[0, :, :, :] = 0 # absorbing state s1 = 0
q[:, 0, :, :] = 0 # absorbing state sk = 0
for s1 in range(2, 302):
    for sk in range(max(2, s1 - 120), s1 + 1):
        for k in range(0, 3):
            if sk >= s1 - k * 60:
                q[s1,sk,k,:] = 0

dart_info = DartInfo()

# possible outcomes of each target action
action_probs = np.array([dart_info.get_prob(a) for a in range(0, 82)]) # list comprehension

# 9 iterations to converge
it = 0
while True:
    max_diff = 0
    it += 1
    print("iteration", it)
    for s1 in range(2, 302):
        # s1 = 1 impossible to reach (bust)
        for sk in range(2, s1+1):
            # sk = 1 impossible to reach (bust)
            # sk < s1 - 120 impossible to reach (max. score per throw = 60)
            for k in range(0, 3):
                if sk < s1 - 60*k:
                    # s1 = sk at the beginning of each turn
                    continue
                for a in range(0, 82):
                    action_stats = action_probs[a]
                    score = action_stats[:, 1]
                    action_prob = action_stats[:, 2]

                    # get the next states (s1,sk,k) and rewards which result from taking action a
                    sk_next = (sk*np.ones_like(score)-score).astype(int)
                    s1_next = sk_next.copy()
                    k_next = np.zeros_like(score).astype(int)
                    reward = np.zeros_like(score)

                    # go bust --> proceed to next turn
                    if (a < 40 or a >= 60) and a != 81:
                        idx_bust = np.where(sk_next <= 1)[0]
                        idx_cont = np.where(sk_next > 1)[0]
                    else:
                        idx_bust = np.where((sk_next < 0) | (sk_next == 1))[0]
                        idx_done = np.where(sk_next == 0)[0]
                        idx_cont = np.where(sk_next > 1)[0]

                    if idx_bust.shape[0] != 0:
                        sk_next[idx_bust] = s1
                        s1_next[idx_bust] = s1
                        reward[idx_bust] = -1

                    if k < 2: # continue turn
                        s1_next[idx_cont] = s1
                        k_next[idx_cont] = k + 1
                    else: # next turn
                        reward[idx_cont] = -1

                    outcome = q[s1_next, sk_next, k_next, 0:82]

                    # debug
                    if np.isnan(q[s1,sk,k,:]).any():
                        raise Exception("ERROR")
                    if np.isnan(outcome).any():
                        print(s1,sk,k)
                        print(s1_next)
                        print(sk_next)
                        print(outcome)
                        raise Exception("ERROR")

                    # take expectations over next states
                    q_new = (np.max(outcome, axis=1)+reward).dot(action_prob)

                    # evaluate convergence criteria
                    if max_diff < abs(q[s1, sk, k, a]-q_new):
                        print(max_diff)
                        max_diff = abs(q[s1, sk, k, a]-q_new)

                    # update q value
                    q[s1, sk, k, a] = q_new

    if max_diff < 0.01:
        break

np.save('qvalues.npy', q)