# Part 1: Improving Performance

### Identify why $\sigma$ is hard to train.
1. Visualize $|V^{invPINN}_{SS} - V^{true}_{SS}|$ over uniform grid (test points). Since the gradient of $\sigma$ depends on $V_{SS}$, the estimation is not accurate if the error is large.
2. After training, print sigma.item(), sigma.grad.item(), and r.grad.item().

### Modify training process to get better result
1. Change learning rate for $sigma$. Try larger learning rate first. If it does not work, try small learning rate.
2. New training strategy. Alternative training process: fix $(r,\sigma)$ train NN, then fix NN weights and train $r,\sigma$.

You also mentioned that we cannot simultaneously have good $r$ and $\sigma$. Can we explain why?



# Part 2: Is it enough to write a paper?

