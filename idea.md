# Part 1: Improving Performance

### Identify why $\sigma$ is hard to train.
1. Visualize $|PINN_{SS} - V_{SS}|$ over uniform grid (test points). Since the gradient of $\sigma$ depends on $V_{SS}$, the estimation is not accurate if the error is large.
2. After training, print sigma.item(), sigma.grad.item(), and r.grad.item().

### Modify training process to get better result
1. Change learning rate for $、sigma$. Try larger learning rate first. If it does not work, try small learning rate.
2. New training strategy. Alternative training process: fix $(r,\sigma)$ train NN, then fix NN weights and train $r,\sigma$.

You also mentioned that we cannot simultaneously have good $r$ and $\sigma$. Can we explain why?



# Part 2: Is it enough to write a paper?

We have three models: inversePINN, paper PINN, and forward PINN, and three datasets: synthetic European call, synthetic American call, and real data.

Inverse PINN: 1) $\sigma$ accuracy (see part 1) 2) balancing several losses by choosing $\lambda$.

Contributions: 1) we have real data, and it outperforms paper PINN and forward PINN. 2) we propose new training strategies (learning rate and alternating training) to improve accuracy. 3) We introduce self-adaptive $\lambda$ (now it is manual selection). 4) We can verify (both theory and numerics) whether $r$ and $\sigma$ are identifiable.

Q:
1. Real data paper PINN, how did they train? Ca they use different training samples at each epoch?

