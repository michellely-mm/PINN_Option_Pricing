# Part 1: Improving Performance

### Identify why $\sigma$ is hard to train.
1. Visualize $|PINN_{SS} - V_{SS}|$ over uniform grid (test points). Since the gradient of $\sigma$ depends on $V_{SS}$, the estimation is not accurate if the error is large.
2. After training, print sigma.item(), sigma.grad.item(), and r.grad.item().

### Modify training process to get better result
1. Change learning rate for $\sigma$. Try larger learning rate first. If it does not work, try small learning rate.
2. New training strategy. Alternative training process: fix $(r,\sigma)$ train NN, then fix NN weights and train $r,\sigma$.

You also mentioned that we cannot simultaneously have good $r$ and $\sigma$. Can we explain why?



# Part 2: Is it enough to write a paper?

#### What we have
We have three models: inversePINN, paper PINN, and forward PINN, and three datasets: synthetic European call, synthetic American call, and real data.

#### Inverse PINN difficulties
Inverse PINN: 1) $\sigma$ accuracy (see part 1) 2) balancing several losses by choosing $\lambda$.

#### Contributions: 
1. We have real data, and it outperforms paper PINN and forward PINN.
2. We propose new training strategies (learning rate and alternating training) to improve accuracy.       ------ Need to be done.
3. We introduce self-adaptive $\lambda$ (now it is manual selection).                                     ------ Need to be done.
4. We can verify (both theory and numerics) whether $r$ and $\sigma$ are identifiable.                    ------ Need to be done. I will check how to do it in theory.

#### Questions:
1. Could you please summarized the real data setting, such as the knowns and unknowns? 
2. Real data paper PINN, how did they train? Can they use different training samples at each epoch?


# Part 3: Future Directions:
1. Comparison between joint optimization and forward-inverse separate optimization.
2. Operator Learning.
3. Inverse problem, but $r$ and $\sigma$ are functions rather than constants.
