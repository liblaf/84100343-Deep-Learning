# Part Two: Generative Model

## 2.1 Variational Auto-Encoder (VAE)

### Task 2.1.1

The closed-form expression for the KL divergence is:

```math
\operatorname{KL}(q(z|x) || p(z)) = \frac{1}{2} \sum_i \left[-(\log{\sigma_i^2} + 1) + \sigma_i^2 + \mu_i^2 \right]
```
