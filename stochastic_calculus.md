# Stochastic Calculus Foundations

## Risk-Neutral SDE Derivation

The stock price in the real-world measure follows:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t^P $$

Using **Girsanov's Theorem**, we change to the risk-neutral measure $\mathbb{Q}$:

Let $ \theta = \frac{\mu - r}{\sigma} $ (market price of risk)

Define the Radon-Nikodym derivative:
$$ \frac{d\mathbb{Q}}{d\mathbb{P}} = \exp\left(-\theta W_t^P - \frac{1}{2}\theta^2 t\right) $$

Then $ W_t^Q = W_t^P + \theta t $ is a $\mathbb{Q}$-Brownian motion.

Substituting into the SDE:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t^P = \mu S_t dt + \sigma S_t (dW_t^Q - \theta dt) $$
$$ dS_t = (\mu - \sigma\theta) S_t dt + \sigma S_t dW_t^Q $$
$$ dS_t = r S_t dt + \sigma S_t dW_t^Q $$

## Ito Calculus in Simulation

When discretizing the GBM:
$$ S_{t+\Delta t} = S_t \exp\left((r - \frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t} Z\right) $$

This is the exact solution obtained by applying **Ito's Lemma** to $\log S_t$:
$$ d(\log S_t) = (r - \frac{1}{2}\sigma^2)dt + \sigma dW_t $$