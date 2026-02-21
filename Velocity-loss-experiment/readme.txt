I am going to do the following experiment:

Objective: choose normalization constants for loss.

Method: run several simulations at different energy levels, record position/velocity variances.

Analysis: compute mean/pooled variance per feature, optionally plot variance vs total energy.

Decision: pick representative variances for weighting the loss; note rationale.

Future notes: mention that different energy regimes might require re-scaling.

Results in file

---------------------------------------------------------------------------------------------------------
Objective: if we use MSE then the differences in X may outweight the differences in V, so we should find the std of V components to weight appropriately

env_jax = ParticleEnvJAX(key, n_env=1, n_particles=10, dt=0.01,interaction_strength=0.2,wall_force_coeff=1000, damping=1,vel_range=2)
gives reasonable params

---------------------------------------------------------------------------------------------------------

Method of getting simulation to correct energy level:

Initialize positions and velocities normally.

Apply initial damping to velocities each step (or anti-damping if below target) to bring kinetic + potential energy toward the desired total energy.

Monitor total energy each step.

Stop damping once the target energy is reached.

From that point on, record velocity and position samples to compute variances for normalization

In order to do this I have

---------------------------------------------------------------------------------------------------------

That was the plan, but after some measurments, I have decided that the variance of velocity is about 1 so I dont need to rescale.