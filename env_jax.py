import jax
import jax.numpy as jnp

class ParticleEnvJAX:
    def __init__(self, key, n_env=10, n_particles=10, dt=0.01, box_size=1.0, k=10.0):
        self.n_env = n_env
        self.n_particles = n_particles
        self.dt = dt
        self.box_size = box_size
        self.k = k

        # state initialized to zeros/ones
        self.pos = jnp.zeros((n_env, n_particles, 2))
        self.vel = jnp.zeros((n_env, n_particles, 2))
        self.charge = jnp.ones((n_env, n_particles, 1))

        # buffers
        self.r = jnp.zeros((n_env, n_particles, n_particles, 2))
        self.dist3 = jnp.zeros((n_env, n_particles, n_particles, 1))
        self.qiqj = jnp.zeros((n_env, n_particles, n_particles, 1))
        self.F = jnp.zeros((n_env, n_particles, n_particles, 2))
        self.F_total = jnp.zeros((n_env, n_particles, 2))
        self.over_pos = jnp.zeros((n_env, n_particles, 2))
        self.under_pos = jnp.zeros((n_env, n_particles, 2))
        self.boundary_force = jnp.zeros((n_env, n_particles, 2))

        self.init_random(key)

        # jit compile the step function
        self._jit_step = jax.jit(self._step)

    def init_random(self, key, pos_range=1.0, vel_range=0.1, charge_range=1.0):
        key1, key2, key3 = jax.random.split(key, 3)
        self.pos = jax.random.uniform(key1, (self.n_env, self.n_particles, 2), minval=-pos_range, maxval=pos_range)
        self.vel = jax.random.uniform(key2, (self.n_env, self.n_particles, 2), minval=-vel_range, maxval=vel_range)
        #self.charge = jax.random.uniform(key3, (self.n_env, self.n_particles, 1), minval=0.0, maxval=charge_range)

    def _step(self, pos, vel, charge, dt, box_size, k):
        eps = 1e-4
        r = pos[:, :, None, :] - pos[:, None, :, :]
        dist3 = jnp.linalg.norm(r, axis=-1, keepdims=True)**3 + eps
        qiqj = charge * jnp.transpose(charge, (0,2,1))
        F = r * qiqj[:, :, :, None] / dist3
        F_total = jnp.sum(F, axis=2)

        over_pos = jnp.clip(pos - box_size, a_min=0.0)
        under_pos = jnp.clip(-box_size - pos, a_min=0.0)
        boundary_force = -k * (over_pos + under_pos)

        vel_new = vel + (F_total + boundary_force) * dt
        pos_new = pos + vel_new * dt
        return pos_new, vel_new

    def step(self):
        self.pos, self.vel = self._jit_step(self.pos, self.vel, self.charge, self.dt, self.box_size, self.k)