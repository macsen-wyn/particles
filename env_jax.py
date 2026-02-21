import jax
import jax.numpy as jnp

class ParticleEnvJAX:
    def __init__(self, key, n_env=10, n_particles=10, dt=0.01, box_size=0.8, interaction_strength = 0.0001, boundary_force=1000.0, damping = 0.10,
                 pos_range=1, vel_range = 0.01):
        self.n_env = n_env
        self.n_particles = n_particles
        self.dt = dt
        self.box_size = box_size
        self.k = boundary_force
        self.damping=damping
        self.interaction_strength = interaction_strength
        self.pos_range = pos_range
        self.vel_range = vel_range


        # state initialized to zeros/ones
        self.pos = jnp.zeros((n_env, n_particles, 2))
        self.vel = jnp.zeros((n_env, n_particles, 2))
        self.charge = jnp.ones((n_env, n_particles, 1))
        self.qiqj = self.charge * jnp.transpose(self.charge, (0,2,1))


        # buffers
        self.KE = jnp.zeros((n_env, n_particles, n_particles, 1))
        self.PE = jnp.zeros((n_env, n_particles, n_particles, 1))
        self.E = jnp.zeros((n_env, n_particles, n_particles, 1))

        self.init_random(key,self.pos_range,self.vel_range)

        # jit compile the step function
        self._jit_step = jax.jit(self._step)
        self._jit_measure_KE = jax.jit(self._measure_KE)

    def init_random(self, key, pos_range=1.0, vel_range=0.0001, charge_range=1.0):
        key1, key2, key3 = jax.random.split(key, 3)
        self.pos = jax.random.uniform(key1, (self.n_env, self.n_particles, 2), minval=-pos_range, maxval=pos_range)
        self.vel = jax.random.uniform(key2, (self.n_env, self.n_particles, 2), minval=-vel_range, maxval=vel_range)
        #self.charge = jax.random.uniform(key3, (self.n_env, self.n_particles, 1), minval=0.0, maxval=charge_range)
        #self.qiqj = self.charge * jnp.transpose(self.charge, (0,2,1))

    def _step(self, pos, vel, qiqj, dt, box_size, k):
        eps = 1e-4
        r = pos[:, :, None, :] - pos[:, None, :, :]
        dist3 = jnp.linalg.norm(r, axis=-1, keepdims=True)**3 + eps
        F = r * qiqj[:, :, :, None] / dist3
        F_total = jnp.sum(F, axis=2) * self.interaction_strength

        over_pos = jnp.clip(pos - box_size, a_min=0.0)**2
        under_pos = -jnp.clip(box_size + pos, a_max=0.0)**2
        boundary_force = -k * (over_pos + under_pos)

        vel_new = vel + (F_total + boundary_force) * dt
        vel_new *= self.damping
        pos_new = pos + vel_new * dt
        return pos_new, vel_new
    
    def _measure_KE(self, vel):
        KE = 0.5 * jnp.sum(vel**2, axis=(-2,-1))
        return KE
    
    def measure_KE(self):
        self.KE = self._jit_measure_KE(self.vel)

    def step(self):
        self.pos, self.vel = self._jit_step(self.pos, self.vel, self.qiqj, self.dt, self.box_size, self.k)