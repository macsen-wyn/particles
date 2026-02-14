import torch

class ParticleEnvPreAlloc:
    def __init__(self, n_env=10, n_particles=10, device='cuda'):
        self.n_env = n_env
        self.n_particles = n_particles
        self.device = device

        # state
        self.pos = torch.zeros((n_env, n_particles, 2), device=device)
        self.vel = torch.zeros((n_env, n_particles, 2), device=device)
        self.charge = torch.ones((n_env, n_particles, 1), device=device)

        # pairwise tensors
        self.r = torch.zeros((n_env, n_particles, n_particles, 2), device=device)
        self.dist3 = torch.zeros((n_env, n_particles, n_particles, 1), device=device)
        self.qiqj = torch.zeros((n_env, n_particles, n_particles, 1), device=device)
        self.F = torch.zeros((n_env, n_particles, n_particles, 2), device=device)
        self.F_total = torch.zeros((n_env, n_particles, 2), device=device)

        # boundary tensors
        self.over_pos = torch.zeros((n_env, n_particles, 2), device=device)
        self.under_pos = torch.zeros((n_env, n_particles, 2), device=device)
        self.boundary_force = torch.zeros((n_env, n_particles, 2), device=device)

        self.init_random()

    def step(self, dt=0.01, box_size=1.0, k=10.0):
        eps = 1e-4

        # pairwise displacement
        self.r.copy_(self.pos[:, :, None, :] - self.pos[:, None, :, :])
        self.dist3.copy_(self.r.norm(dim=-1, keepdim=True).pow(3) + eps)
        self.qiqj.copy_((self.charge * self.charge.transpose(1,2))[:, :, :, None])

        # force
        self.F.copy_(self.r * self.qiqj / self.dist3)
        self.F_total.copy_(self.F.sum(dim=2))

        # boundary force
        self.over_pos.copy_(torch.clamp(self.pos - box_size, min=0.0))
        self.under_pos.copy_(torch.clamp(-box_size - self.pos, min=0.0))
        self.boundary_force.copy_(-k * (self.over_pos + self.under_pos))

        # update
        self.vel += (self.F_total + self.boundary_force) * dt
        self.pos += self.vel * dt

    def init_random(self, pos_range=1.0, vel_range=0.1, charge_range=1.0):
        self.pos.uniform_(-pos_range, pos_range)
        self.vel.uniform_(-vel_range, vel_range)
        #self.charge.uniform_(0.0, charge_range)  # only positive charges