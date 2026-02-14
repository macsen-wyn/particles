import torch

class ParticleEnv:
    def __init__(self, n_env=10, n_particles=10, device='cuda'):
        self.n_env = n_env
        self.n_particles = n_particles
        self.device = device

        self.pos = torch.zeros((n_env, n_particles, 2), device=device)
        self.vel = torch.zeros((n_env, n_particles, 2), device=device)
        self.charge = torch.ones((n_env, n_particles, 1), device=device)

        self.init_random()

    def init_random(self, pos_range=1.0, vel_range=0.1, charge_range=1.0):
        self.pos.uniform_(-pos_range, pos_range)
        self.vel.uniform_(-vel_range, vel_range)
        #self.charge.uniform_(-charge_range, charge_range)

    def step(self, dt=0.01, box_size=1.0, k=10.0):
        eps = 1e-4
        r = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        dist3 = r.norm(dim=-1, keepdim=True).pow(3) + eps
        qiqj = self.charge * self.charge.transpose(1,2)
        F = r * qiqj[:, :, :, None] / dist3
        F_total = F.sum(dim=2)

        # boundary force as true force
        over_pos = torch.clamp(self.pos - box_size, min=0.0)
        under_pos = torch.clamp(-box_size - self.pos, min=0.0)
        F_boundary = -k * (over_pos + under_pos)

        self.vel += (F_total + F_boundary) * dt
        self.pos += self.vel * dt