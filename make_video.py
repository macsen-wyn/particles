import numpy as np
import cv2

def watch(env, n_steps=100, scale=500, output_file="simulation.mp4", fps=30, radius=5):
    """
    Simulate env and save a video with a box outline.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (scale, scale))
    
    for step in range(n_steps):
        env.step()
        env.measure_KE()
        env.measure_PE()
        total_energy = env.KE + env.PE

        pos = np.array(env.pos[0])  # first environment
        img = np.ones((scale, scale, 3), dtype=np.uint8) * 255  # white background

        # draw box outline
        cv2.rectangle(img, (0, 0), (scale-1, scale-1), (0, 0, 0), 2)  # black box

        # map positions to pixels inside the box
        margin = 0.2
        proportion = 1-margin
        #print(np.min(pos[:,0]),np.max(pos[:,0]))
        px = ((proportion * pos[:,0] + 1) / (2) * (scale-1)).astype(int)
        py = ((proportion * pos[:,1] + 1) / (2) * (scale-1)).astype(int)
        px = np.clip(px, 0, scale-1)
        py = np.clip(py, 0, scale-1)

        for x, y in zip(px, py):
            cv2.circle(img, (x, y), radius, (0, 0, 255), -1)  # red particles

        # draw energy text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"KE: {env.KE[0]:.3f}", (10, 20), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"PE: {env.PE[0]:.3f}", (10, 50), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"E: {total_energy[0]:.3f}", (10, 80), font, 0.6, (0, 0, 0), 2)

        out.write(img)

    out.release()
    print(f"Video saved to {output_file}")

#key = jax.random.PRNGKey(0)
#env_jax = ParticleEnvJAX(key, n_env=1, n_particles=10, dt=0.01)
#
#watch(env_jax, n_steps=1000, scale=1000, output_file="simulation.mp4", fps=30)