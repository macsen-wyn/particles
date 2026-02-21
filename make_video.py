import numpy as np
import cv2
import jax
import jax.numpy as jnp

def watch(env, model=None, n_steps=100, dt_pred=5, scale=500, output_file="simulation.mp4", fps=30, radius=5):
    """
    Simulate env and save a video with a box outline.
    model: if provided, predicts future positions dt_pred steps ahead
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (scale, scale))
    
    for step in range(n_steps):
        env.step()
        env.measure_KE()
        env.measure_PE()
        total_energy = env.KE + env.PE

        pos = np.array(env.pos[0])  # actual positions
        img = np.ones((scale, scale, 3), dtype=np.uint8) * 255  # white background

        # draw box outline
        cv2.rectangle(img, (0, 0), (scale-1, scale-1), (0, 0, 0), 2)  # black box

        # map positions to pixels
        margin = 0.2
        proportion = 1-margin
        px = ((proportion * pos[:,0] + 1)/2 * (scale-1)).astype(int)
        py = ((proportion * pos[:,1] + 1)/2 * (scale-1)).astype(int)
        px = np.clip(px, 0, scale-1)
        py = np.clip(py, 0, scale-1)

        # draw red actual particles
        for x, y in zip(px, py):
            cv2.circle(img, (x, y), radius, (0, 0, 255), -1)  # red

        # if a model is provided, predict future positions
        if model is not None:
            state = np.hstack([pos, np.array(env.vel[0])])  # flatten state for model
            state_jax = jnp.array(state)
            pred_state = state_jax
            for _ in range(dt_pred):
                pred_state = model.forward(model.params, pred_state)
            pred_pos = np.array(pred_state[:, :2])

            px_pred = ((proportion * pred_pos[:,0] + 1)/2 * (scale-1)).astype(int)
            py_pred = ((proportion * pred_pos[:,1] + 1)/2 * (scale-1)).astype(int)
            px_pred = np.clip(px_pred, 0, scale-1)
            py_pred = np.clip(py_pred, 0, scale-1)

            # draw blue predicted particles
            for x, y in zip(px_pred, py_pred):
                cv2.circle(img, (x, y), radius, (255, 0, 0), -1)  # blue

            # draw arrows from red to blue
            for x1, y1, x2, y2 in zip(px, py, px_pred, py_pred):
                cv2.arrowedLine(img, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

        # draw energy text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"KE: {env.KE[0]:.3f}", (10, 20), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"PE: {env.PE[0]:.3f}", (10, 50), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"E: {total_energy[0]:.3f}", (10, 80), font, 0.6, (0, 0, 0), 2)

        out.write(img)

    out.release()
    print(f"Video saved to {output_file}")