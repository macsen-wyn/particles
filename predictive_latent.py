import jax
import jax.numpy as jnp

class JaxNetwork:
    def __init__(self, key, layer_sizes):
        """
        layer_sizes: list like [in_dim, h1, h2, out_dim]
        """
        self.params = self._init_params(key, layer_sizes)
        self.forward = jax.jit(JaxNetwork._forward)
        self.sgd_step = jax.jit(JaxNetwork._sgd_step) 

    def _init_params(self, key, layer_sizes):
        params = []
        keys = jax.random.split(key, len(layer_sizes) - 1)
        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            W = jax.random.normal(k, (din, dout)) * 0.1
            b = jnp.zeros(dout)
            params.append((W, b))
        return params
    
    @staticmethod
    def _forward(params, x):
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x
    
    @staticmethod
    def _loss(params, x, y):
        y_pred = JaxNetwork._forward(params, x)
        return jnp.mean((y_pred - y)**2)
    
    @staticmethod
    def _sgd_step(params, x, y, lr=1e-3):

        grads = jax.grad(JaxNetwork._loss)(params, x, y)
        new_params = [(w - lr*dw, b - lr*db) for (w,b), (dw,db) in zip(params, grads)]
        return new_params
    