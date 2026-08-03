
class GradientProduct(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.math.reduce_prod(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


optimizer = GradientProduct(learning_rate=1e-4)
