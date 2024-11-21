class Optimizers:
    def __init__(self):
        pass

    @staticmethod
    def gradient_descent(learning_rate, model):
        for p in model.parameters():
            p.data = p.data - learning_rate * p.grad
        model.zero_grad()
