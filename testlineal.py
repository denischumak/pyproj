import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import regression as rgr


def gen_synth_data(b, w1, a, sigma):
    X = np.linspace(0, 10, 100)
    X = X.reshape(X.shape[0], 1)
    return X, b + w1 * X + np.random.normal(a, sigma, np.shape(X))


def print_data(model):
    print(
        f"loss function value={model.loss_step_values_[-1]}, steps={model.steps_count_}, b={model.intercept_}, W={model.coef_}"
    )


def plot(X, y, models, optimizers):
    fig, ax = plt.subplots(len(models), 2)
    fig.tight_layout()
    lines = []
    for index, model in enumerate(models):
        ax[index, 1].plot(
            np.arange(0, model.steps_count_, 1),
            model.loss_step_values_,
            color="black",
        )
        ax[index, 1].set_xlabel("Times changed weights")
        ax[index, 1].set_ylabel("Loss function value")
        ax[index, 1].grid(True)

        ax[index, 0].scatter(X, y, color="black", alpha=0.6)
        (line,) = ax[index, 0].plot(
            X, model.predict(X), color="red", label="approx. line"
        )
        lines.append(line)
        ax[index, 0].grid(True)
        ax[index, 0].set_title(optimizers[index].desc_)

    def update_line(i):
        for index, opt in enumerate(optimizers):
            if i < models[index].steps_count_:
                lines[index].set_data(
                    X, opt.weight_trace_[i][0] + opt.weight_trace_[i][1] * X
                )
        return lines

    animation = FuncAnimation(
        fig,
        func=update_line,
        frames=np.arange(0, np.max([model.steps_count_ for model in models]), 1),
        interval=30,
        repeat=False,
    )

    for i in range(len(models)):
        ax[i, 0].legend()
    plt.show()


X, y = gen_synth_data(0.9, 3.6, 1, 3)

optimizers = [
    rgr.GradientDescent(),
    rgr.StochasticGradientDescent(),
    rgr.Momentum(),
    rgr.Adam(batch_size=X.shape[0], lr=1e-1),
]
models = [rgr.LinearRegression().fit(X, y, opt) for opt in optimizers]

for model in models:
    print_data(model)

plot(X, y, models, optimizers)
