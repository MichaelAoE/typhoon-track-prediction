"""第8轮入口：累积历史截断样本，并保留先前正式参数先验。"""

from cumulative_training_optimize import run


if __name__ == "__main__":
    run(round_number=8)
