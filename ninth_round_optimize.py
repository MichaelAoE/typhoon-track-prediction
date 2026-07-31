"""第9轮入口：在当前正式参数附近执行细粒度累积训练。"""

from cumulative_training_optimize import run


if __name__ == "__main__":
    run(round_number=9, grid_mode="local")
