"""第6轮入口：复用隔离实验框架，并自动排除前五轮已用台风。"""

from fourth_round_optimize import run


if __name__ == "__main__":
    run(round_number=6)
