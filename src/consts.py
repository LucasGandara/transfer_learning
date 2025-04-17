import datetime
import time
from enum import Enum


def get_stage_name(stage: int):
    if stage == 0:
        return "Main Track"
    elif stage == 1:
        return "TB3 World"
    else:
        raise ValueError("Invalid stage")


def get_stage(stage: int):
    if stage == 0:
        return Stage.MAIN_TRACK
    elif stage == 1:
        return Stage.TB3_WORLD
    else:
        raise ValueError("Invalid stage")


class Stage(Enum):
    MAIN_TRACK = 0
    TB3_WORLD = 1


GOAL_X_LIST = {
    Stage.MAIN_TRACK: [
        1.833015,
        2.833015,
        3.979241,
        5.473178,
        7.737311,
        10.239666,
    ],
    Stage.TB3_WORLD: [
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        0.5,
        0.2,
        -0.8,
        -1,
        -1.5,
        0.5,
        1.5,
        0.5,
        0,
        -0.1,
        -1.5,
    ],
}

GOAL_Y_LIST = {
    Stage.MAIN_TRACK: [
        -0.490867,
        -0.490867,
        -0.841217,
        -1.775312,
        -0.471472,
        -0.494202,
    ],
    Stage.TB3_WORLD: [
        0,
        0,
        0,
        0,
        -0.5,
        1.7,
        -1.5,
        -0.9,
        1,
        1.1,
        -1.5,
        1.5,
        1.7,
        -1,
        1.6,
        -0.8,
    ],
}


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start_time
        print("\nThe training took: ")
        print(str(datetime.timedelta(seconds=duration)), end="\n\n")

    return wrapper


if __name__ == "__main__":
    stage = 0
    print(GOAL_X_LIST)
