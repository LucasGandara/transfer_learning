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
        2.833015,
        5.473178,
        7.737311,
        10.239666,
        10.270091,
        5.632256,
        0.386172,
        0.482948,
    ],
    Stage.TB3_WORLD: [
        0.6,
        0.8,
        1.0,
        1.9,
        0.5,
        0.2,
        -0.8,
        -1,
        -1.9,
        0.5,
        2,
        0.5,
        0,
        -0.1,
        -2,
    ],
}

GOAL_Y_LIST = {
    Stage.MAIN_TRACK: [
        -0.490867,
        -1.775312,
        -0.471472,
        -0.494202,
        0.539621,
        2.043752,
        0.571205,
        -0.440986,
    ],
    Stage.TB3_WORLD: [
        0,
        0,
        0,
        -0.5,
        -1.9,
        1.5,
        -0.9,
        1,
        1.1,
        -1.5,
        1.5,
        1.8,
        -1,
        1.6,
        -0.8,
    ],
}

if __name__ == "__main__":
    stage = 0
    print(GOAL_X_LIST)
