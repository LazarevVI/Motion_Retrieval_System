import numpy as np
import pandas as pd
from tqdm import tqdm


def get_seg_center_coords(df, pairs):
    """
    Get coords of body segment centers in each frame
    :param pairs: pairs of points that form a body segment
    :param df: coordinates of keypoints in each frame
    :return: seg_centers - coords of body segment centers in each frame
    """

    seg_centers = np.zeros((len(df.index.unique(level=0)), int(len(pairs) / 2), 2))

    for j in range(0, len(pairs), 2):
        part_1 = pairs[j]
        part_2 = pairs[j + 1]

        part_1_coord = df.loc[(slice(None), part_1), ["x", "y"]]
        part_2_coord = df.loc[(slice(None), part_2), ["x", "y"]]

        center_x = np.mean(np.asarray([part_1_coord["x"], part_2_coord["x"]]), axis=0)
        center_y = np.mean(np.asarray([part_1_coord["y"], part_2_coord["y"]]), axis=0)

        seg_centers[:, int(j / 2), 0] = center_x
        seg_centers[:, int(j / 2), 1] = center_y

    return seg_centers


def center_of_mass(df, pairs, w):
    """
    Get coords of body center of mass in each frame
    :param df: dataframe with body keypoints coordinates
    :param pairs: pairs of points that form a body segment
    :param w: weight contribution for each segment
    :return: dataframe with added coords of body center of mass
    """
    frames_number = len(df.index.unique(level=0))
    keyp_number = len(df.index.unique(level=1))
    seg_centers = get_seg_center_coords(df, pairs)
    cm_coords = np.zeros((frames_number, 2))

    print('Computing center of mass coordinates...')
    for i in tqdm(range(frames_number)):
        cm_coords[i][0] = np.average(seg_centers[i, :, 0], weights=w)
        cm_coords[i][1] = np.average(seg_centers[i, :, 1], weights=w)

    cm_index = pd.MultiIndex.from_arrays([np.arange(frames_number), np.full(frames_number, keyp_number)],
                                         names=('Frame', 'Point'))
    cm_df = pd.DataFrame(cm_coords, columns=["x", "y"], index=cm_index)

    new_df = pd.concat([df, cm_df]).sort_index()

    return new_df


def speed(df, fps):
    """
    :param df: coordinates of keypoints in each frame
    :param fps: frames per second for video
    :return: dataframe with speed of keypoints in each frame and v_x, v_y components
    """

    df["speed"] = (np.sqrt(df.groupby("Point")["x"].diff() ** 2 + df.groupby("Point")["y"].diff() ** 2) * fps)
    df["speed"] = df["speed"].groupby("Point").shift(-1)
    df["v_x"] = df.groupby("Point")["x"].diff() * fps
    df["v_x"] = df["v_x"].groupby("Point").shift(-1)
    df["v_y"] = df.groupby("Point")["y"].diff() * fps
    df["v_y"] = df["v_y"].groupby("Point").shift(-1)
    return df


def acceleration(df, fps):
    """
    Acceleration of every body point in each frame
    :param df: parameters of keypoints in each frame
    :param fps: frames per second for video
    :return: acceleration of keypoints in each frame
    """

    df["acceleration"] = df.groupby("Point")["speed"].diff() * fps
    df["acceleration"] = df["acceleration"].groupby("Point").shift(-1)
    return df


def weight_effort(df, win):
    """
    Weight effort parameter [Laban movement analysis] for every body point per frame_interval
    :param df: parameters of keypoints in each frame
    :param win: rolling window size
    :return: dataframe with weight effort parameter
    """
    df["weight"] = df.groupby("Point")["speed"].rolling(win).mean().droplevel(0).shift(int(-win / 2))

    return df


def space_effort(df, win):
    """
    Space effort parameter [Laban movement analysis] for every body point per frame_interval
    :param df: parameters of keypoints in each frame
    :param win: rolling window size
    :return: dataframe with space effort parameter
    """

    df["space"] = np.sqrt(df.groupby("Point")["x"].diff() ** 2 + df.groupby("Point")["y"].diff() ** 2) \
                    .groupby("Point").rolling(win) \
                    .sum() \
                    .droplevel(0) / \
                    np.sqrt(df.groupby("Point")["x"].diff(periods=win) ** 2 +
                            df.groupby("Point")["y"].diff(periods=win) ** 2)
    print("Point 0\n", df.xs(0, level=1).loc[([0, 1, 2, 3], ["x", "y", "space"])])

    return df
