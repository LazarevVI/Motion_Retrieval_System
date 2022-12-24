import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline
import featurescomputer as fc

pd.options.plotting.backend = "plotly"

fps = 30

data_dir = "OpenPose/openpose/data"  # directory with keypoints jsons

pose_folder = "pose"
face_folder = "face"
lhand_folder = "lhand"
rhand_folder = "rhand"

# pairs of connected body keypoints
pairs = [1, 8,
         2, 3,
         3, 4,
         5, 6,
         6, 7,
         9, 10,
         10, 11,
         12, 13,
         13, 14,
         1, 0]

weights = [0.33020, 0.03075,  # weight fraction of body-segments
           0.02295, 0.03075,
           0.02295, 0.11125,
           0.06430, 0.11125,
           0.06430, 0.06810]

# face keypoints to be extracted
face_points = [2, 8, 14, 17, 19, 21, 22, 24, 26, 48, 51, 54, 57, 68, 69]

# hand keypoints
hand_points = np.arange(21)

# body keypoints
body_points = np.arange(25)


def delete_outliers(df):
    """
    Delete outliers in dataframe
    :param df: dataframe with outliers
    :return: dataframe without outliers
    """
    len_points = len(face_points) + len(body_points) + 2 * len(hand_points)

    print("\nDeleting outliers...")
    for part in tqdm(range(len_points)):
        df_part = df.loc[df["Point"] == part]
        df_part = df_part.interpolate(method="slinear")

        Q1_x = df_part["x"].quantile(0.05)
        Q3_x = df_part["x"].quantile(0.95)
        IQR_x = Q3_x - Q1_x

        lower_lim_x = Q1_x - 1.5 * IQR_x
        upper_lim_x = Q3_x + 1.5 * IQR_x

        outliers_low_x = (df_part["x"] < lower_lim_x)
        outliers_up_x = (df_part["x"] > upper_lim_x)

        df_part["x"] = df_part["x"][~(outliers_low_x | outliers_up_x)]

        Q1_y = df_part["y"].quantile(0.05)
        Q3_y = df_part["y"].quantile(0.95)
        IQR_y = Q3_y - Q1_y

        lower_lim_y = Q1_y - 1.5 * IQR_y
        upper_lim_y = Q3_y + 1.5 * IQR_y

        outliers_low_y = (df_part["y"] < lower_lim_y)
        outliers_up_y = (df_part["y"] > upper_lim_y)

        df_part["y"] = df_part["y"][~(outliers_low_y | outliers_up_y)]
        df_part = df_part[df_part['x' and 'y'].notna()]
        df[df["Point"] == part] = df_part
    return df


def interpolate_df(df):
    """
    Interpolate data in dataframe
    :param df: dataframe with data to be interpolated
    :return: interpolated dataframe
    """

    len_points = len(face_points) + len(body_points) + 2 * len(hand_points)

    new_df_index = pd.MultiIndex.from_arrays(
        [np.repeat(np.arange(1000), len_points), np.tile(np.arange(len_points), 1000)],
        names=('Frame', 'Point'))
    new_df = pd.DataFrame(index=new_df_index, columns=['x', 'y'])
    new_df.to_csv("interpolated_keypoints_2d.csv")
    new_df = pd.read_csv("interpolated_keypoints_2d.csv")

    print("\nInterpolating dataframe...")
    for part in tqdm(range(len_points)):
        df_part = df.loc[df["Point"] == part]
        df_part = df_part.interpolate(method="slinear")

        x = df_part['x']
        y = df_part['y']
        f = df_part["Frame"]

        spl_x = UnivariateSpline(f, x, k=3)
        spl_y = UnivariateSpline(f, y, k=3)

        fs = np.linspace(0, len(f) - 1, 1000)

        new_df.loc[new_df["Point"] == part, "x"] = spl_x(fs)
        new_df.loc[new_df["Point"] == part, "y"] = spl_y(fs)

    new_df.to_csv("interpolated_keypoints_2d.csv", index=False)
    print("Interpolated dataframe saved to ", os.path.abspath("interpolated_keypoints_2d.csv"))
    return new_df


def preprocess_data(df):
    """
    Delete outliers and interpolate number of frames
    :param df: data to be preprocessed
    :return: preprocessed data
    """
    df = delete_outliers(df)
    df = interpolate_df(df)

    return df


def create_folder(directory):
    """
    Create folder if it doesn't exist
    :param directory: path to folder
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory " + directory + " was successfully created")


def get_jsons(directory):
    """
    Get json files names in directory
    :param directory: path to jsons
    :return: list of json files names
    """
    jsons = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return jsons


def get_csvs(directory):
    """
    Get csv files names in directory
    :param directory: path to csvs
    :return: list of csv files names
    """
    csvs = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return csvs


def create_csvs(data, folder, ind):
    """
    Create csv file for frame
    :param data: data to be saved to csv
    :param folder: folder for csv
    :param ind: frame index
    :return: None
    """
    filename = folder + "_frame_"

    arr = np.asarray(data[0])
    arr = np.reshape(arr, (int(len(arr) / 3), 3))[:, :2]

    np.savetxt(folder + "/" + filename + str(ind) + ".csv", arr, delimiter=",")
    print("Data saved to ", folder + "/" + filename + str(ind) + ".csv")


def extract_data(files):
    """
    Extract key data from jsons and create csv
    :param files: list of json files names
    :return: None
    """
    frame = 0
    mi_keypoints = [[], []]
    keypoints_arr = np.empty((0, 2))

    print("Extracting data from json files...")
    for json_file in tqdm(files):
        f = open(data_dir + "/" + json_file)
        data = json.load(f)

        df = pd.json_normalize(data["people"])

        pose_keypoints = df["pose_keypoints_2d"]
        face_keypoints = df["face_keypoints_2d"]
        lhand_keypoints = df["hand_left_keypoints_2d"]
        rhand_keypoints = df["hand_right_keypoints_2d"]

        pose_frame_arr = np.asarray(pose_keypoints[0])
        pose_frame_arr = np.reshape(pose_frame_arr, (int(len(pose_frame_arr) / 3), 3))[:, :2]
        keypoints_arr = np.vstack((keypoints_arr, pose_frame_arr))

        face_frame_arr = np.asarray(face_keypoints[0])
        face_frame_arr = np.reshape(face_frame_arr, (int(len(face_frame_arr) / 3), 3))[face_points, :2]
        keypoints_arr = np.vstack((keypoints_arr, face_frame_arr))

        lhand_frame_arr = np.asarray(lhand_keypoints[0])
        lhand_frame_arr = np.reshape(lhand_frame_arr, (int(len(lhand_frame_arr) / 3), 3))[:, :2]
        keypoints_arr = np.vstack((keypoints_arr, lhand_frame_arr))

        rhand_frame_arr = np.asarray(rhand_keypoints[0])
        rhand_frame_arr = np.reshape(rhand_frame_arr, (int(len(rhand_frame_arr) / 3), 3))[:, :2]
        keypoints_arr = np.vstack((keypoints_arr, rhand_frame_arr))

        mi_keypoints[0] = np.concatenate(
            (mi_keypoints[0], np.full(len(face_points) + len(body_points) + 2 * len(hand_points), frame)))
        mi_keypoints[1] = np.concatenate(
            (mi_keypoints[1], np.arange(len(face_points) + len(body_points) + 2 * len(hand_points))))

        frame += 1

    keypoints_index = pd.MultiIndex.from_arrays(mi_keypoints, names=('Frame', 'Point'))

    df_keypoints = pd.DataFrame({'x': keypoints_arr[:, 0], 'y': keypoints_arr[:, 1]},
                                index=keypoints_index)
    df_keypoints.replace(0, np.nan, inplace=True)
    df_keypoints.to_csv("keypoints_2d.csv")

    print("Extracted data saved to ", os.path.abspath("keypoints_2d.csv"))

    return df_keypoints


def compute_features():
    data = pd.read_csv('interpolated_keypoints_2d.csv', index_col=[0, 1])
    data = fc.center_of_mass(data, pairs, weights)
    data = fc.speed(data, fps)
    data = fc.acceleration(data, fps)
    data = fc.weight_effort(data, 60)
    data = fc.space_effort(data, 2)
    data.to_csv("keypoints_with_extracted_features.csv")
    return


def plot_keyp_trajectory(keyp=[0]):
    prep_data = pd.read_csv('interpolated_keypoints_2d.csv')
    raw_data = pd.read_csv('keypoints_2d.csv')
    fig = go.Figure()
    fig.update_yaxes(range=[720, 0])
    fig.update_xaxes(range=[0, 1280])

    for key in keyp:
        raw_data_key = raw_data.loc[raw_data["Point"] == key]
        prep_data_key = prep_data.loc[prep_data["Point"] == key]

        fig.add_trace(go.Scatter(x=raw_data_key["x"], y=raw_data_key["y"],
                                 mode='lines',
                                 name='Raw data' + str(key)))
        fig.add_trace(go.Scatter(x=prep_data_key["x"], y=prep_data_key["y"],
                                 mode='lines',
                                 name='Preprocessed data' + str(key)))
    fig.show()

    return
