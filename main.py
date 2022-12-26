import pandas as pd
import keypointsextractor as ke
import featuresextractor as fe
import os


def main():
    # cwd = os.getcwd()
    # ke.extract_keypoints()
    # os.chdir(cwd)
    # json_files = fe.get_jsons("data")
    # fe.extract_data(json_files)
    data = pd.read_csv("keypoints_2d.csv")
    # data = fe.preprocess_data(data)
    # fe.plot_keyp_trajectory([4])
    fe.compute_features()


if __name__ == "__main__":
    main()
