import keypointsextractor as ke
import featuresextractor as fe


def main():
    # ke.extract_keypoints()
    # json_files = fe.get_jsons(data_dir)
    # fe.extract_data(json_files)
    # data = fe.preprocess_data(data)
    fe.compute_features()


if __name__ == "__main__":
    main()
