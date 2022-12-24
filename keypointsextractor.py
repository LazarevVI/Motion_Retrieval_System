import subprocess
import os

def extract_keypoints():
    os.chdir("OpenPose/openpose")
    cwd = os.getcwd()
    cmd = "bin/OpenPoseDemo.exe --video examples/media/outpy.avi --net_resolution -1x304 --face --face_net_resolution 320x320 --hand --write_json data/"
    subprocess.call(cmd)