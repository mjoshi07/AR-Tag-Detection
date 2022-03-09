import argparse
import os
import img_on_AR_tag
import detect_AR_tag
import cube_on_AR_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='1', type=int,  help='problem number you want to solve, options- 1/2/3')
    parser.add_argument('--dataPath', default='..//data//', help='path to data folder, Default: ..//data//')
    parser.add_argument('--videoFile', default='1tagvideo.mp4', help='video file name, Default: 1tagvideo.mp4')
    parser.add_argument('--imgFile', default='testudo.png', help='image file name, Default: testudo.png')

    args = parser.parse_args()
    data_dir = args.dataPath
    problem = int(args.problem)
    video_name = args.videoFile
    img_name = args.imgFile

    if not os.path.exists(data_dir):
        print("[ERROR]: Data directory does not exists. Exiting!!!")
        exit()

    video_file = os.path.join(data_dir, video_name)
    if problem == 1:
        detect_AR_tag.run(video_file)
    elif problem == 2:
        img_file = os.path.join(data_dir, img_name)
        img_on_AR_tag.run(video_file, img_file)
    elif problem == 3:
        cube_on_AR_tag.run(video_file)
