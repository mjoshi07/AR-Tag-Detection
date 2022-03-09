import cv2
import numpy as np
import os
import utils as ut


def superimpose_cube_on_AR(img):
    thresh, thresh1 = ut.preprocess_image(img)
    # cv2.namedWindow('thresh', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('thresh', thresh)

    p1, p2, p3, p4 = ut.get_tag_corners(thresh)

    img_size = ut.IMG_SIZE
    data_points = np.array([[0, 0, p1[0], p1[1]],
                            [img_size[0] - 1, 0, p2[0], p2[1]],
                            [img_size[0] - 1, img_size[0] - 1, p3[0], p3[1]],
                            [0, img_size[0] - 1, p4[0], p4[1]]])

    homo = ut.findHomography(data_points)

    ar_tag_warped = ut.warpPerspective(thresh1.copy(), homo, img_size, 'backward')
    # cv2.namedWindow('ar_tag_warped', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('ar_tag_warped', ar_tag_warped)

    ID_val, orientation = ut.get_tag_info(ar_tag_warped, img_size)
    # print("ID decimal: ", ID_val[0])
    # print("ID binary: ", ID_val[1])
    # print(orientation)

    proj_matrix = ut.get_projection_matrix(homo)
    pts = [(0,0), (ut.IMG_SIZE[0] - 1, 0), (ut.IMG_SIZE[0] - 1, ut.IMG_SIZE[0] - 1), (0, ut.IMG_SIZE[0] - 1)]
    p1_, p2_, p3_, p4_ = ut.get_cube_points(proj_matrix, pts)
    # print(p1_, " ", p2_, " ", p3_, " ", p4_)

    img = ut.draw_cube(img,  [p1, p2, p3, p4], [p1_, p2_, p3_, p4_])

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', img)

    return img


def run(video_file):

    if not os.path.exists(video_file):
        print("[ERROR]: File does not exists {}". format(video_file))

    cap = cv2.VideoCapture(video_file)
    # ret, frame = cap.read()
    # h, w = frame.shape[:2]
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter("cube_on_AR.mp4", fourcc, 30, (w, h))

    while True:
        ret, frame = cap.read()
        if ret:
            frame = superimpose_cube_on_AR(frame)
            # writer.write(frame)
            cv2.waitKey(1)
        else:
            print("[INFO]: Video Finished!!!")
            break

    # writer.release()


if __name__ == "__main__":
    video_file = "..//Data//1tagvideo.mp4"
    run(video_file)