import cv2
import numpy as np
import os
import utils as ut


def superimpose_img_on_AR(img, img_for_ar_tag):

    original_img_size = img.shape[:2]

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

    img_for_ar_tag = cv2.resize(img_for_ar_tag, (img_size[0], img_size[1]))
    rotated_img_for_ar_tag = ut.rotate_img_for_ar(img_for_ar_tag, orientation)

    warped_img_on_ar_tag = ut.warpPerspective(rotated_img_for_ar_tag, homo, (original_img_size[1], original_img_size[0]), 'forward', img)

    cv2.circle(warped_img_on_ar_tag, p1, 10, (255, 0, 0), -1)
    cv2.circle(warped_img_on_ar_tag, p2, 10, (0, 255, 0), -1)
    cv2.circle(warped_img_on_ar_tag, p3, 10, (0, 0, 255), -1)
    cv2.circle(warped_img_on_ar_tag, p4, 10, (0, 255, 255), -1)

    cv2.namedWindow('warped_img_on_ar_tag', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('warped_img_on_ar_tag', warped_img_on_ar_tag)

    return warped_img_on_ar_tag


def run(video_file, img_file):

    if not os.path.exists(video_file):
        print("[ERROR]: File does not exists {}". format(video_file))

    if not os.path.exists(img_file):
        print("[ERROR]: File does not exists {}". format(img_file))

    ar_img = cv2.imread(img_file)

    cap = cv2.VideoCapture(video_file)

    # ret, frame = cap.read()
    # h, w = frame.shape[:2]
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter("img_on_AR.mp4", fourcc, 30, (w, h))

    while True:
        ret, frame = cap.read()
        if ret:
            frame = superimpose_img_on_AR(frame, ar_img)
            # writer.write(frame)
            k = cv2.waitKey(1)
            if k == ord('p'):
                cv2.waitKey(0)
        else:
            print("[INFO]: Video Finished!!!")
            break
    # writer.release()


if __name__ == "__main__":
    video_file = "..//Data//1tagvideo.mp4"
    img_file = "..//Data//testudo.png"
    run(video_file, img_file)