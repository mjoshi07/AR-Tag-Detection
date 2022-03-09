import cv2
import numpy as np
import os
import utils as ut


def detect_AR_tag(img):

    thresh, thresh1 = ut.preprocess_image(img)
    # cv2.namedWindow('thresh', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('thresh', thresh1)

    edges = ut.detect_edges(img)
    edges = ut.remove_noise(edges)
    # cv2.namedWindow('edges', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('edges', edges)

    p1, p2, p3, p4 = ut.get_tag_corners(thresh1, edges)

    img_size = ut.IMG_SIZE
    data_points = np.array([[0, 0, p1[0], p1[1]],
                            [img_size[0] - 1, 0, p2[0], p2[1]],
                            [img_size[0] - 1, img_size[0] - 1, p3[0], p3[1]],
                            [0, img_size[0] - 1, p4[0], p4[1]]])

    homo = ut.findHomography(data_points)

    ar_tag_warped = ut.warpPerspective(thresh1, homo, img_size, 'backward')
    cv2.namedWindow('warped', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('warped', ar_tag_warped)

    ID_val, orientation = ut.get_tag_info(ar_tag_warped, img_size)
    print("ID decimal: ", ID_val[0])
    print("ID binary: ", ID_val[1])
    print("Orientation: ", orientation)

    cv2.putText(img, "AR Tag ID: " + ID_val[1], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, 16)
    cv2.putText(img, "AR Tag ID: " + ID_val[1], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 16)

    print("\n")
    print("AR tag coordinates:")
    print("Blue Circle coordinate: ", p1)
    print("Green Circle coordinate: ", p2)
    print("Red Circle coordinate: ", p3)
    print("Yellow Circle coordinate: ", p4)

    cv2.circle(img, p1, 10, (255, 0, 0), -1)
    cv2.circle(img, p2, 10, (0, 255, 0), -1)
    cv2.circle(img, p3, 10, (0, 0, 255), -1)
    cv2.circle(img, p4, 10, (0, 255, 255), -1)

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', img)

    return img


def run(video_file):
    if not os.path.exists(video_file):
        print("[ERROR]: File does not exists")

    cap = cv2.VideoCapture(video_file)

    ret, frame = cap.read()
    if ret:
        frame = detect_AR_tag(frame)
        cv2.imwrite("AR_tag.png", frame)
        cv2.waitKey(0)


if __name__ == "__main__":
    video_file = "..//Data//1tagvideo.mp4"
    run(video_file)