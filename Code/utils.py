import cv2
import numpy as np
import imutils as im
from imutils import perspective
import matplotlib.pyplot as plt

EPSILON = 1e-6
IMG_SIZE = (240, 240)
K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])
K_inv = np.linalg.inv(K)


def get_A_matrix(data_points):

    A_mat = np.ndarray((2 * data_points.shape[0], 9))
    i, c = 0, 0
    while i < data_points.shape[0]:
        data_row = data_points[i].reshape((len(data_points), 1))
        x = data_row[0].item()
        y = data_row[1].item()
        xp = data_row[2].item()
        yp = data_row[3].item()

        A_mat[c] = np.asarray([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A_mat[c + 1] = np.asarray([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

        c += 2
        i += 1

    return A_mat


def findHomography(data_points):

    A_mat = get_A_matrix(data_points)

    U, Sigma, V_T = np.linalg.svd(A_mat)
    V = V_T.T
    H_col = V[:, V.shape[1] - 1]

    H = H_col.reshape((3, 3))
    H = H / H[2, 2]

    return H


def perspectiveTransform(input_x_points, input_y_points, homo, size):
    cols, rows = size
    homo_points = np.stack((input_x_points.ravel(), input_y_points.ravel(), np.ones(input_y_points.size)))
    transformed_homo_point = homo.dot(homo_points)
    transformed_homo_point /= (transformed_homo_point[2, :] + EPSILON)

    output_x_points = np.int32(transformed_homo_point[0, :])
    output_y_points = np.int32(transformed_homo_point[1, :])

    output_x_points[output_x_points < 0] = 0
    output_x_points[output_x_points >= cols] = cols - 1
    output_y_points[output_y_points < 0] = 0
    output_y_points[output_y_points >= rows] = rows - 1

    return output_x_points, output_y_points


def warpPerspective(img, homo, size, type, background=None):

    if type == 'forward':
        h, w = img.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        src_y, src_x = np.meshgrid(y, x)
        dst_x, dst_y = perspectiveTransform(src_x, src_y, homo, size)
    elif type == 'backward':
        x = np.arange(size[0])
        y = np.arange(size[1])
        dst_y, dst_x = np.meshgrid(y, x)
        src_x, src_y = perspectiveTransform(dst_x, dst_y, homo, (img.shape[1], img.shape[0]))
    else:
        print("[ERROR]: Incorrect argument in warp type")
        exit()

    if background is None:
        if len(img.shape) < 2:
            warped_img = np.zeros((size[0], size[1], 3), np.uint8)
        else:
            warped_img = np.zeros((size[0], size[1]), np.uint8)
    else:
        warped_img = background.copy()

    warped_img[dst_y.ravel(), dst_x.ravel()] = img[src_y.ravel(), src_x.ravel()]
    warped_img = warped_img.astype(np.uint8)

    return warped_img


def get_corner_points(x_points, y_points):
    x_min = np.min(x_points)
    x_min_idx = np.argmin(x_points)
    y_x_min = y_points[x_min_idx]

    y_min = np.min(y_points)
    y_min_idx = np.argmin(y_points)
    x_y_min = x_points[y_min_idx]

    x_max = np.max(x_points)
    x_max_idx = np.argmax(x_points)
    y_x_max = y_points[x_max_idx]

    y_max = np.max(y_points)
    y_max_idx = np.argmax(y_points)
    x_y_max = x_points[y_max_idx]

    p1 = (x_min, y_x_min)
    p2 = (x_y_min, y_min)
    p3 = (x_max, y_x_max)
    p4 = (x_y_max, y_max)

    ordered_pts = perspective.order_points(np.array([p1, p2, p3, p4]))

    p1_ = tuple(np.int32(ordered_pts[0]))
    p2_ = tuple(np.int32(ordered_pts[1]))
    p3_ = tuple(np.int32(ordered_pts[2]))
    p4_ = tuple(np.int32(ordered_pts[3]))

    return p1_, p2_, p3_, p4_


def get_quadrant_non_zero_count(img):
    h, w = img.shape
    top_left_quadrant = img[0: h // 2, 0: w // 2]
    top_right_quadrant = img[0: h // 2, w // 2: w]
    bottom_left_quadrant = img[h // 2: h, 0: w // 2]
    bottom_right_quadrant = img[h // 2: h, w // 2: w]

    top_left_non_zero_count = cv2.countNonZero(top_left_quadrant)
    top_right_non_zero_count = cv2.countNonZero(top_right_quadrant)
    bottom_left_non_zero_count = cv2.countNonZero(bottom_left_quadrant)
    bottom_right_non_zero_count = cv2.countNonZero(bottom_right_quadrant)

    return top_left_non_zero_count, top_right_non_zero_count, bottom_left_non_zero_count, bottom_right_non_zero_count


def get_orientation(ar_tag_4x4):

    tl, tr, bl, br = get_quadrant_non_zero_count(ar_tag_4x4)

    if tl >= max(tr, br, bl):
        orientation = "TL"
    elif tr >= max(br, bl, tl):
        orientation = "TR"
    elif br >= max(bl, tl, tr):
        orientation = "BR"
    elif bl >= max(tl, tr, br):
        orientation = "BL"

    return orientation


def get_ID(ar_tag_2x2):

    tl, tr, bl, br = get_quadrant_non_zero_count(ar_tag_2x2)

    bit_1 = 0
    bit_2 = 0
    bit_3 = 0
    bit_4 = 0

    h, w = ar_tag_2x2.shape

    w = w // 2
    h = h // 2

    if tl >= 0.8*w*h:
        bit_1 = 1
    if tr >= 0.8*w*h:
        bit_2 = 1
    if br >= 0.8*w*h:
        bit_3 = 1
    if bl >= 0.8*w*h:
        bit_4 = 1

    ID_binary = str(bit_4) + str(bit_3) + str(bit_4) + str(bit_1)
    ID = bit_4 * 8 + bit_3 * 4 + bit_2 * 2 + bit_1 * 1

    return [ID, ID_binary]


def get_tag_info(ar_tag, size):

    # center crop to extract 4x4 grid tag
    # divide 240[size[0]] by 8, we get each grid size = 30
    # from center we have to extract a square of size 120(size//2)  [-60, 60]
    grid_size = size[0] // 4
    ar_tag_4x4 = ar_tag[grid_size:grid_size * 3, grid_size:grid_size * 3]

    # for orientation divide the 4x4 tag into 2x2 and check for quadrant with highest white pixels
    orientation = get_orientation(ar_tag_4x4)

    orientation_for_ID = orientation
    while orientation_for_ID != "BR":
        ar_tag_4x4 = im.rotate(ar_tag_4x4, 90)
        orientation_for_ID = get_orientation(ar_tag_4x4)
        if orientation == orientation_for_ID:
            break

    # for ID, center crop the 4x4 into 2x2 and then flatten out the images
    grid_size = size[0] // 8
    ar_tag_2x2 = ar_tag_4x4[grid_size:grid_size * 3, grid_size:grid_size * 3]
    ID = get_ID(ar_tag_2x2)

    return ID, orientation


def rotate_img_for_ar(img_for_ar_tag, orientation):
    if orientation == "BR":
        return img_for_ar_tag
    elif orientation == "TL":
        return im.rotate(img_for_ar_tag, 180)
    elif orientation == "TR":
        return im.rotate(img_for_ar_tag, 90)
    elif orientation == "BL":
        return im.rotate(img_for_ar_tag, -90)


def get_bounding_rect(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]

    x_min = min(p1[0], p2[0], p3[0], p4[0])
    y_min = min(p1[1], p2[1], p3[1], p4[1])

    x_max = max(p1[0], p2[0], p3[0], p4[0])
    y_max = max(p1[1], p2[1], p3[1], p4[1])

    rect = [(x_min, y_min), (x_max, y_max)]

    return rect


def remove_noise(edges):
    edges = cv2.dilate(edges, np.ones((3, 3), np.int32), iterations=2)
    _, edges = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY)
    edges = cv2.erode(edges, np.ones((3, 3), np.int32), iterations=1)
    edges = cv2.medianBlur(edges, 5)
    # edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # edges = cv2.medianBlur(edges, 5)

    return edges


def detect_edges(img):
    """
    Inspired from this article
    https://akshaysin.github.io/fourier_transform.html#.YiZgrOjMLIU
    """

    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh.copy(), np.ones((3, 3), np.int32), iterations=2)
    thresh = cv2.medianBlur(thresh.copy(), 3)

    dft = cv2.dft(np.float32(thresh), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    mag_specturm = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    h, w = img.shape

    center_y, center_x = int(h / 2), int(w / 2)

    mask = np.ones((h, w, 2), np.uint8) * 255

    radius = 200

    y, x = np.ogrid[:h, :w]

    mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    mask[mask_area] = 0

    fshift = dft_shift * mask

    fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(mag_specturm, cmap='gray')
    plt.title('FFT output'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
    plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
    plt.title('Detected Edges'), plt.xticks([]), plt.yticks([])
    plt.show()
    #
    img_back = img_back / (img_back.max() / 255.0)
    img_back = np.asarray(img_back, np.uint8)

    return img_back


def get_tag_corners(thresh, edges=None):
    if edges is not None:
        y, x = np.where(edges == 255)
    else:
        y, x = np.where(thresh == 255)

    p1, p2, p3, p4 = get_corner_points(x, y)

    filled_rect = cv2.fillPoly(thresh.copy(), [np.array([p1, p2, p3, p4])], 255)
    filled_rect = cv2.erode(filled_rect, np.ones((11, 11), np.int32), iterations=3)
    # cv2.namedWindow('filled_rect', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('filled_rect', filled_rect)

    not_filled_rect = cv2.bitwise_not(filled_rect)
    # cv2.namedWindow('not_filled_rect', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('not_filled_rect', not_filled_rect)

    tag_mask = cv2.bitwise_or(not_filled_rect, thresh)
    # cv2.namedWindow('tag_mask', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('tag_mask', tag_mask)

    y, x = np.where(tag_mask == 0)
    p1, p2, p3, p4 = get_corner_points(x, y)

    return p1, p2, p3, p4


def preprocess_image(img):
    gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (11, 11), 0)
    gray_img = cv2.copyMakeBorder(gray_img, 0, 100, 0, 100, cv2.BORDER_CONSTANT, 0)
    # cv2.namedWindow('gray_img', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('gray_img', gray_img)

    _, basic_thresh = cv2.threshold(gray_img.copy(), 185, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('basic_thresh', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('basic_thresh', basic_thresh)
    adv_thresh = cv2.dilate(basic_thresh, np.ones((5, 5), np.int32), iterations=3)
    adv_thresh = cv2.medianBlur(adv_thresh, 7)
    adv_thresh = cv2.medianBlur(adv_thresh, 7)

    return adv_thresh, basic_thresh


def get_projection_matrix(homography):

    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]

    lmbda = 2.0 / (np.linalg.norm(np.dot(K_inv, h1)) + np.linalg.norm(np.dot(K_inv, h2)))

    B_tilde = lmbda * np.dot(K_inv, homography)
    if np.linalg.norm(B_tilde) < 0:
        B_tilde *= -1

    b1 = B_tilde[:, 0]
    b2 = B_tilde[:, 1]
    b3 = B_tilde[:, 2]

    r1 = b1
    r2 = b2
    r3 = np.cross(h1, h2)
    t = b3

    transformation_mat = np.array([r1, r2, r3, t]).T

    projection_matrix = np.dot(K, transformation_mat)

    projection_matrix /= projection_matrix[2, 3]

    return projection_matrix


def get_cube_points(proj_matrix, points):

    num_points = len(points)

    cube_points = []
    for idx in range(num_points):
        cube_points.append(np.array([points[idx][0], points[idx][1], -IMG_SIZE[0] + 1, 1]))

    world_points = np.array(cube_points).T

    camera_points = np.dot(proj_matrix, world_points)

    camera_points /= camera_points[2, :]

    return np.int32(camera_points[:2, 0]), np.int32(camera_points[:2, 1]), np.int32(camera_points[:2, 2]), np.int32(camera_points[:2, 3])


def draw_cube(img, points_bottom, points_top):
    draw_img = img.copy()

    num_points = len(points_bottom)
    for idx in range(num_points):
        cv2.line(draw_img, points_bottom[idx % num_points], points_bottom[(idx + 1) % num_points], (0, 0, 255), 4, 16)
        cv2.line(draw_img, points_top[idx % num_points], points_top[(idx + 1) % num_points], (0, 255, 0), 4, 16)
        cv2.line(draw_img, points_bottom[idx % num_points], points_top[idx % num_points], (255, 0, 0), 4, 16)

    return draw_img
