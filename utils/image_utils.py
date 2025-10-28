import cv2
import numpy as np
import torch

def loadImage(img_path):
    """ Load an image from a file path.
    Args:
        image_path (str): Path to the image file.
    Returns:
        img (numpy.ndarray): Loaded image.
    """

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {img_path}")
    return img

def preprocessImage(img, target_size=(640, 640), device = None):
    """ Preprocess the image for model input.
    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Target size for resizing (width, height).
        device: Device to move the tensor to (e.g., 'cpu' or 'cuda').
    Returns:
        img (torch.Tensor): Preprocessed image tensor.
    """
    
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    img -= (104, 117, 123)  # Mean subtraction
    img = img.transpose(2, 0, 1)  # Change data layout from HWC to CHW
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension and move to device
    if device is not None:
        img = img.to(device)
    return img

def cropFace(img, bbox, landms, expand_ratio = 0.0):
    """ Crop the image base on bounding box and recompute landmarks.
    Args:
        img: Input image.
        bbox: Predicted bouding box.
        landms: Predicted landmarks.
        expand_ratio: Ratio to expand the cropped image.
    Returns:
        cropped_img: Cropped face image.
        cropped_landms: Recomputed landmarks in cropped image.
    """

    # Convert bounding box and landmarks to numpy array 
    bbox = np.array(bbox, dtype = np.float32)
    landms = np.array(landms, dtype = np.float32).reshape(5, 2)

    # Compute expanding size
    x1, y1, x2, y2, _ = bbox
    width, height = x2 - x1, y2 - y1
    crop_x, crop_y = x1 + width / 2, y1 + height / 2

    size = int(max(width, height) * (1 + expand_ratio))
    x1_crop = int(crop_x - size / 2)
    y1_crop = int(crop_y - size / 2)
    x2_crop = int(x1_crop + size)
    y2_crop = int(y1_crop + size)

    # Limit cropping size
    x1_crop = max(0, x1_crop)
    y1_crop = max(0, y1_crop)
    x2_crop = min(img.shape[1], x2_crop)
    y2_crop = min(img.shape[0], y2_crop)

    # Crop image and recompute landmarks
    cropped_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
    cropped_landms = landms - np.array([x1_crop, y1_crop], dtype=np.float32)

    return cropped_img, cropped_landms

def rotateFace(cropped_face, cropped_landms):
    """ Retite the Face image base on landmarks.
    Args:
        cropped_face: Face image.
        cropped_landms: Landmarks of face in cropped_face.
    Returns:
        rotated_face: Rotated face image.
        R: Rotation matrix.
    """

    left_eye, right_eye = cropped_landms[0], cropped_landms[1]
    eyes_center = np.mean([left_eye, right_eye], axis=0)

    # Tính góc xoay
    eye_vector = right_eye - left_eye
    angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

    # Tính ma trận xoay
    R = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)

    rotated_face = cv2.warpAffine(cropped_face, R, (cropped_face.shape[1], cropped_face.shape[0]), flags=cv2.INTER_LINEAR)

    return rotated_face, R

def transformDetection(bbox, landms, tform):
    """ Transform bounding box and landmarks using affine transformation.
    Args:
        bbox: Bounding box to be transformed.
        landms: Landmarks to be transformed.
        tform: Affine transformation matrix.
    Returns:
        aligned_bbox: Transformed bounding box.
        aligned_landms: Transformed landmarks.
    """

    x1, y1, x2, y2, conf_score = bbox
    bbox_points = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    # Thêm một cột các giá trị 1 để biến đổi affine
    bbox_points_3d = np.hstack([bbox_points, np.ones((4, 1), dtype=np.float32)])
    landms_3d = np.hstack([landms.reshape(5, 2), np.ones((5, 1), dtype=np.float32)])
    
    # Áp dụng biến đổi affine
    aligned_landms = (tform @ landms_3d.T).T
    transformed_bbox = (tform @ bbox_points_3d.T).T

    # Lấy bounding box mới
    x_min, y_min = transformed_bbox[:, 0].min(), transformed_bbox[:, 1].min()
    x_max, y_max = transformed_bbox[:, 0].max(), transformed_bbox[:, 1].max()
    aligned_bbox = np.array([x_min, y_min, x_max, y_max, conf_score], dtype=np.float32)
    
    return aligned_bbox, aligned_landms.flatten()

def alignFace(img, landms, bbox, output_size=(640, 640), expand_ratio=0.0):
    """ Align face in the image based on landmarks and bounding box.
    Args:
        img: Input image.
        landms: Landmarks of the face.
        bbox: Bounding box of the face.
        output_size: Desired output size (width, height).
        expand_ratio: Ratio to expand the cropped image.
    Returns:
        aligned_face_resized: Aligned face image resized to output_size.
    """

    # Crop face from image
    if len(bbox) > 5:
        print(f"Warning: bbox has more than 5 elements: {bbox}")
        return None
    face_crop, landms_crop = cropFace(img, bbox, landms, expand_ratio)

    # Xoay khuôn mặt theo mắt
    rotated_face, _ = rotateFace(face_crop, landms_crop)

    # Resize về kích thước chuẩn
    out_width, out_height = int(output_size[0]), int(output_size[1])

    aligned_face_resized = cv2.resize(rotated_face, (out_width, out_height))
    
    return aligned_face_resized

def visualizeDetections(img, detection = None, landm = None, conf_threshold=0.5):
    """ Visualize detections and landmarks on the image.
    Args:
        img: Input image.
        detection: Detected bounding box [x1, y1, x2, y2, score].
        landm: Detected landmarks [x1, y1, x2, y2, ..., x5, y5].
        conf_threshold: Confidence threshold to display bounding box.
    Returns:
        None
    """

    landm = landm.flatten()
    print(f"Visualizing detection with bounding box: {detection}")
    print(f"Landmarks: {landm}")
    if detection is not None:
        if detection[4] >= conf_threshold:
            text = "{:.4f}".format(detection[4])
            detection = list(map(int, detection))
            cv2.rectangle(img, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 2)
            cx, cy = detection[0], detection[1] + 12
            cv2.putText(img, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    if landm is not None:       
        for i in range(5):
            cv2.circle(img, (int(landm[2 * i]), int(landm[2 * i + 1])), 1, (0, 0, 255), 4)

    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resizeWithAspectRatio(image, target_size = (640, 640), pad_value = (0, 0, 0)):
    """ Resize image while maintaining aspect ratio by padding.
    Args:
        image: Input image.
        target_size: Desired output size (width, height).
        pad_value: Padding color value (B, G, R).
    Returns:
        resized_image: Resized image with padding.
    """

    target_width, target_height = target_size
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_width, new_height = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Tính padding để căn giữa
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    pad_right = target_width - new_width - pad_left
    pad_bottom = target_height - new_height - pad_top

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)

    return padded_image, scale, pad_left, pad_top, new_width, new_height