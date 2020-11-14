import cv2
from imgaug import augmenters as iaa
from tensorflow.image import crop_to_bounding_box


def get_bounding_boxes(img, detector):
    result = detector.detect_faces(img)
    bounding_boxes = []
    if len(result) == 0:
        return bounding_boxes
    for res in result:
        bounding_boxes.append(res["box"])
    return bounding_boxes


def draw_bounding_boxes(img, bounding_boxes):
    for bounding_box in bounding_boxes:
        img = cv2.rectangle(
            img,
            (bounding_box[0], bounding_box[1]),
            (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            (255, 255, 255),
            3,
        )
    return img


def crop_by_bounding_box(img, detector, bounding_box=None, vertical_expand=0.05):
    if bounding_box is None:
        result = detector.detect_faces(img)
        if len(result) == 0:
            return None
        bounding_box = result[0]["box"]

    y = bounding_box[1]
    x = bounding_box[0]
    height = bounding_box[3]
    width = bounding_box[2]

    # Account for top left corner outside image
    if y < 0:
        height += y
        y = 0
    if x < 0:
        width += x
        x = 0

    # Expand bounding box vertically by vertical_expand % of bb height in each direction
    # Move y-coord of top-left corner upwards
    old_y = y
    y = max(y - int(vertical_expand * height), 0)
    height += old_y - y

    # Increase height of bounding box
    h, w, _ = img.shape
    height = min(int((1 + vertical_expand) * height), h)

    # Fill horizontally or vertically until width = height
    if height > width:
        if height > w:
            x = 0
            width = w
        else:
            diff_to_fill = height - width

            # Fill leftwards
            old_x = x
            x = max(0, x - int(diff_to_fill / 2))
            x_change = old_x - x
            diff_to_fill -= x_change
            width += x_change

            # Fill rightwards
            width = min(width + diff_to_fill, w - x)
    else:
        diff_to_fill = width - height
        # Fill upwards
        old_y = y
        y = max(0, y - int(diff_to_fill / 2))
        y_change = old_y - y
        diff_to_fill -= y_change
        height += y_change

        # Fill downwards
        height += min(height + diff_to_fill, h - y)
    # Cut height if larger than width
    if height > w:
        diff = height - w
        y += int(diff / 2)
        height -= diff

    crop = crop_to_bounding_box(img, y, x, height, width).numpy()

    return crop


def crop_images(input_img, bounding_boxes):
    if len(bounding_boxes) == 0:
        display_image = input_img
        h, w, _ = display_image.shape
        crop_aug = iaa.CropToFixedSize(
            height=min([h, w]), width=min([h, w]), position="center"
        )
        cropped = [crop_aug(image=display_image)]

    else:
        img_copy = input_img.copy()
        display_image = draw_bounding_boxes(input_img, bounding_boxes)
        cropped = [
            crop_by_bounding_box(
                img_copy,
                None,
                bounding_box=bounding_box,
                vertical_expand=0.1,
            )
            for bounding_box in bounding_boxes
        ]
    return cropped, display_image


def draw_class_names(display_image, bounding_boxes, pred_probs, pred_labels):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    for i, (label, prob) in enumerate(zip(pred_labels, pred_probs)):
        if len(bounding_boxes) == 0:
            x, y, height = 0, 0, 30
        else:
            x, y, _, height = bounding_boxes[i]

        cv2.putText(
            display_image,
            f"{label}: {prob*100:.2f}%",
            (x, y + height + 40),
            font,
            fontScale,
            fontColor,
            lineType,
        )
