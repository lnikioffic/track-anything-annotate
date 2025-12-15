import cv2


def threshold(img, thresh=127, mode='inverse'):
    im = img.copy()
    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    ret, thresh = cv2.threshold(im, thresh, 255, thresh_mode)
    return thresh


def draw_annotations(img, bboxes, thickness=2, color=(0, 255, 0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)
    return annotations


def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)

    return morphed


def get_filtered_bboxes(img, min_area_ratio):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Отсортируем контуры по площади, от большего к меньшему.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    # Удаляем максимальную площадь, самый внешний контур.
    # sorted_cnt.remove(sorted_cnt[0])
    # Container to store filtered bboxes.
    bboxes = []
    # Область изображения.
    im_area = img.shape[0] * img.shape[1]
    for cnt in sorted_cnt:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Удалите очень мелкие дефекты.
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, x + w, y + h))
            
    if len(bboxes) > 0:
        min_x = min([bbox[0] for bbox in bboxes])
        min_y = min([bbox[1] for bbox in bboxes])
        max_x = max([bbox[2] for bbox in bboxes])
        max_y = max([bbox[3] for bbox in bboxes])

        combined_width = max_x - min_x
        combined_height = max_y - min_y

        return [(min_x, min_y, min_x + combined_width, min_y + combined_height)]
        
    return bboxes


def get_filtered_bboxes_xywh(img, min_area_ratio):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    # Область изображения.
    im_area = img.shape[0] * img.shape[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Удалите очень мелкие дефекты.
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, w, h))

    if len(bboxes) > 1:
        # Вычисляем минимальные и максимальные координаты.
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)  # x + width
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)  # y + height

        # Вычисляем ширину и высоту объединенного прямоугольника.
        combined_width = max_x - min_x
        combined_height = max_y - min_y

        # Возвращаем объединенный прямоугольник.
        return [(min_x, min_y, combined_width, combined_height)]
    
    return bboxes


def getting_coordinates(image_mask):
    gray_img = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    thresh_stags = threshold(gray_img, thresh=110, mode='direct')
    bboxes = get_filtered_bboxes_xywh(thresh_stags, min_area_ratio=0.001)
    return bboxes
