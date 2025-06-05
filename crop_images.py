from pathlib import Path
import cv2
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SRC_ROOT   = Path("raw_images")              # original breed folders
DST_ROOT   = Path("cropped_cats")            # where crops will be written
MODEL      = YOLO("yolov8n.pt")              # or yolov8s.pt for a bit more accuracy
CAT_CLASS  = 15                              # COCO class index for “cat”
CONF       = 0.4                             # confidence threshold
AREA_MAX   = 0.90                            # skip box if ≥ 95 % of whole image
IMG_SUFFIX = "_crop"                         # append to filename
# ─────────────────────────────────────────────────────────────────────────────

def best_cat_box(result):
    """
    Return the xyxy box (ints) of the *largest* detected cat whose area is
    < AREA_MAX * whole_image, or None if nothing suitable found.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    h, w = result.orig_shape
    best_area, best_xyxy = 0, None

    for cls, conf, box in zip(result.boxes.cls,
                              result.boxes.conf,
                              result.boxes.xyxy):
        if int(cls.item()) != CAT_CLASS or conf.item() < CONF:
            continue

        x1, y1, x2, y2 = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        area = (x2 - x1) * (y2 - y1)
        if area / (w * h) >= AREA_MAX:
            continue
        if area > best_area:
            best_area = area
            best_xyxy = (x1, y1, x2, y2)

    return best_xyxy

def crop_and_save(src_img, xyxy, dest_path):
    x1, y1, x2, y2 = xyxy
    crop = src_img[y1:y2, x1:x2]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), crop)
    print(f"{dest_path.relative_to(DST_ROOT.parent)} "
          f"(box: {x1},{y1} – {x2},{y2})")

def main():
    for img_path in SRC_ROOT.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Couldn’t read {img_path}")
            continue

        # Run detector
        result = MODEL.predict(img, classes=[CAT_CLASS], conf=CONF, verbose=False)[0]
        print(f'result: {result.boxes.xyxy.shape} boxes, '
              f'{result.boxes.conf.shape} confidences, '
              f'{result.boxes.cls.shape} classes')
        xyxy = best_cat_box(result)

        if xyxy is None:
            print(f"No usable cat box: {img_path}")
            continue

        # Build destination path:  dst_root/<breed>/<file>_crop.jpg
        rel = img_path.relative_to(SRC_ROOT)
        dest_file = rel.with_stem(rel.stem + IMG_SUFFIX)
        dest_path = DST_ROOT / dest_file

        crop_and_save(img, xyxy, dest_path)

if __name__ == "__main__":
    main()
