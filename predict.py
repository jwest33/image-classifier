import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config – must match the training script
# ---------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (128, 128)
MODEL_PATH = Path("cat_breed_classifier.keras")
DATA_DIR   = Path("cropped_cats")  # used only to rebuild class-name list
TOP_K      = 5                      # how many best predictions to display

# ---------------------------------------------------------------------------
# Helper – load image, *center-crop* to square, resize to 128×128, batch-dim
# ---------------------------------------------------------------------------

def load_and_prepare(img_path: Path) -> tf.Tensor:
    """Read an image, center-crop the longest edge off, resize, expand dims."""
    img_raw = tf.io.read_file(str(img_path))
    img     = tf.io.decode_jpeg(img_raw, channels=3)              # uint8

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    side = tf.minimum(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2

    img = tf.image.crop_to_bounding_box(img, top, left, side, side)
    img = tf.image.resize(img, IMG_SIZE)                         # float32
    img = tf.expand_dims(img, axis=0)                            # (1, H, W, 3)
    img = tf.cast(img, tf.float32)
    return img

# ---------------------------------------------------------------------------
# Main – load model, rebuild label map, predict, print top-k
# ---------------------------------------------------------------------------

def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cat_image_test_3.jpg")

    # 1. Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Re-create class-name list (sorted for stable ordering)
    class_names = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    num_classes = len(class_names)

    # 3. Build input tensor & run prediction
    x = load_and_prepare(img_path)
    probs = model.predict(x, verbose=0)[0]                        # shape (C,)

    # 4. Get top-k indices (handle case where C < TOP_K)
    k = min(TOP_K, num_classes)
    top_indices = np.argsort(probs)[-k:][::-1]                    # high → low

    print(f"Top {k} predictions for {img_path.name} (model: {MODEL_PATH}):\n")
    for rank, idx in enumerate(top_indices, start=1):
        label = class_names[idx]
        confidence = probs[idx]
        print(f"  {rank:>2}. {label:<20} {confidence:6.1%}")

if __name__ == "__main__":
    main()
