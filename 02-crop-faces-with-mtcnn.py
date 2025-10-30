import cv2
from mtcnn import MTCNN
import sys, os, json
from keras import backend as K
import tensorflow as tf

print(tf.__version__)
# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --- GPU Handling Section ---
physical_devices = tf.config.list_physical_devices('GPU')
print("Detected GPU devices:", physical_devices)

if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU memory growth enabled.")
    except Exception as e:
        print(f"⚠️ Could not set memory growth: {e}")
else:
    print("⚠️ No GPU found. Running on CPU instead.")

# --- Base Path ---
base_path = './train-sample-videos/'  

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# --- Load Metadata ---
metadata_path = os.path.join(base_path, 'metadata.json')
if not os.path.exists(metadata_path):
    print(f"❌ metadata.json not found at {metadata_path}")
    sys.exit(1)

with open(metadata_path) as metadata_json:
    metadata = json.load(metadata_json)
    print(f"Loaded metadata for {len(metadata)} files")

# --- Face Extraction Loop ---
for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    if not os.path.exists(tmp_path):
        print(f"⚠️ Skipping {tmp_path} (folder not found)")
        continue

    print(f"\nProcessing Directory: {tmp_path}")
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]

    faces_path = os.path.join(tmp_path, 'faces')
    os.makedirs(faces_path, exist_ok=True)
    print(f"Created/Using Directory: {faces_path}")

    print("Cropping Faces from Images...")
    detector = MTCNN()

    for frame in frame_images:
        print(f"Processing {frame}")
        image_path = os.path.join(tmp_path, frame)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping {frame} (could not read image)")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print(f"Faces Detected: {len(results)}")

        count = 0
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']

            # Skip uncertain faces
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3
                margin_y = bounding_box[3] * 0.3
                x1 = max(int(bounding_box[0] - margin_x), 0)
                y1 = max(int(bounding_box[1] - margin_y), 0)
                x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])

                crop_image = image[y1:y2, x1:x2]
                new_filename = f"{os.path.join(faces_path, get_filename_only(frame))}-{count:02d}.png"
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                count += 1
            else:
                print("Skipped a low-confidence face.")
