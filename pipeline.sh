 #!/bin/bash
set -e  # Exit immediately on error

# -------- CONFIGURATION --------
OPENPOSE_DIR="openpose"
IMAGES_DIR="hit_and_takes/images"
KEYPOINTS_DIR="../hit_and_takes/keypoints"
SMPLIFYX_DIR="smplify-x"
SMPLIFYX_CFG="cfg_files/fit_smpl.yaml"
MODEL_DIR="../smplx_models"
VPOSER_CKPT="./V02_05"
HIT_DIR="HIT"
OUT_DIR="../hit_and_takes"

# -------- FIND FIRST IMAGE (png or jpeg) --------
FIRST_IMAGE=$(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | head -n 1)

if [ -z "$FIRST_IMAGE" ]; then
  echo " No PNG or JPEG image found in $IMAGES_DIR"
  exit 1
fi

IMG_NAME=$(basename "$FIRST_IMAGE")
IMG_BASE="${IMG_NAME%.*}"   # strip extension (e.g., hi.png → hi)
echo "Found first image: $IMG_NAME"

# -------- RUN OPENPOSE --------
echo "Running OpenPose..."
cd "$OPENPOSE_DIR"
./build/examples/openpose/openpose.bin \
  --image_dir "../$IMAGES_DIR" \
  --face --hand \
  --write_json "$KEYPOINTS_DIR" \
  --display 0 --render_pose 0

# -------- RUN SMPLIFY-X --------
echo " Running SMPLify-X..."
cd "../$SMPLIFYX_DIR"
python smplifyx/main.py \
  --config "$SMPLIFYX_CFG" \
  --data_folder "$OUT_DIR" \
  --output_folder "$OUT_DIR" \
  --visualize="False" \
  --model_folder "$MODEL_DIR" \
  --vposer_ckpt "$VPOSER_CKPT"

# -------- RUN HIT --------
cd "../$HIT_DIR"
PKL_FILE="${OUT_DIR}/results/${IMG_BASE}/000.pkl"

if [ ! -f "$PKL_FILE" ]; then
  echo "Expected SMPLify-X output not found: $PKL_FILE"
  exit 1
fi

echo " Running HIT inference for $IMG_BASE ..."
python demos/infer_smpl.py \
  --exp_name=hit_male \
  --to_infer smpl_file \
  --target_body "$PKL_FILE" \
  --output='slices' \
  --betas 0.0 0.0 \
  --out_folder "$OUT_DIR"

echo " Pipeline completed successfully for $IMG_NAME!"
