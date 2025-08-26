# **SkyVision: Real-Time Object Detection for Drone Navigation (YOLO-based)**

This repo contains a Jupyter notebook that prepares a drone-vision dataset, converts annotations into YOLO format, and runs **YOLOv3** inference via OpenCV’s DNN module. It also includes simple **TensorFlow/Keras baselines** (a quick image classifier and an experimental fixed-box multi-object detector) to help you benchmark or extend the pipeline.

> **Why this exists:** Rapidly prototype onboard/in-the-loop detection for drones, starting from raw annotations → YOLO labels → ready-to-run inference and quick learning baselines.

---

## **Key Features**

* 📦 **Dataset parsing (VisDrone-style):** Reads `_annotations.txt` lines like
  `image.jpg x1,y1,x2,y2,class_id [x1,y1,x2,y2,class_id]...`
* 🏷️ **YOLO label export:** Converts annotations to YOLO txt per image (`class x_center y_center width height`, normalized).
* 👁️ **Visualization tools:** Plot images with ground-truth boxes to verify labels.
* 🚀 **YOLOv3 inference (OpenCV DNN):** Load `yolov3.cfg` + `yolov3.weights`, run detection, and visualize predictions.
* 🔬 **Baselines in Keras (optional):**

  * **Classifier baseline:** Predicts the primary object class in an image (fast sanity check).
  * **Experimental multi-object head:** A simple CNN predicting up to *N* boxes + classes (for learning/ablation, not SOTA).
* 📊 **Training curves & quick metrics:** Basic history plots and evaluation helpers.

> **Note:** The notebook *runs YOLO inference* but does **not** fine-tune YOLO. For training YOLO, consider Ultralytics YOLOv5/YOLOv8/YOLO11 or Darknet forks.

> In the notebook there are references to both `labels/` and `data/labels/`. Use **one** consistently. The conversion step writes to `labels/` by default.

---

## **Setup**

### **1) Environment**

* Python 3.9+
* Recommended: create a virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Minimal `requirements.txt`:**

```
numpy
pandas
matplotlib
opencv-python
tensorflow>=2.10
```

> If you have a compatible GPU, consider `tensorflow-gpu` and proper CUDA/cuDNN drivers.

---

### **2) Weights & Config (YOLOv3)**

Download the official **YOLOv3** files and place them in `yolo3/`:

* `yolov3.cfg`
* `yolov3.weights`
* `coco_classes.txt` (80 COCO labels, one per line)

> The notebook loads COCO class names when running the pre-trained YOLOv3 model. Your dataset’s `_classes.txt` is used for *ground truth* visualization and label conversion.

---

## **Data & Annotations**

* Place all images in `data/images/`.
* Create/verify these files:

  * `data/images/_annotations.txt`
    Each line follows:

    ```
    <image_name> x1,y1,x2,y2,class_id [x1,y1,x2,y2,class_id] ...
    ```
  * `data/images/_classes.txt`
    One class per line, e.g.:

    ```
    pedestrian
    people
    motor
    car
    awning-tricycle
    object
    van
    bicycle
    truck
    bus
    others
    tricycle
    ```

---

### **Convert to YOLO format**

The notebook writes YOLO labels to `labels/<image>.txt` with **normalized** coordinates:

```
class_id x_center y_center width height
```

All values are in **\[0,1]** relative to `img_width`/`img_height` (default: 640×640 in the notebook).

---

## **How to Run (Notebook)**

1. Open `python.ipynb` in Jupyter/VS Code.
2. **Dataset sanity checks:**

   * Parse `_annotations.txt` → `df`
   * Visualize a sample with GT boxes (`show_image_with_boxes`)
3. **Export YOLO labels:**
   Convert annotations with `df.groupby('image')` and save to `labels/`.
4. **YOLOv3 inference (OpenCV DNN):**

   ```python
   cfg_path = "yolo3/yolov3.cfg"
   weights_path = "yolo3/yolov3.weights"
   classes_path = "yolo3/coco_classes.txt"
   ```

   Load and run:

   ```python
   net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
   layer_names = net.getLayerNames()
   output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
   ```

   Use `run_yolo_visdrone(image_path, conf_threshold=0.5, nms_threshold=0.4)` for testing.
5. **Baselines (optional):**

   * **Classifier:** Simple CNN predicting primary object per image.
   * **Experimental multi-object:** Educational model with fixed-box predictions.

---

## **Real-Time & Drone Integration**

To make it **real-time**:

```python
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # parse outs → boxes/classes/conf → draw
    cv2.imshow("SkyVision - YOLOv3", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
cap.release(); cv2.destroyAllWindows()
```

For drones:

* Add trackers like Deep SORT.
* Integrate with ROS 2 or PX4/MAVSDK.
* Migrate to YOLOv5/v8 for fine-tuning.

---

## **Tips & Gotchas**

* Use consistent paths.
* Check class name alignment between your `_classes.txt` and YOLO configs.
* Normalize images correctly during conversion.
* Enable GPU backend in OpenCV for better performance.

---

## **Results**

* Ground-truth and YOLOv3 visualizations.
* Baseline classifier accuracy (varies).
* Simple CNN experiment curves for educational purposes.

---

## **Roadmap**

* [ ] Real-time webcam/RTSP pipeline
* [ ] Drone telemetry + control integration
* [ ] Object tracking (Deep SORT, ByteTrack)
* [ ] Fine-tune YOLO models on dataset
* [ ] Export to ONNX/TensorRT

---

## **Acknowledgements**

* **YOLOv3** (Redmon & Farhadi)
* **VisDrone** dataset
* OpenCV, TensorFlow/Keras, NumPy, pandas, Matplotlib

---

## **License**

This project is for research/educational purposes. Check third-party component licenses (YOLOv3 weights, datasets) before commercial use.
