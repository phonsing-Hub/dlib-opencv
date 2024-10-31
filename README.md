# Face Recognition with Landmark Detection

This project uses OpenCV and dlib libraries to perform face recognition and landmark detection on images. It detects faces in images, computes face encodings, and compares them to determine if they match.

## Requirements

- Python 3.x
- OpenCV
- dlib
- numpy

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
# Files Needed
- `shape_predictor_68_face_landmarks.dat` This file is used for landmark detection. You can download it from dlib’s model files and extract it.
- `dlib_face_recognition_resnet_model_v1.dat` This file is used for face recognition. You can download it from dlib’s model files and extract it.
- Two images named Image1.png and Image2.png containing the faces you want to compare.
