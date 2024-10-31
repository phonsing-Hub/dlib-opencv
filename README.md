# Face Recognition with Landmark Detection

This project uses OpenCV and dlib libraries to perform face recognition and landmark detection on images. It detects faces in images, computes face encodings, and compares them to determine if they match.

## Requirements

- Python 3.x
- OpenCV
- dlib
- numpy

You can install the required libraries using the following command
1. ```bash
   git clone https://github.com/phonsing-Hub/dlib-opencv.git
   cd dlib-opencv
   ```
2. ```bash
   python3 -m venv venv
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```
3. ```bash
   pip install -r requirements.txt
   ```

## Files Needed
- `shape_predictor_68_face_landmarks.dat` This file is used for landmark detection. You can download it from [dlib’s model files](https://github.com/davisking/dlib-models) files and extract it.
- `dlib_face_recognition_resnet_model_v1.dat` This file is used for face recognition. You can download it from [dlib’s model files](https://github.com/davisking/dlib-models) and extract it.
- Two images named Image1.png and Image2.png containing the faces you want to compare.

## How to Use
1.	Ensure you have the required libraries and the necessary model files in the same directory as your script.
2.	Place two images (Image1.png and Image2.png) in the same directory.
3.	Run the script
```bash
python main.py
```
4.	The script will display the images with landmarks drawn on the detected faces. It will also print whether the faces match based on the computed encodings.
5.	Press q to close the displayed images.

## How It Works
- The script loads the required models for face detection and recognition.
- It defines a function get_face_encoding_and_landmarks that reads an image, detects faces, computes landmarks, and generates face encodings.
- The script compares the face encodings of two images to determine if they match based on a defined threshold.

## Output
- The script will output a message indicating whether the faces match or not based on the distance between the face encodings.
- It will also display the images with landmarks for visual confirmation.

## Example Output
```bash
Faces match!
```
or
```bash
Faces do not match.
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.
