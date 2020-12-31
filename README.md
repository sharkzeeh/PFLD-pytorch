# Face rotation evaluation

Implementation of the euler angles estimation via `PFLD` for faces cropped by the object detection model named `yolov3`.
## Getting started guide

0. Change cwd to the main folder of the project
1. Run `python3 get_euler_angles.py` with a source image:
    * `--source https://` (you can give a web link to the image)
    * `--source /path/to/file/image.jpg` (single image file)
    * `--source some/dir` (directory with images)
    * default source will locate images in `./data/samples/`
2. Watch command line to get the euler angles

### Docker container

To build your own docker container do as follows in your Unix terminal:
0. `make build` - this will only build the container
1. `make start` - this will build and start your  newly built docker container
2. `python3 get_euler_angles.py --source ./data/samples/driver.jpg` - type it inside docker container terminal to see the cropped image and the corresponding euler angles
3. For a custom web image please refer to point 1 in `Getting started guide`
4. To see the results for a batch of images please also refer to point 1 in `Getting started guide`

Optionally, you can download a prebuilt docker container from Dockerhub via the link:

~~~shell
docker pull sharkzeeh/face:v1
~~~
and then do the same steps

#### Install requirements

~~~shell
pip3 install -r requirements.txt
~~~

#### Datasets

Euler angles are calculated on the
[Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) face dataset. It contains 10000 faces (7500 for training and 2500 for testing)  with 98 fully manual annotated landmarks.
#### Reference: 

 PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf