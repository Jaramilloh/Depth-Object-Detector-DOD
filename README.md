# Depth-Object-Detector-DOD

Implementation of the novel method DOD for object detection with depth estimation integrated within the prediction heads of a convolutional neural network of one million of parameters.

The architecture of the proposed method is heavily inspired in the architecture and implementation of the object detector model YOLOv8 [1] implemented by [Ultralitycs](https://docs.ultralytics.com/models/yolov8/) under the AGPL 3.0 License.

The model itself was trained and evaluated with the Common Objects in Context (COCO) 2017 [2] dataset for object detection task, and with the MinneApple [3,4,5] dataset for fruits detection for edge IoT applications. These datasets were proccesed in order to synthetize a representative depth value for each object label using the monocular depth map predictor model MiDas [6,7] implemented by [Intelligent Systems Lab Org](https://github.com/isl-org).


References:

[1] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics
[2] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollár, “Microsoft coco: Common objects in context,” 2015.
[3] N. Häni, P. Roy, and V. Isler, “Minneapple: A benchmark dataset for apple detection and segmentation,” 2019.
[4] N. Häni, P. Roy, and V. Isler, “A comparative study of fruit detection and counting methods for yield mapping in apple orchards,” Journal of Field Robotics, vol. 37, no. 2, pp. 263–282, aug 2019. [Online]. Available: 
https://doi.org/10.1002%2Frob.21902
[5] N. Häni, P. Roy, and V. Isler, “Apple counting using convolutional neural networks,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 2559–2565.
[6] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” ICCV, 2021
[7] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 3, 2022.
