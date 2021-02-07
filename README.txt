The repo stores the code accompanying to my master thesis "Semantic Classification of 3D Point Cloud by Random Forest-based Feature Learning". The code can **by no means** used directly in a prodution environment because the focus of the project to verify a novel idea, which is, to make use of the feature learning capability at a leaf node in a tree to automatically choose the feature that can split the data points optimally. To improve the final results, many tree (a forest) is combined. This technique is one example of the ensemble learning methods.

The code is not reviewed by a third party, so it's not well-organized or optimized. Some basic operations, such as point cloud IO, can be replaced by [Point Cloud Library](github.com/PointCloudLibrary/pcl). Bugs are expected.

### Workflow:
![workflow](https://user-images.githubusercontent.com/32868278/107156977-c57e0480-6981-11eb-8709-ecaaf2bc43ae.png)

### Test result:
The dataset in test is [Oakland 3-D Point Cloud Dataset](https://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)

#### Ground truth:
![oak_truth_part1](https://user-images.githubusercontent.com/32868278/107157007-f827fd00-6981-11eb-8ab2-9e6b500e95a7.jpg)

#### Prediction:
![oak_predict_part1](https://user-images.githubusercontent.com/32868278/107157013-00803800-6982-11eb-890f-0c1744082317.jpg)

#### Confusion Matrix:
![Screenshot 2021-02-07 205458](https://user-images.githubusercontent.com/32868278/107157885-d8470800-6986-11eb-98f2-ab07992e1d10.jpg)
For more details, please refer to the pdf.

### In order to run the code:

Place the dataset under the 'dataset' directory and update the path in main.cpp (line 17/18/39) The current files found in the dataset directory are toy datasets used for testing during the developing of the programm. (Further todo: input these parameters as command arguments)

Modify the directives concerning the dependent Eigen libary in the Sample.h, InOut.h and utils.h 

