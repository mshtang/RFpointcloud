The repo stores the code accompanying to my master thesis "Semantic Classification of 3D Point Cloud by Random Forest-based Feature Learning". The code can **by no means** used directly in a prodution environment because the focus of the project to verify a novel idea, which is, to make use of the feature learning capability at a leaf node in a tree to automatically choose the feature that can split the data points optimally. To improve the final results, many tree (a forest) is combined. This technique is one example of the ensemble learning methods.

The code is not reviewed by a third party, so it's not well-organized or optimized. Some basic operations, such as point cloud IO, can be replaced by [Point Cloud Library](github.com/PointCloudLibrary/pcl). Bugs are expected.

### Workflow:
![workflow](https://user-images.githubusercontent.com/32868278/107156977-c57e0480-6981-11eb-8709-ecaaf2bc43ae.png)

### Test result:
The dataset in test is [Oakland 3-D Point Cloud Dataset](https://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)

#### Ground truth:
![Ground truth](https://user-images.githubusercontent.com/32868278/107157007-f827fd00-6981-11eb-8ab2-9e6b500e95a7.jpg)

#### Prediction:
![Prediction](https://user-images.githubusercontent.com/32868278/107157013-00803800-6982-11eb-890f-0c1744082317.jpg)

#### Confusion Matrix:
\begin{table}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{c|ccccc|c||ccccc|c}
 & \multicolumn{6}{c||}{This Work} & \multicolumn{6}{c}{Baseline} \\ \hline
 & v & w & p & g & f & Re. & v & w & p & g & f & Re. \\ \hline
v & 242515 & 3575 & 1332 & 7287 & 12616 & 0.907 & 223148 & 19042 & 10308 & 208 & 724 & 0.835 \\
w & 933 & 1228 & 628 & 203 & 802 & 0.324 & 906 & 1831 & 125 & 208 & 724 & 0.483 \\
p & 3557 & 937 & 1565 & 341 & 1533 & 0.197 & 1432 & 1378 & 4729 & 8 & 386 & 0.596 \\
g & 8638 & 1599 & 420 & 915344 & 8145 & 0.98 & 11356 & 4690 & 501 & 917418 & 191 & 0.982 \\
f & 19514 & 1584 & 1268 & 17063 & 71683 & 0.645 & 9128 & 7506 & 5577 & 237 & 88664 & 0.798 \\ \hline
Pre. & 0.881 & 0.138 & 0.300 & 0.974 & 0.756 &  & 0.907 & 0.053 & 0.223 & 0.996 & 0.876 &  \\ \cline{1-6} \cline{8-12}
IoU & 0.808 & 0.107 & 0.135 & 0.954 & 0.534 &  & 0.769 & 0.0502 & 0.194 & 0.978 & 0.542 & 
\end{tabular}%
}
\caption{Confusion matrix for the test results of the trained classifier on test set of the Oakland 3D point cloud dataset. v -- vegetation, w -- wire, p -- pole/trunk, g -- ground, f -- facade. Re.: Recall, Pre.: precision. Reference labels are shown in the left column, the predicted labels in the upper row. Overall accuracy of our method: 0.931, baseline: 0.933. Average intersection over union of our method: 0.508, baseline: 0.542.}
\label{tb:oakland_result_table}
\end{table}

For more details, please refer to the pdf.

### In order to run the code:

Place the dataset under the 'dataset' directory and update the path in main.cpp (line 17/18/39) The current files found in the dataset directory are toy datasets used for testing during the developing of the programm. (Further todo: input these parameters as command arguments)

Modify the directives concerning the dependent Eigen libary in the Sample.h, InOut.h and utils.h 

