//#include "c:\Eigen\Eigen\Dense"
#include <iostream>
#include "InOut.h"
#include "Sample.h"
#include "FeatureFactory.h"
#include "Node.h"
#include "Forest.h"
#include "utils.h"
#include <chrono>

int main(int argc, char** argv)
{
	// prepare dataset
	// instantiate a training object
	InOut trnObj;
	Eigen::MatrixXf cloud;
	//Eigen::MatrixXf points;
	Eigen::VectorXi cloudLabels;
	//Eigen::VectorXi pointLabels;

	//trnObj.readPoints("./datasets/bildstein_station1_xyz_intensity_rgb_trainData.txt", cloud);
	trnObj.readPoints("./toy_dataset/downsampled.txt", cloud);
	//trnObj.readLabels("./datasets/bildstein_station1_xyz_intensity_rgb_train.labels", cloudLabels);
	trnObj.readLabels("./toy_dataset/downsampled.labels", cloudLabels);

	// search for the k nearest neighbors for each point in the dataset
	// and store the indices and dists in two matrices for later use by
	// indexing instead of searching again

	Eigen::MatrixXi trnIndices;
	Eigen::MatrixXf trnDists;
	trnObj.searchNN(cloud, trnIndices, trnDists);

	/*Sample trnSamples(&trnData, &trnLabels, &trnIndices, &trnDists, 8, 1);
	Eigen::MatrixXf local_nei = trnSamples.buildNeighborhood(0);
	std::cout << "Neighborhood of Point 1 is:\n" << local_nei << std::endl;
	trnSamples.randomSampleFeatures();
	std::vector<Features> feats = trnSamples.getSelectedFeatures();
	std::cout << "No. of voxels are " << feats.at(0)._numVoxels << std::endl;
	Features testFeat;
	testFeat._featType = 16;
	testFeat._numVoxels = 1;
	testFeat._pointId = { 5 };
	testFeat._voxelSize = { 7 };
	FeatureFactory ff(local_nei, testFeat);

	std::cout << "Local neighbors are:\n" << ff.getLocalNeighbors() << std::endl;
	
	std::vector<Eigen::Matrix3f> cov = ff.computeCovarianceMatrix();
	for (auto i : cov)
		std::cout << i << std::endl << std::endl;

	std::vector<Eigen::Vector3f> eigv = ff.computeEigenValues(cov);
	for (auto i : eigv)
		std::cout << i.transpose() << std::endl << std::endl;
	 ff.computeFeature();*/
	
	int numTrees = 3;
	int maxDepths = 5;
	int minSamplesPerLeaf = 20;
	//RandomForest rf(numTrees, maxDepths, minSamplesPerLeaf);
	int numClasses = 8;
	int featsPerNode = 5;

	/*auto start = std::chrono::system_clock::now();
	rf.train(&cloud, &cloudLabels, &trnIndices, &trnDists, numClasses, featsPerNode);
	auto end = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
	std::cout << "Training takes: " << elapsed << "s" << std::endl;
	
	rf.saveModel("./models/test.model");*/

	RandomForest rf("./models/test.model");
	
	
	Eigen::VectorXi predictedLabels;

	auto start2 = std::chrono::system_clock::now();
	//rf.predict("./datasets/bildstein_station1_xyz_intensity_rgb_valData.txt", predictedLabels);
	rf.predict("./toy_dataset/testset.txt", predictedLabels);
	auto end2 = std::chrono::system_clock::now();

	auto elapsed2 = std::chrono::duration_cast<std::chrono::duration<double>>(end2 - start2).count();
	std::cout << "Predicting takes: " << elapsed2 << "s" << std::endl;
	
	//trnObj.writeToDisk("./datasets/predict.labels", predictedLabels);
	trnObj.writeToDisk("./toy_dataset/predict_from_restore.labels", predictedLabels);
	system("pause");
	return 0;
}