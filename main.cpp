//#include "c:\Eigen\Eigen\Dense"
#include <iostream>
#include "InOut.h"
#include "Sample.h"
#include "FeatureFactory.h"
#include "Node.h"
#include "Forest.h"
#include "utils.h"

int main(int argc, char** argv)
{
	// prepare dataset
	// instantiate a training object
	InOut trnObj;
	Eigen::MatrixXf trnData;
	Eigen::VectorXi trnLabels;

	trnObj.readPoints("./dataset/downsampled.txt", trnData);
	trnObj.readLabels("./dataset/downsampled.labels", trnLabels);

	// search for the k nearest neighbors for each point in the dataset
	// and store the indices and dists in two matrices for later use by
	// indexing instead of searching again
	Eigen::MatrixXi trnIndices;
	Eigen::MatrixXf trnDists;
	trnObj.searchNN(trnData, trnIndices, trnDists);

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

	int numTrees = 1;
	int maxDepths = 3;
	int minSamplesPerLeaf = 20;
	float infoGainThresh = 0;
	RandomForest rf(numTrees, maxDepths, minSamplesPerLeaf, infoGainThresh);
	int numClasses = 9;
	int featsPerNode = 10;
	rf.train(&trnData, &trnLabels, &trnIndices, &trnDists, numClasses, featsPerNode);


	Eigen::VectorXi predictedLabels;
	rf.predict("./dataset/testset.txt", predictedLabels);
	std::cout << predictedLabels << std::endl;
	system("pause");
	return 0;
}