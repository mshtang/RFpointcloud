//#include "c:\Eigen\Eigen\Dense"
#include <iostream>
#include "InOut.h"
#include "Sample.h"
#include "FeatureFactory.h"
#include "Node.h"
#include "Forest.h"

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
	const int numNeighbors = 10;
	trnObj.searchNN(trnData, numNeighbors, trnIndices, trnDists);
	
	int numTrees = 1;
	int maxDepths = 4;
	int minSamplesPerLeaf = 20;
	float infoGainThresh = 0;
	RandomForest rf(numTrees, maxDepths, minSamplesPerLeaf, infoGainThresh);
	int numClasses = 9;
	int featsPerNode = 3;
	rf.train(&trnData, &trnLabels, &trnIndices, &trnDists, numClasses, featsPerNode);
	
	
	Eigen::VectorXi predictedLabels;
	rf.predict("./dataset/testset.txt", predictedLabels);
	std::cout << predictedLabels << std::endl;
	system("pause");
	return 0;
}