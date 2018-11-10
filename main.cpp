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
	if (1)
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
		const int numNeighbors = 20;
		trnObj.searchNN(trnData, numNeighbors, trnIndices, trnDists);

		Sample trnSamples(&trnData, &trnLabels, &trnIndices, &trnDists, 8, 1);
		Eigen::MatrixXf local_nei = trnSamples.buildNeighborhood(0);
		std::cout << "Neighborhood of Point 1 is:\n" << local_nei << std::endl;
		trnSamples.randomSampleFeatures();
		std::vector<Features> feats = trnSamples.getSelectedFeatures();
		std::cout << "No. of voxels are " << feats.at(0)._numVoxels << std::endl;
		FeatureFactory ff(local_nei, feats[0]);
		//ff.localNeighbors();
		//std::cout << "Local neighbors are:\n" << ff.getLocalNeighbors() << std::endl;
		//ff.buildVoxels();
		std::vector<Eigen::Matrix3f> res = ff.computeCovarianceMatrix();

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
	}
	else
	{
		Eigen::VectorXf point(7);
		point(0) = 1;
		point(1) = 2;
		point(2) = 2;
		point(3) = 2;
		point(4) = 112;
		point(5) = 178;
		point(6) = 29;
		std::cout << point << std::endl;
		Eigen::VectorXf pointprime;
		pointprime = toHSV(point);
		std::cout << point << std::endl;
		std::cout << pointprime << std::endl;
	}
	system("pause");
	return 0;
}