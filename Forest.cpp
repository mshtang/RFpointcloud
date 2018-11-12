#include "Forest.h"
#include "InOut.h"
#include "Sample.h"
#include "utils.h"

RandomForest::RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf, float giniThresh) :
	_numTrees(numTrees),
	_maxDepth(maxDepth),
	_minSamplesPerLeaf(minSamplesPerLeaf),
	_giniThresh(giniThresh),
	_trainSample(nullptr),
	_forest(_numTrees, nullptr)
{
	std::cout << "The number of trees in the forest is: " << _numTrees << std::endl;
	std::cout << "The max depth of a single tree is: " << _maxDepth << std::endl;
	std::cout << "The minimal number of samples at a leaf node is: " << _minSamplesPerLeaf << std::endl;
}

RandomForest::~RandomForest()
{
	if (_trainSample != nullptr)
	{
		delete _trainSample;
		_trainSample = nullptr;
	}
}

void RandomForest::train(Eigen::MatrixXf *trainset, Eigen::VectorXi *labels, Eigen::MatrixXi *indices,
						 Eigen::MatrixXf *dists, int numClasses, int numFeatsPerNode)
{
	if (_numTrees < 1)
	{
		std::cout << "Total number of trees must be bigger than 0." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}
	if (_maxDepth < 1)
	{
		std::cout << "The max depth must be bigger than 0." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}
	if (_minSamplesPerLeaf < 2)
	{
		std::cout << "The minimal number of samples at a leaf node must be greater than 1." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}

	int _numSamples = trainset->rows();
	_numClasses = numClasses;
	_numFeatsPerNode = numFeatsPerNode;

	// bagging: only about two thirds of the entire dataset is used
	// for each tree. This sampling process is with replacement.
	int _numSelectedSamples = _numSamples * 0.7;

	// initializing the trees
	for (int i = 0; i < _numTrees; ++i)
	{
		_forest[i] = new Tree(_maxDepth, _numFeatsPerNode, _minSamplesPerLeaf, _giniThresh);
	}

	// this object holds the whole training dataset
	_trainSample = new Sample(trainset, labels, indices, dists, _numClasses, _numFeatsPerNode);

	// selected samples
	Eigen::VectorXi selectedSamplesId(_numSelectedSamples);
	
	// tree training starts
	for (int i = 0; i < _numTrees; ++i)
	{
		std::cout << "Training tree No. " << i << std::endl;

		// randomly sample 2/3 of the points with replacement from training set
		Sample *sample = new Sample(_trainSample);
		sample->randomSampleDataset(selectedSamplesId, _numSelectedSamples);
		
		_forest[i]->train(sample);
		delete sample;
	}
}

void RandomForest::predict(const char* testDataPath, Eigen::VectorXi &predictedLabels)
{
	// looking for the nearest neighbors of each point in the test dataset
	Eigen::MatrixXf testset;
	Eigen::MatrixXi testIndices;
	Eigen::MatrixXf testDists;
	InOut testObj;
	testObj.readPoints(testDataPath, testset);
	testObj.searchNN(testset, testIndices, testDists);

	int numTests = testset.rows();
	predictedLabels.resize(numTests);
	Sample* testSamples = new Sample(&testset, &predictedLabels, &testIndices, &testDists, _numClasses, _numFeatsPerNode);
	
	for (int i = 0; i < numTests; ++i)
	{
		//Eigen::VectorXi datapoint = testSamples->_dataset->row(i);
		Eigen::MatrixXf testDataNeigh = testSamples->buildNeighborhood(i);
		predictedLabels[i] = predict(testDataNeigh);
	}
}

int RandomForest::predict(Eigen::MatrixXf &testNeigh)
{
	Eigen::VectorXf predictedProbs(_numClasses);
	for (int i = 0; i < _numClasses; ++i)
		predictedProbs[i] = 0;

	std::vector<float> predictedProbsVec;

	// accumulate the class distribution of every tree
	for (int i = 0; i < _numTrees; ++i)
	{
		predictedProbsVec = _forest[i]->predict(testNeigh);
		predictedProbs += toEigenVec(predictedProbsVec);
	}
	// average the class distribution
	predictedProbs /= _numTrees;
	// find the max value in the distribution and return its index
	// as the class label
	float prob = predictedProbs[0];
	int label = 0;
	for (int i = 1; i < _numClasses; ++i)
	{
		if (predictedProbs[i] > prob)
		{
			prob = predictedProbs[i];
			label = i;
		}
	}
	return label;
}