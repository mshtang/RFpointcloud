#include "Forest.h"
#include "InOut.h"
#include "Sample.h"
#include "utils.h"

RandomForest::RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf) :
	_numTrees(numTrees),
	_maxDepth(maxDepth),
	_minSamplesPerLeaf(minSamplesPerLeaf),
	_trainSample(nullptr),
	_forest(_numTrees, nullptr)
{
	std::cout << "The number of trees in the forest is: " << _numTrees << std::endl;
	std::cout << "The max depth of a single tree is: " << _maxDepth << std::endl;
	std::cout << "The minimal number of samples at a leaf node is: " << _minSamplesPerLeaf << std::endl;
}

RandomForest::RandomForest(const char* modelPath)
{
	readModel(modelPath);
}

RandomForest::~RandomForest()
{
	if (_trainSample != nullptr)
	{
		delete _trainSample;
		_trainSample = nullptr;
	}
}

void RandomForest::train(Eigen::MatrixXf *trainset, Eigen::VectorXi *labels, 
						 Eigen::MatrixXi *indices, Eigen::MatrixXf *dists, 
						 int numClasses, int numFeatsPerNode,
						 Eigen::MatrixXf *cloud)
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
		_forest[i] = new Tree(_maxDepth, _numFeatsPerNode, _minSamplesPerLeaf);
	}

	// this object holds the whole training dataset
	_trainSample = new Sample(trainset, labels, indices, dists, _numClasses, _numFeatsPerNode, cloud);

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

		std::vector<Node*> nodes = _forest[i]->getTreeNodes();
		_forest[i]->computeStats(nodes);

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
	Sample* testSamples = new Sample(&testset, &predictedLabels, &testIndices, &testDists, _numClasses, _numFeatsPerNode, &testset);
	
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

void RandomForest::saveModel(const char* path, const char* statFilePath)
{
	printf("Saving model ... ");
	FILE *saveFile = fopen(path, "wb");
	FILE *statFile = nullptr;
	if (statFilePath != nullptr)
		statFile = fopen(statFilePath, "w");
	fwrite(&_numTrees, sizeof(int), 1, saveFile);
	fwrite(&_maxDepth, sizeof(int), 1, saveFile);
	fwrite(&_numClasses, sizeof(int), 1, saveFile);
	fprintf(statFile, "There are %d trees in the forest.\n", _numTrees);
	fprintf(statFile, "Each tree has a max height of %d.\n", _maxDepth);
	int numNodes = static_cast<int>(pow(2.0, _maxDepth) - 1);
	int isLeaf = 0;
	for (int i = 0; i < _numTrees; ++i)
	{
		// write some statistics to file
		if (statFilePath != nullptr)
		{
			fprintf(statFile, "----------------------------------Stats for Tree %d-----------------------------:\n", i);
			int totalNum = _forest[i]->getTotalNumSamples();
			fprintf(statFile, "%d samples are trained in this tree.\n", totalNum);
			float balance = _forest[i]->getBalance();
			fprintf(statFile, "Tree balance is: %.3f\n", balance);
			int numSamplesInLargestLeaf = _forest[i]->getNumSamplesInLargestLeaf();
			fprintf(statFile, "The number of samples in the largest leaf: %d\n", numSamplesInLargestLeaf);
			float gradeSorting = _forest[i]->getSortingGrade();
			fprintf(statFile, "Grade of sorting is: %.3f\n", gradeSorting);
			float largestLeafGini = _forest[i]->getLargestLeafGini();
			fprintf(statFile, "Gini of the largest leaf node: %.3f\n", largestLeafGini);
			std::vector<float> largestLeafDistr = _forest[i]->getLargestLeafDistr();
			fprintf(statFile, "The posterior of the largest leaf node: \n");
			for (int k = 0; k < largestLeafDistr.size(); ++k)
			{
				fprintf(statFile, "%.3f   ", largestLeafDistr[k]);
			}
			std::vector<float> bestFeatTypeDistr = _forest[i]->getBestFeatTypeDistr();
			fprintf(statFile, "\nThe posterior of the projection type of the selected best feature at each node:\n");
			for (int k = 0; k < bestFeatTypeDistr.size(); ++k)
			{
				fprintf(statFile, "%.3f   ", bestFeatTypeDistr[k]);
			}
			fprintf(statFile, "\n");
		}

		std::vector<Node*> arr = _forest[i]->getTreeNodes();
		isLeaf = 0;
		for (int j = 0; j < numNodes; ++j)
		{
			if (arr[j] != nullptr)
			{
				if (arr[j]->isLeaf())
				{
					isLeaf = 1;
					fwrite(&isLeaf, sizeof(int), 1, saveFile);
					int clas = arr[j]->getClass();
					float prob = arr[j]->getProb();
					fwrite(&clas, sizeof(int), 1, saveFile);
					fwrite(&prob, sizeof(float), 1, saveFile);
					// save the posterior distr of each leaf node
					// it is again a vector
					for (int k = 0; k < _numClasses; ++k)
					{
						float tmpProb = arr[j]->_probs[k];
						fwrite(&tmpProb, sizeof(float), 1, saveFile);
					}
				}
				else
				{
					isLeaf = 0;
					fwrite(&isLeaf, sizeof(int), 1, saveFile);

					// Features is a user defined datatype
					// saving it is a little complicated
					Features bestFeat = arr[j]->getBestFeature();
					int numVoxel = bestFeat._numVoxels;
					fwrite(&numVoxel, sizeof(int), 1, saveFile);
					int featType = bestFeat._featType;
					fwrite(&featType, sizeof(int), 1, saveFile);
					float thresh = bestFeat._thresh;
					fwrite(&thresh, sizeof(float), 1, saveFile);
					// save the vector size -> then save its content
					int pointIdSize = bestFeat._pointId.size();
					fwrite(&pointIdSize, sizeof(int), 1, saveFile);
					for (int k = 0; k < pointIdSize; ++k)
					{
						int pointId = bestFeat._pointId[k];
						fwrite(&pointId, sizeof(int), 1, saveFile);
					}
					int voxelSizeSize = bestFeat._voxelSize.size();
					fwrite(&voxelSizeSize, sizeof(int), 1, saveFile);
					for (int k = 0; k < voxelSizeSize; ++k)
					{
						int voxelSize = bestFeat._voxelSize[k];
						fwrite(&voxelSize, sizeof(int), 1, saveFile);
					}
				}
			}
		}
	}
	fclose(statFile);
	fclose(saveFile);
	printf("Model saved!\n");
}

void RandomForest::readModel(const char* path)
{
	printf("Reading model ... \n");
	_minSamplesPerLeaf = 0;
	_numFeatsPerNode = 0;
	FILE* modelFile = fopen(path, "rb");
	fread(&_numTrees, sizeof(int), 1, modelFile);
	fread(&_maxDepth, sizeof(int), 1, modelFile);
	fread(&_numClasses, sizeof(int), 1, modelFile);
	int numNodes = static_cast<int>(pow(2.0, _maxDepth) - 1);
	_trainSample = nullptr;
	printf("The number of trees in the forest is: %d\n", _numTrees);
	printf("The max depth of a single tree is: %d\n", _maxDepth);
	printf("The number of classes is: %d\n", _numClasses);
	_forest.resize(_numTrees);
	for (int i = 0; i < _numTrees; ++i)
	{
		_forest[i] = new Tree(_maxDepth, _numFeatsPerNode, _minSamplesPerLeaf);
	}
	int* nodeTable = new int[numNodes];
	int isLeaf = -1;
	Features bestFeat;
	int clas = 0;
	float prob = 0;
	std::vector<float> probs(_numClasses, 0);
	for (int i = 0; i < _numTrees; ++i)
	{
		//std::vector<Node*> treeNodes = _forest[i]->getTreeNodes();
		memset(nodeTable, 0, sizeof(int)*numNodes);
		nodeTable[0] = 1;
		for (int j = 0; j < numNodes; ++j)
		{
			if (nodeTable[j] == 0)
				continue;
			fread(&isLeaf, sizeof(int), 1, modelFile);
			if(isLeaf == 0)
			{
				nodeTable[j * 2 + 1] = 1;
				nodeTable[j * 2 + 2] = 1;
				// read Features members
				// first is the numVoxels
				int numVoxels = 0;
				fread(&numVoxels, sizeof(int), 1, modelFile);
				bestFeat._numVoxels = numVoxels;
				// then the featType
				int featType = 0;
				fread(&featType, sizeof(int), 1, modelFile);
				bestFeat._featType = featType;
				// after that is the thresh
				float thresh = 0;
				fread(&thresh, sizeof(float), 1, modelFile);
				bestFeat._thresh = thresh;
				// read the two vectors, a little complicated
				// vector I:
				int pointIdSize = 0;
				fread(&pointIdSize, sizeof(int), 1, modelFile);
				// initialize a vector
				std::vector<int> pointId(pointIdSize, 0);
				for (int k = 0; k < pointIdSize; ++k)
				{
					int tmpPointId = 0;
					fread(&tmpPointId, sizeof(int), 1, modelFile);
					pointId[k] = tmpPointId;
				}
				bestFeat._pointId = pointId;
				// vector II:
				int voxelSize = 0;
				fread(&voxelSize, sizeof(int), 1, modelFile);
				// initialize a vector
				std::vector<int> voxels(voxelSize, 0);
				for (int k = 0; k < voxelSize; ++k)
				{
					int tmpVoxel = 0;
					fread(&tmpVoxel, sizeof(int), 1, modelFile);
					voxels[k] = tmpVoxel;
				}
				bestFeat._voxelSize = voxels;
				_forest[i]->createNode(j, bestFeat);
			}
			else
			{
				fread(&clas, sizeof(int), 1, modelFile);
				fread(&prob, sizeof(float), 1, modelFile);
				//std::vector<float> probs(_numClasses, 0);
				for (int k = 0; k < _numClasses; ++k)
				{
					float tmpProb = 0;
					fread(&tmpProb, sizeof(float), 1, modelFile);
					probs[k] = tmpProb;
				}
				_forest[i]->createLeaf(j, clas, prob, probs);
			}
		}
	}
	fclose(modelFile);
	delete[] nodeTable;
	printf("Model read!\n");
}