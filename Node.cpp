#include "Node.h"
#include <vector>
#include "FeatureFactory.h"
#include <iostream>
#include "utils.h"

Node::Node():
	_isLeaf(false),
	_samples(nullptr),
	_bestFeat(Features()),
	_class(-1),
	_prob(0.0)
{}


float Node::computeGini(std::vector<int> & samplesId, std::vector<float> &probs)
{
	int numSamples = samplesId.size();
	int numClasses = _samples->getNumClasses();
	probs.resize(numClasses, 0);
	// reset probs;
	for (int i = 0; i < numClasses; ++i)
		probs[i] = 0;
	
	for (int i = 0; i < numSamples; ++i)
	{
		// count the number of instances of each class
		probs[(*(_samples->_labels))[samplesId[i]]]++;
	}
	
	float gini = 0;
	for (int i = 0; i < numClasses; ++i)
	{
		probs[i] = probs[i] / (float)numSamples;
		gini += (probs[i] * probs[i]);
	}
	
	return 1 - gini;
}
// calculate the gini of the samples in this node
void Node::computeNodeGini()
{
	Eigen::VectorXi samplesId = _samples->getSelectedSamplesId();
	// convert Eigen::Vector to std::vector
	std::vector<int> samplesIdVec = toStdVec(samplesId);
	_gini = computeGini(samplesIdVec, _probs);
}

// split the node into two children and compute the gini in each child node
void Node::computeInfoGain(std::vector<Node*> &nodes, int nodeId, float thresh)
{
	Eigen::MatrixXf data = *(_samples->_dataset);
	Eigen::VectorXi labels =*( _samples->_labels);
	// randomly samples some points from the cloud
	// and store the selected points id in sampleId
	Eigen::VectorXi sampleId = _samples->getSelectedSamplesId();
	int numSamples = sampleId.size();
	// for this work, numClasses is 8
	int numClasses = _samples->getNumClasses();
	_samples->randomSampleFeatures();
	std::vector<Features> featCandidates = _samples->getSelectedFeatures();
	int numFeats = _samples->getNumFeatures();

	// variables for storing the final parameters
	float bestInfoGain = 0;
	float bestLeftGini = 0;
	float bestRightGini = 0;
	Features bestFeat = featCandidates[0];
	std::vector<int> bestLeftChild;
	std::vector<int> bestRightChild;
	std::vector<float> bestLeftProbs;
	std::vector<float> bestRightProbs;

	// temporary variables for storing intermediate parameters
	float tBestInfoGain = 0;
	float tBestLeftGini = 0;
	float tBestRightGini = 0;
	Features tBestFeat = featCandidates[0];
	std::vector<int> tBestLeftChild;
	std::vector<int> tBestRightchild;
	std::vector<float> tBestLeftProbs;
	std::vector<float> tBestRightProbs;

	for (int i = 0; i < numFeats; ++i)
	{
		// apply each candidate feature to the samples at this node
		float infoGain = 0;
		float leftGini = 0;
		float rightGini = 0;
		Features bestFeat = featCandidates[0];
		std::vector<int> leftChildSamples;
		std::vector<int> rightChildSamples;
		std::vector<float> leftProbs;
		std::vector<float> rightProbs;
		
	
		for (int j = 0; j < numSamples; ++j)
		{
			Eigen::MatrixXf neigh = _samples->buildNeighborhood(sampleId[j]);
			FeatureFactory nodeFeat(neigh, featCandidates[i]);
			if (nodeFeat.computeFeature() == false)
				leftChildSamples.push_back(sampleId[j]);
			else
				rightChildSamples.push_back(sampleId[j]);
		}
		
		leftGini = computeGini(leftChildSamples, leftProbs);
		rightGini = computeGini(rightChildSamples, rightProbs);
		float leftRatio = leftChildSamples.size() / (float)numSamples;
		float rightRatio = rightChildSamples.size() / (float)numSamples;
		infoGain = _gini - leftGini * leftRatio - rightGini * rightRatio;
		if (infoGain > tBestInfoGain)
		{
			tBestInfoGain = infoGain;
			tBestLeftGini = leftGini;
			tBestRightGini = rightGini;
			tBestFeat = bestFeat;
			tBestLeftChild = leftChildSamples;
			tBestRightchild = rightChildSamples;
			tBestLeftProbs = leftProbs;
			tBestRightProbs = rightProbs;
		}
		
	}
	if (tBestInfoGain > bestInfoGain)
	{
		bestInfoGain = tBestInfoGain;
		bestLeftGini = tBestLeftGini;
		bestRightGini = tBestRightGini;
		bestLeftChild = tBestLeftChild;
		bestRightChild = tBestRightchild;
		bestFeat = tBestFeat;
		bestLeftProbs = tBestLeftProbs;
		bestRightProbs = tBestRightProbs;
	}

	if (bestInfoGain < thresh)
	{
		createLeaf();
	}
	else
	{
		_bestFeat = bestFeat;
		nodes[nodeId * 2 + 1] = new Node();
		nodes[nodeId * 2 + 2] = new Node();
		(nodes[nodeId * 2 + 1])->_gini = bestLeftGini;
		(nodes[nodeId * 2 + 2])->_gini = bestRightGini;
		(nodes[nodeId * 2 + 1])->_probs = bestLeftProbs;
		(nodes[nodeId * 2 + 2])->_probs = bestRightProbs;

		Eigen::VectorXi bestLeftChildVec = toEigenVec(bestLeftChild);
		Eigen::VectorXi bestRightChildVec = toEigenVec(bestRightChild);

		Sample *leftSamples = new Sample(_samples, bestLeftChildVec);
		Sample *rightSamples = new Sample(_samples, bestRightChildVec);
		(nodes[nodeId * 2 + 1])->_samples = leftSamples;
		(nodes[nodeId * 2 + 2])->_samples = rightSamples;
	}
}

// make the most frequent class as the class of this leaf node
void Node::createLeaf()
{
	_class = 0;
	_prob = _probs[0];
	for (int i = 1; i < _samples->getNumClasses(); ++i)
	{
		if (_probs[i] > _prob)
		{
			_class = i;
			_prob = _probs[i];
		}
	}
	_isLeaf = true;
}


bool Node::isHomogenous(){
	int numSamples = _samples->getSelectedSamplesId().size();
	for (int i = 0; i < numSamples-1; ++i)
	{
		int pointId1 = _samples->getSelectedSamplesId()[i];
		int pointId2 = _samples->getSelectedSamplesId()[i+1];

		if ((*(_samples->_labels))[pointId1] !=
			(*(_samples->_labels))[pointId2])
		{
			return false;
		}
	}
	return true;
}