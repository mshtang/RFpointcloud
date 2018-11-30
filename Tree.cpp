#include "Tree.h"
#include "utils.h"

Tree::Tree(int maxDepth, int numFeatPerNode, int minNumSamplesPerLeaf):
	_maxDepth(maxDepth),
	_numFeatPerNode(numFeatPerNode),
	_minNumSamplesPerLeaf(minNumSamplesPerLeaf),
	_numNodes(static_cast<int>(std::pow(2, _maxDepth)-1)),
	_treeNodes(_numNodes, nullptr)
{
}

void Tree::train(Sample *sample)
{
	// for all the possible features, only numFeats features at each node
	// is calculated
	//int numFeats = sample->getNumFeatures();
	Eigen::VectorXi selectedSamplesId = sample->getSelectedSamplesId();
	Sample *nodeSample = new Sample(sample, selectedSamplesId);
	_treeNodes[0] = new Node();
	_treeNodes[0]->_samples = nodeSample;
	std::vector<float> priorDistr;
	// calculate the probability and gini
	_treeNodes[0]->computeNodeGini();
	priorDistr = _treeNodes[0]->_probs;
	for (int i = 0; i < _numNodes; ++i)
	{
		int parentId = (i - 1) / 2;
		// if current node's parent is null, continue
		if (_treeNodes[parentId] == nullptr)
			continue;
		// if current node's parent is a leaf, continue
		if (i > 0 and _treeNodes[parentId]->isLeaf())
			continue;
		// if maxDepth is reached, set current node as a leaf node
		if (i * 2 + 1 >= _numNodes)
		{
			_treeNodes[i]->createLeaf(priorDistr);
			continue;
		}
		// if current samples in this node is less than the threshold
		// set current node as a leaf node and continue
		if ((_treeNodes[i]->_samples->getNumSelectedSamples() <= _minNumSamplesPerLeaf) or
			_treeNodes[i]->isHomogenous())
		{
			_treeNodes[i]->createLeaf(priorDistr);
			continue;
		}
		//_treeNodes[i]->_samples->randomSampleFeatures();
		_treeNodes[i]->computeInfoGain(_treeNodes, i);
	}
}

void Tree::createLeaf(int nodeId, int classLabel, float prob, std::vector<float> probs)
{
	_treeNodes[nodeId] = new Node();
	_treeNodes[nodeId]->setLeaf(true);
	_treeNodes[nodeId]->setClass(classLabel);
	_treeNodes[nodeId]->setProb(prob);
	_treeNodes[nodeId]->_probs = probs;
}

std::vector<float> Tree::predict(Eigen::MatrixXf &testNeigh)
{
	// starting from the 0th node
	// do a recursive breadth first search in searchNode()
	int nodeId = 0;
	nodeId = searchNode(testNeigh, 0);
	return _treeNodes[nodeId]->_probs;
}

int Tree::searchNode(Eigen::MatrixXf &testNeigh, int nodeId)
{
	if (_treeNodes[nodeId]->isLeaf())
	{
		return nodeId;
	}
	else
	{
		Features testFeat = _treeNodes[nodeId]->getBestFeature();
		FeatureFactory testNodeFeat(testNeigh,testFeat);
		if (testNodeFeat.castProjection() < testFeat._thresh)
		{
			return searchNode(testNeigh, nodeId * 2 + 1);
		}
		else
		{
			return searchNode(testNeigh, nodeId * 2 + 2);
		}
	}
}

void Tree::createNode(int nodeId, Features bestFeat)
{
	_treeNodes[nodeId] = new Node();
	_treeNodes[nodeId]->setLeaf(false);
	_treeNodes[nodeId]->setBestFeature(bestFeat);
}

void Tree::computeStats(std::vector<Node*> nodes)
{
	int nonNull = 0;
	int largestLeaf = 0;
	int largestLeafId = 0;
	int nonLeaf = 0;
	_bestFeatTypeDistr.resize(FeatureFactory::numOfPossibleProjections, 0);
	for (int i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] != nullptr)
		{
			nonNull++;
			if (nodes[i]->isLeaf())
			{
				int samplesInNode = nodes[i]->_samples->getNumSelectedSamples();
				if (samplesInNode > largestLeaf)
				{
					largestLeaf = samplesInNode;
					largestLeafId = i;
				}
			}
			else
			{
				nonLeaf++;
				int featType = nodes[i]->getBestFeature()._featType;
				_bestFeatTypeDistr[featType]++;
			}
		}
	}
	_balance = nonNull / static_cast<float>(nodes.size());
	_totalNumSamples = nodes[0]->_samples->getNumSelectedSamples();
	_numSamplesInLargestLeaf = largestLeaf;
	_gradeSorting = 1 - largestLeaf / static_cast<float>(_totalNumSamples);
	Node* node = nodes[largestLeafId];
	Eigen::VectorXi ids = node->_samples->getSelectedSamplesId();
	std::vector<int> idsVec = toStdVec(ids);
	_largestLeafGini = node->computeGini(idsVec, _largestLeafDistr);
	for (int i = 0; i < _bestFeatTypeDistr.size(); ++i)
	{
		_bestFeatTypeDistr[i] /= nonLeaf;
	}
}
