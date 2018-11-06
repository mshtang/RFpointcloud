
#pragma once

#include <vector>
#include "Sample.h"


/*********************************************************************************
 * This class contains the basic element (node) of a tree. Samples at each node
 * are split into two parts based on the result of test function (from FeatureFactory).
**********************************************************************************/
class Node
{
public:
	Node();


	// split the node into two child nodes or set it as a leaf node
	// based on the info gain of this node
	// info gain is defined as:
	// gini_of_node - left_child_proportion*left_child_gini - right_child_proportion*
	// right_child_gini
	void computeInfoGain(std::vector<Node*>& nodes, int nodeId, float threshGini);
	// compute the Gini Index at this node
	void computeNodeGini();
	// compute the Gini Index given the class label vector and returns a class 
	// distribution
	float computeGini(std::vector<int> & labelVec, std::vector<float> &probs);

	// create a leaf node
	void createLeaf();

	// to tell if the samples at this node belong to the same class in which 
	// case the splitting will terminate and a node is set as a leaf node
	bool isHomogenous();

	// the samples stored in this node
	Sample *_samples;

	// some setters/getters
	inline bool isLeaf() { return _isLeaf; }
	inline void setLeaf(bool flag) { _isLeaf = flag; }
	inline void setClass(int classLabel) { _class = classLabel; }
	inline void setProb(float prob) { _prob = prob; }
	inline Features getBestFeature() { return _bestFeat; }
	
	float _gini;
	std::vector<float> _probs;

private:
	bool _isLeaf;
	Features _bestFeat;
	int _class;
	float _prob;
};