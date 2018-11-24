#pragma once
#include "C:/Eigen/Eigen/Dense"
#include <algorithm>
#include <vector>
#include <random>
#include "InOut.h"
#include <iostream>

/******************************************************************************************
 * this class will contain the features at each node, a "feature" consists of one or two or 
 * four randomly selected points (ID) from the neighborhood with each has its own voxel size
 * (number of points around each selected point), thus one/two/four voxels are built and they
 * will be further projected to a real value by one of the feature type functions.
 *****************************************************************************************/
struct Features{
	Features()
	{
	}
	Features(const Features& rhs):
		_numVoxels(rhs._numVoxels),
		_pointId(rhs._pointId),
		_voxelSize(rhs._voxelSize),
		_featType(rhs._featType)		
	{
	}
	Features& operator=(const Features& rhs)
	{
		_numVoxels = rhs._numVoxels;
		_pointId = rhs._pointId;
		_voxelSize = rhs._voxelSize;
		_featType = rhs._featType;
		return *this;
	}
	int _numVoxels;
	std::vector<int> _pointId;
	std::vector<int> _voxelSize;
	int _featType;
	// to ensure there are at least 10 points in each voxel, so that 3d features can be calculated
	int minSamples = static_cast<int>(0.1*InOut::numOfNN);
	const int minSamplesPerVoxel = minSamples > 10 ? minSamples : 10;
	int maxSamples = static_cast<int>(0.5*InOut::numOfNN);
	int tmpMaxSamples = (maxSamples > minSamplesPerVoxel + 10) ? maxSamples : (minSamplesPerVoxel + 10);
	const int maxSamplesPerVoxel = tmpMaxSamples > InOut::numOfNN ? InOut::numOfNN : tmpMaxSamples;
};


/**************************************************
This class is used for drawing samples and features
(test node functions) from the dataset. 
***************************************************/
class Sample {
public:
	/***************************************************************************************** 
	 * construct a new Sample object using given dataset, labels, the index matrix (representing
	 * the points ID in neighborhoods), the distance matrix of corresponding dists, together  
	 * with number of classes, number of features means how many features are considered at each
	 * node.
	 *****************************************************************************************/
	Sample(Eigen::MatrixXf *dataset, Eigen::VectorXi *labels, 
		   Eigen::MatrixXi *indexMat, Eigen::MatrixXf *distMat, int numClass, int numFeature);
	
	/* to new a Sample object using a pointer to Sample, so that the dataset/labels etc. will 
	 * not be copied but referenced*/
	Sample(Sample* samples);

	/* to reference only a part of the dataset with the point ID*/
	Sample(const Sample* sample, Eigen::VectorXi &samplesId);


	// randomly select numSelectedSamples samples from dataset with replacement (bagging)
	void randomSampleDataset(Eigen::VectorXi &selectedSamplesId, int numSelectedSamples);

	/***************************************************************************************
	 * randomly sample features from each neighborhoood
	 * given a neighborhood consisting of k points, the number of possible features
	 * are k*(k-1)*n, where n is the projection operations, but only numSelectedFeatures
	 * are randomly chosen from all these features
	 ***************************************************************************************/
	void randomSampleFeatures();


	/***************************************************************************************
	 * return a matrix representing the neighborhood of the pointId-th point
	 * whose shape is (k, d), where k is the number of nearest neighbors
	 * and d is the dimention of each datapoint*
	 **************************************************************************************/
	Eigen::MatrixXf buildNeighborhood(int pointId);

	// keep track of the number of different classes in a Sample obejct
	inline int getNumClasses() { return _numClass; }

	// get the selected sample indices
	inline Eigen::VectorXi getSelectedSamplesId() { return _selectedSamplesId; }
	inline Eigen::VectorXi getSelectedSamplesId() const { return _selectedSamplesId; }

	// get the number of selected samples
	inline int getNumSelectedSamples() { return _numSelectedSamples; }

	// get the selected features
	inline std::vector<Features> getSelectedFeatures() { return _features; }

	// get the number of features sampled at each node
	inline int getNumFeatures() { return _numFeature; }
	inline int getNeighborhoodSize() { return _indexMat->cols(); }

	Eigen::VectorXi *_labels;
	Eigen::MatrixXf *_dataset;

private:

	// _indexMat stores the indices of nearest neighbors for each datapoint
	Eigen::MatrixXi *_indexMat;
	// _distMat stores the dists of nearest neighbors to each datapoint
	Eigen::MatrixXf *_distMat;

	// _selectedSamplesId stores the indices of selected datapoints
	Eigen::VectorXi _selectedSamplesId;
	std::vector<Features> _features;
	int _numClass;
	int _numSelectedSamples;
	int _numFeature;

	// randomly sample from {1, 2, 4} because there are 1/2/4 possible
	// voxels in a neighborhood
	int randomFrom124();
};


/*****************************************************************
This class is used for aquiring samples without replacement.
 * It makes use of the shuffle method from the STL, which randomly
 * shuffles a vector using the given seed. If sampleSize samples 
 * are needed, then the first sampleSize elements of the vector
 * after being shuffled are kept. 
 *****************************************************************/
class Random
{
public:
	Random(int popSize, int sampleSize):
		_popSize(popSize),
		_sampleSize(sampleSize)
	{}

	std::vector<int> sampleWithoutReplacement()
	{
		std::vector<int> population;
		candidates(population);

		std::random_device rd;
		std::mt19937 gen(rd());
		// DEBUG to uncomment
		// for debugging purposes, to generate deterministic numbers
		// std::mt19937 gen(123);
		std::shuffle(population.begin(), population.end(), gen);
		std::vector<int> samples(population.begin(), population.begin() + _sampleSize);
		return samples;
	}

private:
	void candidates(std::vector<int> &nums)
	{
		for (int i = 0; i < _popSize; ++i)
			nums.push_back(i);
	}
	int _popSize;
	int _sampleSize;
};
