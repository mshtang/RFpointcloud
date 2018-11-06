#include "Sample.h"
#include <ctime>
#include <iostream>


Sample::Sample(Eigen::MatrixXf *dataset, Eigen::VectorXi *labels, 
			   Eigen::MatrixXi *indexMat, Eigen::MatrixXf *distMat, int numClass, int numFeature):
	_dataset(dataset),
	_labels(labels),
	_indexMat(indexMat),
	_distMat(distMat),
	_numClass(numClass),
	_numFeature(numFeature)
{ 
}


Sample::Sample(Sample* sample):
	_dataset(sample->_dataset),
	_labels(sample->_labels),
	_indexMat(sample->_indexMat),
	_distMat(sample->_distMat),
	_numClass(sample->_numClass),
	_numFeature(sample->_numFeature),
	_numSelectedSamples(sample->getNumSelectedSamples()),
	_selectedSamplesId(sample->getSelectedSamplesId())
{ 
}

Sample::Sample(const Sample* sample, Eigen::VectorXi &samplesId) :
	_dataset(sample->_dataset),
	_labels(sample->_labels),
	_indexMat(sample->_indexMat),
	_distMat(sample->_distMat),
	_numClass(sample->_numClass),
	_numFeature(sample->_numFeature),
	_selectedSamplesId(samplesId),
	_numSelectedSamples(samplesId.rows())
{
}

void Sample::randomSampleDataset(Eigen::VectorXi &selectedSamplesId, int numSelectedSamples)
{
	int numTotalSamples = _dataset->rows();
	_numSelectedSamples = numSelectedSamples;

	// to generate a uniform distribution and sample with replacement
	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::uniform_int_distribution<int> distribution(0, numTotalSamples - 1);

	for (int i = 0; i < _numSelectedSamples; ++i)
	{
		selectedSamplesId[i] = distribution(generator);
		//selectedSamplesId[i] = rand() % numTotalSamples;
	}

	_selectedSamplesId = selectedSamplesId;
}

void Sample::randomSampleFeatures()
{
	// reset member variable
	// _features.resize(_numFeature);
	Features feature;
	// total population
	int numPoints = _indexMat->cols();
	// for each feauture, it contains two random points from the neighborhood
	// and a node test function from the FeatureFactory
	for (int i = 0; i < _numFeature; ++i)
	{
		// randomly select two*_numFeature points from neighboorhood
		Random randomPoints(numPoints, 2);
		std::vector<int> selectedPointId = randomPoints.sampleWithoutReplacement();
	
		// randomly select one of the features from FeatureFactory
		//int selectedFeatType = rand() % 6; // this way of generating random number
		// is not reliable, ie. is not truely random
		Random randomFeatType(6, 1); // 6 features in the factory now
		int selectedFeatType = randomFeatType.sampleWithoutReplacement()[0];
		feature._point1 = selectedPointId[0];
		feature._point2 = selectedPointId[1];
		feature._featType = selectedFeatType;
		_features.push_back(feature);
	}
}

Eigen::MatrixXf Sample::buildNeighborhood(int pointId) const
{
	// number of points in the neighborhood
	int k = _indexMat->cols();
	// datapoint dimension
	int d = _dataset->cols();
	/*std::cout << "pointID\n";
	std::cout << pointId << std::endl;*/
	Eigen::MatrixXf neighborhood(k, d);
	Eigen::VectorXi candidatePointIndices;
	candidatePointIndices = _indexMat->row(pointId);
	// std::cout << "candidates\n";
	// std::cout << candidatePointIndices << std::endl;
	//neighborhood.row(0) = _dataset.row(pointId);
	for (int i = 0; i < k; ++i) {
		neighborhood.row(i) = _dataset->row(candidatePointIndices[i]);
	}
	return neighborhood;
}
