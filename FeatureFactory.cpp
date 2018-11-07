#include "FeatureFactory.h"
#include <ctime>
#include <iostream>
#include "nanoflann.hpp"

FeatureFactory::FeatureFactory(Eigen::MatrixXf& neighborhood, Features feat) :
	_neighborhood(neighborhood),
	_feat(feat)
{}

void FeatureFactory::localNeighbors()
{
	typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> Matrix3fRow;
	Matrix3fRow coords = _neighborhood.leftCols(3);

	// different max_leaf values only affect the search speed 
	// and any value between 10 - 50 is reasonable
	const int max_leaf = 10;
	nanoflann::KDTreeEigenMatrixAdaptor<Matrix3fRow> mat_index(coords, max_leaf);
	mat_index.index->buildIndex();

	// maximal number of nn is half size the neighborhood
	int k1 = _neighborhood.rows() / 2 + 1;

	_localIndices.resize(_neighborhood.rows(), k1);
	_localDists.resize(_neighborhood.rows(), k1);

	// do a knn search
	for (int i = 0; i < coords.rows(); ++i)
	{
		// coords is RowMajor so coords.data()[i*3+0 / +1  / +2] represents the ith row of coords
		std::vector<float> query_pt{ coords.data()[i * 3 + 0], coords.data()[i * 3 + 1], coords.data()[i * 3 + 2] };

		std::vector<size_t> ret_indices(k1);
		std::vector<float> out_dists_sqr(k1);
		nanoflann::KNNResultSet<float> resultSet(k1);
		resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		for (size_t j = 0; j < k1; ++j)
		{
			_localIndices(i, j) = ret_indices[j];
			_localDists(i, j) = std::sqrt(out_dists_sqr[j]);
		}
	}
	//std::cout << _localIndices << std::endl;
}

void FeatureFactory::buildVoxels()
{
	for (int i = 0; i < _feat._numVoxels; ++i)
	{
		Eigen::MatrixXf voxel;
		int rowsOfVoxel = _feat._voxelSize[i];
		voxel.resize(rowsOfVoxel, 7);
		int pointId = _feat._pointId[i];
		Eigen::MatrixXi pointIdsForVoxel = _localIndices.row(pointId);
		for (int j = 0; j < rowsOfVoxel; ++j)
		{
			
			voxel.row(j) = _neighborhood.row(pointIdsForVoxel(j));
		}
		std::cout << "Voxel " << i << " is:\n" << voxel << std::endl;
		_voxels.push_back(voxel);
	}

}


bool FeatureFactory::computeFeature()
{	
	bool result = false;
	switch (_feat._featType)
	{
		case 0: 
			result = FeatureFactory::redColorDiff(); 
			break;
		case 1: 
			result = FeatureFactory::greenColorDiff(); 
			break;
		case 2: 
			result = FeatureFactory::blueColorDiff(); 
			break;
		case 3: 
			result = FeatureFactory::xDiff(); 
			break;
		case 4: 
			result = FeatureFactory::yDiff(); 
			break;
		case 5: 
			result = FeatureFactory::zDiff(); 
			break;
		/*case 7: absRedColorDiff(); break;
		case 8: absGreenColorDiff(); break;
		case 9: absBlueColorDiff(); break;
		case 10: localDensity(); breakneighborhood
		case 11: localPlanarity(); break;
		case 12: localSufaceDev(); break;*/
	}
	return result;
}

inline bool FeatureFactory::redColorDiff()
{
	//return _neighborhood(_feat._point1, 4) < _neighborhood(_feat._point2, 4) ? true : false;
	return 1;
}

inline bool FeatureFactory::greenColorDiff()
{
	//return _neighborhood(_feat._point1, 5) < _neighborhood(_feat._point2, 5) ? true : false;
	return 1;
}

inline bool FeatureFactory::blueColorDiff()
{
	//return _neighborhood(_feat._point1, 6) < _neighborhood(_feat._point2, 6) ? true : false;
	return 1;
}

inline bool FeatureFactory::xDiff()
{
	//return _neighborhood(_feat._point1, 0) < _neighborhood(_feat._point2, 0) ? true : false;
	return 1;
}

inline bool FeatureFactory::yDiff()
{
	//return _neighborhood(_feat._point1, 1) < _neighborhood(_feat._point2, 1) ? true : false;
	return 1;
}

inline bool FeatureFactory::zDiff()
{
	//return _neighborhood(_feat._point1, 2) < _neighborhood(_feat._point2, 2) ? true : false;
	return 1;
}
