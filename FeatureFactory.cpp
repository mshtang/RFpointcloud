#include "FeatureFactory.h"
#include <ctime>
#include <iostream>
#include "nanoflann.hpp"
#include "utils.h"

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
	int k1 = _neighborhood.rows();

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
		int rowsOfVoxel = _feat._voxelSize[i]+1;
		voxel.resize(rowsOfVoxel, 7);
		int pointId = _feat._pointId[i];
		Eigen::MatrixXi pointIdsForVoxel = _localIndices.row(pointId);
		for (int j = 0; j < rowsOfVoxel; ++j)
		{
			voxel.row(j) = _neighborhood.row(pointIdsForVoxel(j));
		}
		std::cout << "Voxel " << i << " is:\n" << voxel << std::endl;
		/*InOut inout;
		std::string filename = "voxel " + std::to_string(i) + ".txt";
		inout.writeToDisk(filename.c_str(), voxel);*/
		_voxels.push_back(voxel);
	}

}


std::vector<Eigen::MatrixXf> FeatureFactory::averageVoxels()
{
	std::vector<Eigen::MatrixXf> voxels_avg;
	for (int i = 0; i < _voxels.size(); ++i)
	{
		Eigen::MatrixXf res(1, 7);
		res = _voxels[i].colwise().mean();
		voxels_avg.push_back(res);
	}
	return voxels_avg;
}

bool FeatureFactory::computeFeature()
{
	bool testResult = false;
	if (_feat._featType <= 8) // radiometric features
	{
		localNeighbors();
		buildVoxels();
		std::vector<Eigen::MatrixXf> avg_voxels = averageVoxels();
		// red channel diff
		if (_feat._featType == 0)
			testResult = compareChannels(avg_voxels, 4);

		// green channel diff
		else if (_feat._featType == 1)
			testResult = compareChannels(avg_voxels, 5);

		// blue channel diff
		else if (_feat._featType == 2)
			testResult = compareChannels(avg_voxels, 6);

		// h value diff
		else if (_feat._featType == 3)
			testResult = compareChannels(avg_voxels, 4, false);

		// s value diff
		else if (_feat._featType == 4)
			testResult = compareChannels(avg_voxels, 5, false);

		// v value diff
		else if (_feat._featType == 5)
			testResult = compareChannels(avg_voxels, 6, false);

		// x diff
		else if (_feat._featType == 6)
			testResult = compareChannels(avg_voxels, 0);

		// y diff
		else if (_feat._featType == 7)
			testResult = compareChannels(avg_voxels, 1);

		// z diff
		else if (_feat._featType == 8)
			testResult = compareChannels(avg_voxels, 2);
	}
	else // features based on local voxel tensors (geometric features)
	{
		//Eigen::Matrix3f tensor = computeCovarianceMatrix();
		if (_feat._featType == 9)
		{

		}

	}
	return testResult;
}

bool FeatureFactory::compareChannels(std::vector<Eigen::MatrixXf> avg_voxels, int channelNo, bool isInRGBSpace)
{
	// if point cloud should be in HSV space
	if (!isInRGBSpace)
	{
		for (int i = 0; i < avg_voxels.size(); ++i)
			avg_voxels[i] = toHSV(avg_voxels[i]);
	}
	if (_feat._numVoxels == 1)
		return avg_voxels[0](channelNo) < 0 ? true : false;
	else if (_feat._numVoxels == 2)
		return avg_voxels[0](channelNo) < avg_voxels[1](channelNo) ? true : false;
	else
		return (avg_voxels[0](channelNo) - avg_voxels[2](channelNo)) < (avg_voxels[1](channelNo) - avg_voxels[3](channelNo)) ? true : false;

}

std::vector<Eigen::Matrix3f> FeatureFactory::computeCovarianceMatrix()
{
	localNeighbors();
	std::cout << _localIndices << std::endl;
	buildVoxels();
	std::vector<Eigen::MatrixXf> avg_voxels = averageVoxels();
	std::vector<Eigen::Matrix3f> res;
	for (int i = 0; i < avg_voxels.size(); ++i)
	{
		Eigen::Matrix3f tensor;
		tensor.setZero();
		for (int j = 0; j < _voxels[i].rows(); ++j) // for every point in each voxel
		{
			float px = _voxels[i].row(j).x() - avg_voxels[i].row(0).x();
			float py = _voxels[i].row(j).y() - avg_voxels[i].row(0).y();
			float pz = _voxels[i].row(j).z() - avg_voxels[i].row(0).z();
			tensor(1, 1) += py * py;
			tensor(1, 2) += py * pz;
			tensor(2, 2) += pz * pz;
			float pxx = px * px;
			float pxy = px * py;
			float pxz = px * pz;
			tensor(0, 0) += pxx;
			tensor(0, 1) += pxy;
			tensor(0, 2) += pxz;
		}
		tensor(1, 0) = tensor(0, 1);
		tensor(2, 0) = tensor(0, 2);
		tensor(2, 1) = tensor(1, 2);
		tensor /= _voxels[i].rows();
		res.push_back(tensor);
	}
	return res;
}