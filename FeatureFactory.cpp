#include "FeatureFactory.h"
#include <ctime>
#include <iostream>
#include "nanoflann.hpp"
#include "utils.h"
#include "c:/Eigen/Eigen/src/Eigenvalues/EigenSolver.h"

FeatureFactory::FeatureFactory(std::vector<Eigen::MatrixXf> &voxels, Features feat,
							   Eigen::VectorXf &ptEValues, Eigen::VectorXf &ptEVectors,
							   Eigen::MatrixXf &voxelEValues, Eigen::MatrixXf &voxelEVectors) :
	_voxels(voxels),
	_feat(feat),
	_ptEValues(ptEValues),
	_ptEVectors(ptEVectors),
	_voxelEValues(voxelEValues),
	_voxelEVectors(voxelEVectors)

{}

//std::vector<Eigen::VectorXf> FeatureFactory::averageVoxels()
//{
//	std::vector<Eigen::VectorXf> voxels_avg;
//	for (int i = 0; i < _voxels.size(); ++i)
//	{
//		Eigen::VectorXf res;
//		res = _voxels[i].colwise().mean();
//		voxels_avg.push_back(res);
//	}
//	return voxels_avg;
//}

//bool FeatureFactory::project(float &res)
//{
//	std::vector<std::vector<Eigen::VectorXf>> partitions;
//	partitions = partitionSpace(_neighborhood);
//	buildVoxels(partitions);
//	res = 0.0f;
//	// one-voxel comparison
//	if (_voxels.size() == 1)
//	{
//		// if the voxel has no points
//		if (_voxels[0].size() == 1)
//		{
//			res = -1000.0f;
//			return false;
//		}
//		else // compare the voxel to the neighborhood center point
//		{
//			float voxelValue = 0;
//			float centerPointValue = 0;
//			bool success1 = true;
//			bool success2 = true;
//			// sometimes projection cannot be performed (e.g. some _featTypes require
//			// at least three points)
//			success1 = castProjection(_voxels[0], _feat._featType, voxelValue);
//			if (!success1)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			success2 = castProjection(_neighborhood.row(0), _feat._featType, centerPointValue);
//			if (!success2)
//			{ 
//				res = -1000.0f;
//				return false; 
//			}
//			res = centerPointValue - voxelValue;
//			return true;
//		}
//	}
//	// two-voxel comparison
//	else if (_voxels.size() == 2)
//	{
//		// if both voxels are not empty
//		if (_voxels[0].size() != 1 and _voxels[1].size() != 1)
//		{
//			float voxelValue1 = 0.0f, voxelValue2 = 0.0f;
//			bool success1 = castProjection(_voxels[0], _feat._featType, voxelValue1);
//			if (!success1)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success2 = castProjection(_voxels[1], _feat._featType, voxelValue2);
//			if (!success2)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			res =  voxelValue1 - voxelValue2;
//			return true;
//		}
//		// downgrade to 1 voxel case
//		// first voxel is not empty but second is
//		else if (_voxels[0].size() != 1 and _voxels[1].size() == 1)
//		{
//			float voxelValue = 0.0f;
//			float centerPointValue = 0.0f;
//			bool success1 = castProjection(_voxels[0], _feat._featType, voxelValue);
//			if (!success1)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success2 = castProjection(_neighborhood.row(0), _feat._featType, centerPointValue);
//			if (!success2)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			res = centerPointValue - voxelValue;
//			return true;
//		}
//		// second voxel is not empty but first is
//		else if (_voxels[1].size() != 1 and _voxels[0].size()==1)
//		{
//			float voxelValue = 0.0f;
//			float centerPointValue = 0.0f;
//			bool success1 = castProjection(_voxels[1], _feat._featType, voxelValue);
//			if (!success1)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success2 = castProjection(_neighborhood.row(0), _feat._featType, centerPointValue);
//			if (!success2)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			res = centerPointValue - voxelValue;
//			return true;
//		}
//		// both voxels are empty
//		else
//		{
//			res = -1000.0f;
//			return false;
//		}
//	}
//	// four voxel comparison
//	else
//	{
//		// if all voxels are not empty
//		if (_voxels[0].size() != 1 and _voxels[1].size() != 1
//			and _voxels[2].size() != 1 and _voxels[3].size() != 1)
//		{
//			float voxelValue1 = 0.0f, voxelValue2 = 0.0f, voxelValue3 = 0.0f, voxelValue4 = 0.0f;
//			bool success1 = castProjection(_voxels[0], _feat._featType, voxelValue1);
//			if (!success1)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success2 = castProjection(_voxels[1], _feat._featType, voxelValue2);
//			if (!success2)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success3 = castProjection(_voxels[2], _feat._featType, voxelValue3);
//			if (!success3)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			bool success4 = castProjection(_voxels[3], _feat._featType, voxelValue4);
//			if (!success4)
//			{
//				res = -1000.0f;
//				return false;
//			}
//			res =  (voxelValue1 - voxelValue2) - (voxelValue3 - voxelValue4);
//			return true;
//		}
//		// special cases
//		else
//		{
//			res = -1000.0f;
//			return false;
//		}
//	}
//}

bool FeatureFactory::project(float &res)
{
	res = 0.0f;
	// select voxels
	std::vector<Eigen::MatrixXf> selectedVoxels;
	std::vector<Eigen::VectorXf> selectedEigenValues;
	std::vector<Eigen::VectorXf> selectedEigenVectors;
	for (int i = 0; i < _feat._numVoxels; ++i)
	{
		int selectedId = _feat._pointId[i];
		selectedVoxels.push_back(_voxels[selectedId]);
		selectedEigenValues.push_back(_voxelEValues.row(selectedId));
		selectedEigenVectors.push_back(_voxelEVectors.row(selectedId));
	}
	// one-voxel comparison
	if (selectedVoxels.size()==1)
	{
		if (selectedVoxels[0].size()==1) // empty voxel
		{
			res = -1000.0;
			return false;
		}
		// voxel not empty but no enough points for eigen based features
		else if (selectedVoxels[0].rows() < 3 and _feat._featType >= 9)
		{
			res = -1000.0;
			return false;
		}
		else
		{
			float res1 = 0.0f, res2 = 0.0f;
			castProjection(selectedVoxels[0], _feat._featType, selectedEigenValues[0], selectedEigenVectors[0], res1);
			Eigen::MatrixXf neighborhood = recoverNeighborhood(_voxels);
			castProjection(neighborhood, _feat._featType, _ptEValues, _ptEVectors, res2);
			res = res1 - res2;

		}
	}
	// two-voxel comparion
	else if (selectedVoxels.size() == 2)
	{
		if (selectedVoxels[0].size() != 1 and selectedVoxels[1].size() != 1) // both are not empty
		{
			if ((selectedVoxels[0].rows() < 3 or selectedVoxels[1].rows() < 3) and _feat._featType >= 9)
			{
				res = -1000.0;
				return false;
			}
			else
			{
				float res1 = 0.0f, res2 = 0.0f;
				castProjection(selectedVoxels[0], _feat._featType, selectedEigenValues[0], selectedEigenVectors[0], res1);
				castProjection(selectedVoxels[1], _feat._featType, selectedEigenValues[1], selectedEigenVectors[1], res2);
				res = res1 - res2;
			}
		}
		else
		{
			res = -1000.0;
			return false;
		}
	}
	// four-voxel comparion
	else
	{
		if (selectedVoxels[0].size() != 1 and selectedVoxels[1].size() != 1 and selectedVoxels[2].size()!=1 and selectedVoxels[3].size()!=1) // all are not empty
		{
			if ((selectedVoxels[0].rows() < 3 or selectedVoxels[1].rows() < 3 or selectedVoxels[2].rows()<3 or selectedVoxels[3].rows()<3) and _feat._featType >= 9)
			{
				res = -1000.0;
				return false;
			}
			else
			{
				float res1 = 0.0f, res2 = 0.0f, res3 = 0.0f, res4 = 0.0f;
				castProjection(selectedVoxels[0], _feat._featType, selectedEigenValues[0], selectedEigenVectors[0], res1);
				castProjection(selectedVoxels[1], _feat._featType, selectedEigenValues[1], selectedEigenVectors[1], res2);
				castProjection(selectedVoxels[2], _feat._featType, selectedEigenValues[2], selectedEigenVectors[2], res3);
				castProjection(selectedVoxels[3], _feat._featType, selectedEigenValues[3], selectedEigenVectors[3], res4);
				res = (res1 - res2) - (res3 - res4);
			}
		}
		else
		{
			res = -1000.0f;
			return false;
		}
	}
	return true;
}

// a voxel may contain only one or two points, in which case, projections of featType greater than
// 8 (eigen value based projection) can not be performed, a flase flag will be returned.
bool FeatureFactory::castProjection(const Eigen::MatrixXf &voxel, int featType, Eigen::VectorXf &selectedEigenValues,  
									Eigen::VectorXf &selectedEigenVectors, float &testResult)
{
	testResult = 0.0f;

	if (_feat._featType <= 8) // radiometric features
	{
		Eigen::VectorXf avg_voxel;
		if (voxel.rows() != 1)
			avg_voxel = voxel.colwise().mean();
		else
			avg_voxel = voxel.row(0);

		// red channel
		if (featType == 0)
			testResult = selectChannel(avg_voxel, 4);

		// green channel
		else if (featType == 1)
			testResult = selectChannel(avg_voxel, 5);

		// blue channel
		else if (featType == 2)
			testResult = selectChannel(avg_voxel, 6);

		// h value
		else if (featType == 3)
			testResult = selectChannel(avg_voxel, 4, true);

		// s value
		else if (featType == 4)
			testResult = selectChannel(avg_voxel, 5, true);

		// v value
		else if (featType == 5)
			testResult = selectChannel(avg_voxel, 6, true);

		// x
		else if (featType == 6)
			testResult = selectChannel(avg_voxel, 0);

		// y
		else if (featType == 7)
			testResult = selectChannel(avg_voxel, 1);

		// z
		else // if (featType == 8)
			testResult = selectChannel(avg_voxel, 2);
	}
	// features based on local voxel covariance matrices 
	// (eigenvalue-based 3d geometric features)
	else // if (featType >= 9 and featType <= 26)
	{
		
		// eigenvalues
		float eigv0 = selectedEigenValues.x();
		float eigv1 = selectedEigenValues.y();
		float eigv2 = selectedEigenValues.z();
		// eigenvectors;
		Eigen::Vector3f vec0;
		vec0 << selectedEigenVectors(0), selectedEigenVectors(1), selectedEigenVectors(2);
		Eigen::Vector3f vec1;
		vec1 << selectedEigenVectors(3), selectedEigenVectors(4), selectedEigenVectors(5);
		Eigen::Vector3f vec2;
		vec2 << selectedEigenVectors(6), selectedEigenVectors(7), selectedEigenVectors(8);

		// compare linearity
		if (featType == 9)
			testResult = (eigv0 - eigv1) / eigv0;

		// compare planarity
		else if (featType == 10)
			testResult = (eigv1 - eigv2) / eigv0;

		// compare scattering
		else if (featType == 11)
			testResult = eigv2 / eigv0;

		// omnivariance
		else if (featType == 12)
			testResult = std::pow(eigv0*eigv1*eigv2, 1.0 / 3.0);

		// anisotropy
		else if (featType == 13)
			testResult = (eigv0 - eigv2) / eigv0;

		// eigenentropy
		else if (featType == 14)
			testResult = -(eigv0*std::log(eigv0) + eigv1 * std::log(eigv1) + eigv2 * std::log(eigv2));

		// change of curvature
		else if (_feat._featType == 15)
			testResult = eigv2 / (eigv0 + eigv1 + eigv2);
		// 1st eigenvector verticality
		// = |pi/2 - arccos(eigenvec0[2])|
		else if (_feat._featType == 16)
			testResult = fabs(std::_Pi*0.5 - std::acos(vec0(2)));
		// 3rd eigenvector verticality
		else if (_feat._featType == 17)
			testResult = fabs(std::_Pi*0.5 - std::acos(vec2(2)));
		// absolute moment
		// 1/N*|sumof((p-p_0).dot(eigenvector0)^1)|
		else if (_feat._featType == 18)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += ((voxel.row(i).x() - meanval.x()) * vec0.x()
							   + (voxel.row(i).y() - meanval.x()) * vec0.y()
							   + (voxel.row(i).z() - meanval.x()) * vec0.z());

			}
			testResult = fabs(testResult) / n;
		}
		else if (_feat._featType == 19)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += ((voxel.row(i).x() - meanval.x()) * vec1.x()
							   + (voxel.row(i).y() - meanval.x()) * vec1.y()
							   + (voxel.row(i).z() - meanval.x()) * vec1.z());

			}
			testResult = fabs(testResult) / n;
		}
		else if (_feat._featType == 20)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += ((voxel.row(i).x() - meanval.x()) * vec2.x()
							   + (voxel.row(i).y() - meanval.x()) * vec2.y()
							   + (voxel.row(i).z() - meanval.x()) * vec2.z());

			}
			testResult = fabs(testResult) / n;
		}
		// second order moment
		else if (_feat._featType == 21)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += std::pow(((voxel.row(i).x() - meanval.x()) * vec0.x()
										+ (voxel.row(i).y() - meanval.x()) * vec0.y()
										+ (voxel.row(i).z() - meanval.x()) * vec0.z()), 2);

			}
			testResult /= n;
		}
		else if (_feat._featType == 22)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += std::pow(((voxel.row(i).x() - meanval.x()) * vec1.x()
										+ (voxel.row(i).y() - meanval.x()) * vec1.y()
										+ (voxel.row(i).z() - meanval.x()) * vec1.z()), 2);

			}
			testResult /= n;
		}
		else if (_feat._featType == 23)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += std::pow(((voxel.row(i).x() - meanval.x()) * vec2.x()
										+ (voxel.row(i).y() - meanval.x()) * vec2.y()
										+ (voxel.row(i).z() - meanval.x()) * vec2.z()), 2);

			}
			testResult /= n;
		}
		// vertical moment
		// = 1/N * |sumof(p-p_0).dot([0, 0, 1])^1)|
		else if (_feat._featType == 24)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += (voxel.row(i).z() - meanval.z());
			}
			testResult/= n;
		}
		else if (_feat._featType == 25)
		{
			int n = voxel.rows();
			Eigen::VectorXf meanval = voxel.colwise().mean();
			for (int i = 0; i < n; ++i)
			{
				testResult += ((voxel.row(i).z() - meanval.z())
							   *(voxel.row(i).z() - meanval.z()));
			}
			testResult /= n;
		}
		else //if (_feat._featType == 26)
			testResult = eigv0 + eigv1 + eigv2;
	}
	return true;
}


float FeatureFactory::selectChannel(Eigen::VectorXf avg_voxel, int channelNo, bool convertToHSV)
{
	// if point cloud should be in HSV space
	if (convertToHSV == true)
		avg_voxel = toHSV(avg_voxel);
	return avg_voxel[channelNo];
}

Eigen::MatrixXf FeatureFactory::recoverNeighborhood(std::vector<Eigen::MatrixXf> voxels)
{
	std::vector<Eigen::VectorXf> points;
	for (int i = 0; i < voxels.size(); ++i)
	{
		if (voxels[i].size() == 1) // empty voxel
			continue;
		for (int j = 0; j < voxels[i].rows(); ++j)
		{
			points.push_back(voxels[i].row(j));
		}
	}
	Eigen::MatrixXf res(points.size(), points[0].size());
	for (int i = 0; i < points.size(); ++i)
	{
		res.row(i) = points[i].transpose();
	}
	return res;
}