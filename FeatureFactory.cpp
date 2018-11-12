#include "FeatureFactory.h"
#include <ctime>
#include <iostream>
#include "nanoflann.hpp"
#include "utils.h"
#include "c:/Eigen/Eigen/src/Eigenvalues/EigenSolver.h"

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
	// DEBUG to uncomment
	//std::cout << _localIndices << std::endl;
	//std::cout << _localDists << std::endl;
}

void FeatureFactory::buildVoxels()
{
	for (int i = 0; i < _feat._numVoxels; ++i)
	{
		Eigen::MatrixXf voxel;
		int rowsOfVoxel = _feat._voxelSize[i]+1;
		// here because the neighborhood already contains the dist information
		// (but it's relative to the original point cloud), so the dim is 8
		voxel.resize(rowsOfVoxel, 8);
		int pointId = _feat._pointId[i];
		Eigen::MatrixXi pointIdsForVoxel = _localIndices.row(pointId);
		Eigen::VectorXf dists = _localDists.row(pointId).leftCols(rowsOfVoxel);
		Eigen::MatrixXf newdists = Eigen::Map<Eigen::Matrix<float, -1, 1>>(dists.data(), dists.size());
		for (int j = 0; j < rowsOfVoxel; ++j)
		{
			voxel.row(j) = _neighborhood.row(pointIdsForVoxel(j));
		}
		voxel << voxel.leftCols(7), newdists;
		// DEBUG to uncomment
		//std::cout << "Voxel " << i << " is:\n" << voxel << std::endl;
		/*InOut inout;
		std::string filename = "voxel " + std::to_string(i) + ".txt";
		inout.writeToDisk(filename.c_str(), voxel);*/
		_voxels.push_back(voxel);
	}
}


std::vector<Eigen::VectorXf> FeatureFactory::averageVoxels()
{
	std::vector<Eigen::VectorXf> voxels_avg;
	for (int i = 0; i < _voxels.size(); ++i)
	{
		Eigen::VectorXf res;
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
		std::vector<Eigen::VectorXf> avg_voxels = averageVoxels();
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
	// features based on local voxel covariance matrices 
	// (eigenvalue-based 3d geometric features)
	else if(_feat._featType>=9 and _feat._featType<=15)
	{
		std::vector<Eigen::Matrix3f> tensor = computeCovarianceMatrix();
		std::vector<Eigen::Vector3f> eigv = computeEigenValues(tensor);
		// compare linearity
		if (_feat._featType == 9)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float lin1 = (neighEigv.x() - neighEigv.y()) / neighEigv.x();
				float lin2 = (eigv[0].x() - eigv[0].y()) / eigv[0].x();
				testResult = lin1 < lin2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float lin1 = (eigv[0].x() - eigv[0].y()) / eigv[0].x();
				float lin2 = (eigv[1].x() - eigv[1].y()) / eigv[1].x();
				testResult = lin1 < lin2 ? true : false;
			}
			else
			{
				float lin1 = (eigv[0].x() - eigv[0].y()) / eigv[0].x();
				float lin2 = (eigv[1].x() - eigv[1].y()) / eigv[1].x();
				float lin3 = (eigv[2].x() - eigv[2].y()) / eigv[2].x();
				float lin4 = (eigv[3].x() - eigv[3].y()) / eigv[3].x();
				testResult = (lin1 - lin2) < (lin3 - lin4) ? true : false;
			}
		}
		
		// compare planarity
		else if (_feat._featType == 10)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float pla1 = (neighEigv.y() - neighEigv.z()) / neighEigv.x();
				float pla2 = (eigv[0].y() - eigv[0].z()) / eigv[0].x();
				testResult = pla1 < pla2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float pla1 = (eigv[0].y() - eigv[0].z()) / eigv[0].x();
				float pla2 = (eigv[1].y() - eigv[1].z()) / eigv[1].x();
				testResult = pla1 < pla2 ? true : false;
			}
			else
			{
				float pla1 = (eigv[0].y() - eigv[0].z()) / eigv[0].x();
				float pla2 = (eigv[1].y() - eigv[1].z()) / eigv[1].x();
				float pla3 = (eigv[2].y() - eigv[2].z()) / eigv[2].x();
				float pla4 = (eigv[3].y() - eigv[3].z()) / eigv[3].x();
				testResult = (pla1 - pla2) < (pla3 - pla4) ? true : false;
			}
		}
		
		// compare scattering
		else if (_feat._featType == 11)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float sca1 = neighEigv.z() / neighEigv.x();
				float sca2 = eigv[0].z() / eigv[0].x();
				testResult = sca1 < sca2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float sca1 = eigv[0].z() / eigv[0].x();
				float sca2 = eigv[1].z() / eigv[1].x();
				testResult = sca1 < sca2 ? true : false;
			}
			else
			{
				float sca1 = eigv[0].z() / eigv[0].x();
				float sca2 = eigv[1].z() / eigv[1].x();
				float sca3 = eigv[2].z() / eigv[2].x();
				float sca4 = eigv[3].z() / eigv[3].x();
				testResult = (sca1 - sca2) < (sca3 - sca4) ? true : false;
			}
		}
		
		// omnivariance
		else if (_feat._featType == 12)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float omn1 = std::pow(neighEigv.x()*neighEigv.y()*neighEigv.z(), 1.0/3.0);
				float omn2 = std::pow(eigv[0].x()*eigv[0].y()*eigv[0].z(), 1.0 / 3.0);
				testResult = omn1 < omn2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float omn1 = std::pow(eigv[0].x()*eigv[0].y()*eigv[0].z(), 1.0 / 3.0);
				float omn2 = std::pow(eigv[1].x()*eigv[1].y()*eigv[1].z(), 1.0 / 3.0);
				testResult = omn1 < omn2 ? true : false;
			}
			else
			{
				float omn1 = std::pow(eigv[0].x()*eigv[0].y()*eigv[0].z(), 1.0 / 3.0);
				float omn2 = std::pow(eigv[1].x()*eigv[1].y()*eigv[1].z(), 1.0 / 3.0);
				float omn3 = std::pow(eigv[2].x()*eigv[2].y()*eigv[2].z(), 1.0 / 3.0);
				float omn4 = std::pow(eigv[3].x()*eigv[3].y()*eigv[3].z(), 1.0 / 3.0);
				testResult = (omn1 - omn2) < (omn3 - omn4) ? true: false;
			}
		}

		// anisotropy
		else if (_feat._featType == 13)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float ani1 = (neighEigv.x() - neighEigv.z()) / neighEigv.x();
				float ani2 = (eigv[0].x() - eigv[0].z()) / eigv[0].x();
				testResult = ani1 < ani2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float ani1 = (eigv[0].x() - eigv[0].z()) / eigv[0].x();
				float ani2 = (eigv[1].x() - eigv[1].z()) / eigv[1].x();
				testResult = ani1 < ani2 ? true : false;
			}
			else
			{
				float ani1 = (eigv[0].x() - eigv[0].z()) / eigv[0].x();
				float ani2 = (eigv[1].x() - eigv[1].z()) / eigv[1].x();
				float ani3 = (eigv[2].x() - eigv[2].z()) / eigv[2].x();
				float ani4 = (eigv[3].x() - eigv[3].z()) / eigv[3].x();
				testResult = (ani1 - ani2) < (ani3 - ani4) ? true : false;
			}
		}

		// eigenentropy
		else if (_feat._featType == 14)
		{
			if (_feat._numVoxels == 1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float ent1 = -(neighEigv.x()*std::log(neighEigv.x())
							   + neighEigv.y()*std::log(neighEigv.y())
							   + neighEigv.z()*std::log(neighEigv.z()));
				float ent2 = -(eigv[0].x()*std::log(eigv[0].x())
							   + eigv[0].y()*std::log(eigv[0].y())
							   + eigv[0].z()*std::log(eigv[0].z()));
				testResult = ent1 < ent2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float ent1 = -(eigv[0].x()*std::log(eigv[0].x())
							 + eigv[0].y()*std::log(eigv[0].y())
							 + eigv[0].z()*std::log(eigv[0].z()));
				float ent2 = -(eigv[1].x()*std::log(eigv[1].x())
							 + eigv[1].y()*std::log(eigv[1].y())
							 + eigv[1].z()*std::log(eigv[1].z()));
				testResult = ent1 < ent2 ? true : false;
			}
			else
			{
				float ent1 = -(eigv[0].x()*std::log(eigv[0].x())
							   + eigv[0].y()*std::log(eigv[0].y())
							   + eigv[0].z()*std::log(eigv[0].z()));
				float ent2 = -(eigv[1].x()*std::log(eigv[1].x())
							   + eigv[1].y()*std::log(eigv[1].y())
							   + eigv[1].z()*std::log(eigv[1].z()));
				float ent3 = -(eigv[2].x()*std::log(eigv[2].x())
							   + eigv[2].y()*std::log(eigv[2].y())
							   + eigv[2].z()*std::log(eigv[2].z()));
				float ent4 = -(eigv[3].x()*std::log(eigv[3].x())
							   + eigv[3].y()*std::log(eigv[3].y())
							   + eigv[3].z()*std::log(eigv[3].z()));
				testResult = (ent1 - ent2) < (ent3 - ent4) ? true : false;
			}
		}

		// change of curvature
		else // _feat._featType == 15
		{
			if(_feat._numVoxels==1)
			{
				Eigen::Matrix3f neighCov = computeCovarianceMatrix(_neighborhood);
				Eigen::Vector3f neighEigv = computeEigenValues(neighCov);
				float cha1 = neighEigv.z() / (neighEigv.x() + neighEigv.y() + neighEigv.z());
				float cha2 = eigv[0].z() / (eigv[0].x() + eigv[0].y() + eigv[0].z());
				testResult = cha1 < cha2 ? true : false;
			}
			else if (_feat._numVoxels == 2)
			{
				float cha1 = eigv[0].z() / (eigv[0].x() + eigv[0].y() + eigv[0].z());
				float cha2 = eigv[1].z() / (eigv[1].x() + eigv[1].y() + eigv[1].z());
				testResult = cha1 < cha2 ? true : false;
			}
			else
			{
				float cha1 = eigv[0].z() / (eigv[0].x() + eigv[0].y() + eigv[0].z());
				float cha2 = eigv[1].z() / (eigv[1].x() + eigv[1].y() + eigv[1].z());
				float cha3 = eigv[2].z() / (eigv[2].x() + eigv[2].y() + eigv[2].z());
				float cha4 = eigv[3].z() / (eigv[3].x() + eigv[3].y() + eigv[3].z());
				testResult = cha1 < cha2 ? true : false;
			}
		}
	}
	else // _feat._featType==16: local density
	{
		localNeighbors();
		buildVoxels();
		if (_feat._numVoxels == 1)
		{
			// the furtherst point to the center is the radius of this neighborhood
			float dist1 = _neighborhood.bottomRightCorner(1,1)(0, 0);
			float dist2 = _voxels[0].bottomRightCorner(1, 1)(0, 0);
			float volumn1 = 4.0 / 3.0*std::_Pi *dist1*dist1*dist1;
			float volumn2 = 4.0 / 3.0*std::_Pi *dist2*dist2*dist2;
			float density1 = _neighborhood.rows() / volumn1;
			float density2 = _voxels[0].rows() / volumn2;
			testResult = density1 < density2 ? true : false;
		}
		else if (_feat._numVoxels == 2)
		{
			
			float dist1 = _voxels[0].bottomRightCorner(1, 1)(0, 0);
			float dist2 = _voxels[1].bottomRightCorner(1, 1)(0, 0);
			float volumn1 = 4.0 / 3.0*std::_Pi *dist1*dist1*dist1;
			float volumn2 = 4.0 / 3.0*std::_Pi *dist2*dist2*dist2;
			float density1 = _voxels[0].rows() / volumn1;
			float density2 = _voxels[1].rows() / volumn2;
			testResult = density1 < density2 ? true : false;
		}
		else
		{
			float dist1 = _voxels[0].bottomRightCorner(1, 1)(0, 0);
			float dist2 = _voxels[1].bottomRightCorner(1, 1)(0, 0);
			float dist3 = _voxels[2].bottomRightCorner(1, 1)(0, 0);
			float dist4 = _voxels[3].bottomRightCorner(1, 1)(0, 0);
			float volumn1 = 4.0 / 3.0*std::_Pi *dist1*dist1*dist1;
			float volumn2 = 4.0 / 3.0*std::_Pi *dist2*dist2*dist2;
			float volumn3 = 4.0 / 3.0*std::_Pi *dist3*dist3*dist3;
			float volumn4 = 4.0 / 3.0*std::_Pi *dist4*dist4*dist4;
			float density1 = _voxels[0].rows() / volumn1;
			float density2 = _voxels[1].rows() / volumn2;
			float density3 = _voxels[2].rows() / volumn3;
			float density4 = _voxels[3].rows() / volumn4;
			testResult = (density1 - density2) < (density3 - density4) ? true : false;
		}
	}
	return testResult;
}


bool FeatureFactory::compareChannels(std::vector<Eigen::VectorXf> avg_voxels, int channelNo, bool isInRGBSpace)
{
	// if point cloud should be in HSV space
	if (!isInRGBSpace)
	{
		for (int i = 0; i < avg_voxels.size(); ++i)
			avg_voxels[i] = toHSV(avg_voxels[i]);
		_voxels[0].row(0) = toHSV(_voxels[0].row(0));

	}
	if (_feat._numVoxels == 1)
		// _voxels[0].row(0) is the center point of the first voxel
		return avg_voxels[0](channelNo) < _voxels[0].row(0)(channelNo) ? true : false;
	else if (_feat._numVoxels == 2)
		return avg_voxels[0](channelNo) < avg_voxels[1](channelNo) ? true : false;
	else
		return (avg_voxels[0](channelNo) - avg_voxels[2](channelNo)) < (avg_voxels[1](channelNo) - avg_voxels[3](channelNo)) ? true : false;

}

Eigen::Matrix3f FeatureFactory::computeCovarianceMatrix(Eigen::MatrixXf neigh)
{
	Eigen::Matrix3f covMat;
	covMat.setZero();
	Eigen::VectorXf	meanNeigh = neigh.colwise().mean();
	for (int j = 0; j < neigh.rows(); ++j) // for every point in each neighborhood
	{
		float px = neigh.row(j).x() - meanNeigh[0];
		float py = neigh.row(j).y() - meanNeigh[1];
		float pz = neigh.row(j).z() - meanNeigh[2];
		covMat(1, 1) += py * py;
		covMat(1, 2) += py * pz;
		covMat(2, 2) += pz * pz;
		float pxx = px * px;
		float pxy = px * py;
		float pxz = px * pz;
		covMat(0, 0) += pxx;
		covMat(0, 1) += pxy;
		covMat(0, 2) += pxz;
	}
	covMat(1, 0) = covMat(0, 1);
	covMat(2, 0) = covMat(0, 2);
	covMat(2, 1) = covMat(1, 2);
	covMat /= neigh.rows();
	return covMat;
}

std::vector<Eigen::Matrix3f> FeatureFactory::computeCovarianceMatrix()
{
	// to obtain the local indices
	localNeighbors();
	// DEBUG to uncomment
	// std::cout << _localIndices << std::endl;
	// build voxels based on local indices with pointId and voxelSize;
	buildVoxels();
	std::vector<Eigen::Matrix3f> res;
	for (int i = 0; i < _voxels.size(); ++i)
	{
		Eigen::Matrix3f covMat = computeCovarianceMatrix(_voxels[i]);
		res.push_back(covMat);
	}
	return res;
}

Eigen::Vector3f FeatureFactory::computeEigenValues(Eigen::Matrix3f covMat)
{
	Eigen::Vector3f eigv;
	Eigen::EigenSolver<Eigen::Matrix3f> es(covMat);
	// eigenvalues are not in a particular order
	eigv = es.eigenvalues().real(); 
	// in descending order
	std::sort(eigv.data(), eigv.data() + eigv.size(), [](float a, float b) { return a > b; });
	//std::sort(eigv.data(), eigv.data() + eigv.size(), std::greater<float>());
	//if eigenvalues are less than or equal to 0, set them to 1e-6 to avoid dividing by NaN;
	if (eigv(0) <= 0) eigv(0) = 1e-6;
	if (eigv(1) <= 0) eigv(1) = 1e-6;
	if (eigv(2) <= 0) eigv(2) = 1e-6;
	// normalize eigenvalues
	eigv /= eigv.sum();
	return eigv;
}

std::vector<Eigen::Vector3f> FeatureFactory::computeEigenValues(std::vector<Eigen::Matrix3f> covTensor)
{
	std::vector<Eigen::Vector3f> res;
	for (int i = 0; i < covTensor.size(); ++i)
	{
		Eigen::Vector3f eigv = computeEigenValues(covTensor[i]);
		res.push_back(eigv);
	}
	return res;
}
