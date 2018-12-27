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


void FeatureFactory::buildVoxels(std::vector<std::vector<Eigen::VectorXf>> &partitions)
{
	for (int i = 0; i < _feat._numVoxels; ++i)
	{
		int idx = _feat._pointId[i];
		if (partitions[idx].size() == 0)
		{
			Eigen::MatrixXf tmp;
			tmp << 0;
			_voxels.push_back(tmp);
		}
		else if (partitions[idx].size() == 1)
		{
			_voxels.push_back(partitions[idx][0]);
		}
		else
		{
			std::vector<Eigen::VectorXf> points = partitions[idx];
			Eigen::MatrixXf voxel(points.size(), 7);
			for (int j = 0; j < points.size(); ++j)
				voxel.row(j) = points[j].transpose();
			_voxels.push_back(voxel);
		}
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

float FeatureFactory::project()
{
	std::vector<std::vector<Eigen::VectorXf>> partitions;
	partitions = partitionSpace(_neighborhood);
	buildVoxels(partitions);
	float testResult = 0.0f;
	// one voxel comparison
	if (_voxels.size() == 1)
	{
		// if the voxel has no points
		if (_voxels[0].size() == 1)
			return -1;
		else
		{
			float voxelValue = castProjection(_voxels[0], _feat._featType);
			float centerPointValue = castProjection(_neighborhood.row(0), _feat._featType);
			return centerPointValue - voxelValue;
		}
	}
	// two voxel comparison
	else if (_voxels.size() == 2)
	{
		// if both voxels are not empty
		if (_voxels[0].size() != 1 and _voxels[1].size() != 1)
		{
			float voxelValue1 = castProjection(_voxels[0], _feat._featType);
			float voxelValue2 = castProjection(_voxels[1], _feat._featType);
			return voxelValue1 - voxelValue2;
		}
		// downgrade to 1 voxel case
		else if (_voxels[0].size() == 1)
		{
			float voxelValue = castProjection(_voxels[1], _feat._featType);
			float centerPointValue = castProjection(_neighborhood.row(0), _feat._featType);
			return centerPointValue - voxelValue;
		}
		else if (_voxels[1].size() == 1)
		{
			float voxelValue = castProjection(_voxels[0], _feat._featType);
			float centerPointValue = castProjection(_neighborhood.row(0), _feat._featType);
			return centerPointValue - voxelValue;
		}
		// both voxels are empty
		else return -1;
	}
	// four voxel comparison
	else
	{
		// if all voxels are not empty
		if (_voxels[0].size() != 1 and _voxels[1].size() != 1
			and _voxels[2].size() != 1 and _voxels[3].size() != 1)
		{
			float voxelValue1 = castProjection(_voxels[0], _feat._featType);
			float voxelValue2 = castProjection(_voxels[1], _feat._featType);
			float voxelValue3 = castProjection(_voxels[2], _feat._featType);
			float voxelValue4 = castProjection(_voxels[3], _feat._featType);
			return (voxelValue1 - voxelValue2) - (voxelValue3 - voxelValue4);
		}
		// special cases
		else
		{
			return -1;
		}
	}
}
float FeatureFactory::castProjection(const Eigen::MatrixXf &voxel, int featType)
{

	float testResult = 0;

	if (_feat._featType <= 8) // radiometric features
	{
		Eigen::MatrixXf avg_voxel;
		if (voxel.rows() != 1)
			avg_voxel = voxel.colwise.mean();
		else
			avg_voxel = voxel;

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
			testResult = selectChannel(avg_voxel, 4, false);

		// s value
		else if (featType == 4)
			testResult = selectChannel(avg_voxel, 5, false);

		// v value
		else if (featType == 5)
			testResult = selectChannel(avg_voxel, 6, false);

		// x
		else if (featType == 6)
			testResult = selectChannel(avg_voxel, 0);

		// y
		else if (featType == 7)
			testResult = selectChannel(avg_voxel, 1);

		// z
		else if (featType == 8)
			testResult = selectChannel(avg_voxel, 2);
	}
	// features based on local voxel covariance matrices 
	// (eigenvalue-based 3d geometric features)
	else if (featType >= 9 and featType <= 15)
	{
		if (voxel.rows() == 1 or voxel.rows() == 2)
			return -1;
		
		// eigenvalues
		float eigv0 = 0;
		float eigv1 = 0;
		float eigv2 = 0;
		// eigenvectors;
		Eigen::Vector3f vec0;
		Eigen::Vector3f vec1;
		Eigen::Vector3f vec2;

		computeEigens(voxel, eigv0, eigv1, eigv2, vec0, vec1, vec2);
		
		// compare linearity
		if (featType == 9)
			return (eigv0 - eigv1) / eigv0;

		// compare planarity
		else if (featType == 10)
			return (eigv1 - eigv2) / eigv0;

		// compare scattering
		else if (featType == 11)
			return eigv2 / eigv0;

		// omnivariance
		else if (featType == 12)
			return std::pow(eigv0*eigv1*eigv2, 1.0 / 3.0);

		// anisotropy
		else if (featType == 13)
			return (eigv0 - eigv2) / eigv0;

		// eigenentropy
		else if (featType == 14)
			return -(eigv0*std::log(eigv0) + eigv1 * std::log(eigv1) + eigv2 * std::log(eigv2));

		// change of curvature
		else // _feat._featType == 15
			return eigv2 / (eigv0 + eigv1 + eigv2);
	}
	else
		return -1;
}


float FeatureFactory::selectChannel(Eigen::VectorXf avg_voxel, int channelNo, bool convertToHSV)
{
	// if point cloud should be in HSV space
	if (convertToHSV == true)
		avg_voxel = toHSV(avg_voxel);
	return avg_voxel[channelNo];
}

Eigen::Matrix3f FeatureFactory::computeCovarianceMatrix(Eigen::MatrixXf neigh)
{
	Eigen::Matrix3f covMat;
	covMat.setZero();
	Eigen::VectorXf	 meanNeigh = neigh.colwise().mean();
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


// analytically solve the eigenvalues
//Eigen::Vector3f FeatureFactory::computeEigenValues(Eigen::Matrix3f covMat)
//{
//	const double SQRT3 = 1.73205080756887729352744634151; // sqrt(3)
//	double m, c1, c0;
//	Eigen::Vector3f eigenvals(0, 0, 0);
//	// the covMat is real symmetric in the form of
//	//			/a	d	f/
//	//	covMat=	/d	b	e/
//	//			/f  e   c/
//	double de = covMat(0, 1) * covMat(1, 2); 	// d*e
//	double dd = covMat(0, 1) * covMat(1, 0);	// d*d
//	double ee = covMat(1, 2) * covMat(2, 1);	// e*e
//	double ff = covMat(2, 0) * covMat(0, 2);	// f*f
//	m = covMat(0, 0) + covMat(1, 1) + covMat(2, 2);	// a+b+c
//	c1 = (covMat(0, 0)*covMat(1, 1) + covMat(0, 0)*covMat(2, 2) + covMat(1, 1)*covMat(2, 2))
//		- (dd + ee + ff); 	// a*b + a*c + b*c - d^2 - e^2 - f^2
//	c0 = covMat(2, 2)*dd + covMat(0, 0)*ee + covMat(1, 1)*ff - covMat(0, 0)*covMat(1, 1)*covMat(2, 2)
//		- 2.0*covMat(0, 2)*de;	//c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e
//
//	double p, sqrt_p, q, c, s, phi;
//	p = m * m - 3.0*c1;
//	q = m * (p - (3.0 / 2.0)*c1) - (27.0 / 2.0)*c0;
//	sqrt_p = std::sqrt(std::abs(p));
//	phi = 27.0*(0.25*c1*c1*(p - c1) + c0 * (q + 27.0 / 4.0*c0));
//	phi = (1.0 / 3.0)*std::atan2(std::sqrt(std::abs(phi)), q);
//
//	c = sqrt_p * std::cos(phi);
//	s = (1.0 / SQRT3)*sqrt_p*std::sin(phi);
//	eigenvals(1) = (1.0 / 3.0)*(m - c);
//	eigenvals(2) = eigenvals(1) + s;
//	eigenvals(0) = eigenvals(1) + c;
//	eigenvals(1) -= s;
//
//	//if eigenvalues are less than or equal to 0, set them to 1e-6 to avoid dividing by NaN;
//	if (eigenvals(0) <= 0) eigenvals(0) = 1e-6;
//	if (eigenvals(1) <= 0) eigenvals(1) = 1e-6;
//	if (eigenvals(2) <= 0) eigenvals(2) = 1e-6;
//	// normalize eigenvalues
//	eigenvals /= eigenvals.sum();
//	return eigenvals;
//}


void FeatureFactory::computeEigens(const Eigen::Matrix3f &covMat, float &majorVal, float &middleVal, float &minorVal, 
								   Eigen::Vector3f &majorAxis, Eigen::Vector3f &middleAxis, Eigen::Vector3f &minorAxis)
{
	Eigen::EigenSolver<Eigen::Matrix3f> es;
	// eigenvalues are not in a particular order
	es.compute(covMat);
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvectorsType eigenVecs;
	Eigen::EigenSolver <Eigen::Matrix3f>::EigenvalueType eigenVals;
	eigenVecs = es.eigenvectors();
	eigenVals = es.eigenvalues();

	// sort the eigenvalues in descending order
	unsigned int temp = 0;
	unsigned int majorIdx = 0;
	unsigned int middleIdx = 1;
	unsigned int minorIdx = 2;
	if (eigenVals.real() (majorIdx) < eigenVals.real() (middleIdx))
	{
		temp = majorIdx;
		majorIdx = middleIdx;
		middleIdx = temp;
	}
	if (eigenVals.real() (majorIdx) < eigenVals.real() (minorIdx))
	{
		temp = majorIdx;
		majorIdx = minorIdx;
		minorIdx = temp;
	}
	if (eigenVals.real() (middleIdx) < eigenVals.real() (minorIdx))
	{
		temp = minorIdx;
		minorIdx = middleIdx;
		middleIdx = temp;
	}

	majorVal = eigenVals.real() (majorIdx);
	middleVal = eigenVals.real() (middleIdx);
	minorVal = eigenVals.real() (minorIdx);

	majorAxis = eigenVecs.col(majorIdx).real();
	middleAxis = eigenVecs.col(middleIdx).real();
	minorAxis = eigenVecs.col(minorIdx).real();

	// normalize
	majorAxis.normalize();
	middleAxis.normalize();
	minorAxis.normalize();

	float det = majorAxis.dot(middleAxis.cross(minorAxis));
	if (det <= 0.0f)
	{
		majorAxis(0) = -majorAxis(0);
		majorAxis(1) = -majorAxis(1);
		majorAxis(2) = -majorAxis(2);
	}
}

// compute the axis algined orientated bounding box
void FeatureFactory::computeOBB(Eigen::MatrixXf &neigh, Eigen::MatrixXf &neighR, Eigen::Vector3f &obbMinP, Eigen::Vector3f &obbMaxP, 
								Eigen::Matrix3f &obbR, Eigen::Vector3f &obbPos)
{
	Eigen::MatrixXf coords = neigh.leftCols(3);
	Eigen::Matrix3f covMat = computeCovarianceMatrix(coords);
	Eigen::Vector3f meanVal = coords.colwise().mean();

	// get the eigenvalues and eigenvector of the covmat
	float majorVal = 0, midVal = 0, minorVal = 0;
	Eigen::Vector3f majorAxis(0, 0, 0);
	Eigen::Vector3f midAxis(0, 0, 0);
	Eigen::Vector3f minorAxis(0, 0, 0);
	computeEigens(covMat, majorVal, midVal, minorVal, majorAxis, midAxis, minorAxis);

	obbMinP << 0, 0, 0;
	obbMaxP << 0, 0, 0;
	obbMinP.x() = std::numeric_limits<float>::max();
	obbMinP.y() = std::numeric_limits<float>::max();
	obbMinP.z() = std::numeric_limits<float>::max();
	
	obbMaxP.x() = std::numeric_limits<float>::min();
	obbMaxP.y() = std::numeric_limits<float>::min();
	obbMaxP.z() = std::numeric_limits<float>::min();

	neighR = neigh;
	
	// express the points in the local frame formed by the eigenvectors
	// P_A = P_AB + R_AB * P_B
	// point expressed in frame {A} is equivalent to this point expressed
	// in frame {B} rotated by R_AB (rotation of frame {B} relative to frame {A})
	// plus the translation of origin of {B} to {A}
	// suppose {A} is the universal frame and {B} is the local frame
	// then P_B = R_AB^-1 * (P_A - P_AB) = R_AB^T * (P_A - P_AB).
	for (int i = 0; i < coords.rows(); ++i)
	{
		float x = (coords(i, 0) - meanVal(0))*majorAxis(0) 
				+ (coords(i, 1) - meanVal(1))*midAxis(0) 
				+ (coords(i, 2) - meanVal(2))*minorAxis(0);
		float y = (coords(i, 0) - meanVal(0))*majorAxis(1)
				+ (coords(i, 1) - meanVal(1))*midAxis(1)
				+ (coords(i, 2) - meanVal(2))*minorAxis(1);
		float z = (coords(i, 0) - meanVal(0))*majorAxis(2)
				+ (coords(i, 1) - meanVal(1))*midAxis(2)
				+ (coords(i, 2) - meanVal(2))*minorAxis(2);

		neighR(i, 0) = x;
		neighR(i, 1) = y;
		neighR(i, 2) = z;

		if (x <= obbMinP.x()) 
			obbMinP.x()=x;
		if (y <= obbMinP.y()) 
			obbMinP.y()=y;
		if (z <= obbMinP.z()) 
			obbMinP.z()=z;

		if (x >= obbMaxP.x()) 
			obbMaxP.x()=x;
		if (y >= obbMaxP.y()) 
			obbMaxP.y()=y;
		if (z >= obbMaxP.z()) 
			obbMaxP.z()=z;
	}

	// rotation matrix
	obbR.setZero();
	obbR << majorAxis(0), midAxis(0), minorAxis(0),
			majorAxis(1), midAxis(1), minorAxis(1),
			majorAxis(2), midAxis(2), minorAxis(2);

	// translation vector
	Eigen::Vector3f trans((obbMaxP.x() - obbMinP.x()) / 2.0,
						  (obbMaxP.y() - obbMinP.y()) / 2.0,
						  (obbMaxP.z() - obbMinP.z()) / 2.0);
	// translate the origin of the local frame to the minimal point
	neighR.leftCols(3).rowwise() -= obbMinP.transpose();
	obbMaxP -= obbMinP;
	obbMinP -= obbMinP;
	obbPos << 0, 0, 0;
	obbPos = meanVal + obbR * trans;
}


// partition the obb of neigh into 27 (3*3*3) small cubes
// points in each cube form the respective subvoxels
// note that some of the subvoxels may contain no points or
// too few (less than 3) points to compute the covariance based
// features, such cases should be treated carefully
std::vector<std::vector<Eigen::VectorXf>> FeatureFactory::partitionSpace(Eigen::MatrixXf &neigh)
{
	// get the obb
	Eigen::Vector3f minp(0, 0, 0);  // the minimal dimensions
	Eigen::Vector3f maxp(0, 0, 0);  // the maximal dimensions
	Eigen::Matrix3f rot;  // rotation matrix
	rot.setZero();
	Eigen::MatrixXf neighR;
	Eigen::Vector3f pos(0, 0, 0); // translation matrix
	computeOBB(neigh, neighR, minp, maxp, rot, pos);
	// divide the length/width/height of the bounding box into three parts
	// that is the dimensions of the small cubes
	// to avoid divisions in successive steps, the inverse of each is computed
	float inverse_dlength = 3.0f/(maxp(0) - minp(0))-1e-5;
	float inverse_dwidth = 3.0f/(maxp(1) - minp(1))-1e-5;
	float inverse_dheight = 3.0f/(maxp(2) - minp(2))-1e-5;
	// compute the minimal/maximal bounding box values
	Eigen::Vector3i minbb(0, 0, 0);
	Eigen::Vector3i maxbb(0, 0, 0);
	minbb(0) = static_cast<int>(floor(minp(0) * inverse_dlength));
	//maxbb(0) = static_cast<int>(floor(maxp(0) * inverse_dlength));
	minbb(1) = static_cast<int>(floor(minp(1) * inverse_dwidth));
	//maxbb(1) = static_cast<int>(floor(maxp(1) * inverse_dwidth));
	minbb(2) = static_cast<int>(floor(minp(2) * inverse_dheight));
	//maxbb(2) = static_cast<int>(floor(maxp(2) * inverse_dheight));
	std::vector<std::vector<Eigen::VectorXf>> voxels(27);
	for (int i = 0; i < neigh.rows(); ++i)
	{
		int ijk0 = static_cast<int>(floor(neighR(i, 0)*inverse_dlength));
		int ijk1 = static_cast<int>(floor(neighR(i, 1)*inverse_dwidth));
		int ijk2 = static_cast<int>(floor(neighR(i, 2)*inverse_dheight));
		int idx = ijk0 + ijk1 * 3 + ijk2 * 9;
		voxels[idx].push_back(neigh.row(i));
	}
	return voxels;
}