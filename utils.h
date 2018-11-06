#pragma once

#include<vector>
#include "C:/Eigen/Eigen/Dense"
/***********************************************************************
This header is used for converting a std::vector to an Eigen::vector
or vice versa conveniently. Conversion between the two data structures
is necessary because sometimes the size of the vector can't be determined
beforhand and thus std::vector is needed which has method `push_back()`.
Most of the time, Eigen::vector is used due to its efficiency.
************************************************************************/

// convert Eigen::Vector to std::vector
inline std::vector<int> toStdVec(Eigen::VectorXi eigVec)
{
	std::vector<int> stdVec(&eigVec[0], eigVec.data() + eigVec.size());
	return stdVec;
}


//// convert std::vector to Eigen::VectorX
//template <typename _Scalar>
//Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>  toEigenVec(std::vector<_Scalar> stdVec)
//{
//	_Scalar* ptr = &stdVec[0];
//	return Eigen::Map<Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>>(ptr, stdVec.size());
//}

inline Eigen::VectorXi toEigenVec(std::vector<int> stdVec)
{
	int* ptr = &stdVec[0];
	return Eigen::Map<Eigen::VectorXi>(ptr, stdVec.size());
}

inline Eigen::VectorXf toEigenVec(std::vector<float> stdVec)
{
	float* ptr = &stdVec[0];
	return Eigen::Map<Eigen::VectorXf>(ptr, stdVec.size());
}