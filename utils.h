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

inline std::vector<float> toStdVec(Eigen::VectorXf eigVec)
{
	std::vector<float> stdVec(&eigVec[0], eigVec.data() + eigVec.size());
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



inline bool isequal(float a, float b)
{
	if ((a - b) < 0.00001)
		return true;
	else
		return false;
}


// convert a voxel in RGB space to HSV space
// the last 3 columns of a point represent the r/g/b value
inline Eigen::VectorXf toHSV(Eigen::VectorXf point)
{
	float r = point(4)/255;
	float g = point(5)/255;
	float b = point(6)/255;
	// find the min and max of the three
	float max = r > g ? r : g;
	max = max > b ? max : b;
	float min = r < g ? r : g;
	min = min < b ? min : b;
	
	float delta = max - min;
	
	// calculate H in degrees
	float h = 0;
	if (delta < 0.00001) // max==min
		h = 0;
	else if (isequal(max, r)) // max == r channel
		h = 60 * (g - b) / delta;
	else if (isequal(max, g))
		h = 60 * (2 + (b - r) / delta);
	else
		h = 60 * (4 + (r - g) / delta);

	// calculate S
	float s = 0;
	if (isequal(max, 0))
		s = 0;
	else
		s = delta / max;

	float v = max;

	point(4) = h;
	point(5) = s;
	point(6) = v;
	return point;
}