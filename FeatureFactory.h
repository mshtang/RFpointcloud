/***************************************************
This class is used for projecting a high dimensional
data to a real value (1d) so that simple comparison 
at a given node is possible
***************************************************/

#pragma once
//#include "C:/Eigen/Eigen/Dense"
#include "Sample.h"

class FeatureFactory
{
public:

	FeatureFactory(Eigen::MatrixXf& neighborhood, Features feat);
	bool computeFeature();

private:
	bool redColorDiff();
	bool greenColorDiff();
	bool blueColorDiff();
	bool xDiff();
	bool yDiff();
	bool zDiff();
	Eigen::MatrixXf _neighborhood;
	Features _feat;
};



