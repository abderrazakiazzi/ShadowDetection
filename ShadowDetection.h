#pragma once
#include <opencv2\/opencv.hpp>
class ShadowDetection
{
public:
	ShadowDetection();
	~ShadowDetection();
	void detectShadow(cv::Mat inputcurrent, cv::Mat inputreference, cv::Mat& outputMask);
	void detectShadowHSVColor(cv::Mat imageBack, cv::Mat imageCurrent, cv::Mat& imageResult);
	void detectShadowGradiant(cv::Mat& imageBack, cv::Mat& imageCurrent, cv::Mat& imageResult, cv::Mat& imageFinal);
	
	void gradientmagnitude(cv::Mat Xdirect, cv::Mat Ydirect, cv::Mat& outputmang);
	void gradientOrientDirect(cv::Mat Xdirect, cv::Mat Ydirect, cv::Mat& outputorient);
	void gradientSobel(cv::Mat image, cv::Mat& directimage, int a);
	void gradientdiffer(cv::Mat bx, cv::Mat by, cv::Mat cx, cv::Mat cy, cv::Mat& out);
	void gradientCorrelation(cv::Mat Rm, cv::Mat& outRmC);
	void thresholdMagnitude(cv::Mat outRmC, cv::Mat& mask);


	void NormalizeHSV(cv::Mat& imageHSV, cv::Mat& imageHSVNormalise);
	bool satifieRules(cv::Mat imageCurrent, cv::Mat imageBack, cv::Mat& imageResultMask);
	// 
	double maxMatrice(cv::Mat image);
	double minMatrice(cv::Mat image);
	float multiply2Matrice(cv::Mat image1, cv::Mat image2, cv::Mat output);
	float Addition2Matrice(cv::Mat image1, cv::Mat image2, cv::Mat output);
	//float divide2Matrice(cv::Mat image1, cv::Mat image2, cv::Mat output);
	// 
	void loadParameters();
	void findNonzerosPixel(cv::Mat image);
	void DisplayDetectPix(cv::Mat& image);
	void DisplayContours(cv::Mat& image);
	cv::Mat postprocesShadows(cv::Mat image);

private:
	cv::Mat imageBack;
	cv::Mat imageCurrent;
	cv::Mat imageResultHSV;
	cv::Mat imageResultGradiant;
	cv::Mat imageFinal;
	cv::Mat imageHSV, imageH, imageS, imageV;
	cv::Mat imageHSVN, imageHN, imageSN, imageVN;
	cv::vector<cv::Point> listnonzeros;
	cv::vector<cv::vector<cv::Point>> listcontours;
	//parametres of color based detection
	float alpha;
	float beta;
	float th_s;
	float th_h;
	//parametres of gradient based detection
	float th_m;
	float th_a;
	float th_c;

};

