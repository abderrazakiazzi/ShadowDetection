#include "ShadowDetection.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <vector>

ShadowDetection::ShadowDetection()
{	
}

ShadowDetection::~ShadowDetection()
{
}

void ShadowDetection::loadParameters(){

	//parametres of color based detection
	alpha = 0.50 /*0.50 /*0.21*/;
	beta = 0.93 /*0.93 /*0.99*/;
	th_s = 76;
	th_h = 93;
	//parametres of gradient based detection
	th_m = 6;
	th_a = 0.314/10.0;
	th_c = 0.2;
}

void ShadowDetection::findNonzerosPixel(cv::Mat image){

		  cv::Mat listpointnonzeros;
		 cv::findNonZero(image, listpointnonzeros);
		 int list = listpointnonzeros.total();
		 for(int ii = 0; ii < list; ii++)
			 this->listnonzeros.push_back(listpointnonzeros.at<cv::Point>(ii));
}

void ShadowDetection::DisplayDetectPix(cv::Mat& image){
	for(int k = 0; k < listnonzeros.size(); k++)
	{
	 image.at<cv::Vec3b>(listnonzeros[k]) = 255;
	}

}
// Apply Detection of shadow by HSV color and Gradient  
void ShadowDetection::detectShadow(cv::Mat inputCurrent, cv::Mat inputReference, cv::Mat& outputMask){

	// Apply HSV colcor shadow detection
	detectShadowHSVColor(inputCurrent, inputReference, outputMask);

	//Apply morphological operation
	cv::Mat kernel1 = cv::getStructuringElement(CV_SHAPE_CROSS, cv::Size(3,3));
	cv::dilate(outputMask, outputMask, kernel1);
	cv::erode(outputMask, outputMask, kernel1, cvPoint(-1,-1), 2);
	outputMask = postprocesShadows(outputMask);
	//cv::imshow(" mask final", outputMask);
	
	
	// remove shadow region by gradient operation
	cv::Mat imgcurr = cv::Mat(inputReference.size(), CV_8U);
    cv::Mat imgref = cv::Mat(inputReference.size(), CV_8U);
	inputCurrent.copyTo(imgcurr, outputMask);
	inputReference.copyTo(imgref, outputMask);
	this->findNonzerosPixel(outputMask);
	std::cout <<  " size of nozeros " << this->listnonzeros.size() <<std::endl;

	//  Apply shadows detection using gradient    
	// Message de iazzi : cette methode contient encore des problemes. ils contients des erreurs, essayer de les corrigées 
	/* if(this->listcontours.size() > 0)		
	    detectShadowGradiant(imgcurr, imgref, outputMask, imageFinal);
	this->listnonzeros.clear();*/
}

    // apply detection shadow by HSV color 
void ShadowDetection::detectShadowHSVColor(cv::Mat imageCurrent, cv::Mat imageReference, cv::Mat& imageResultMask){
	bool a = satifieRules(imageCurrent, imageReference,  imageResultMask);
	//cv::imshow(" mask final", imageResultMask);
}

void ShadowDetection::gradientmagnitude(cv::Mat Xdirect, cv::Mat Ydirect, cv::Mat& outputmagn){
      magnitude(Xdirect, Ydirect, outputmagn);
	}
void ShadowDetection::gradientOrientDirect(cv::Mat Xdirect, cv::Mat Ydirect, cv::Mat& outputorient){
	phase(Xdirect, Ydirect, outputorient, true);
}
void ShadowDetection::gradientSobel(cv::Mat image, cv::Mat& diretimage, int a){
	//Mat Sx;
	if(a == 1)
      Sobel(image, diretimage, CV_32F, 1, 0, 3);
    if(a == 0)
    //Mat Sy;
      Sobel(image, diretimage, CV_32F, 0, 1, 3);
	if(a!= 0 && a!= 1)
		std::cout << " error , try to use third input as 1 fo overtical gradient or 0 for orizontal gradient " << std::endl; 
}

// Compute diffrenc gradient direction  C and B 
void ShadowDetection::gradientdiffer(cv::Mat bx, cv::Mat by, cv::Mat cx, cv::Mat cy, cv::Mat& out){	
	 //std::cout << " size of listnonzeros " << listnonzeros.size() <<  std:: endl;
	 for(int k = 0; k < listnonzeros.size(); k++)
		       {   
				   double bcx = bx.at<double>(listnonzeros[k]) * cx.at<double>(listnonzeros[k]);
				   double bcy = by.at<double>(listnonzeros[k]) * cy.at<double>(listnonzeros[k]);
				   //std::cout << " bcx and bcy  " << bcx <<  "  " << bcy <<  "   " << listnonzeros[k] <<std:: endl;
				   double sumbcxy = bcx + bcy;
				   double bx2 = std::pow(bx.at<double>(listnonzeros[k]), 2), by2 = std::pow(bx.at<double>(listnonzeros[k]), 2);
				   double sumbx2by2 = bx2 + by2;
				   double cx2 = std::pow(cx.at<double>(listnonzeros[k]), 2), cy2 = std::pow(cx.at<double>(listnonzeros[k]), 2);
				   double sumcx2cy2 = cx2 + cy2;
				   double produitBC2 = sumbx2by2 * sumcx2cy2;
				   double sqrtproduitBC2 = std::sqrt(produitBC2);

				   double divbcxy = (sqrtproduitBC2 != 0 ? sumbcxy/sqrtproduitBC2:0);
				   double pixdiff = std::acos(divbcxy);	
				   out.at<double>(listnonzeros[k]) = pixdiff;			   
				   //std::cout << " result of a/b " << divbcxy << std::endl;
		           //std::cout << " result of acos(a/b) " << pixdiff << std::endl;
	           }


}

// Apply shadow gradient detection
void ShadowDetection::detectShadowGradiant(cv::Mat& imageBack, cv::Mat& imageCurrent, cv::Mat& imageResult, cv::Mat& imageFinal){

	// reference image
	cv::Mat BX /*= cv::Mat(imageBack.size(), CV_32F)*/, BY /*= cv::Mat(imageBack.size(), CV_32F)*/, ORB;
	cv::Mat  gradmanB = cv::Mat(imageBack.size(), CV_64F);
	cv::Mat  gradmanC = cv::Mat(imageBack.size(), CV_64F);
	cv::Mat maskB = cv::Mat::zeros(imageBack.size(),CV_64F);
	cv::Mat maskC = cv::Mat::zeros(imageBack.size(), CV_64F);

	//CV_Assert( lImg.type() == CV_64FC3 && rImg.type() == CV_64FC3 );
	int hei = imageBack.rows;
	int wid = imageBack.cols;
	cv::Mat lGray;
	cv::Mat lGrdX, lGrdY;
	cv::Mat cGray;
	cv::Mat cGrdX, cGrdY;
	cv::Mat tmp, tmp2;
	imageBack.convertTo( tmp, CV_64F );
	imageCurrent.convertTo( tmp2, CV_64F );
	cvtColor( tmp, lGray, CV_RGB2GRAY );
	cvtColor( tmp2, cGray, CV_RGB2GRAY );
	// sobel size must be 1
	Sobel( lGray, lGrdX, CV_64F, 1, 0, 1 );
	Sobel( lGray, lGrdY, CV_64F, 0, 1, 1 );

	Sobel( cGray, cGrdX, CV_64F, 1, 0, 1 );
	Sobel( cGray, cGrdY, CV_64F, 0, 1, 1 );
	// eliminate negativ values
	lGrdX += 0.5;
	lGrdY += 0.5;

	cGrdX += 0.5;
	cGrdY += 0.5;
// convert to uchar 0>>> 255
    //lGrdX.convertTo( lGrdX, CV_8U, 255 );
	//lGrdY.convertTo( lGrdY, CV_8U, 255 );
// compute magnitude;
	gradientmagnitude(lGrdX, lGrdY, gradmanB);
	thresholdMagnitude(gradmanB, maskB);
	gradientmagnitude(cGrdX, cGrdY, gradmanC);
	thresholdMagnitude(gradmanC, maskC);

    cv::Mat out = cv::Mat::zeros(lGrdX.size(), CV_64F);	
	gradientdiffer(lGrdX, lGrdY, cGrdX, cGrdY, out);

	//out.convertTo(out, CV_8U, 255);
	//cv::imshow(" result of gd direction", out);

	cv::Mat outRmc = cv::Mat(out.size(), CV_8U);
	gradientCorrelation(out, outRmc);

	//cv::imshow("resultat of correlation", outRmc);
}

void ShadowDetection::thresholdMagnitude(cv::Mat outRmC, cv::Mat& mask){
	
	//cv::normalize(outRmC, outRmC, 1, CV_MINMAX);
	 for(int i = 0; i < outRmC.size().height; i++)
		 for(int j = 0; j < outRmC.size().width; j++)
	 {
		double  pix = outRmC.at<double>(i,j);
	    if(std::abs(pix) <= th_m)
			outRmC.at<double>(i,j) = 0;
	 }
	 
}
// compute correlation of gradirent direction 
void ShadowDetection::gradientCorrelation(cv::Mat Rm, cv::Mat& outRmC){

		cv::vector<cv::vector<cv::Point>> listcontours2;

		for(int j = 0; j < listcontours.size(); j++)
			{
				double ar = cv::contourArea(listcontours[j]);
				int nb = 0;
				for(int i = 0; i < listnonzeros.size(); i++)
				 { 
					 cv::Point pt = listnonzeros[i];
					 double dist = cv::pointPolygonTest(listcontours[j], cv::Point2f(pt), true );
				     if(dist <= 0 )
				      { 
						double pix = Rm.at<double>(pt);
					   double val = th_a -  pix ;
				       if( val >= 0 )
					      { nb ++;}
					   
				     }
				}
				if(nb/ar >= 0.2)
					listcontours2.push_back(listcontours[j]);
	       }
		listcontours.clear();
	    listcontours = listcontours2;

		for(int p = 0; p<listcontours.size(); p++)
			cv::drawContours(outRmC, listcontours, p, cv::Scalar(255), -1,8);
	}
	



double ShadowDetection::maxMatrice(cv::Mat image){
  double tmp = image.at<uchar>(0,0);
 
  for (int i = 0; i < image.rows; i++)
	  for(int j = 0; j < image.cols; j++)
    if (image.at<uchar>(i, j) > tmp) 
		tmp = image.at<uchar>(i, j);
  return tmp;
  
}

double ShadowDetection::minMatrice(cv::Mat image){
  double tmp = image.at<uchar>(0,0);
 
  for (int i = 0; i < image.rows; i++)
	  for(int j = 0; j < image.cols; j++)
    if (image.at<uchar>(i, j) < tmp) 
		tmp = image.at<uchar>(i, j);
  return tmp;
}


void ShadowDetection::NormalizeHSV(cv::Mat& imageHSV, cv::Mat& imageHSVNormalise){
	
	   // channels[2] =   channels[2];
	   double minMat = minMatrice(imageHSV);
	   double maxMat = maxMatrice(imageHSV);



	   for(int i=0; i < imageHSV.size().height; i++)
		   for(int j = 0; j < imageHSV.size().width; j++)
		   {
		    double pix = imageHSV.at<uchar>(i,j);
			pix = (pix -  minMat) / (maxMat - minMat);
			imageHSVNormalise.at<double>(i,j) = pix;
		   }


		   //std::cout << " maxnVal = " << maxMat << std::endl;
			/*double* minVal = 0 ;
			double* maxVal = 0;
           minMaxLoc(imageHSV, minVal, maxVal);
	      
		   std::cout << " maxnVal = " << maxVal << std::endl;
		   cv::normalize(imageHSV, imageHSVNormalise, 0, 1, cv::NORM_MINMAX);*/
		   //imshow("hhhhh", imageHSVNormalise);
}
	 
bool ShadowDetection::satifieRules(cv::Mat imageCurrent, cv::Mat imageBack, cv::Mat& imageResultMask){
	// convert bgr to hsv;
		cv::Mat BimageHSV, CimageHSV, CimageHSVNormalise, BimageHSVNormalise;
		cv::cvtColor(imageCurrent, BimageHSV, CV_RGB2HSV);
	    cv::cvtColor(imageBack, CimageHSV, CV_RGB2HSV);
	// Normaliser image HSV
		CimageHSVNormalise = cv::Mat(CimageHSV.size(), CV_64F);
		BimageHSVNormalise = cv::Mat(CimageHSV.size(), CV_64F);
	   cv::vector<cv::Mat> channels_C;
	   cv::vector<cv::Mat> channels_B;
	   cv::split(BimageHSV, channels_B);
	   cv::split(CimageHSV, channels_C);	   
	   NormalizeHSV(channels_B[2], BimageHSVNormalise);
       NormalizeHSV(channels_C[2], CimageHSVNormalise);
	   for(int i=0; i < CimageHSVNormalise.size().height; i++)
		   for(int j = 0; j < CimageHSVNormalise.size().width; j++)
	       {
			double pixCV = CimageHSVNormalise.at<double>(i, j);
			double pixBV = BimageHSVNormalise.at<double>(i, j);
			cv::Vec3b pixC = CimageHSV.at<cv::Vec3b >(i, j);
			cv::Vec3b  pixB = BimageHSV.at<cv::Vec3b >(i, j);
			double Ch = pixC.val[0], Bh = pixB.val[0];
			double Cs = pixC.val[1], Bs = pixB.val[1];
			double Cv = pixCV, Bv = pixBV;
		    //  test rules 
			int r1 =   ((Cv/Bv >= this-> alpha ? 1 :0) && (Cv/Bv <= this->beta ? 1 : 0 )) ;
			int r2 =   ((Cs - Bs) <= th_s ? 1 : 0);
			int r3 = (std::abs(Ch - Bh) <= th_h ? 1 : 0 );
			if(r1 && r2 && r3)
			   imageResultMask.at<uchar>(i,j) = 255;		     
			else
				imageResultMask.at<uchar>(i,j) =  0;
	       }
		  // cv::imshow("mask shadow", imageResultMask);
		   if(cv::countNonZero(imageResultMask) > 0)
			   return true;
		   else
			   return false;
	   }	
cv::Mat ShadowDetection::postprocesShadows(cv::Mat image){	
	 cv::Mat result = cv::Mat::zeros(image.size(), CV_8U); 
	 image.copyTo(result);
	 cv::vector<cv::vector<cv::Point>> contour;
	 cv::vector<cv::Vec4i> hierarchy;
	 cv::findContours(image, contour, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
	 for(int i = 0; i < contour.size(); i++)
	 {
		 if(cv::contourArea(contour[i]) < 50.0)
			 cv::drawContours(result, contour, i, cv::Scalar(0), -1,  8, hierarchy, 0, cv::Point());
	 }
	 //imshow(" shdows", result);
   return result;
}


void ShadowDetection::DisplayContours(cv::Mat& image){
	for(int i = 0; i < this->listcontours.size(); i++)
	{
		cv::vector<cv::Point> ct = listcontours[i];
		cv::Rect  a  = cv::boundingRect(ct);
		std::ostringstream str;
		str << i;
		cv::string text = "" + str.str();
		cv::Point center = cv::Point(a.x+a.height/2, a.y+a.width/2);		
		cv::drawContours(image, listcontours, i, cv::Scalar(255, 100, 100), 2); 
		cv::putText(image, text, center, CV_FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(255,0,0), 2,8, 0);		
	}
	//cv::imshow("Result of cout", image);
}