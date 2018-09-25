#include <stdio.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "ShadowDetection.h"

void StartAffiche(){
	std::cout << " *****    iazzi@abderrazak- @LRIT - @RIITM - @FS - @UM5 - @Rabat ************** " << std::endl;

	std::cout << " // Shadow detection  " << std::endl;
	std::cout << " // HSV and gradient methods" << std::endl;
	std::cout << " // "<< std::endl;
	std::cout << " // "<< std::endl;
	std::cout << " // "<< std::endl;
	std::cout << " // "<< std::endl;
	std::cout << " // "<< std::endl;
	std::cout << " "<< std::endl;
	std::cout << " "<< std::endl;
	std::cout << " "<< std::endl;
	/*std::cout << " Lancement du programme  version " << _MSC_VER << std::endl;*/
	 
}

void readfilevideo(const cv::string& file){
		cv::VideoCapture cap;
		//CvCapture* capt;
		//const cv::string& file2 = "video1.mpg";
		//capt = cvCreateFileCapture(file);
	    if(file.size() == 0)
		const cv::string& file = "F:/myvideo.avi";	

		cap.open(file);
	    if(!cap.isOpened())
			return;
		cv::Mat frame;
		int i = 0;
		for(;;)
			{
              cap >> frame;
			  if(frame.empty())
				break;
			  std::ostringstream str;
			  str << i;
			  cv::string& name = "shadowimages/frame" + str.str() + ".jpg";
			  cv::imwrite(name, frame);
			  i++;
			  cv::waitKey(25);
			}
}
//-----------------------------------------------------------------


int main(){

    // Debut  du programme	
	StartAffiche();
    // debut de programme	
	ShadowDetection obj = ShadowDetection();

	// read two images, one is without shadows, other is with shadow
	cv::Mat imageRef = cv::imread("imageReference.jpg");
	cv::Mat imageCurr = cv::imread("imageCurrant.jpg");
	cv::Mat imagChro = cv::Mat(imageCurr.size(), imageCurr.depth());
	imageCurr.copyTo(imagChro);

	// shadow detection 
	cv::Mat res = cv::Mat(imageRef.size(), CV_8UC1);
	obj.loadParameters();
	obj.detectShadow(imageCurr, imageRef, res);
        obj.findNonzerosPixel(res);
        obj.DisplayDetectPix(imageCurr);

	    /*
	    // HSV color step 
	    cv::Mat res = cv::Mat(imageRef.size(), CV_8UC1);
		obj.loadParameters();
	    obj.detectShadowHSVColor(imageCurr, imageRef, res);
	    obj.findNonzerosPixel(res);
	    obj.DisplayDetectPix(imageCurr);
	    // Gradient step
		cv::Mat imresC, imresB;
		cv::Mat res2 = cv::Mat(imageRef.size(), CV_8UC1);
		imageRef.copyTo(imresB, res);
		imageCurr.copyTo(imresC, res);
		obj.detectShadowGradiant(imresB, imresC, res2, res2);	
		cv::Mat ch = obj.chromacity(imagChro, 255, 78, "adaptive", 0, false);
		imshow("chromacity", ch);*/

	// Display results
	cv::imshow(" image curr", imageCurr);
	cv::imshow("Shadow detection", res);
	//cv::imshow("Shadow detection2", res2);
	
	
	cv::waitKey(0);
	system("pause");
 return 0;

}
