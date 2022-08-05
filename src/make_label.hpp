#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "mxnet_mtcnn.hpp"
#include "feature_extract.hpp"

void getFiles(std::string path, std::vector<std::string>& files, std::vector<std::string> &ownname)
{
    DIR* dir = opendir(path.c_str());
    dirent* p = NULL;
    while((p = readdir(dir)) != NULL)
    {
        if(p->d_name[0] != '.')
        {
            std::string name = path + "/" + std::string(p->d_name);
            ownname.push_back(std::string(p->d_name));
            files.push_back(name);
            std::cout<<"name: "<<p->d_name<<", file:"<<name<<std::endl;
        }
    }
    closedir(dir);
}

void make_label(std::vector<std::string>images, std::vector<std::string>labels, std::string mtcnn_model, std::string params, std::string json)
{
	cv::Mat features;
	cv::Mat names;

	MxNetMtcnn mtcnn;
	mtcnn.LoadModule(mtcnn_model);
	Mxnet_extract extract;
	extract.LoadExtractModule(params, json, 1, 3, 112, 112);

	cv::Mat src(5, 2, CV_32FC1, norm_face);

	std::ofstream newfile("../labels.txt");

	for (int i = 0; i < images.size(); i++)
	{
		std::cout << "extract feature for image: " << images[i] << std::endl;
		std::vector<std::string> tmp;
		SplitString(labels[i], tmp, ".");

		std::cout << tmp[0] << " ";
		newfile << tmp[0] << " ";

		cv::Mat img = cv::imread(images[i]);

		std::vector<face_box> face_info;
		mtcnn.Detect(img, face_info);

		if (face_info.size() > 1)
		{
			std::cout << "more than 1 face in image: " << images[i] << std::endl;
		}

		if (face_info.empty())
		{
			std::cout << "no face in image: " << images[i] << std::endl;
		}

		face_box face= face_info[0];

		float v2[5][2] =
		{ { face.landmark.x[0] , face.landmark.y[0] },
			{ face.landmark.x[1] , face.landmark.y[1] },
			{ face.landmark.x[2] , face.landmark.y[2] },
			{ face.landmark.x[3] , face.landmark.y[3] },
			{ face.landmark.x[4] , face.landmark.y[4] } };

		cv::Mat dst(5, 2, CV_32FC1, v2);
		cv::Mat m = similarTransform(dst, src);

		cv::Mat aligned(112,112,CV_32FC3);
		cv::Size size(112, 112);

		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
		std::cout << m << std::endl;
		cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);

		cv::Mat output = extract.extractFeature(aligned);
		features.push_back(output);
		std::vector<int> com;
		com.push_back(cv::IMWRITE_JPEG_QUALITY);
		com.push_back(30);
		std::string add = images[i];
		add.replace(17, 5, "avatar");
std::cout << add << std::endl;
		cv::imwrite(add, img, com);
	}

	cv::FileStorage fs("../features.xml", cv::FileStorage::WRITE);
	fs << "features" << features;
	fs.release();

}

