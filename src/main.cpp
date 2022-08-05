#include<stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>
#include <math.h>
#include "face_align.hpp"
#include "mxnet_mtcnn.hpp"
#include "feature_extract.hpp"
#include "make_label.hpp"
#include "comm_lib.hpp"
#include <algorithm>
#include <unistd.h>
#include <time.h>
#include <bits/stdc++.h>
#include <regex>

void make_student(std::string path, std::vector<std::string>& _name, std::vector<std::string>& _class)
{
	std::ifstream file(path);
	if (file)
	{
		std::stringstream ss;
		ss << file.rdbuf();
		std::string word;
		while (ss >> word) {
			
				int index = word.find('@');
				_name.push_back(word.substr(0, index));
				_class.push_back(word.substr(index+1));
			
		}
		file.close();
	}
	for (auto& n : _name)
		std::cout << n << std::endl;
	for (auto& n : _class)
		std::cout << n << std::endl;
}
void test_make_label(std::string path)
{
	std::vector<std::string> imagePath;
	std::vector<std::string> imageLabel;

	getFiles(path, imagePath, imageLabel);
	make_label(imagePath, imageLabel, "../mtcnn_model", "../feature_model/model-0000.params", "../feature_model/model-symbol.json");
}
void test_camera()
{
	int __index = -1 ;
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule("../mtcnn_model");
	Mxnet_extract extract;
	extract.LoadExtractModule("../feature_model/model-0000.params", "../feature_model/model-symbol.json", 1, 3, 112, 112);

	//loading features
	cv::FileStorage fs("../features.xml", cv::FileStorage::READ);
	cv::Mat features;
	fs["features"] >> features;

	//Load list name student
	std::vector<std::string> _name;
	std::vector<std::string> _class;
	make_student("../labels.txt", _name, _class);
	//Setup
	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) return;
	char ID[50] = {0};
	// //Setup images
	cv::Mat result(cv::Size(W_PADDING, H_PADDING), CV_8UC3, CV_RGB(255,255,255));
	display(result, __index);
	
	while (1)
	{
		cv::Mat frame;
		/*Get Frame*/
		cap >> frame;	
		if (frame.empty()) 
		{
			std::cout << "Frame empty" << std::endl;
			break;
		}
		memset(ID, 0 , 30);

		/*Get Time*/
		time_t 	t = time(NULL);
		struct tm tm = *localtime(&t);
		char timer[70];
		sprintf(timer, "Time: %02d:%02d %02d/%d/%02d", tm.tm_hour, tm.tm_min, tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900);
		run_recognition(mtcnn, extract, frame, result, features, _name, _class, ID, timer);
		/*Check student ?*/
		if (strlen(ID) == 0)
		{
			//std::cout << "face-not-detect" << std::endl;
			cv::imshow("HieuPV - Program", result);
			cv::waitKey(1);	
			continue;
		}
		cv::imshow("HieuPV - Program", result);
		cv::waitKey(1);
		usleep(10);
	}
	cap.release();
}

int main(int argc, char* argv[]) 
{
	system("rm Ketqua.txt");
	if (argc > 1)
	{
		std::cout << argv[1] << std::endl;
		if (strcmp(argv[1], "get") == 0)
		{
			std::string path = "/home/pi/DOAN/FR/image";
			test_make_label(path);
		}
	}
	test_camera();
	return 0;
}
