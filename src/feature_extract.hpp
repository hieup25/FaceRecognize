#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "comm_lib.hpp"
#include "buffer_file.hpp"
#include "Config.hpp"
#include <unistd.h>

using namespace mxnet::cpp;
//compare input features with labeled features, get classification result with minimal distance and class index
struct class_info
{
	double min_distance;
	int index;
};

class_info classify(const cv::Mat& img, const  cv::Mat& cmp)
{
	int rows = cmp.rows;
	cv::Mat broad;
	cv::repeat(img, rows, 1, broad);

	broad = broad - cmp;
	cv::pow(broad,2,broad);
	cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

	double dis;
	cv::Point point;
	cv::minMaxLoc(broad, &dis, 0, &point, 0);

	return class_info{dis, point.y};
}


class Mxnet_extract 
{
public:
	~Mxnet_extract()
	{
		if(pred_feature)
		    MXPredFree(pred_feature);
	}
	int LoadModel(const std::string & fname, std::vector<char>& buf)
	{
		std::ifstream fs(fname, std::ios::binary | std::ios::in);
		if (!fs.good())
		{
			std::cerr << fname << " does not exist" << std::endl;
			return -1;
		}
		fs.seekg(0, std::ios::end);
		int fsize = fs.tellg();
		fs.seekg(0, std::ios::beg);
		buf.resize(fsize);
		fs.read(buf.data(), fsize);
		fs.close();
		return 0;
	}
	int LoadExtractModule(const std::string& param_file, const std::string& json_file, int batch, int channel, int input_h, int input_w)
	{

		std::vector<char> param_buffer;
		std::vector<char> json_buffer;

		if (LoadModel(param_file, param_buffer)<0)
			return -1;

		if (LoadModel(json_file, json_buffer)<0)
			return -1;

		int device_type = 1;
		int dev_id = 0;
		mx_uint  num_input_nodes = 1;
		const char * input_keys[1];
		const mx_uint input_shape_indptr[] = { 0, 4 };
		const mx_uint input_shape_data[] = {
			static_cast<mx_uint>(batch),
			static_cast<mx_uint>(channel),
			static_cast<mx_uint>(input_h),
			static_cast<mx_uint>(input_w)
		};

		input_keys[0] = "data";

		int ret = MXPredCreate(json_buffer.data(),
			param_buffer.data(),
			param_buffer.size(),
			device_type,
			dev_id,
			num_input_nodes,
			input_keys,
			input_shape_indptr,
			input_shape_data,
			&pred_feature
		);
		
		return ret;
	}
	cv::Mat extractFeature(const cv::Mat& img)
	{
		int width = img.cols;
		int height = img.rows;
        cv::Mat img_rgb(height, width, CV_32FC3);
		img.convertTo(img_rgb, CV_32FC3);
		cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);
		std::vector<float> input(3 * height * width);
		std::vector<cv::Mat> input_channels;
		set_input_buffer(input_channels, input.data(), height, width);
		cv::split(img_rgb, input_channels);
		MXPredSetInput(pred_feature, "data", input.data(), input.size());
		MXPredForward(pred_feature);
		mx_uint *shape = NULL;
		mx_uint shape_len = 0;
		MXPredGetOutputShape(pred_feature, 0, &shape, &shape_len);
		int feature_size = 1;
		for (unsigned int i = 0;i<shape_len;i++)
			feature_size *= shape[i];
		std::vector<float> feature(feature_size);
		MXPredGetOutput(pred_feature, 0, feature.data(), feature_size);
		cv::Mat output = cv::Mat(feature, true).reshape(1, 1);
		cv::normalize(output, output);
		return output;
	}
private:
	PredictorHandle pred_feature;
};

void display(cv::Mat& img_out, int __index = -1, std::string path_image = "")
{
	//add background
	cv::Mat bg = cv::imread("../data/images/bg.jpeg");
	cv::resize(bg, bg, cv::Size(X_PADDING, H_RESIZE));
	//add logo
	cv::Mat logo = cv::imread("../data/images/epu_logo1.png", -1);
	/**/
	cv::Mat mask1;
	std::vector<cv::Mat> ch;
	cv::split(logo, ch);
	if (logo.channels() == 4)
	{
		cv::split(logo, ch);
		cv::Mat cs[3] = {ch[0], ch[1], ch[2]};
		cv::merge(cs, 3, logo);
		mask1 = ch[3];
	}
	/**/
	logo.copyTo(bg(cv::Rect(35, 10, logo.cols, logo.rows)), mask1);
	/**/
	//add strokes id student
    if(__index != -1)
	{
		std::string add = path_image;
		add.replace(3, 5, "avatar");
std::cout << "\n\n\n" << add << std::endl;
		cv::Mat images_display = cv::imread(add);
		cv::resize(images_display, images_display, cv::Size(X_PADDING - 70, X_PADDING - 70));
	    	images_display.copyTo(bg(cv::Rect(35, logo.rows + 30, X_PADDING - 70, X_PADDING - 70)));
	}
	else
	{
		cv::Mat strokes = cv::imread("../data/images/unknow.png");
		cv::resize(strokes, strokes, cv::Size(X_PADDING - 70, X_PADDING - 70));
		strokes.copyTo(bg(cv::Rect(35, logo.rows + 30, X_PADDING - 70, X_PADDING - 70)));
	}
	//add outline
	cv::rectangle(bg, cv::Rect(0, 0, bg.cols, bg.rows), CV_RGB(0, 0, 255), 3);
	bg.copyTo(img_out(cv::Rect(0, 0, bg.cols, bg.rows)));
}
void draw_focus(cv::Mat& in, cv::Rect& mask)
{
	/*
	1	2
	4	3
	*/
	cv::Point P1_1, P1_2, P1_3;
	uint32_t PAD = 50;
	//1
	P1_1.x = W_RESIZE/4;
	P1_1.y = H_RESIZE/4 - PAD;
	P1_2.x = P1_1.x + 10;
	P1_2.y = P1_1.y;
	P1_3.y = P1_1.y + 10;
	P1_3.x = P1_1.x;
	cv::line(in, P1_1, P1_2,CV_RGB(0,255,0) ,2);
	cv::line(in, P1_1, P1_3,CV_RGB(0,255,0) ,2);
	mask.x = P1_1.x;
	mask.y = P1_1.y;
	//2
	P1_1.x = W_RESIZE - (W_RESIZE/4);
	P1_1.y = H_RESIZE/4 - PAD;
	P1_2.x = P1_1.x - 10;
	P1_2.y = P1_1.y;
	P1_3.y = P1_1.y + 10;
	P1_3.x = P1_1.x;
	cv::line(in, P1_1, P1_2,CV_RGB(0,255,0) ,2);
	cv::line(in, P1_1, P1_3,CV_RGB(0,255,0) ,2);
	mask.width = P1_1.x - (W_RESIZE/4);
	//3
	P1_1.x = W_RESIZE - (W_RESIZE/4);
	P1_1.y = (H_RESIZE - (H_RESIZE/4)) + PAD;
	P1_2.x = P1_1.x - 10;
	P1_2.y = P1_1.y;
	P1_3.y = P1_1.y - 10;
	P1_3.x = P1_1.x;
	cv::line(in, P1_1, P1_2,CV_RGB(0,255,0) ,2);
	cv::line(in, P1_1, P1_3,CV_RGB(0,255,0) ,2);
	mask.height = (P1_1.y - (H_RESIZE/4)) + PAD;
	//4
	P1_1.x = W_RESIZE/4;
	P1_1.y = (H_RESIZE - (H_RESIZE/4)) + PAD;
	P1_2.x = P1_1.x + 10;
	P1_2.y = P1_1.y;
	P1_3.y = P1_1.y - 10;
	P1_3.x = P1_1.x;
	cv::line(in, P1_1, P1_2,CV_RGB(0,255,0) ,2);
	cv::line(in, P1_1, P1_3,CV_RGB(0,255,0) ,2);
}
void writeFile(std::string path, std::string content, bool append = true)
{
	std::ofstream out;
	out.open(path, std::ios_base::app);
	out << content;
	out.close();
}
void run_recognition(MxNetMtcnn& mtcnn, Mxnet_extract& extract, 
					 cv::Mat& img_in, cv::Mat& img_out, cv::Mat& data,
					 V_STR_T labels, V_STR_T _class, char* id, char* time_label)
{
	cv::Mat src(5, 2, CV_32FC1, norm_face);
	std::vector<face_box> face_info;
	//detect with mtcnn, get detect reuslt with bounding box and landmark point
	cv::resize(img_in, img_in, cv::Size(W_RESIZE, H_RESIZE));
	cv::Rect mask;
	draw_focus(img_in, mask);
	//
	cv::Mat fmask;
	fmask = img_in(mask);
	cv::resize(fmask, fmask, cv::Size(112, 112));
	//
	mtcnn.Detect(fmask, face_info);
	//
	cv::Point _ptime_(10, img_in.rows-15);
	cv::putText(img_in, time_label, _ptime_, cv::FONT_ITALIC, 0.75, CV_RGB(255,255,0), 2);
	/*check_display*/
	if (face_info.size() < 1)
	{
		display(img_out);
	}

	for (int i = 0; i < face_info.size(); ++i)
	{
		NDArray::WaitAll();	
	    face_box face = face_info[i];
		float v2[5][2] =
			{ { face.landmark.x[0] , face.landmark.y[0] },
			{ face.landmark.x[1] , face.landmark.y[1] },
			{ face.landmark.x[2] , face.landmark.y[2] },
			{ face.landmark.x[3] , face.landmark.y[3] },
			{ face.landmark.x[4] , face.landmark.y[4] }};

		cv::Mat dst(5, 2, CV_32FC1, v2);

		//do similar transformation according normal face
		cv::Mat m = similarTransform(dst, src);

		cv::Mat aligned(112, 112, CV_32FC3);
		cv::Size size(112, 112);

		//get aligned face with transformed matrix and resize to 112*112
		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));		
		// cv::warpAffine(img_in, aligned, transfer, size, 1, 0, 0);
		cv::warpAffine(fmask, aligned, transfer, size, 1, 0, 0);

		//extract feature from aligned face and do classification with labels 
		cv::Mat output = extract.extractFeature(aligned);
        class_info result = classify(output, data);

		//draw landmark points
		for (int j = 0; j < 5; j++)
		{
			cv::Point p(face.landmark.x[j], face.landmark.y[j]);
			cv::circle(img_out, p, 2, cv::Scalar(0, 0, 255), -1);
		}
		cv::Point pt1(face.x0, face.y0);
		cv::Point pt2(face.x1, face.y1);
		cv::Point pt3(face.x0 + 400 , face.y1 + 380);
        // check distancing
		if (result.min_distance <= 1)
		{
			strcpy(id, labels[result.index].c_str());
			std::string path_show_image = "../image/" + labels[result.index] + "@" + _class[result.index] + ".jpg";
		        display(img_out, result.index, path_show_image);
			cv::putText(img_out, "Ten: " + labels[result.index], cv::Point(30, img_out.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0));
			cv::putText(img_out, "Lop: " + _class[result.index], cv::Point(30, img_out.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0));
			std::string result2 = "Hoc sinh: "  + labels[result.index] + " da diem danh"  + "\t Luc: " +  time_label + "\n";
			writeFile("Ketqua.txt", result2);
		}
		else
		{
			display(img_out);
			cv::putText(img_out, "Chu y: ", cv::Point(30, img_out.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0));
			cv::putText(img_out, "Can chinh khoang cach", cv::Point(105, img_out.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0));
			cv::putText(img_out, "Khuon mat", cv::Point(130, img_out.rows - 45), cv::FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0,0,0));
		}
	}
	// nhan dien mat khung hinh o giua
	img_in.copyTo(img_out(cv::Rect(X_PADDING, Y_PADDING, W_RESIZE, H_RESIZE)));
}
