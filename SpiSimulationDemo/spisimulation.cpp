#include "spisimulaiton.h"

//参数设置
unsigned N = 128;
char imgFilePath[] = "E:\\Programming\\MATLAB\\SPI_Simulation\\images\\im_lena512.jpg";

cv::Mat GetImage(char* imgPath) {
	cv::Mat img = cv::imread(imgPath);
	cv::Mat output_img;
	cv::resize(img, output_img, cv::Size(N, N));
	cv::cvtColor(output_img, output_img, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	//cv::normalize(output_img, output_img, 0.0, 1.0, cv::NormTypes::NORM_MINMAX);
	output_img.convertTo(output_img, CV_64F, 1.0 / 255.0);
	return output_img;
}
Mat4Step GetPattern(unsigned x, unsigned y) {
	Mat4Step patterns(N, N, CV_64F);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++){
			//If matrix is of type CV_64F then use Mat.at<double>(y,x).
			patterns.Mat1.at<double>(j, i) =
				0.5 + 0.5 * cos(2 * PI * x * i / N + 2 * PI * y * j / N);
			patterns.Mat2.at<double>(j, i) =
				0.5 + 0.5 * cos(2 * PI * x * i / N + 2 * PI * y * j / N + PI / 2);
			patterns.Mat3.at<double>(j, i) =
				0.5 + 0.5 * cos(2 * PI * x * i / N + 2 * PI * y * j / N + PI);
			patterns.Mat4.at<double>(j, i) =
				0.5 + 0.5 * cos(2 * PI * x * i / N + 2 * PI * y * j / N + PI * 3 / 2);
		}
	return patterns;
}
int main() {
	cv::Mat img = GetImage(imgFilePath);
	//cv::normalize(img, img, 0, 255, cv::NormTypes::NORM_MINMAX);
	Mat4Step output4Step(N, N, CV_64F);
	clock_t clk_begin = clock();
	for (int x = 0; x < N; x++)
		for (int y = 0; y < N; y++) {
			Mat4Step patterns = GetPattern(x, y);
			output4Step.Mat1.at<double>(y, x) = cv::sum(img.mul(patterns.Mat1))[0];
			output4Step.Mat2.at<double>(y, x) = cv::sum(img.mul(patterns.Mat2))[0];
			output4Step.Mat3.at<double>(y, x) = cv::sum(img.mul(patterns.Mat3))[0];
			output4Step.Mat4.at<double>(y, x) = cv::sum(img.mul(patterns.Mat4))[0];
		}
	cv::Mat output(N, N, CV_64FC2);
	cv::Mat outputs[] = {
		output4Step.Mat1 - output4Step.Mat3,output4Step.Mat2 - output4Step.Mat4 };
	cv::merge(outputs, 2, output);
	cv::Mat rebuild;
	cv::idft(output, rebuild);
	cv::extractChannel(rebuild, rebuild, 0);
	cv::normalize(rebuild, rebuild, 0, 1, cv::NormTypes::NORM_MINMAX);
	clock_t clk_end = clock();
	std::cout << "用时" << double(clk_end - clk_begin) / CLOCKS_PER_SEC;
	std::cout << "秒" << std::endl;
	//return -1;
	cv::namedWindow("Rebuild Image", 0);
	cv::resizeWindow("Rebuild Image", 768, 768);
	cv::imshow("Rebuild Image", rebuild);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}