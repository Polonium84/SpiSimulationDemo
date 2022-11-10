#include "spisimulaiton.h"

//参数设置
unsigned N = 128;//此设置无实际意义，分辨率由运行时手动输入
const char imgFilePath[] = "E:\\Programming\\MATLAB\\SPI_Simulation\\images\\im_lena512.jpg";

cv::Mat GetImage(const char* imgPath) {
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
void InputN() {
	std::cout << "请输入分辨率：" ;
	std::cin >> N;
	std::cout << "正在仿真..." << std::endl;
}
void NormalizeSpectrum(cv::Mat& spectrum) {
	cv::Mat s_array[2];
	cv::split(spectrum, s_array);
	cv::Mat normalized = s_array[0].mul(s_array[0]) + s_array[1].mul(s_array[1]);
	cv::sqrt(normalized, normalized);
	cv::log(normalized, normalized);
	normalized = normalized / cv::log(10);
	cv::normalize(normalized, normalized, 0, 1, cv::NormTypes::NORM_MINMAX);
	normalized.convertTo(normalized, CV_8UC1,255);
	cv::applyColorMap(normalized, normalized, cv::ColormapTypes::COLORMAP_PARULA);
	spectrum = normalized;
}
cv::Mat FftShift(cv::Mat spectrum) {
	//throw "FftShift函数暂未实现";
	if (spectrum.rows % 2 != 0 || spectrum.cols % 2 != 0) {
		std::cout << "频谱不是偶数分辨率，暂不可中心化\n";
		return cv::Mat();
	}
	int x = spectrum.cols - 1;
	int y = spectrum.rows - 1;
	int cx = spectrum.cols / 2;
	int cy = spectrum.rows / 2;
	//printf_s("%d %d %d %d", x, y, cx, cy);
	cv::Mat shifted(spectrum);
	spectrum(cv::Rect(cx, cy, cx, cy)).copyTo(shifted(cv::Rect(0, 0, cx, cy)));//右下->左上
	spectrum(cv::Rect(0, cx, cx, cy)).copyTo(shifted(cv::Rect(cx, 0, cx, cy)));//左下->右上
	spectrum(cv::Rect(cx, 0, cx, cy)).copyTo(shifted(cv::Rect(0, cx, cx, cy)));//右上->左下
	spectrum(cv::Rect(0, 0, cx, cy)).copyTo(shifted(cv::Rect(cx, cy, cx, cy)));//左上->右下
	cv::flip(shifted(cv::Rect(0, cx, cx, cy)), shifted(cv::Rect(0, cx, cx, cy)), -1);
	cv::flip(shifted(cv::Rect(cx, cy, cx, cy)), shifted(cv::Rect(cx, cy, cx, cy)), -1);
	return shifted;
}
int main() {
	InputN();
	cv::Mat img = GetImage(imgFilePath);
	//cv::normalize(img, img, 0, 255, cv::NormTypes::NORM_MINMAX);
	Mat4Step output4Step(N, N, CV_64F);
	int total_loop_num = N * N;
	clock_t clk_begin = clock();
	for (int x = 0; x < N; x++)
		for (int y = 0; y < N; y++) {
			Mat4Step patterns = GetPattern(x, y);
			output4Step.Mat1.at<double>(y, x) = cv::sum(img.mul(patterns.Mat1))[0];
			output4Step.Mat2.at<double>(y, x) = cv::sum(img.mul(patterns.Mat2))[0];
			output4Step.Mat3.at<double>(y, x) = cv::sum(img.mul(patterns.Mat3))[0];
			output4Step.Mat4.at<double>(y, x) = cv::sum(img.mul(patterns.Mat4))[0];
			//char name[80];
			//cv::Mat save;
			//sprintf_s(name, ".\\patterns\\%03d_%03d_1.bmp", x, y);
			//patterns.Mat1.convertTo(save, CV_8UC1, 255);
			//cv::imwrite(name, save);
			//sprintf_s(name, ".\\patterns\\%03d_%03d_2.bmp", x, y);
			//patterns.Mat2.convertTo(save, CV_8UC1, 255);
			//cv::imwrite(name, save);
			//sprintf_s(name, ".\\patterns\\%03d_%03d_3.bmp", x, y);
			//patterns.Mat3.convertTo(save, CV_8UC1, 255);
			//cv::imwrite(name, save);
			//sprintf_s(name, ".\\patterns\\%03d_%03d_4.bmp", x, y);
			//patterns.Mat4.convertTo(save, CV_8UC1, 255);
			//cv::imwrite(name, save);
			std::cout << "进度：" << x * N + y + 1 << "/" << total_loop_num << '\r';
			//注意：进度显示会导致效率降低
		}
	std::cout << std::endl;
	cv::Mat output(N, N, CV_64FC2);
	cv::Mat outputs[] = {
		output4Step.Mat1 - output4Step.Mat3,output4Step.Mat2 - output4Step.Mat4 };
	cv::merge(outputs, 2, output);
	cv::Mat spectrum;
	cv::extractChannel(output, spectrum, 0);
	NormalizeSpectrum(spectrum);
	spectrum = FftShift(spectrum);
	cv::Mat rebuild;
	cv::idft(output, rebuild);
	cv::extractChannel(rebuild, rebuild, 0);
	cv::normalize(rebuild, rebuild, 0, 1, cv::NormTypes::NORM_MINMAX);
	clock_t clk_end = clock();
	std::cout << "用时" << double(clk_end - clk_begin) / CLOCKS_PER_SEC;
	std::cout << "秒" << std::endl;
	//return -1;
	cv::namedWindow("Rebuild Image", 0);
	cv::resizeWindow("Rebuild Image", 512, 512);
	cv::imshow("Rebuild Image", rebuild);
	cv::namedWindow("Spectrum", 0);
	cv::resizeWindow("Spectrum", 512, 512);
	cv::imshow("Spectrum", spectrum);
	rebuild.convertTo(rebuild, CV_8UC1, 255);
	cv::imwrite(".\\output\\rebuild.jpg", rebuild);
	//spectrum.convertTo(spectrum, CV_8UC3);
	cv::imwrite(".\\output\\spectrum.jpg", spectrum);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}