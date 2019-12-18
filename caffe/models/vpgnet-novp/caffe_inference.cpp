#include <caffe/caffe.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include<algorithm> 
using namespace caffe;
using namespace cv;
#define MAX_INPUT_SIDE 640;
#define MIN_INPUT_SIDE 480;
std::string caffe_root = "/LANE/VPGNet/caffe/models/vpgnet-novp/inference/";
//dump caffe feature map
class CaffeDump {
public:
    CaffeDump(const std::string& net_file, const std::string& weight_file, const int GPUID);
    ~CaffeDump();
    void caffe_forward(cv::Mat, const std::string& , int, const std::string&, int);
private:
    void preprocess(const cv::Mat cv_image);
    cv::Mat image_translation(cv::Mat & srcImage, int x0ffset, int y0ffset);
private:
    shared_ptr<Net<float> > net_;
    int num_channels_;
    float threshold_;
};
CaffeDump::CaffeDump(const std::string& net_file, const std::string& weights_file, const int GPUID)
{
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    net_.reset(new Net<float>(net_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK_EQ(num_channels_, 3) << "Input layer should have 3 channels.";
}
CaffeDump::~CaffeDump() {}
void CaffeDump::caffe_forward(cv::Mat cv_image, const std::string& layer_name, int channel, const std::string& filepath, int factor)
{
    if (cv_image.empty()) {
        std::cout << "Can not reach the image" << std::endl;
        return;
    }
    preprocess(cv_image);
    net_->ForwardFrom(0);
    shared_ptr<caffe::Blob<float>> layerData =net_->blob_by_name(layer_name);
    int batch_size = layerData->num();  
    int dim_features = layerData->count() / batch_size; 
    int channels = layerData->channels();
    int height = layerData->height();
    int width = layerData->width();
    std::cout << "batch size:" << batch_size << std::endl;
    std::cout << "dimension:" << dim_features << std::endl;
    std::cout << "channels:" << channels << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "width:" << width << std::endl;
    std::cout << "channels*height*width:" << channels*height*width << std::endl;
    CHECK_LT(channel, channels) << "Input channel number should small than channels.";
    float* feature_blob_data;  
    for (int n = 0; n < batch_size; ++n)  
    { 
        feature_blob_data = layerData->mutable_cpu_data() +  
            layerData->offset(n); 
        float *arr = (float(*))malloc(height*width*sizeof(float));
        int idx = 0;
        for (int d = 0; d < dim_features; ++d)  
        { 
            if (idx < height*width){
                arr[idx] = feature_blob_data[idx+channel*height*width];
                idx++;
            }
        } 
        int len = height*width; 
        float min_val = *std::min_element (arr,arr+len);
        float max_val = *std::max_element(arr,arr+len);  
        std::cout << "size of feature:" << idx << ",max " 
            << *std::max_element(arr,arr+len) << ",min " <<*std::min_element (arr,arr+len)<<std::endl;
        for (int i=0;i<len;i++){
            arr[i] = 255*(arr[i]-min_val)/(max_val-min_val);
        } 
        const cv::Mat img(cv::Size(width, height), CV_32FC1, arr);
        cv::imwrite(filepath+".jpg", img); 
        const cv::Mat scale_img(cv::Size(width*factor, height*factor), CV_32FC1, arr);
        cv::imwrite(filepath+"_"+std::to_string(factor)+"x.jpg", scale_img); 
        cv::Mat srcImage=cv::imread(filepath+"_"+std::to_string(factor)+"x.jpg");
        int x0ffset = -180;
        int y0ffset = -180;
        //图像左平移不改变大小
        cv::Mat resultImage1 = image_translation(srcImage, x0ffset, y0ffset);   
        cv::imwrite(filepath+"_"+std::to_string(factor)+"x-off.jpg", resultImage1);
        std::cout << ("\n");  
    }  // for (int n = 0; n < batch_size; ++n) 
}
cv::Mat CaffeDump::image_translation(cv::Mat & srcImage, int x0ffset, int y0ffset)
{
    int nRows = srcImage.rows;
    int nCols = srcImage.cols;
    cv::Mat resultImage(srcImage.size(), srcImage.type());
    //int nRows = srcImage.rows + abs(y0ffset);
    //int nCols = srcImage.cols + abs(x0ffset);
    //cv::Mat resultImage(nRows, nCols, srcImage.type());
    //遍历图像
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            //映射变换
            int x = j - x0ffset;
            int y = i - y0ffset;
            //边界判断
            if (x >= 0 && y >= 0 && x < nCols && y < nRows)
            {
                resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
            }
        }
    }
    return resultImage;
}
void CaffeDump::preprocess(const cv::Mat cv_image) {
    cv::Mat cv_new(cv_image.rows, cv_image.cols, CV_32FC3, cv::Scalar(0, 0, 0));
    int height = cv_image.rows;
    int width = cv_image.cols;
    /* Mean normalization (in this case it may not be the average of the training) */
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[0]);// - float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[1]);// - float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[2]) ;//- float(122.7717);
        }
    }
    /* Max image size comparation to know if resize is needed */
    int max_side = MAX(height, width);
    int min_side = MIN(height, width);
    float max_side_scale = float(max_side) / MAX_INPUT_SIDE;
    float min_side_scale = float(min_side) / MIN_INPUT_SIDE;
    float max_scale = MAX(max_side_scale, min_side_scale);
    float img_scale = 1;
    if (max_scale > 1)
        img_scale = float(1) / max_scale;
    int height_resized = int(height * img_scale);
    int width_resized = int(width * img_scale);
    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width_resized, height_resized));
    float data_buf[height_resized*width_resized * 3];
    for (int h = 0; h < height_resized; ++h)
    {
        for (int w = 0; w < width_resized; ++w)
        {
            data_buf[(0 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }
    net_->blob_by_name("data")->Reshape(1, num_channels_, height_resized, width_resized);
    Blob<float> * input_blobs = net_->input_blobs()[0];
    switch (Caffe::mode()) {
        case Caffe::CPU:
            memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
            break;
        case Caffe::GPU:
            caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
            break;
        default:
            LOG(FATAL) << "Unknow Caffe mode";
    }
}
int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        printf("Usage caffe_test <net.prototxt> <net.caffemodel> <inputFile_txt> <outputDirectory> <output_prefix>\n");
        return -1;
    }
    int GPUID = 0;
    std::string  prototxt_file = argv[1];
    std::string caffemodel_file = argv[2];
    const char * input_files_path = argv[3];
    const char * output_directory = argv[4];
    std::cout << "Reading the given prototxt file : " << prototxt_file << std::endl;
    std::cout << "Reading the given caffemodel file: " << caffemodel_file << std::endl;
    FILE * fs;
    char * image_path = NULL;
    size_t buff_size = 0;
    ssize_t read;
    fs = fopen(input_files_path, "r");
    if (!fs) {
        std::cout << "Unable to open the file." << input_files_path << std::endl;
        return -1;
    }
    CaffeDump dump(prototxt_file.c_str(), caffemodel_file.c_str(), GPUID);
    cv::Mat image = cv::imread(input_files_path, CV_LOAD_IMAGE_COLOR);
    std::cout << input_files_path << std::endl;
    //dump.caffe_forward(image, "bb-output-tiled", 0);
    for (int i=0; i<5; i++){
        dump.caffe_forward(image, "multi-label", i, "./l"+std::to_string(i), 8);}
    dump.caffe_forward(image, "data" ,2, "./org", 1);
    //dump.caffe_forward(image, "type-conv-tiled" ,0);
    BlobProto blob_proto; 
    string mean_file = "/data1/workspace/liyiran/VPGNet/caffe/models/vpgnet-novp/driving_mean_train.binaryproto"; 
    ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
    Blob<float> data_mean_; 
    data_mean_.FromProto(blob_proto);  
    std::cout << "mean file:" << data_mean_.num() << std::endl;  
    std::cout << data_mean_.channels() << std::endl;  
    std::cout << data_mean_.height() << std::endl;  
    std::cout << data_mean_.width() << std::endl;
    return 0;
}