#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "orb_extractor.h"

using namespace cv;
using namespace std;
using namespace orb_exteactor;

inline void show_matches(cv::Mat image_ref,cv::Mat image_cur,
                       const vector<cv::Point2f> points_ref,const vector<cv::Point2f> points_cur)
{
    if(image_ref.channels()<3||image_cur.channels()<3)
    {
        cv::cvtColor(image_ref, image_ref, CV_GRAY2BGR);
        cv::cvtColor(image_cur, image_cur, CV_GRAY2BGR);
    }    
    cv::Mat showMerge;
    int merge_rows = 2*image_ref.rows,merge_cols=image_ref.cols;
    CV_Assert(image_ref.type () == image_cur.type ());
    showMerge.create(merge_rows,merge_cols,image_cur.type());
    image_ref.copyTo( showMerge(Rect(0,0,             image_ref.cols,image_ref.rows)) );
    image_cur.copyTo( showMerge(Rect(0,image_cur.rows,image_cur.cols,image_cur.rows)) );
    for(int j = 0; j < points_ref.size(); j++)
    {
        float b = 255*float (rand()) / RAND_MAX;
        float g = 255*float (rand()) / RAND_MAX;
        float r = 255*float (rand()) / RAND_MAX;
        line(showMerge,points_ref[j],cv::Point2f(points_cur[j].x,image_ref.rows+points_cur[j].y),Scalar(b, g, r), 1, 8 );
        circle(showMerge,points_ref[j],3,Scalar(b, g, r),1);
        circle(showMerge,cv::Point2f(points_cur[j].x,image_ref.rows+points_cur[j].y),3,Scalar(b, g, r),1);
    }
    cv::imshow("aligned points",showMerge);
}

inline void show_stereo_match(string window,cv::Mat imgLeft,cv::Mat imgRight,
                              vector<cv::Point2f> ptsLeft,vector<cv::Point2f> ptsRight)
{
    if(imgLeft.channels()<3||imgRight.channels()<3)
    {
        cvtColor(imgLeft,imgLeft,CV_GRAY2BGR);
        cvtColor(imgRight,imgRight,CV_GRAY2BGR);
    }
    cv::Mat mergeImg,showMerge,img0,img1;
    int merge_rows = imgLeft.rows , merge_cols = imgLeft.cols+10+imgRight.cols;
    CV_Assert(imgLeft.type () == imgRight.type ());
    mergeImg.create(merge_rows,merge_cols,imgLeft.type());
    imgLeft.copyTo( mergeImg(Rect(0,0,imgLeft.cols,imgLeft.rows)) );
    imgRight.copyTo( mergeImg(Rect(imgRight.cols+10,0,imgRight.cols,imgRight.rows)) );
    for(int j = 0; j < ptsLeft.size(); j++)
    {
        float b = 255*float (rand()) / RAND_MAX;
        float g = 255*float (rand()) / RAND_MAX;
        float r = 255*float (rand()) / RAND_MAX;
        line(mergeImg,ptsLeft[j],cv::Point2f(ptsRight[j].x+imgRight.cols+10,ptsRight[j].y),Scalar(b, g, r), 1, 8 );
    }
    for(int j = 0; j < ptsLeft.size(); j++)
    {
        float b = 255*float (rand()) / RAND_MAX;
        float g = 255*float (rand()) / RAND_MAX;
        float r = 255*float (rand()) / RAND_MAX;
        circle(mergeImg,ptsLeft[j],3,Scalar(b, g, r),1);
        circle(mergeImg,cv::Point2f(ptsRight[j].x+imgRight.cols+10,ptsRight[j].y),3,Scalar(b, g, r),1);
    }
    cv::imshow(window,mergeImg);
}


void cout_kps(vector<cv::KeyPoint> kps)
{
    for(auto kp:kps)
        cout<<"uv: "<<kp.pt.x<<" "<<kp.pt.y<<endl;
}



std::string img1str = "../kitti0_l.png";

int main()
{
    //extract
    cv::Mat mDescriptors1,  imout1  ;
    vector<cv::KeyPoint> vKps1;

    /**
     * @brief 构造函数
     * @detials 之所以会有两种响应值的阈值，原因是，程序先使用初始的默认FAST响应值阈值提取图像cell中的特征点；如果提取到的
     * 特征点数目不足，那么就降低要求，使用较小FAST响应值阈值进行再次提取，以获得尽可能多的FAST角点。
     * @param[in] nfeatures         指定要提取出来的特征点数目
     * @param[in] scaleFactor       图像金字塔的缩放系数
     * @param[in] nlevels           指定需要提取特征点的图像金字塔层
     * @param[in] iniThFAST         初始的默认FAST响应值阈值
     * @param[in] minThFAST         较小的FAST响应值阈值
    */
    ORBextractor *mpIniORBextractor = new ORBextractor(2000,1.2,8,20,5);

    cv::Mat im1 = cv::imread(img1str,cv::IMREAD_UNCHANGED);
    
    if(im1.empty())
        cout<<"read image error! \n"<<img1str<<endl;
    cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);

    // 但是注释掉这个函数就每次的结果都会一样了，如果不注释掉的话会影响
    cv::imshow("im1",im1);

    auto t0 = chrono::steady_clock::now();
    mpIniORBextractor->extract_orb_fts(im1,cv::Mat(),vKps1,mDescriptors1);
    
    // 打印特征点的坐标，可以看的出来每次的结果都不一样
    cout_kps(vKps1);

    cout<<"mvKeys1 size:"<<vKps1.size()<<endl;
    
    cv::drawKeypoints(im1, vKps1, imout1);
    cv::imshow("Image1", imout1);
    cv::waitKey();
    cv::destroyAllWindows();

    return -1;
}
