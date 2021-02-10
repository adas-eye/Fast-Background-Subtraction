#include "capture_thread.h"
#include "utilities.h"
#include <string>
#include <QDebug>
#include <QTime>
#include <iostream>
#include <vector>
#include <QVector>
#include <opencv2/highgui.hpp>

capture_thread::capture_thread(std::string camname, QMutex *lock):
    running(false), camname(camname), videopath(""), data_lock(lock)
{
    fps_calculating = false;
    fps = 0.0;

    frame_width=frame_height=0;
    video_writer=nullptr;
    video_saving_status=STOPPED;
}

capture_thread::capture_thread(QString videopath, QMutex *lock):
    running(false), camname("nocam"), videopath(videopath), data_lock(lock)
{
    fps_calculating = false;
    fps = 0.0;

    frame_width=frame_height=0;
    video_writer=nullptr;
    video_saving_status=STOPPED;
    saved_video_name="";

}

capture_thread::~capture_thread()
{

}

void capture_thread::setRunning(bool run){
    data_lock->lock();
    running=run;
    data_lock->unlock();
}

void capture_thread::startCalcFPS(bool start){
    data_lock->lock();
    fps_calculating = start;
    data_lock->unlock();
}

bool capture_thread::isFPSCalculating(){return fps_calculating;}

bool capture_thread::generateFrames(cv::VideoCapture &cap, cv::Mat &tmp_frame){

    cap >> tmp_frame;
    if (tmp_frame.empty()){
        return false;
    }

    cv::cvtColor(tmp_frame, tmp_frame, cv::COLOR_BGR2RGB);
    data_lock->lock();
    cv::flip( tmp_frame, frame, 1);
    data_lock->unlock();
    emit frameCaptured(&frame);
    return true;


}

void capture_thread::run()
{
    // a helpful code snippet
    // https://www.kurokesu.com/main/2020/07/12/pulling-full-resolution-from-a-webcam-with-opencv-windows/

    // set thread running
    setRunning(true);

    // open webcam
    cv::VideoCapture cap(camname);
    if (!cap.isOpened()){
        CV_Assert("Failed to open camera.");
    }

    //set framerate, width and height
//    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // get the actual frame height and width we got.
    frame_width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    frame_height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps = cap.get(cv::CAP_PROP_FPS);
    qDebug() << QString("frame %1(w) x %2(h) @fps %3").arg(frame_width).arg(frame_height).arg(fps);

    // create the blank frame.
    blankFrame = new cv::Mat(frame_height, frame_width, CV_8U, 255);

    // create segmentor
    segmentor = cv::createBackgroundSubtractorMOG2(500, 16, true);

    // tmp_frames for resource allocation.
    cv::Mat tmp_frame;
    cv::Mat tmp_frame2;

    // for fps calculation.
    int frame_count=0;
    QTime timer;
    bool first_time=true;
    const int n_frames_to_consider=30;

    while(running){

        if ( pause )
            continue;

        cap >> tmp_frame;
        if(tmp_frame.empty())
            break;

        if (doMirror)
        {
            cv::flip(tmp_frame, tmp_frame2, 1);
            tmp_frame = tmp_frame2;
        }

        if (motion_detecting_status && segmentor != nullptr)
            motionDetect(tmp_frame);

        if (fps_calculating)
        {
            if(first_time)
            {
                timer.start();
                first_time=false;
            }

            else if(frame_count != n_frames_to_consider)
                frame_count++;

            else if(frame_count == n_frames_to_consider)
            {
                int elapsed_ms = timer.elapsed();
                fps = frame_count / (elapsed_ms / 1000.0);
                startCalcFPS(false);
                emit fpsChanged(fps, frame_width, frame_height);
                first_time = true;
                frame_count=0;
            }

        }

        if (video_saving_status != STOPPED )
        {
            if(video_saving_status == STARTING)
                startSavingVideo(tmp_frame);

            else if(video_saving_status == STARTED)
                video_writer->write(tmp_frame);

            else if(video_saving_status == STOPPING)
                stopSavingVideo();
        }

        // convert color to visualise at screen
        cv::cvtColor(tmp_frame, tmp_frame, cv::COLOR_BGR2RGB);

        // write and emit frame
        data_lock->lock();
        frame=tmp_frame;
        data_lock->unlock();
        emit frameCaptured(&frame);

    }

    if(video_saving_status != STOPPED)
        stopSavingVideo();

    emit frameCaptured(blankFrame);
    emit fgMaskCaptured(blankFrame);
    emit bgImageCaptured(blankFrame);
    cap.release();
    setRunning(false);

    qDebug()<<"stopped running.";
    emit RunComplete(true);
}

void capture_thread::setVideoSavingStatus(VideoSavingStatus status)
{
    data_lock->lock();
    video_saving_status = status;
    data_lock->unlock();
}

void capture_thread::startSavingVideo(cv::Mat &firstFrame)
{
    saved_video_name = utilities::newSavedVideoName();

    // generate a cover image for video.
    QString cover_path = utilities::getSavedVideoPath(saved_video_name, "jpg");
    cv::imwrite(cover_path.toStdString(), firstFrame);

    // video save path.
    QString video_saved_path = utilities::getSavedVideoPath(saved_video_name, "avi");
    video_writer = new cv::VideoWriter(
                video_saved_path.toStdString(),
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps? fps:30,
                cv::Size(frame_width, frame_height)
                );
    setVideoSavingStatus(STARTED);
    saved_video_name = video_saved_path;
    emit videoRecordStatus(video_saving_status, saved_video_name);

}

void capture_thread::stopSavingVideo()
{
    setVideoSavingStatus( STOPPED );
    if (video_writer != nullptr )
    {
        video_writer->release();
        delete video_writer;
        video_writer=nullptr;

    }
    emit videoRecordStatus(video_saving_status, saved_video_name);
}

void capture_thread::setMirror(bool mirror)
{
    data_lock->lock();
    doMirror=mirror;
    data_lock->unlock();
}

bool capture_thread::isMirror(){
    return doMirror;
}

capture_thread::VideoSavingStatus capture_thread::getVideoSavingStatus()
{
    return video_saving_status;
}

void capture_thread::setMotionDetectingStatus(bool status){

    data_lock->lock();
    motion_detecting_status=status;
    motion_detected = false;
    data_lock->unlock();
}

void capture_thread::motionDetect(cv::Mat &frame)
{
    cv::Mat fgMask;
    segmentor->apply(frame, fgMask);

    if (fgMask.empty())
        return;

    //apply thresholding on fgmask
    cv::threshold(fgMask, fgMask, 25, 255, cv::THRESH_BINARY);

    // remove noise by erosion than dilation.
    int noise_size = 9;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(noise_size, noise_size));
    cv::erode(fgMask, fgMask, kernel);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(noise_size, noise_size));
    cv::dilate(fgMask, fgMask, kernel, cv::Point(-1, -1), 3);

    // update fgMaskToEmit
    data_lock->lock();
    cv::cvtColor(fgMask, fgMaskToEmit, cv::COLOR_GRAY2RGB);
    data_lock->unlock();

    // update background image
    data_lock->lock();
    segmentor->getBackgroundImage(bgImageToEmit);
    cv::cvtColor(bgImageToEmit, bgImageToEmit, cv::COLOR_BGR2RGB);
    data_lock->unlock();

    // emit background image and fgMaskToEmit
    emit fgMaskCaptured(&fgMaskToEmit);
    emit bgImageCaptured(&bgImageToEmit);

    // find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // motion detecting using contours.
    bool has_motion = contours.size() > 0;

    // update the statuses
    if(!motion_detected && has_motion)
    {
        motion_detected = true;
        setVideoSavingStatus(STARTING);
//        qDebug() << "new motion detected. ";
    }
    else if (motion_detected && !has_motion)
    {
        motion_detected=false;
        setVideoSavingStatus(STOPPING);
    }

    // create rectangles around moving objects
    cv::Scalar color = cv::Scalar(0, 0, 255);
    int max_area=0;
    cv::Rect choosen_rect;

    // find the biggest rectangle and draw it.
    for(size_t i=0; i<contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.area() > max_area)
        {
            max_area = rect.area();
            choosen_rect = rect;
        }
    }
    cv::rectangle(frame, choosen_rect, color, 1);

}


void capture_thread::setPause(bool doPause)
{
    data_lock->lock();
    pause = doPause;
    data_lock->unlock();
}

void capture_thread::setVideoMode(QString videoFile)
{
    data_lock->lock();
    webcam_mode = false;
    videoFilePath = videoFile;
    data_lock->unlock();
}

void capture_thread::setWebcamMode()
{
    data_lock->lock();
    webcam_mode = true;
    data_lock->unlock();
}








