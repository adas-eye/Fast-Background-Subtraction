#include "mainwindow.h"
#include <QMenuBar>
#include <QApplication>
#include <QMessageBox>
#include <QCameraInfo>
#include <QScrollBar>
#include <QList>
#include <QPushButton>
#include <QGridLayout>
#include <string>
#include <QShortcut>
#include "capture_thread.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent), fileMenu(nullptr), capturer(nullptr)
{
    initUI();
    toggleHideActions(false);
    data_lock = new QMutex();
}

MainWindow::~MainWindow(){

}

void MainWindow::initUI(){
    this->resize(640, 800);

    initUIFileMenu();
    initUICameraMenu();
    initUIStatusBar();
    initUIViewArea();
}

void MainWindow::initUIFileMenu()
{
    // add menu
    fileMenu = menuBar()->addMenu("File");
//    fileToolBar = addToolBar("File");

    // add action exit
    exitAction = new QAction("Exit", this);
    fileMenu->addAction(exitAction);
    connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
    exitAction->setShortcut(QKeySequence("Ctrl+F4"));
}

void MainWindow::initUICameraMenu()
{
    /*
     * TODO
     * 1.Add actions
     * 2.Add Shortcuts for each
     * 3.Check visibilty : default ON
     * 4.connect to a proper slot
     */
    // add menu and toolbar

    cameraMenu = menuBar()->addMenu("Camera(s)");
    cameraToolBar = addToolBar("Camera(s)");

    // add cameraInfoAction
    cameraInfoAction = new QAction("Camera(s) Info", this);
    cameraMenu->addAction(cameraInfoAction);
    connect(cameraInfoAction, SIGNAL(triggered(bool)), this,SLOT(cameraInfo()));
    cameraInfoAction->setShortcut(QKeySequence("Alt+I"));

    // add cameraOpenAction
    cameraOpenAction = new QAction("Open Camera", this);
    cameraMenu->addAction(cameraOpenAction);
    cameraToolBar->addAction(cameraOpenAction);
    connect(cameraOpenAction, SIGNAL(triggered(bool)), this, SLOT(cameraOpen()));
    cameraOpenAction->setShortcut(QKeySequence("Alt+O"));

    // add stop camera action
    // set visibility off initially
    stopCameraAction = new QAction("Stop", this);
    cameraMenu->addAction(stopCameraAction);
    cameraToolBar->addAction(stopCameraAction);
    connect(stopCameraAction, SIGNAL(triggered(bool)), this, SLOT(stopCamera()));
    stopCameraAction->setShortcut(QKeySequence("Alt+S"));

    // add FPS calculation action
    // set visibility off initially
    fpsCalculationAction = new QAction("FPS", this);
    cameraMenu->addAction(fpsCalculationAction);
    cameraToolBar->addAction(fpsCalculationAction);
    connect(fpsCalculationAction, SIGNAL(triggered(bool)), this, SLOT(calculateFPS()));
    fpsCalculationAction->setShortcut(QKeySequence("Alt+F"));

    // add mirror checkbox
    // set visibility off initially
   cameraMirrorAction = new QAction("Mirror", this);
   cameraMenu->addAction(cameraMirrorAction);
   cameraToolBar->addAction(cameraMirrorAction);
   connect(cameraMirrorAction, SIGNAL(triggered(bool)), this, SLOT(doCameraMirror()));
   cameraMirrorAction->setShortcut(QKeySequence("Alt+M"));

}

void MainWindow::initUIViewArea()
{
    // grid layout system to divide up the screen.
    QGridLayout *main_layout = new QGridLayout();

    // Image Scene1 for real video
    imageScene1 = new QGraphicsScene(0, 0, 6, 6, this);
    imageScene1->addText("Real");
    imageView1 = new QGraphicsView(imageScene1);
    imageView1->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    imageView1->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // image Scene2 for foreground mask
    imageScene2 = new QGraphicsScene(0, 6, 6, 6, this);
    imageScene2->addText("predicted Foreground Mask");
    imageView2 = new QGraphicsView(imageScene2);
    imageView2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    imageView2->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // image Scene3 for background image
    imageScene3 = new QGraphicsScene(6, 0, 6, 6, this);
    imageScene3->addText("Background Image");
    imageView3 = new QGraphicsView(imageScene3);
    imageView3->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    imageView3->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    imageScene4 = new QGraphicsScene(6, 6, 6, 6, this);
    imageScene4->addText("Empty.");
    imageView4 = new QGraphicsView(imageScene4);
    imageView4->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    imageView4->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    main_layout->addWidget(imageView1, 0, 0, 6, 6);
    main_layout->addWidget(imageView2, 0, 6, 6, 6);
    main_layout->addWidget(imageView3, 6, 0, 6, 6);
    main_layout->addWidget(imageView4, 6, 6, 6, 6);

    // a panel for monitor button , record button
    QGridLayout *tools_layout = new QGridLayout();
    main_layout->addLayout(tools_layout, 12, 0, 1, 12);

    // monitor box
    monitorCheckBox = new QCheckBox(this);
    monitorCheckBox->setText("Monitor on/off");
    tools_layout->addWidget(monitorCheckBox, 0, 0);
    connect(monitorCheckBox, SIGNAL(stateChanged(int)), this, SLOT(updateMonitorStatus(int)));

    //record button
    recordButtonText = new QList<QString>({"Record", "Stop Record"});
    recordButton = new QPushButton(recordButtonText->at(0), this);
    tools_layout->addWidget(recordButton, 0, 1);
    connect(recordButton, SIGNAL(clicked(bool)), this, SLOT(recordingStartStop()));

    // pause button
    playPauseButtonText = new QList<QString>({"Play", "Pause"});
    playPauseButton = new QPushButton(playPauseButtonText->at(1), this);
    playPauseButton->setCheckable(true);
    tools_layout->addWidget(playPauseButton, 0, 2);
    connect(playPauseButton, SIGNAL(clicked(bool)), this, SLOT(togglePlayPause(bool)));

//    // list view bottom
//    saved_list = new QListView(this);
//    main_layout->addWidget(saved_list, 13, 0, 4, 1);

    QWidget *widget = new QWidget();
    widget->setLayout(main_layout);
    setCentralWidget(widget);
}

void MainWindow::initUIStatusBar()
{
    mainStatusBar = statusBar();

    mainStatusBarData = new QMap<QString, QString>();
    mainStatusBarData->insert("default", "Ready");
    mainStatusBarData->insert("Camera Name", "");
    mainStatusBarData->insert("Resolution", "");
    mainStatusBarData->insert("FPS", "");

    mainStatusLabel = new QLabel(mainStatusBarData->value("default"));
    mainStatusBar->addWidget(mainStatusLabel);
    updateStatusBar(true);
}

void MainWindow::updateStatusBar(QString key, QString value, bool includeDefault=false){
    mainStatusBarData->insert(key, value);
    QString label = "";
    foreach(QString key, mainStatusBarData->keys()){
        if(key =="default" && !includeDefault)
            continue;

        if (mainStatusBarData->value(key) != "")
        {
            label += mainStatusBarData->value(key) + " | ";
        }
    }
    mainStatusLabel->setText(label);
}

void MainWindow::updateStatusBar(bool includeDefault){

    QString label = "";
    foreach(QString key, mainStatusBarData->keys()){
        if(key =="default" && !includeDefault)
            continue;

        if (mainStatusBarData->value(key) != "")
        {
            label += mainStatusBarData->value(key) + " | ";
        }
    }
    mainStatusLabel->setText(label);
}

void MainWindow::cameraInfo(){
    QMessageBox msgBox;
//    QPushButton *button = new QPushButton("new button", this);

    msgBox.setText("Camera(s) Info");
    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    QString info = QString("Available Cameras (%1):\n").arg(cameras.length());
    QString detailed_info;
    if (cameras.empty()){
        QMessageBox::information(this, "Information", "No Camera available.");
        return;
    }

    foreach(QCameraInfo c_info, cameras)
    {
        info += "    - " + c_info.deviceName() + "\n";
        detailed_info += " - " + c_info.description() + "\n";
    }

    msgBox.setInformativeText(info);
    msgBox.setDetailedText(detailed_info);
//    msgBox.addButton(button, QMessageBox::RejectRole);
    msgBox.exec();
}

QString MainWindow::selectCamera(){

    QList<QCameraInfo> cameras = QCameraInfo::availableCameras();

    // no camera available
    if (cameras.empty()){
        QMessageBox::information(this, "Information", "No Camera available.");
        return nullptr;
    }

   // single camera available
   if (cameras.length()==1){
       return cameras.at(0).deviceName();
   }

   // multi cameras available - let user select camera.
    QMessageBox msgBox;
    foreach(QCameraInfo camera, cameras){
        msgBox.addButton(camera.deviceName(), QMessageBox::ActionRole);
    }

    QPushButton *cancelButton = msgBox.addButton(QMessageBox::Cancel);
    msgBox.exec();

    if(msgBox.clickedButton()==cancelButton){
        return nullptr;
    }
    else{
        return msgBox.clickedButton()->text();
    }
}

void MainWindow::cameraOpen(){

        if (isCameraOpen){
            QMessageBox::information(this, "Information", "Close the opened camera first");
            return;
        }

        QString camname = selectCamera();
        if (camname == nullptr){
           return;
        }

        // get the camera name and convert to std string
        std::string std_camname = camname.toUtf8().constData();

        // open the concurrent thread object for frame capture.
        capturer = new capture_thread(std_camname, data_lock);
        connect(capturer, &capture_thread::frameCaptured, this, &MainWindow::updateFrame);
        connect(capturer, &capture_thread::fpsChanged, this, &MainWindow::updateFPS);
        connect(capturer, &capture_thread::RunComplete, this, &MainWindow::closeCapturer);
        capturer->start();
        // add the text to status label
        updateStatusBar("Camera Name", camname, false);

        // change the variable indicating camera is open
        isCameraOpen = true;

        // open some buttons as visible.
        toggleHideActions(true);
        capturer->setWebcamMode();

}

void MainWindow::stopCamera()
{
    if (!isCameraOpen){
        return;
    }

    // stop the thread.
    capturer->setRunning(false);

    // close the variable also.
    isCameraOpen = false;
    toggleHideActions(false);

    //mainStatuslevel text reset
    foreach(QString key, mainStatusBarData->keys())
    {
        if(key=="default")
            continue;
        mainStatusBarData->insert(key, "");
    }

    mainStatusLabel->setText(mainStatusBarData->value("default"));

}

void MainWindow::updateFrame(cv::Mat *mat)
{    
    data_lock->lock();
    currentframe = *mat;
    data_lock->unlock();
    updateView(imageScene1, imageView1, currentframe);

}

void MainWindow::updateFgMask(cv::Mat *mat){

    // update the data.
    data_lock->lock();
    currentFgMask = *mat;
    data_lock->unlock();
    updateView(imageScene2, imageView2, currentFgMask);
}

void MainWindow::updateBackgroundImage(cv::Mat *mat)
{
    data_lock->lock();
    currentBgImage = *mat;
    data_lock->unlock();
    updateView(imageScene3, imageView3, currentBgImage);

}

void MainWindow::updateView(QGraphicsScene *scene, QGraphicsView *view, cv::Mat &image)
{
    QImage frame(
                image.data,
                image.cols,
                image.rows,
                image.step,
                QImage::Format_RGB888);

    QPixmap Image = QPixmap::fromImage(frame);

    scene->clear();
    view->resetMatrix();

    if(!Image.isNull())
    {
        Image = Image.scaledToWidth(view->width());
        Image = Image.scaledToHeight(view->height());

//        if (view->width() > view->height())
//            Image = Image.scaledToWidth(view->width());
//        else
//            Image = Image.scaledToHeight(view->height());

        scene->addPixmap(Image);
        scene->update();
        view->setSceneRect(Image.rect());

    }
}

void MainWindow::calculateFPS()
{
    if (capturer != nullptr ){

        if(capturer->isFPSCalculating()){
            QMessageBox::information(this, "Information", "FPS is being calculated.");
        }
        else{
            capturer->startCalcFPS(true);
            mainStatusBarData->insert("Resolution", "...Calculating");
            updateStatusBar("FPS" ,"...Calculating");
        }
    }
}

void MainWindow::updateFPS(float fps, int width, int height)
{
    QString fps_update = mainStatusLabel->text();
    if (capturer != nullptr && capturer->isRunning())
    {
       mainStatusBarData->insert("Resolution", QString("%1(w) x %2(h)").arg(width).arg(height));
       updateStatusBar( "FPS", QString("%1").arg(fps));

    }
}

void MainWindow::doCameraMirror()
{
    if (capturer!=nullptr)
        capturer->setMirror( !capturer->isMirror());
}

void MainWindow::toggleHideActions(bool show)
{
    // close some buttons from visible.
    stopCameraAction->setVisible(show);
    fpsCalculationAction->setVisible(show);
    cameraMirrorAction->setVisible(show);
    monitorCheckBox->setVisible(show);
}

void MainWindow::recordingStartStop()
{
    if(capturer == nullptr)
    {
        // clear record.
        if (mainStatusLabel->text() != mainStatusBarData->value("default"))
            updateStatusBar("Record Status", "", true);

        return;
    }

   clickedRecord = !clickedRecord;

   if (clickedRecord)
   {
       recordButton->setText(recordButtonText->at(1));
       capturer->setVideoSavingStatus(capture_thread::STARTING);
       connect(capturer, &capture_thread::videoRecordStatus, this, &MainWindow::updateVideoRecordStatus);
   }

   else
   {
       recordButton->setText(recordButtonText->at(0));
       capturer->setVideoSavingStatus(capture_thread::STOPPING);
   }

}

void MainWindow::updateVideoRecordStatus(int status, QString saved_video_name)
{
    QList<QString> textualStatus =  QList<QString> ( {"Starting", "Started", "Stopping", "Stopped"} );
    updateStatusBar("Record Status", textualStatus.at(status) + "@" + saved_video_name);
    if (status==3)
    {
        recordButton->setText(recordButtonText->at(0));
        disconnect(capturer, &capture_thread::videoRecordStatus, this, &MainWindow::updateVideoRecordStatus);
    }

}

void MainWindow::closeCapturer(bool)
{

    delete capturer;
    capturer = nullptr;
    qDebug() <<"Closed the thread.";
    clickedRecord=false;

    playPauseButton->setText(playPauseButtonText->at(0));
    playPauseButton->setChecked(false);
}

void MainWindow::updateMonitorStatus(int checked)
{
    if (capturer == nullptr)
        return;

    qDebug() << QString("Updating monitor status with status : %1").arg(checked);

    if (checked)
    {
        connect(capturer, &capture_thread::fgMaskCaptured, this, &MainWindow::updateFgMask);
        connect(capturer, &capture_thread::bgImageCaptured, this, &MainWindow::updateBackgroundImage);
        capturer->setMotionDetectingStatus(true);
    }

    else
    {
        capturer->setMotionDetectingStatus(false);
    }

    qDebug() << "done updateMonitorStatus";
}


void MainWindow::togglePlayPause(bool clicked)
{
    if (capturer == nullptr)
        return;

    if (!playPauseButton->isChecked())
    {
        playPauseButton->setText( playPauseButtonText->at(1) );
        capturer->setPause(false);
    }

    else{
        playPauseButton->setText( playPauseButtonText->at(0));
        capturer->setPause(true);
    }
}
