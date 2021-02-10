#include <QApplication>
//#include <QMainWindow>
#include "mainwindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    MainWindow window;
    window.setWindowTitle("Software");
    window.show();
    // or use showFullScreen
    // or show maximized
    return app.exec();
}
