#include "utilities.h"
#include <QStandardPaths>
#include <QDir>
#include <QDebug>
#include <QDateTime>

QString utilities::getDataPath()
{
    // get the standard locations
    QString user_movie_path = QStandardPaths::standardLocations( QStandardPaths::MoviesLocation)[0];
//    qDebug() <<QString("user movie path : " ) + user_movie_path;
    // initialise a movie dir and make a new folder
    QDir movie_dir(user_movie_path);
    movie_dir.mkpath("software");

    // return the path of newly created folder.
    QString directory = movie_dir.absoluteFilePath("software");
//    qDebug() <<"directory created : " + directory;
    return directory;
}

QString utilities::newSavedVideoName()
{
    QDateTime time = QDateTime::currentDateTime();
    return time.toString("yyyy-MM-dd+HH:mm:ss");
}

QString utilities::getSavedVideoPath(QString name, QString postfix)
{
    return QString("%1/%2.%3").arg(utilities::getDataPath(), name, postfix);
}
