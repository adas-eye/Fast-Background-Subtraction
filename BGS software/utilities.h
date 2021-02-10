#ifndef UTILITIES_H
#define UTILITIES_H
#include <QString>

class utilities
{
public:
    static QString getDataPath();
    static QString newSavedVideoName();
    static QString getSavedVideoPath(QString name, QString postfix);
    static void notifyMobile(QString camname);
};

#endif // UTILITIES_H
