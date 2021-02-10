/****************************************************************************
** Meta object code from reading C++ file 'capture_thread.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../capture_thread.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'capture_thread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_capture_thread_t {
    QByteArrayData data[15];
    char stringdata0[157];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_capture_thread_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_capture_thread_t qt_meta_stringdata_capture_thread = {
    {
QT_MOC_LITERAL(0, 0, 14), // "capture_thread"
QT_MOC_LITERAL(1, 15, 13), // "frameCaptured"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 8), // "cv::Mat*"
QT_MOC_LITERAL(4, 39, 4), // "data"
QT_MOC_LITERAL(5, 44, 14), // "fgMaskCaptured"
QT_MOC_LITERAL(6, 59, 15), // "bgImageCaptured"
QT_MOC_LITERAL(7, 75, 10), // "fpsChanged"
QT_MOC_LITERAL(8, 86, 3), // "fps"
QT_MOC_LITERAL(9, 90, 5), // "width"
QT_MOC_LITERAL(10, 96, 6), // "height"
QT_MOC_LITERAL(11, 103, 17), // "videoRecordStatus"
QT_MOC_LITERAL(12, 121, 6), // "status"
QT_MOC_LITERAL(13, 128, 16), // "saved_video_name"
QT_MOC_LITERAL(14, 145, 11) // "RunComplete"

    },
    "capture_thread\0frameCaptured\0\0cv::Mat*\0"
    "data\0fgMaskCaptured\0bgImageCaptured\0"
    "fpsChanged\0fps\0width\0height\0"
    "videoRecordStatus\0status\0saved_video_name\0"
    "RunComplete"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_capture_thread[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       6,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   44,    2, 0x06 /* Public */,
       5,    1,   47,    2, 0x06 /* Public */,
       6,    1,   50,    2, 0x06 /* Public */,
       7,    3,   53,    2, 0x06 /* Public */,
      11,    2,   60,    2, 0x06 /* Public */,
      14,    1,   65,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, QMetaType::Float, QMetaType::Int, QMetaType::Int,    8,    9,   10,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,   12,   13,
    QMetaType::Void, QMetaType::Bool,    2,

       0        // eod
};

void capture_thread::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        capture_thread *_t = static_cast<capture_thread *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->frameCaptured((*reinterpret_cast< cv::Mat*(*)>(_a[1]))); break;
        case 1: _t->fgMaskCaptured((*reinterpret_cast< cv::Mat*(*)>(_a[1]))); break;
        case 2: _t->bgImageCaptured((*reinterpret_cast< cv::Mat*(*)>(_a[1]))); break;
        case 3: _t->fpsChanged((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 4: _t->videoRecordStatus((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 5: _t->RunComplete((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (capture_thread::*_t)(cv::Mat * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::frameCaptured)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (capture_thread::*_t)(cv::Mat * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::fgMaskCaptured)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (capture_thread::*_t)(cv::Mat * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::bgImageCaptured)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (capture_thread::*_t)(float , int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::fpsChanged)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (capture_thread::*_t)(int , QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::videoRecordStatus)) {
                *result = 4;
                return;
            }
        }
        {
            typedef void (capture_thread::*_t)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&capture_thread::RunComplete)) {
                *result = 5;
                return;
            }
        }
    }
}

const QMetaObject capture_thread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_capture_thread.data,
      qt_meta_data_capture_thread,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *capture_thread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *capture_thread::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_capture_thread.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int capture_thread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 6)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 6;
    }
    return _id;
}

// SIGNAL 0
void capture_thread::frameCaptured(cv::Mat * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void capture_thread::fgMaskCaptured(cv::Mat * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void capture_thread::bgImageCaptured(cv::Mat * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void capture_thread::fpsChanged(float _t1, int _t2, int _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void capture_thread::videoRecordStatus(int _t1, QString _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void capture_thread::RunComplete(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
