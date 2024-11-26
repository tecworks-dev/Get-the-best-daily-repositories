// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_run_guard.hpp"
#include "nau/app/nau_qt_app.hpp"
#include "baseWidgets/nau_widget_utility.hpp"

#include <QApplication>
#include <QCryptographicHash>
#include <QMessageBox>
#include <QSharedMemory>
#include <QSystemSemaphore>


// ** NauRunGuardPrivate
// 
// Based on https://stackoverflow.com/questions/5006547/qt-best-practice-for-a-single-instance-app-protection

class NauRunGuardPrivate
{
public:
    NauRunGuardPrivate();
    ~NauRunGuardPrivate();

    bool tryToAcquire();

private:
    void release();
    bool isAnotherRunning();
    QString generateKeyHash(const QString& key, const QString& salt);

private:
    QSystemSemaphore m_lock;

    // Note that native key of this must be syncronized with NSIS installer script's
    // installer/win64/installer.nsi:SharedMemoryKey.
    QSharedMemory m_memory;
};


NauRunGuardPrivate::NauRunGuardPrivate()
    : m_lock(generateKeyHash("editor", "_lock_key"), 1)
    , m_memory(generateKeyHash("editor", "_memory_key"))
{
}

NauRunGuardPrivate::~NauRunGuardPrivate()
{
    release();
}

void NauRunGuardPrivate::release()
{
    m_lock.acquire();
    if (m_memory.isAttached()) m_memory.detach();
    m_lock.release();
}

bool NauRunGuardPrivate::isAnotherRunning()
{
    if (m_memory.isAttached()) return false;

    m_lock.acquire();
    const bool isRunning = m_memory.attach();
    if (isRunning)  m_memory.detach();
    m_lock.release();

    return isRunning;
}

QString NauRunGuardPrivate::generateKeyHash(const QString& key, const QString& salt)
{
    QByteArray data;
    data.append(key.toUtf8());
    data.append(salt.toUtf8());
    data = QCryptographicHash::hash(data, QCryptographicHash::Sha1).toHex();
    return data;
}

bool NauRunGuardPrivate::tryToAcquire()
{
    if (isAnotherRunning()) return false;

    m_lock.acquire();
    const bool result = m_memory.create(sizeof(quint64));
    m_lock.release();

    if (!result) {
        release();
        return false;
    }

    return true;
}


// ** NauRunGuard

NauRunGuard::NauRunGuard()
    : m_impl(new NauRunGuardPrivate)
{
}

NauRunGuard::~NauRunGuard() = default;

bool NauRunGuard::tryToAcquire()
{
    return m_impl->tryToAcquire();
}

namespace Nau
{
    // ** ShowAlreadyRunningWarning

    int ShowAlreadyRunningWarning(int argc, char* argv[])
    {
        NauApp application(argc, argv);

        const auto buttonClicked = QMessageBox::warning(
            nullptr,
            NauApp::name(),
            QObject::tr("Another instance of %1 is already running! Terminate that instance?" 
                "(Warning: this can result in unsaved progress!)").arg(NauApp::name()),
            QMessageBox::Ok | QMessageBox::Cancel,
            QMessageBox::Cancel
        );

        // Kill the current instance
        // TODO: ifdef out of the production builds
        if (buttonClicked == QMessageBox::StandardButton::Ok) {
            #ifdef Q_OS_WIN
            NauProcess processTerminator;
            processTerminator.start("taskkill", { "/im", "NauEditor.exe", "/f" });
            processTerminator.waitForFinished();
            #else
            #error Not implemented
            #endif
        }

        return application.exec();
    }
}