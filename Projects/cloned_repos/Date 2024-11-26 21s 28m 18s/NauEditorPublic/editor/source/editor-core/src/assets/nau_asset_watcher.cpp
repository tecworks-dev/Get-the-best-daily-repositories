// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/assets/nau_asset_watcher.hpp"

#include "nau_log.hpp"
#include "nau_assert.hpp"
#include <QFileInfo>
#include <QDirIterator>


// ** NauAssetWatcher

NauAssetWatcher::NauAssetWatcher(const QString& directory)
    : m_skipFlags({ false, false, false })
    , m_rootDirectory(std::move(directory))
{
    updateDirectory(rootDirectory(), false);

    connect(&m_assetsWatcher, &QFileSystemWatcher::fileChanged, [this](const QString& filePath) {
        const QFileInfo fileInfo { filePath };
        if (fileInfo.exists()) {
            updateAsset(filePath, UpdateReason::Changed);
        }
    });

    connect(&m_assetsWatcher, &QFileSystemWatcher::directoryChanged, this, [this](const QString& path) {
        updateDirectory(path);
    });
}

const QString& NauAssetWatcher::rootDirectory() const
{
    return m_rootDirectory;
}

void NauAssetWatcher::skipNextEvent(UpdateReason reason, bool flag)
{
    NED_ASSERT(reason < UpdateReason::CountReasons);
    m_skipFlags[static_cast<unsigned>(reason)] = flag;
}

void NauAssetWatcher::updateDirectory(const QString& dirPath, bool isInitialized)
{
    QStringList pathList;
    pathList.reserve(static_cast<qsizetype>(m_watchedPaths.size()));
    for (const auto& [path, type]: m_watchedPaths) {
        const QFileInfo info{path };
        if (!path.startsWith(dirPath) || info.exists()) {
            continue;
        }
        if (type == FileSystemObject::File) {
            updateAsset(path, UpdateReason::Removed);
        }
        pathList.emplace_back(path);
    }
    if (!pathList.isEmpty()) {
        for (const QString& path: pathList) {
            m_watchedPaths.erase(path);
        }
        m_assetsWatcher.removePaths(pathList);
        pathList.clear();
    }
    pathList.push_back(dirPath);
    {
        QDirIterator directoryIt(dirPath, QDir::Files, QDirIterator::Subdirectories);
        while (directoryIt.hasNext()) {
            directoryIt.next();
            QString filePath = std::move(directoryIt.filePath());
            if (m_watchedPaths.contains(filePath)) {
                continue;
            }
            m_watchedPaths.emplace(filePath, FileSystemObject::File);

            // TODO: If the function is called from the constructor,
            // resources may be registered twice each.
            // The engine allows this. Fix it in the future. 
            if (isInitialized) {
                updateAsset(filePath, UpdateReason::Added);
            }

            pathList.emplace_back(std::move(filePath));
        }
    }
    {
        QDirIterator directoryIt(dirPath, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
        while (directoryIt.hasNext()) {
            directoryIt.next();
            QString filePath = std::move(directoryIt.filePath());
            if (m_watchedPaths.contains(filePath)) {
                continue;
            }
            m_watchedPaths.emplace(filePath, FileSystemObject::Directory);
            pathList.emplace_back(std::move(filePath));
        }
    }
    if (!pathList.isEmpty()) {
        m_assetsWatcher.addPaths(pathList);
    }
}

void NauAssetWatcher::updateAsset(const QString& assetPath, NauAssetWatcher::UpdateReason reason)
{
    if (m_skipFlags[static_cast<unsigned>(reason)]) {
        m_skipFlags[static_cast<unsigned>(reason)] = false;
        return;
    }

    switch (reason) {
        case UpdateReason::Changed:
            emit eventAssetChanged(assetPath.toUtf8().constData());
            break;
        case UpdateReason::Added:            
            emit eventAssetAdded(assetPath.toUtf8().constData());
            break;
        case UpdateReason::Removed:
            emit eventAssetRemoved(assetPath.toUtf8().constData());
            break;
        case UpdateReason::CountReasons:
            NED_ASSERT(false);
            break;
    }
}