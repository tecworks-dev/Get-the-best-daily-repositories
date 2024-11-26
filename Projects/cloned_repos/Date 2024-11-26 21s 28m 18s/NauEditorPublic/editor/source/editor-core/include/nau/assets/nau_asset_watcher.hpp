// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Asset modify observer

#pragma once

#include "nau/nau_editor_config.hpp"

#include <QObject>
#include <QFileSystemWatcher>

#include <string>
#include <memory>


// ** NauAssetWatcher
//
// Asset modify observer

class NAU_EDITOR_API NauAssetWatcher : public QObject
{
Q_OBJECT
public:
    enum class UpdateReason : uint8_t
    {
        Changed,
        Added,
        Removed,
        CountReasons
    };

    explicit NauAssetWatcher(const QString& watchDir);

    [[nodiscard]]
    const QString& rootDirectory() const;
    void skipNextEvent(UpdateReason reason, bool flag = true);

protected:
    enum class FileSystemObject : uint8_t
    {
        Directory,
        File
    };

    // It uses virtual functions, so it must be called in the constructor of the final class
    void updateDirectory(const QString& dirPath, bool isInitialized = true);

private:
    void updateAsset(const QString& assetPath, UpdateReason reason);

signals:
    void eventAssetRemoved(const std::string& assetPath);
    void eventAssetAdded(const std::string& assetPath);
    void eventAssetChanged(const std::string& assetPath);

private:
    std::array<bool, static_cast<unsigned>(UpdateReason::CountReasons)> m_skipFlags;
    std::unordered_map<QString, FileSystemObject> m_watchedPaths;
    QFileSystemWatcher m_assetsWatcher;
    QString m_rootDirectory;
};