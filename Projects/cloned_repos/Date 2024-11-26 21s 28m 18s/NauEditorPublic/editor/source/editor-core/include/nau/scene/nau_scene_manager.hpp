// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Responsible for managing and saving scenes.

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"

#include <QJsonObject>
#include <QWidget>


// ** NauNewSceneDialog

class NAU_EDITOR_API NauNewSceneDialog : public NauDialog
{
    Q_OBJECT

public:
    NauNewSceneDialog(QWidget& parent, const NauDir& defaultDirectory);

signals:
    void eventCreateScene(const QString& path);

private:
    void createScene();

private:
    const NauDir m_defaultDirectory;
    NauLineEdit* m_inputName;
    QPushButton* m_buttonCreate;
};


// ** NauSceneManager

class NAU_EDITOR_API NauSceneManager : public QObject
{
    Q_OBJECT

public:

    struct SceneInfo
    {
        QString name;
        QString path;  // Relative to the root of the project

        bool operator==(const SceneInfo& other) const
        {
            return (this->name == other.name) && (this->path == other.path);
        }
    };

    NauSceneManager(const NauDir& root);

    void createScene(const QString& name);
    void switchScene(const NauSceneManager::SceneInfo& scene);
    void browseForScenes(QWidget& parent, const QString& projectPath);

    SceneInfo mainScene() const;
    QString mainScenePath() const;
    SceneInfo currentScene() const;
    QString currentSceneName() const;
    QString currentSceneAbsolutePath() const;
    NauDir defaultScenesDirectory() const;
    SceneInfo absolutePathToScene(const QString& path) const;
    bool sceneExists(const SceneInfo& info) const;

    static QString sceneToPath(const SceneInfo& scene);

    // Serialization
    void load(const QJsonObject& data);
    void save(QJsonObject& data);
    bool isDirty() const { return m_dirty; }

    static SceneInfo deserializeScene(const QJsonObject& data);
    static QJsonObject serializeScene(const SceneInfo& data);

signals:
    void eventAboutToSwitchScenes(bool showCancel = false);  // Needs to be emitted before any attempt to switch scenes
    void eventCreateScene(const std::string& path);
    void eventLoadScene(const NauSceneManager::SceneInfo& scene);

public:
    // TODO: Move to USD scene manager
    inline static const QString sceneSuffix = "nausd_scene";
    inline static const QString defaultDirectory = "/content/scenes";

private:
    SceneInfo     m_mainScene;
    SceneInfo     m_currentScene;
    const NauDir  m_root;
    bool          m_dirty = false;
};
