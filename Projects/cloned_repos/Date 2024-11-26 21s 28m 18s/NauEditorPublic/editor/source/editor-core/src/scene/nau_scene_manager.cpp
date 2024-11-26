// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_assert.hpp"
#include "nau_label.hpp"
#include "nau_log.hpp"
#include "nau/scene/nau_scene_manager.hpp"
#include "nau_widget_utility.hpp"
#include "nau/app/nau_qt_app.hpp"

#include <QCoreApplication>
#include <QFileDialog>
#include <QJsonArray>
#include <QMessageBox>


// ** NauNewSceneDialog

NauNewSceneDialog::NauNewSceneDialog(QWidget& parent, const NauDir& defaultDirectory)
    : NauDialog(&parent)
    , m_defaultDirectory(defaultDirectory)
    , m_inputName(new NauLineEdit(this))
    , m_buttonCreate(new QPushButton(tr("Create"), this))
{
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(tr("Create New Scene"));
    setFixedWidth(400);

    // Scene name
    auto layout = new NauLayoutVertical(this);
    layout->addWidget(new NauLabel(tr("Scene Name:")));

    QRegularExpression rx("^[A-Za-z0-9_-]{1,32}$");
    m_inputName->setValidator(new QRegularExpressionValidator(rx, this));
    layout->addWidget(m_inputName);

    // Create scene
    m_buttonCreate->setEnabled(false);
    layout->addWidget(m_buttonCreate);
    connect(m_buttonCreate, &QPushButton::clicked, this, &NauNewSceneDialog::createScene);

    // Enable create button
    connect(m_inputName, &NauLineEdit::textChanged, [this] {
        m_buttonCreate->setEnabled(!m_inputName->text().isEmpty());
    });
}

void NauNewSceneDialog::createScene()
{
    setEnabled(false);
    const auto name = m_inputName->text();
    NED_ASSERT(!name.isEmpty());

    const NauFile sceneFile(m_defaultDirectory.absolutePath() + NauDir::separator() + name + "." + NauSceneManager::sceneSuffix);
    if (sceneFile.exists()) {
        QMessageBox::warning(this, NauApp::name(),
            tr("Scene %1 at %2 already exists!").arg(name).arg(m_defaultDirectory.absolutePath()), QMessageBox::Ok);
        setEnabled(true);
        return;
    }

    emit eventCreateScene(name);
    accept();
}


// ** NauSceneManager

NauSceneManager::NauSceneManager(const NauDir& root)
    : m_mainScene({ "main", defaultDirectory })   // Default for new projects
    , m_currentScene(m_mainScene)
    , m_root(root)
{
    NED_DEBUG("Editor scene manager initialised");
}

void NauSceneManager::load(const QJsonObject& data)
{
    NED_DEBUG("Saving scene data");

    // Main scene
    m_mainScene = deserializeScene(data["MainScene"].toObject());
    switchScene(m_mainScene);
}

void NauSceneManager::save(QJsonObject& data)
{
    NED_DEBUG("Loading scene data");

    // Main scene
    data["MainScene"] = serializeScene(m_mainScene);
    
    m_dirty = false;
}

void NauSceneManager::createScene(const QString& name)
{
    emit eventAboutToSwitchScenes();

    NED_DEBUG("Creating new scene:", name);
    NED_ASSERT(!name.isEmpty());

    // Create scenes path
    const auto scenesPath = defaultScenesDirectory().absolutePath();
    NauDir scenesDirectory(scenesPath);
    if (!scenesDirectory.exists() && !scenesDirectory.mkdir(scenesPath)) {
        NED_CRITICAL("Failed to create the default scenes directory!");
        return;
    }

    // Create the file
    const QString scenePath = scenesDirectory.absoluteFilePath(name + "." + sceneSuffix);
    emit eventCreateScene(scenePath.toUtf8().constData());

    // Update scenes
    m_currentScene = { name, defaultDirectory };

    NED_DEBUG("Success!");
    emit eventLoadScene(m_currentScene);
}

void NauSceneManager::switchScene(const SceneInfo& info)
{
    emit eventAboutToSwitchScenes();

    NED_ASSERT(!info.name.isEmpty());
    NED_ASSERT(!info.path.isEmpty());
    NED_TRACE("Scene manager: switching from scene {} to scene {}", m_currentScene.name, info.name);

    m_currentScene = info;
    
    emit eventLoadScene(m_currentScene);
}

void NauSceneManager::browseForScenes(QWidget& parent, const QString& projectPath)
{
    NED_DEBUG("Attempting load a scene from file browser...");
    
    const QString scenePath = QFileDialog::getOpenFileName(&parent, QObject::tr("Open scene"), projectPath, QObject::tr("Nau scene (*.%1)").arg(sceneSuffix));
    if (scenePath.isEmpty()) {
        return;
    }
    
    switchScene(absolutePathToScene(scenePath));
}

NauSceneManager::SceneInfo NauSceneManager::mainScene() const
{
    return m_mainScene;
}

QString NauSceneManager::mainScenePath() const
{
    return sceneToPath(mainScene());
}

NauSceneManager::SceneInfo NauSceneManager::currentScene() const
{
    return m_currentScene;
}

QString NauSceneManager::currentSceneName() const
{
    return m_currentScene.name;
}

QString NauSceneManager::currentSceneAbsolutePath() const
{
    return m_root.absolutePath() + QDir::separator() + sceneToPath(m_currentScene);
}

NauDir NauSceneManager::defaultScenesDirectory() const
{
    return m_root.absolutePath() + defaultDirectory;
}

QString NauSceneManager::sceneToPath(const NauSceneManager::SceneInfo& scene)
{
    NED_ASSERT(!scene.name.isEmpty());
    NED_ASSERT(!scene.path.isEmpty());
    const auto fullPath = QString("%1/%2.%3").arg(scene.path).arg(scene.name).arg(sceneSuffix);  // Relative to the root of the project
    return fullPath;
}

NauSceneManager::SceneInfo NauSceneManager::absolutePathToScene(const QString& path) const
{
    NED_ASSERT(!path.isEmpty());
    
    // Convert absolute path to relative
    const auto relativePath = QFileInfo(m_root.relativeFilePath(path));
    return { relativePath.baseName(), relativePath.path() };
}

bool NauSceneManager::sceneExists(const SceneInfo& info) const
{
    NauFile sceneFile(m_root.absolutePath() + QDir::separator() + NauSceneManager::sceneToPath(info));
    return sceneFile.exists();
}

NauSceneManager::SceneInfo NauSceneManager::deserializeScene(const QJsonObject& data)
{
    return { data["Name"].toString(), data["Path"].toString() };
}

QJsonObject NauSceneManager::serializeScene(const NauSceneManager::SceneInfo& data)
{
    QJsonObject result;
    result["Name"] = data.name;
    result["Path"] = data.path;
    return result;
}
