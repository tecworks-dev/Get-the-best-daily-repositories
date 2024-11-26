// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_script_manager.hpp"
#include "nau/nau_editor_delegates.hpp"
#include "nau/scene/nau_scene_settings.hpp"

#include <QDir>
#include <QStringList>
#include <QDirIterator>
#include <QObject>


// ** NauScriptManager

NauScriptManager::NauScriptManager()
{
    auto scriptChangesWatch = [this](NauEditorMode mode) {
        if (mode == NauEditorMode::Play) {
            startWatch();
        } else {
            stopWatch();
        }
    };

    m_scriptChangesWatchCbId = NauEditorEngineDelegates::onEditorModeChanged.addCallback(scriptChangesWatch);
}

NauScriptManager::~NauScriptManager()
{
    NauEditorEngineDelegates::onEditorModeChanged.deleteCallback(m_scriptChangesWatchCbId);
}

void NauScriptManager::startWatch()
{
    QStringList scriptFiles;

    // Add initial scripts to watch
    //const auto& initialLoadedScripts = ecs::g_scenes->GetSceneInitScripts();
    //for (const auto& script : initialLoadedScripts) {
    //    scriptFiles.push_back(script.c_str());
    //}

    //// Add runtime scripts to watch
    //const auto& runtimeLoadedScripts = ecs::g_scenes->GetSceneRuntimeScripts();
    //for (const auto& script : runtimeLoadedScripts) {
    //    scriptFiles.push_back(script.c_str());
    //}

    if (scriptFiles.isEmpty()) {
        return;
    }

    m_scriptsWatcher.addPaths(scriptFiles);
    m_scriptsWatcher.connect(&m_scriptsWatcher, &QFileSystemWatcher::fileChanged, this, &NauScriptManager::reloadScript);
}

void NauScriptManager::stopWatch()
{
    QStringList filesPaths = m_scriptsWatcher.files();
    if (filesPaths.isEmpty()) {
        return;
    }

    m_scriptsWatcher.removePaths(filesPaths);
    m_scriptsWatcher.disconnect(&m_scriptsWatcher, &QFileSystemWatcher::fileChanged, this, &NauScriptManager::reloadScript);
}

void NauScriptManager::reloadScript(const QString& path)
{  
    //if (!bind_dascript::is_script_loaded(path.toUtf8().constData())) {
    //    return;
    //}

    //// TODO: Made single file hotreload
    //gamescripts::unload_all_das_scripts_without_debug_agents();
    //gamescripts::reload_das_init();
    //gamescripts::reload_das_modules();
    //gamescripts::main_thread_post_load();
    //g_entity_mgr->broadcastEvent(EventDaScriptReloaded());
}