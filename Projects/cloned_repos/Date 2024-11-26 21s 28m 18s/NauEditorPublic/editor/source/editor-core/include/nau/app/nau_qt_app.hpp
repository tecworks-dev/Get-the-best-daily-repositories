// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Nau qt app wrapper

#pragma once

#include "nau/nau_editor_config.hpp"
#include "nau/app/nau_editor_translator.hpp"

#include <QApplication>
#include <QString>


// ** NauApp

class NAU_EDITOR_API NauApp
{
public:
    NauApp(int argc, char* argv[]);

    void setStyleSheet(const QString& resourcePath);
    void setLanguage(const QString& lang);

    int exec();
    void processEvents() { m_app.processEvents(); }

    bool ignoreProjectModules() { return m_ignoreProjectModules; }

    static std::string sessionID();
    static QString name();

private:
    QApplication m_app;

    std::unique_ptr<NauEditorTranslator> m_translator;

    bool m_ignoreProjectModules = false;
};