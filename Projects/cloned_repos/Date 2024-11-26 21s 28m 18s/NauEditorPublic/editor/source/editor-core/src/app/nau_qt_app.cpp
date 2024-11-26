// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/app/nau_qt_app.hpp"
#include "nau_editor_version.hpp"
#include "nau_log.hpp"
#include "src/utility/nau_guid.hpp"

#include <QCommandLineParser>
#include <QFile>


// ** NauApp

NauApp::NauApp(int argc, char* argv[])
    : m_app(argc, argv)
{
    m_app.setApplicationName("Editor");  // Used for system paths, for display purposes use NauApp::name()!
    m_app.setApplicationVersion(NauEditorVersion::current().asQtString());
    m_app.setOrganizationName("Nau");
    m_app.setOrganizationDomain("nauengine.org");  // Among other things, used to store settings on Apple operating systems

    QCommandLineParser parser;
    QCommandLineOption ignoreProjectModules(QStringList() << "ignore-project-modules" << "ipm",
            QStringLiteral("Do not try to compile and load project modules"));
    parser.addOption(ignoreProjectModules);
    parser.process(m_app);

     m_ignoreProjectModules = parser.isSet(ignoreProjectModules);
}

void NauApp::setStyleSheet(const QString& resourcePath)
{
    QFile styleSheetFile(resourcePath);
    if (!styleSheetFile.open(QFile::ReadOnly)) {
        NED_ERROR("Failed to load application style sheet: {}", resourcePath);
        return;
    }
    const QString styleSheet = QLatin1String(styleSheetFile.readAll());
    m_app.setStyleSheet(styleSheet);
    NED_DEBUG("Loaded global application style sheet: {}", resourcePath);
}

void NauApp::setLanguage(const QString& lang)
{
    m_translator = std::make_unique<NauEditorTranslator>(lang);
}

std::string NauApp::sessionID()
{
    static NauObjectGUID sessionId;
    return sessionId;
}

QString NauApp::name()
{
    return "Nau Editor";
}

int NauApp::exec()
{
    return m_app.exec();
}