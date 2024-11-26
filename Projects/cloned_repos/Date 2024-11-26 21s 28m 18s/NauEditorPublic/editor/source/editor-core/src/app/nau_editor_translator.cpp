// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/app/nau_editor_translator.hpp" 
#include "nau_log.hpp"

#include <QCoreApplication>


// ** NauEditorTranslator

NauEditorTranslator::NauEditorTranslator(const QString& lang)
{
    QString const appDir = QCoreApplication::applicationDirPath();
    QString const translationsDir = appDir + QStringLiteral("/translations");

    load(m_qtTranslator, QStringLiteral("qt_") + lang, translationsDir);
    load(m_editorTranslator, QStringLiteral("editor_") + lang, translationsDir);
}

NauEditorTranslator::~NauEditorTranslator()
{
    QCoreApplication::removeTranslator(&m_qtTranslator);
    QCoreApplication::removeTranslator(&m_editorTranslator);
}

void
NauEditorTranslator::load(QTranslator& translator, QString const& fileNameBase, QString const& dir)
{
    if (translator.load(fileNameBase, dir))
    {
        QCoreApplication::installTranslator(&translator);
    }
    else
    {
        NED_ERROR("Error loading translation file {}", fileNameBase);
    }
}