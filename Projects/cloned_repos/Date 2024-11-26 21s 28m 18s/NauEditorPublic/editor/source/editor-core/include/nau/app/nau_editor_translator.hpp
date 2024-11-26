// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Editor application ui translation tools

#pragma once

#include <QString>
#include <QTranslator>


// ** NauEditorTranslator

class NauEditorTranslator
{
public:
    explicit NauEditorTranslator(const QString& lang);
    ~NauEditorTranslator();

private:
    void load(QTranslator& translator, QString const& fileNameBase, QString const& dir);

private:
    QTranslator m_qtTranslator;
    QTranslator m_editorTranslator;
};