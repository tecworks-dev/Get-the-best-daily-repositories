// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Enhances the standard qt label.

#pragma once

#include "nau_widget.hpp"

#include <QLabel>


// ** NauLabel

class NAU_EDITOR_API NauLabel : public QLabel
{
    Q_OBJECT
public:
    NauLabel(const QString& text, QWidget* parent = nullptr);
    NauLabel(const QPixmap& px, QWidget* parent = nullptr);
    NauLabel(QWidget* parent = nullptr);

signals:
    void clicked();

protected:
    virtual void mousePressEvent(QMouseEvent* event) override;
    virtual void mouseReleaseEvent(QMouseEvent* event) override;
};