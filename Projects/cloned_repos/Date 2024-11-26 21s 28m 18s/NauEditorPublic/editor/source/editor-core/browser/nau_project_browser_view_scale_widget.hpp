// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// A widget to manipulate the scale of the content view in the project browser.

#pragma once

#include "baseWidgets/nau_widget.hpp"


// ** NauProjectBrowserAppearanceSlider

class NAU_EDITOR_API NauProjectBrowserAppearanceSlider : public NauSlider
{
    Q_OBJECT
public:
    explicit NauProjectBrowserAppearanceSlider(QWidget* widget = nullptr);

    void paintEvent(QPaintEvent* event) override;
    QSize sizeHint() const override;
};


// ** NauProjectBrowserViewScaleWidget

class NAU_EDITOR_API NauProjectBrowserViewScaleWidget : public NauWidget
{
    Q_OBJECT
public:
    explicit NauProjectBrowserViewScaleWidget(NauWidget* parent);

    // Set current scale value. Valid values between [0..1].
    // Emits eventScaleValueChanged().
    void setScale(float scale);

    float scale() const;

signals:
    // Emitted when user changes the scale. scale is always in range [0..1].
    void eventScaleValueChanged(float scale);

private:
    NauProjectBrowserAppearanceSlider* m_slider = nullptr;
};