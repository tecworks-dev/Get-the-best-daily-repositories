// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Base toolbar

#pragma once

#include "nau_widget.hpp"

#include <functional>


// ** NauToolbarSection

class NauIcon;

class NAU_EDITOR_API NauToolbarSection : public NauWidget
{
    Q_OBJECT

    friend class NauToolbarBase;

public:
    
    // Which side children widgets are squeezed towards
    enum Alignment
    {
        Left,      // [w__]
        Center,    // [_w_]
        Right      // [__w]
    };

    template<typename Sender, typename Signal>
    NauToolButton* addButton(NauIcon icon, const QString& tooltip, Sender* sender, Signal signal);
    NauToolButton* addButton(NauIcon icon, const QString& tooltip, std::function<void()> callback);
    void addMenu();
    void addSeparator();

    void addExternalWidget(QWidget* widget);

protected:
    NauToolbarSection(Alignment alignment, NauToolbarBase* parent);

private:
    NauToolButton* addButtonInternal(NauIcon icon, const QString& tooltip);

private:
    NauLayoutHorizontal* m_layout;
};


// ** NauToolbarBase

class NAU_EDITOR_API NauToolbarBase : public NauWidget
{
    Q_OBJECT

public:

    enum Section {
        Left,
        Middle,
        Right
    };

    NauToolbarBase(QWidget* parent);

    NauToolbarSection* addSection(NauToolbarSection::Alignment alignment);

protected:
    void paintEvent(QPaintEvent* event) override;
};


template<typename Sender, typename Signal>
inline NauToolButton* NauToolbarSection::addButton(NauIcon icon, const QString& tooltip, Sender* sender, Signal signal)
{
    auto button = addButtonInternal(icon, tooltip);
    connect(button, &NauToolButton::clicked, sender, signal);
    return button;
}
