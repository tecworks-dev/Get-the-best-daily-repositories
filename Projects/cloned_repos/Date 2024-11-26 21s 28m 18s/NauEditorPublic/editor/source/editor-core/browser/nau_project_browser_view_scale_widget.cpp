// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_project_browser_view_scale_widget.hpp"
#include "nau_label.hpp"
#include "nau_buttons.hpp"
#include "themes/nau_theme.hpp"


// ** NauProjectBrowserAppearanceSlider

NauProjectBrowserAppearanceSlider::NauProjectBrowserAppearanceSlider(QWidget* widget)
    : NauSlider(widget)
{
    setPalette(Nau::Theme::current().paletteWidgetAppearanceSlider());
    setMouseTracking(true);
}

void NauProjectBrowserAppearanceSlider::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    const QRect rect = event->rect();

    // Groove painting.
    painter.setBrush(QBrush(nauPalette().brush(NauPalette::Role::Background)));
    painter.setPen(Qt::NoPen);
    QRect grooveRect = rect;
    grooveRect.setHeight(2);
    grooveRect.moveCenter(event->rect().center());

    static const int rectRadius = 1;
    painter.drawRoundedRect(grooveRect, rectRadius, rectRadius);

    // Middle tick painting
    static const QSize tickSize{2, 6};
    QRect tickRect{QPoint(0, 0), tickSize};
    tickRect.moveCenter(grooveRect.center());
    painter.drawRoundedRect(tickRect, rectRadius, rectRadius);

    // Handler painting.
    static const int hoveredHandleSide = 14;
    static const int normalHandleSide = 8;
    const auto side = hovered() ? hoveredHandleSide : normalHandleSide;
    const auto handleRadius = side / 2;

    const float alpha = maximum() - minimum() != 0 
        ? (1.0f * minimum() + value()) / (maximum() - minimum())
        : 0.0f;

    const QRect handleAreaRect = rect - QMargins(handleRadius, 0, handleRadius, 0);
    const int horizontalShift = static_cast<int>(handleAreaRect.width() * alpha);

    QRect handlerRect{0, 0, side, side};
    handlerRect.moveCenter(QPoint(horizontalShift + handleAreaRect.left() - 1, rect.center().y()));

    painter.setBrush(QBrush(nauPalette().brush(NauPalette::Role::AlternateBackground, hovered() 
        ? NauPalette::State::Hovered
        : NauPalette::State::Normal)));
    painter.drawEllipse(handlerRect);
}

QSize NauProjectBrowserAppearanceSlider::sizeHint() const
{
    return QSize(108, 14);
}


// ** NauProjectBrowserViewScaleWidget

NauProjectBrowserViewScaleWidget::NauProjectBrowserViewScaleWidget(NauWidget* parent)
    : NauWidget(parent)
{
    m_slider = new NauProjectBrowserAppearanceSlider(this);
    m_slider->setMinimum(0);
    m_slider->setMaximum(100);
    m_slider->setSingleStep(10);
    m_slider->setValue(100);
    m_slider->setOrientation(Qt::Horizontal);

    auto layout = new NauLayoutHorizontal(this);
    layout->setSpacing(8);
    layout->setContentsMargins(0, 0, 0, 0);

    auto tableView = new NauMiscButton(this);
    tableView->setIcon(Nau::Theme::current().iconTableViewIcon());
    tableView->setToolTip(tr("Click to view data in table presentation"));

    auto tileView = new NauMiscButton(this);
    tileView->setIcon(Nau::Theme::current().iconTileViewIcon());
    tileView->setToolTip(tr("Click to view data as tiles"));

    layout->addWidget(tableView);
    layout->addWidget(m_slider);
    layout->addWidget(tileView);

    connect(m_slider, &NauProjectBrowserAppearanceSlider::valueChanged, this, [this](int value){
        emit eventScaleValueChanged(1.0f * value / (m_slider->maximum() - m_slider->minimum()));
    });
    connect(tableView, &NauMiscButton::clicked, [this]{
        m_slider->setValue(m_slider->minimum());
    });
    connect(tileView, &NauMiscButton::clicked, [this]{
        m_slider->setValue(m_slider->maximum());
    });
}

void NauProjectBrowserViewScaleWidget::setScale(float newScale)
{
    newScale = std::clamp(newScale, 0.0f, 1.0f);

    static const auto eps = 0.001f;
    if (std::abs(newScale - scale()) < eps)
        return;

    m_slider->setValue(m_slider->minimum() + 
        newScale * (m_slider->maximum() - m_slider->minimum()));

    emit eventScaleValueChanged(newScale);
}

float NauProjectBrowserViewScaleWidget::scale() const
{
    assert(m_slider->maximum() != m_slider->minimum());
    return 1.0f * m_slider->value() / (m_slider->maximum() - m_slider->minimum());
}
