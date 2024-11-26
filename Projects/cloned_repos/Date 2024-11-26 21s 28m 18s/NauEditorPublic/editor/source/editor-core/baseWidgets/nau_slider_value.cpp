// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_slider_value.hpp"
#include "nau_static_text_label.hpp"
#include "themes/nau_theme.hpp"

#include <QPainterPath>


// ** NauSliderController

NauSliderController::NauSliderController(NauWidget* parent)
    : NauWidget(parent)
{
}

void NauSliderController::updateSlider(double positionX, Reason reason) noexcept
{
    if (m_pressed) {
        const double ratio = positionX / static_cast<double>(width());
        emit eventChangedValue(std::clamp(ratio, 0.0, 1.0), reason);
    }
}

void NauSliderController::mousePressEvent(QMouseEvent* event)
{
    m_pressed = true;
    updateSlider(event->position().x(), Reason::Begin);
}

void NauSliderController::mouseReleaseEvent(QMouseEvent* event)
{
    updateSlider(event->position().x(), Reason::End);
    m_pressed = false;
}

void NauSliderController::mouseMoveEvent(QMouseEvent* event)
{
    updateSlider(event->position().x(), Reason::Move);
}


// ** NauSliderValueController

NauSliderValueController::NauSliderValueController(NauLineEdit* lineEdit)
    : QObject(lineEdit)
{
}

bool NauSliderValueController::editing() const noexcept
{
    return static_cast<NauLineEdit*>(parent())->editing();
}

bool NauSliderValueController::eventFilter(QObject* object, QEvent* event)
{
    const QEvent::Type eventType = event->type();
    const bool isFocusEvent = eventType == QEvent::FocusAboutToChange;
    const auto* focusEvent = isFocusEvent ? static_cast<QFocusEvent*>(event) : nullptr;

    if (isFocusEvent && (focusEvent->reason() == Qt::MouseFocusReason)) {
        static_cast<NauLineEdit*>(parent())->clearFocus();
    }
    return QObject::eventFilter(object, event);
}


// ** NauAbstractSlider

NauAbstractSlider::NauAbstractSlider(NauWidget* parent)
    : NauWidget(parent)
    , m_prevRatioCache(0.0)
    , m_controller(new NauSliderController(this))
    , m_sliderSize(8, 7)
    , m_hovered(false)
{
    constexpr int borderWidth = 1;
    auto* contentLayout = new NauLayoutVertical(this);
    contentLayout->setContentsMargins(borderWidth, borderWidth, borderWidth, borderWidth);
    contentLayout->addWidget(m_controller);

    setLayout(contentLayout);
    setFixedHeight(32);
    setAttribute(Qt::WA_Hover);

    m_controller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_controller->setFixedHeight(m_sliderSize.height());

    connect(m_controller, &NauSliderController::eventChangedValue, [this] (double ratio, NauSliderController::Reason reason) {
        updateSlider(ratio, reason);
    });
}

bool NauAbstractSlider::editing() const noexcept
{
    bool result = false;
    for (const auto* controller : m_valueControllers) {
        result = result || controller->editing();
    }
    return result;
}

QSize NauAbstractSlider::sliderSize() const noexcept
{
    return sliding() ? m_sliderSize : QSize{ 2, m_sliderSize.height() / 2 };
}

NauSliderValueController* NauAbstractSlider::addValueController(NauLineEdit* lineEdit)
{
    return m_valueControllers.emplace_back(new NauSliderValueController(lineEdit));
}

void NauAbstractSlider::paintEvent(QPaintEvent* event)
{
    constexpr int borderWidth = 1;
    constexpr uint32_t borderRound = 2;
    constexpr QPointF offset = QPointF{ borderWidth, borderWidth };

    const NauAbstractTheme& theme = Nau::Theme::current();
    const NauPalette& palette = theme.paletteNumericSlider();

    const bool isEditing = editing();
    const QSize pathSize = size() - QSize{ borderWidth, borderWidth } * 2;

    QPainter painter{ this };
    painter.setRenderHint(QPainter::Antialiasing);

    QPainterPath path;
    path.addRoundedRect(offset.x(), offset.y(), pathSize.width(), pathSize.height(), borderRound, borderRound);
    painter.fillPath(path, palette.color(NauPalette::Role::Background));

    const auto [sliderWidth, sliderHeight] = sliderSize();
    if (sliderHeight > 0 && !isEditing) {
        const int halfSliderWidth = sliderWidth / 2;
        const auto [fillLeftRatio, fillRightRatio] = ratioValues();
        const int widgetFilledLeft = static_cast<int>(std::round(pathSize.width() * fillLeftRatio));
        const int widgetFilledRight = static_cast<int>(std::round(pathSize.width() * fillRightRatio));

        const int sliderPositionY = static_cast<int>(borderWidth + pathSize.height() - sliderHeight);
        const int sliderPositionLeft = std::clamp(widgetFilledLeft - 1, borderWidth, pathSize.width() - halfSliderWidth);
        const int sliderPositionRight = std::clamp(widgetFilledRight - 1, borderWidth, pathSize.width() - halfSliderWidth);

        paintRect(painter, palette.color(NauPalette::Role::AlternateBackground), borderWidth, sliderPositionY, pathSize.width(), sliderHeight);
        paintRect(painter, palette.color(NauPalette::Role::Foreground), widgetFilledLeft, sliderPositionY, widgetFilledRight - widgetFilledLeft, sliderHeight);

        // Has left and right range values
        if (m_hovered && (m_valueControllers.size() > 1)) {
            paintRect(painter, palette.color(NauPalette::Role::AlternateForeground), sliderPositionLeft, sliderPositionY, sliderWidth, sliderHeight);
        }
        if (m_hovered) {
            paintRect(painter, palette.color(NauPalette::Role::AlternateForeground), sliderPositionRight, sliderPositionY, sliderWidth, sliderHeight);
        }
    }

    if (m_hovered || isEditing) {
        painter.setPen({palette.color(NauPalette::Role::Border), borderWidth});
        painter.drawPath(path);
    }
}

bool NauAbstractSlider::event(QEvent* qEvent)
{
    if (qEvent->type() == QEvent::Leave) {
        m_hovered = false;
    } else if (qEvent->type() == QEvent::Enter) {
        m_hovered = true;
    }
    return NauWidget::event(qEvent);
}

void NauAbstractSlider::paintRect(QPainter& painter, const NauColor& color, int x, int y, int width, int height, float rx, float ry)
{
    QPainterPath path;
    path.addRoundedRect(x, y, width, height, rx, ry);
    painter.fillPath(path, color);
}


// ** NauSliderIntValue

NauSliderIntValue::NauSliderIntValue(NauWidget* parent)
    : NauAbstractSlider(parent)
    , m_spinBox(new NauSpinBox(this))
{
    static_cast<NauLayoutVertical*>(layout())->insertWidget(0, m_spinBox);

    setRangedValue({});
    setupSpinBox(m_spinBox);

    connect(m_spinBox, &QSpinBox::valueChanged, this, &NauSliderIntValue::eventRangeChanged);
}

void NauSliderIntValue::setValue(int value) noexcept
{
    m_spinBox->setValue(value);
}

void NauSliderIntValue::setMinimum(int minimum) noexcept
{
    m_spinBox->setMinimum(minimum);
}

void NauSliderIntValue::setMaximum(int maximum) noexcept
{
    m_spinBox->setMaximum(maximum);
}

void NauSliderIntValue::setRangedValue(const NauRangedValue<int>& range) noexcept
{
    setMinimum(range.minimum());
    setMaximum(range.maximum());
    setValue(range.value());
}

NauRangedValue<int> NauSliderIntValue::rangedValue() const noexcept
{
    return { value(), m_spinBox->minimum(), m_spinBox->maximum() };
}

int NauSliderIntValue::value() const noexcept
{
    return m_spinBox->value();
}

void NauSliderIntValue::updateSlider(double ratio, NauSliderController::Reason reason)
{
    QSignalBlocker blocker{ this };
    if (reason == NauSliderController::Reason::End) {
        blocker.unblock();
    }
    const double rangeLength = static_cast<double>(m_spinBox->maximum()) - static_cast<double>(m_spinBox->minimum());
    setValue(static_cast<int>(std::round(rangeLength * ratio + static_cast<double>(m_spinBox->minimum()))));
    emit eventRangeChanged();
    update();
}

NauAbstractSlider::RangeRatio NauSliderIntValue::ratioValues() const noexcept
{
    const double rangeLength = static_cast<double>(m_spinBox->maximum()) - static_cast<double>(m_spinBox->minimum());
    const double filledLength = static_cast<double>(m_spinBox->value()) - static_cast<double>(m_spinBox->minimum());
    const double fillRatio = filledLength / rangeLength;
    return { 0.0, fillRatio };
}


// ** NauSliderFloatValue

NauSliderFloatValue::NauSliderFloatValue(NauWidget* parent)
    : NauAbstractSlider(parent)
    , m_spinBox(new NauDoubleSpinBox(this))
{
    static_cast<NauLayoutVertical*>(layout())->insertWidget(0, m_spinBox);

    setRangedValue({});
    setupSpinBox(m_spinBox);

    connect(m_spinBox, &QDoubleSpinBox::valueChanged, this, &NauAbstractSlider::eventRangeChanged);
}

void NauSliderFloatValue::setValue(float value) noexcept
{
    m_spinBox->setValue(value);
}

void NauSliderFloatValue::setMinimum(float minimum) noexcept
{
    m_spinBox->setMinimum(minimum);
}

void NauSliderFloatValue::setMaximum(float maximum) noexcept
{
    m_spinBox->setMaximum(maximum);
}

void NauSliderFloatValue::setRangedValue(const NauRangedValue<float>& range) noexcept
{
    setMinimum(range.minimum());
    setMaximum(range.maximum());
    setValue(range.value());
}

NauRangedValue<float> NauSliderFloatValue::rangedValue() const noexcept
{
    return { value(), static_cast<float>(m_spinBox->minimum()), static_cast<float>(m_spinBox->maximum()) };
}

float NauSliderFloatValue::value() const noexcept
{
    return static_cast<float>(m_spinBox->value());
}

NauAbstractSlider::RangeRatio NauSliderFloatValue::ratioValues() const noexcept
{
    const double rangeLength = m_spinBox->maximum() - m_spinBox->minimum();
    const double filledLength = m_spinBox->value() - m_spinBox->minimum();
    const double fillRatio = filledLength / rangeLength;
    return { 0.0, fillRatio };
}

void NauSliderFloatValue::updateSlider(double ratio, NauSliderController::Reason reason)
{
    QSignalBlocker blocker{ this };
    if (reason == NauSliderController::Reason::End) {
        blocker.unblock();
    }
    const double rangeLength = m_spinBox->maximum() - m_spinBox->minimum();
    setValue(static_cast<float>(rangeLength * ratio + m_spinBox->minimum()));
    emit eventRangeChanged();
    update();
}


// ** NauSliderIntPair

NauSliderIntPair::NauSliderIntPair(NauWidget* parent)
    : NauAbstractSlider(parent)
    , m_spinBoxLeft(new NauSpinBox(this))
    , m_spinBoxRight(new NauSpinBox(this))
    , m_leftEditing(false)
    , m_rightEditing(false)
{
    auto* separator = new NauStaticTextLabel("-", this);
    const QFont& font = separator->font();
    separator->setFont(NauFont(font.family(), 12, font.weight(), font.italic()));
    separator->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    separator->setFixedHeight(16);

    auto* dataLayout = new NauLayoutHorizontal;
    dataLayout->addWidget(m_spinBoxLeft);
    dataLayout->addWidget(separator);
    dataLayout->addWidget(m_spinBoxRight);

    auto* dataWidget = new NauWidget(this);
    dataWidget->setLayout(dataLayout);
    dataWidget->setFixedHeight(22);
    dataWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    static_cast<NauLayoutVertical*>(layout())->insertWidget(0, dataWidget);

    setRangedPair({});
    setupSpinBox(m_spinBoxLeft);
    setupSpinBox(m_spinBoxRight);

    connect(m_spinBoxLeft, &QSpinBox::valueChanged, this, &NauAbstractSlider::eventRangeChanged);
    connect(m_spinBoxRight, &QSpinBox::valueChanged, this, &NauAbstractSlider::eventRangeChanged);
}

void NauSliderIntPair::setMinimum(int minimum) noexcept
{
    m_spinBoxLeft->setMinimum(minimum);
    m_spinBoxRight->setMinimum(minimum);
}

void NauSliderIntPair::setMaximum(int maximum) noexcept
{
    m_spinBoxLeft->setMaximum(maximum);
    m_spinBoxRight->setMaximum(maximum);
}

void NauSliderIntPair::setRangedPair(const NauRangedPair<int>& range) noexcept
{
    m_range = range;
    setMinimum(range.minimum());
    setMaximum(range.maximum());
    setPair(range.left(), range.right());
}

void NauSliderIntPair::setPair(int left, int right) noexcept
{
    m_spinBoxLeft->setValue(left);
    m_spinBoxRight->setValue(right);
}

NauRangedPair<int> NauSliderIntPair::rangedPair() const noexcept
{
    return {
        std::min(m_spinBoxLeft->value(), m_range.right()),
        std::max(m_range.left(), m_spinBoxRight->value()),
        m_range.minimum(),
        m_range.maximum()
    };
}

int NauSliderIntPair::left() const noexcept
{
    return m_spinBoxLeft->value();
}

int NauSliderIntPair::right() const noexcept
{
    return m_spinBoxRight->value();
}

NauAbstractSlider::RangeRatio NauSliderIntPair::ratioValues() const noexcept
{
    const double rangeLength = static_cast<double>(m_spinBoxLeft->maximum()) - static_cast<double>(m_spinBoxLeft->minimum());
    const double left = static_cast<double>(m_spinBoxLeft->value()) - static_cast<double>(m_spinBoxLeft->minimum());
    const double right = static_cast<double>(m_spinBoxRight->value()) - static_cast<double>(m_spinBoxRight->minimum());

    const double leftRatio = left / rangeLength;
    const double rightRatio = right / rangeLength;

    return { leftRatio, rightRatio };
}

void NauSliderIntPair::updateSlider(double ratio, NauSliderController::Reason reason)
{
    const double rangeLength = static_cast<double>(m_spinBoxLeft->maximum()) - static_cast<double>(m_spinBoxLeft->minimum());
    const int value = static_cast<int>(std::round(ratio * rangeLength));

    QSignalBlocker blocker{ this };
    if (reason == NauSliderController::Reason::Begin) {
        const double halfSliderWidth = static_cast<double>(sliderInputWidth()) * 0.5;
        const int extendArea = static_cast<int>(rangeLength / static_cast<double>(width()) * halfSliderWidth);
        m_leftEditing = (value - extendArea) <= (m_spinBoxLeft->value() - m_spinBoxLeft->minimum());
        m_rightEditing = !m_leftEditing && (value + extendArea) >= (m_spinBoxRight->value() - m_spinBoxRight->minimum());
        m_prevRatioCache = ratio;
    } else if (reason == NauSliderController::Reason::End) {
        m_leftEditing = false;
        m_rightEditing = false;
        blocker.unblock();
    }
    if (m_leftEditing) {
        const double newLeftValue = std::min(value + m_spinBoxLeft->minimum(), m_spinBoxRight->value());
        setPair(static_cast<int>(newLeftValue), m_spinBoxRight->value());
    } else if (m_rightEditing) {
        const double newRightValue = std::max(value + m_spinBoxLeft->minimum(), m_spinBoxLeft->value());
        setPair(m_spinBoxLeft->value(), static_cast<int>(newRightValue));
    } else {
        const double pairLength = (static_cast<double>(m_spinBoxRight->value()) - static_cast<double>(m_spinBoxLeft->value()));
        const auto [left, right] = ratioValues();
        const double newLeftRatio = std::clamp(left + ratio - m_prevRatioCache, 0.0, 1.0 - pairLength / rangeLength);
        const double newLeftValue = std::round(rangeLength * newLeftRatio + static_cast<double>(m_spinBoxLeft->minimum()));
        setPair(static_cast<int>(newLeftValue), static_cast<int>(newLeftValue + pairLength));
    }
    emit eventRangeChanged();
    m_prevRatioCache = ratio;
    update();
}


// ** NauSliderFloatPair

NauSliderFloatPair::NauSliderFloatPair(NauWidget* parent)
    : NauAbstractSlider(parent)
    , m_spinBoxLeft(new NauDoubleSpinBox(this))
    , m_spinBoxRight(new NauDoubleSpinBox(this))
    , m_leftEditing(false)
    , m_rightEditing(false)
{
    auto* separator = new NauStaticTextLabel("-", this);
    const QFont& font = separator->font();
    separator->setFont(NauFont(font.family(), 12, font.weight(), font.italic()));
    separator->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    separator->setFixedHeight(16);

    auto* dataLayout = new NauLayoutHorizontal;
    dataLayout->addWidget(m_spinBoxLeft);
    dataLayout->addWidget(separator);
    dataLayout->addWidget(m_spinBoxRight);

    auto* dataWidget = new NauWidget(this);
    dataWidget->setLayout(dataLayout);
    dataWidget->setFixedHeight(22);
    dataWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    static_cast<NauLayoutVertical*>(layout())->insertWidget(0, dataWidget);

    setRangedPair({});
    setupSpinBox(m_spinBoxLeft);
    setupSpinBox(m_spinBoxRight);

    connect(m_spinBoxLeft, &QDoubleSpinBox::valueChanged, this, &NauAbstractSlider::eventRangeChanged);
    connect(m_spinBoxRight, &QDoubleSpinBox::valueChanged, this, &NauAbstractSlider::eventRangeChanged);
}

void NauSliderFloatPair::setMinimum(float minimum) noexcept
{
    m_spinBoxLeft->setMinimum(minimum);
    m_spinBoxRight->setMinimum(minimum);
}

void NauSliderFloatPair::setMaximum(float maximum) noexcept
{
    m_spinBoxLeft->setMaximum(maximum);
    m_spinBoxRight->setMaximum(maximum);
}

void NauSliderFloatPair::setRangedPair(const NauRangedPair<float>& range) noexcept
{
    m_range = range;
    setMinimum(range.minimum());
    setMaximum(range.maximum());
    setPair(range.left(), range.right());
}

void NauSliderFloatPair::setPair(float left, float right) noexcept
{
    m_spinBoxLeft->setValue(left);
    m_spinBoxRight->setValue(right);
}

NauRangedPair<float> NauSliderFloatPair::rangedPair() const noexcept
{
    return {
        std::min<float>(m_spinBoxLeft->value(), m_range.right()),
        std::max<float>(m_range.left(), m_spinBoxRight->value()),
        m_range.minimum(),
        m_range.maximum()
    };
}

float NauSliderFloatPair::left() const noexcept
{
    return static_cast<float>(m_spinBoxLeft->value());
}

float NauSliderFloatPair::right() const noexcept
{
    return static_cast<float>(m_spinBoxRight->value());
}

NauAbstractSlider::RangeRatio NauSliderFloatPair::ratioValues() const noexcept
{
    const double rangeLength = m_spinBoxLeft->maximum() - m_spinBoxLeft->minimum();
    const double left = m_spinBoxLeft->value() - m_spinBoxLeft->minimum();
    const double right = m_spinBoxRight->value() - m_spinBoxRight->minimum();

    const double leftRatio = left / rangeLength;
    const double rightRatio = right / rangeLength;

    return { leftRatio, rightRatio };
}

void NauSliderFloatPair::updateSlider(double ratio, NauSliderController::Reason reason)
{
    const double rangeLength = m_spinBoxLeft->maximum() - m_spinBoxLeft->minimum();
    const double value = ratio * rangeLength;

    QSignalBlocker blocker{ this };
    if (reason == NauSliderController::Reason::Begin) {
        const double halfSliderWidth = static_cast<double>(sliderInputWidth()) * 0.5;
        const int extendArea = static_cast<int>(rangeLength / static_cast<double>(width()) * halfSliderWidth);
        m_leftEditing = (value - extendArea) <= (m_spinBoxLeft->value() - m_spinBoxLeft->minimum());
        m_rightEditing = !m_leftEditing && (value + extendArea) >= (m_spinBoxRight->value() - m_spinBoxRight->minimum());
        m_prevRatioCache = ratio;
    } else if (reason == NauSliderController::Reason::End) {
        m_leftEditing = false;
        m_rightEditing = false;
        blocker.unblock();
    }
    if (m_leftEditing) {
        const double newLeftValue = std::min(value + m_spinBoxLeft->minimum(), m_spinBoxRight->value());
        setPair(static_cast<float>(newLeftValue), static_cast<float>(m_spinBoxRight->value()));
    } else if (m_rightEditing) {
        const double newRightValue = std::max(value + m_spinBoxLeft->minimum(), m_spinBoxLeft->value());
        setPair(static_cast<float>(m_spinBoxLeft->value()), static_cast<float>(newRightValue));
    } else {
        const double pairLength = m_spinBoxRight->value() - m_spinBoxLeft->value();
        const auto [left, right] = ratioValues();
        const double newLeftRatio = std::clamp(left + ratio - m_prevRatioCache, 0.0, 1.0 - pairLength / rangeLength);
        const double newLeftValue = rangeLength * newLeftRatio + m_spinBoxLeft->minimum();
        setPair(static_cast<float>(newLeftValue), static_cast<float>(newLeftValue + pairLength));
    }
    emit eventRangeChanged();
    m_prevRatioCache = ratio;
    update();
}
