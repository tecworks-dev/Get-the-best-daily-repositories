// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// The widget changes the value depending on the position of the slider

#pragma once

#include "nau_widget.hpp"
#include "nau_types.hpp"
#include "nau_concepts.hpp"

// ** NauSliderController

class NAU_EDITOR_API NauSliderController : public NauWidget
{
    Q_OBJECT

public:
    explicit NauSliderController(NauWidget* parent);
    [[nodiscard]]
    bool sliding() const noexcept { return m_pressed; }
    
    enum class Reason
    {
        Begin,
        Move,
        End
    };

signals:
    void eventChangedValue(double ratio, Reason reason);

private:
    void updateSlider(double positionX, Reason reason) noexcept;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    bool m_pressed = false;
};


// ** NauSliderValueController

class NAU_EDITOR_API NauSliderValueController : public QObject
{
public:
    NauSliderValueController(NauLineEdit* lineEdit);
    
    [[nodiscard]]
    bool editing() const noexcept;

protected:
    bool eventFilter(QObject* object, QEvent* event) override;
};

// ** NauAbstractSlider

// TODO: In the future it is worth developing a system based on an abstract slider widget factory,
// which will consist of an abstract class with template methods and its successor classes with partial template specification if necessary.

class NAU_EDITOR_API NauAbstractSlider : public NauWidget
{
    Q_OBJECT

public:
    explicit NauAbstractSlider(NauWidget* parent);

signals:
    void eventRangeChanged();
    
protected:
    struct RangeRatio
    {
        double left;
        double right;
    };

    [[nodiscard]]
    virtual RangeRatio ratioValues() const noexcept = 0;
    virtual void updateSlider(double ratio, NauSliderController::Reason reason) = 0;
    void paintEvent(QPaintEvent* event) override;
    bool event(QEvent* qEvent) override;

    static void paintRect(QPainter& painter, const NauColor& color, int x, int y, int width, int height, float rx = 0.f, float ry = 0.f) ;

    template <class T>
    void setupSpinBox(T* spinBox)
    {
        spinBox->setStyleSheet("background: 0#00000000; font-size: 12px;");
        spinBox->setAlignment(Qt::AlignBottom | Qt::AlignHCenter);
        spinBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinBox->setFrame(false);
        spinBox->setFixedHeight(22);
        spinBox->installEventFilter(addValueController(spinBox->lineEdit()));
    }

    [[nodiscard]]
    QSize sliderSize() const noexcept;
    [[nodiscard]]
    int sliderInputWidth() const noexcept { return m_sliderSize.width(); }
    [[nodiscard]]
    bool editing() const noexcept;
    [[nodiscard]]
    bool sliding() const noexcept { return m_controller->sliding(); }
    NauSliderValueController* addValueController(NauLineEdit* lineEdit);

protected:
    double m_prevRatioCache;
    
private:
    NauSliderController* m_controller;
    std::vector<NauSliderValueController*> m_valueControllers;
    QSize m_sliderSize;
    bool m_hovered;
};

// ** NauSliderIntValue

class NAU_EDITOR_API NauSliderIntValue final : public NauAbstractSlider
{
    Q_OBJECT

public:
    explicit NauSliderIntValue(NauWidget* parent);

    void setMinimum(int minimum) noexcept;
    void setMaximum(int maximum) noexcept;
    void setRangedValue(const NauRangedValue<int>& range) noexcept;
    void setValue(int value) noexcept;

    [[nodiscard]]
    NauRangedValue<int> rangedValue() const noexcept;
    [[nodiscard]]
    int value() const noexcept;

private:
    [[nodiscard]]
    RangeRatio ratioValues() const noexcept override;
    void updateSlider(double ratio, NauSliderController::Reason reason) override;

private:
    NauSpinBox* m_spinBox;
};

// ** NauSliderFloatValue

class NAU_EDITOR_API NauSliderFloatValue : public NauAbstractSlider
{
Q_OBJECT

public:
    explicit NauSliderFloatValue(NauWidget* parent);

    void setMinimum(float minimum) noexcept;
    void setMaximum(float maximum) noexcept;
    void setRangedValue(const NauRangedValue<float>& range) noexcept;
    void setValue(float value) noexcept;

    [[nodiscard]]
    NauRangedValue<float> rangedValue() const noexcept;
    [[nodiscard]]
    float value() const noexcept;

private:
    [[nodiscard]]
    RangeRatio ratioValues() const noexcept override;
    void updateSlider(double ratio, NauSliderController::Reason reason) override;

private:
    NauDoubleSpinBox* m_spinBox;
};

// ** NauSliderFloatValue

class NAU_EDITOR_API NauSliderIntPair : public NauAbstractSlider
{
Q_OBJECT

public:
    explicit NauSliderIntPair(NauWidget* parent);

    void setMinimum(int minimum) noexcept;
    void setMaximum(int maximum) noexcept;
    void setRangedPair(const NauRangedPair<int>& range) noexcept;
    void setPair(int left, int right) noexcept;

    [[nodiscard]]
    NauRangedPair<int> rangedPair() const noexcept;
    [[nodiscard]]
    int left() const noexcept;
    [[nodiscard]]
    int right() const noexcept;

private:
    [[nodiscard]]
    RangeRatio ratioValues() const noexcept override;
    void updateSlider(double ratio, NauSliderController::Reason reason) override;

private:
    NauRangedPair<int> m_range;
    NauSpinBox* m_spinBoxLeft;
    NauSpinBox* m_spinBoxRight;
    bool m_leftEditing;
    bool m_rightEditing;
};

// ** NauSliderFloatValue

class NAU_EDITOR_API NauSliderFloatPair : public NauAbstractSlider
{
Q_OBJECT

public:
    explicit NauSliderFloatPair(NauWidget* parent);

    void setMinimum(float minimum) noexcept;
    void setMaximum(float maximum) noexcept;
    void setRangedPair(const NauRangedPair<float>& range) noexcept;
    void setPair(float left, float right) noexcept;

    [[nodiscard]]
    NauRangedPair<float> rangedPair() const noexcept;
    [[nodiscard]]
    float left() const noexcept;
    [[nodiscard]]
    float right() const noexcept;

private:
    [[nodiscard]]
    RangeRatio ratioValues() const noexcept override;
    void updateSlider(double ratio, NauSliderController::Reason reason) override;

private:
    NauRangedPair<float> m_range;
    NauDoubleSpinBox* m_spinBoxLeft;
    NauDoubleSpinBox* m_spinBoxRight;
    bool m_leftEditing;
    bool m_rightEditing;
};
