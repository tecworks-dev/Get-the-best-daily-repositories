// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau/widgets/nau_timeline_parameters.hpp"
#include "nau/widgets/nau_common_timeline_widgets.hpp"

#include "baseWidgets/nau_static_text_label.hpp"
#include "themes/nau_theme.hpp"
#include "nau_assert.hpp"


// ** NauTimelineModesWidget

NauTimelineModesWidget::NauTimelineModesWidget(NauWidget* parent)
    : QTabBar(parent)
{
    const QSize SWITCH_SIZE{ tabSizeHint(0).width() * 2, 24};

    const std::vector<NauIcon> icons = Nau::Theme::current().iconsTimelineParameters();

    NED_ASSERT(icons.size() >= 2);

    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setFixedSize(SWITCH_SIZE);
    setUsesScrollButtons(false);
    setTabsClosable(false);

    addTab(icons[0], "");
    addTab(icons[1], "");

    connect(this, &QTabBar::currentChanged, [this](int index) {
        if (index == 0) {
            emit eventDopeSheetModeEnabled();
        } else {
            emit eventCurveModeEnabled();
        }
    });
}

QSize NauTimelineModesWidget::tabSizeHint(int index) const
{
    constexpr QSize CASE_SIZE{ 48, 24 };
    return CASE_SIZE;
}

void NauTimelineModesWidget::paintEvent(QPaintEvent* event)
{
    const std::array icons{ tabIcon(0), tabIcon(1) };
    const QSize tabSize{ tabSizeHint(0) };
    const QSize iconSize = icons[0].actualSize(tabSize);

    const NauPalette palette = Nau::Theme::current().paletteTimelineMode();

    QPainterPath path;
    path.addRoundedRect(QRect(0, 0, tabSize.width() * 2, tabSize.height()), tabSize.height() / 2, tabSize.height() / 2);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setClipPath(path);

    const QSize halfFreeSpace = (tabSize - iconSize) / 2;
    QRect tabRect{ 0, 0, tabSize.width(), tabSize.height() };
    QRect iconRect{ halfFreeSpace.width(), halfFreeSpace.height(), iconSize.width(), iconSize.height() };
    const int currentTabIndex = currentIndex();
    for (int index = 0, tabsCount = count(); index < tabsCount; ++index) {
        const auto iconState = (index == currentTabIndex) ? QIcon::On : QIcon::Off;
        const auto colorState = (index == currentTabIndex) ? NauPalette::Selected : NauPalette::Normal;

        painter.fillRect(tabRect, palette.color(NauPalette::Role::Background, colorState));
        tabRect.translate(tabSize.width(), 0);

        icons[index].paint(&painter, iconRect, Qt::AlignCenter, QIcon::Normal, iconState);
        iconRect.translate(tabSize.width(), 0);
    }
}


// ** NauTimelineParameters

NauTimelineParameters::NauTimelineParameters(NauWidget* parent)
    : NauWidget(parent)
    , m_step(new NauTimelineSuffixedWidget<NauDoubleSpinBox>(QObject::tr("sec"), this))
    , m_duration(new NauTimelineSuffixedWidget<NauSpinBox>(QObject::tr("sec"), this))
{
    constexpr int DURATION_SPACING = 8;
    constexpr int SELECTOR_SPACING = 8;
    constexpr QSize SELECTOR_SIZE{ 80, 24 };
    constexpr QSize STEP_SIZE{ 64, 24 };
    constexpr QSize STEP_BTN_SIZE{ 24, 24 };
    constexpr QSize TIME_SELECTOR_CONTAINER_SIZE = SELECTOR_SIZE + STEP_SIZE + STEP_BTN_SIZE + QSize{ SELECTOR_SPACING * 2, -(SELECTOR_SIZE.height() + STEP_BTN_SIZE.height())};
    constexpr QSize DURATION_SIZE{ 96, 24 };
    constexpr QSize TOTAL_SIZE{ 32, 24 };
    constexpr QSize DURATION_CONTAINER_SIZE = DURATION_SIZE + TOTAL_SIZE + QSize{ DURATION_SPACING, -TOTAL_SIZE.height() };

    const std::vector<NauIcon> icons = Nau::Theme::current().iconsTimelineParameters();

    m_step->setDecimals(1);
    m_step->setValue(0.1);
    m_step->setFixedSize(STEP_SIZE);
    m_duration->setValue(50);
    m_duration->setFixedSize(DURATION_SIZE);

    auto* selector = new NauComboBox(this);
    // TODO: delete after completion of work on custom combobox
    selector->setObjectName("NauTL_headerBox");
    selector->setStyleSheet("QComboBox#NauTL_headerBox { background-color: #222222; }"
                            "QComboBox#NauTL_headerBox::down-arrow { image: url(:/inputEditor/icons/inputEditor/vArrowDown.svg); }");
    selector->addItems({ QObject::tr("Time"), QObject::tr("Frame") });
    selector->setFixedSize(SELECTOR_SIZE);

    auto* stepSwitcher = new NauTimelineButton(this);
    stepSwitcher->setIcon(icons[2]);

    auto* timeContainer = new NauWidget(this);
    timeContainer->setLayout(new NauLayoutHorizontal);
    timeContainer->setFixedSize(TIME_SELECTOR_CONTAINER_SIZE);
    timeContainer->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    timeContainer->layout()->setSpacing(SELECTOR_SPACING);
    timeContainer->layout()->addWidget(selector);
    timeContainer->layout()->addWidget(m_step);
    timeContainer->layout()->addWidget(stepSwitcher);

    auto* textContainer = new NauWidget(this);
    textContainer->setFixedSize(TOTAL_SIZE);
    auto* text = new NauStaticTextLabel(QObject::tr("Total"), textContainer);
    text->setColor(NauColor{ 128, 128, 128 });
    text->move(0, 4);

    auto* durationContainer = new NauWidget(this);
    durationContainer->setLayout(new NauLayoutHorizontal);
    durationContainer->setFixedSize(DURATION_CONTAINER_SIZE);
    durationContainer->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    durationContainer->layout()->setSpacing(DURATION_SPACING);
    durationContainer->layout()->addWidget(textContainer);
    durationContainer->layout()->addWidget(m_duration);

    auto* layout = new NauLayoutHorizontal;
    layout->setContentsMargins(16, 8, 16, 8);
    layout->setSpacing(16);
    layout->addWidget(new NauTimelineModesWidget(this));
    layout->addWidget(timeContainer);
    layout->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding));
    layout->addWidget(durationContainer);

    setLayout(layout);
    setFixedHeight(40);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    setDisabled(true);
}

void NauTimelineParameters::paintEvent(QPaintEvent* event)
{
    const NauPalette palette = Nau::Theme::current().paletteTimelineTrackList();
    QPainter painter{ this };
    painter.fillRect(QRectF(0, 0, width(), height()), palette.color(NauPalette::Role::BackgroundFooter));
}