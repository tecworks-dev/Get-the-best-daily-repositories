// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_spoiler.hpp"
#include "themes/nau_theme.hpp"
#include "nau_assert.hpp"

#include <QPropertyAnimation>
#include <QPixmap>


// ** NauHeader

NauHeader::NauHeader(const QString& title, NauWidget* parent)
    : NauWidget(parent)
    , m_toggleButton(new QToolButton())  // TODO: should be NauToolButton, but completely non-functional (yet, not disabled)
    , m_mainLayout(new NauLayoutVertical(this))
    , m_headerLayout(new NauLayoutHorizontal())
{
    setObjectName("headerWidget");

    // TODO: -4 is a hack, need to investigate
    m_headerLayout->setContentsMargins(OuterMargin, OuterMargin, OuterMargin - 4, OuterMargin);
    m_headerLayout->setSpacing(Spacing);
    m_headerLayout->addWidget(m_toggleButton, Qt::AlignLeft);
    m_headerLayout->addStretch(1);
    
    m_toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_toggleButton->setFixedHeight(HeaderHeight);
    m_toggleButton->setText(title);
    m_toggleButton->setFont(Nau::Theme::current().fontObjectInspectorSpoiler());

    m_mainLayout->addLayout(m_headerLayout);
}


NauToolButton* NauHeader::addStandardButton(Button kind)
{
    auto button = new NauToolButton(this);
    button->setAutoRaise(true);
    button->setEnabled(false);
    button->setContentsMargins(ButtonMargin, ButtonMargin, ButtonMargin, ButtonMargin);
    button->setIconSize(QSize(ButtonIconSize, ButtonIconSize));

    if (kind == Menu) {
        button->setIcon(Nau::paintPixmapCopy(Nau::Theme::current().iconDockAreaMenu().pixmap(ButtonIconSize, ButtonIconSize), ColorDefaultButton));
    } else if (kind == Help) {
        button->setIcon(Nau::paintPixmapCopy(Nau::Theme::current().iconQuestionMark().pixmap(ButtonIconSize, ButtonIconSize), ColorDefaultButton));
    } else {
        NED_ASSERT(false && "Unknown button");
    }

    m_headerLayout->addWidget(button);
    return button;
}


// ** NauSpoiler

NauSpoiler::NauSpoiler(const QString& title, int animationDuration, NauWidget* parent)
    : NauHeader(title, parent)
    , m_headerLine(new QFrame())
    , m_toggleAnimation(new QParallelAnimationGroup())
    , m_contentArea(new QScrollArea())
    , m_animationDuration(animationDuration)
    , m_userWidgetsCount(0)
{
    setObjectName("spoilerWidget");

    const auto& theme = Nau::Theme::current();
    m_toggleButton->setIcon(theme.iconSpoilerIndicator());
    m_toggleButton->setCheckable(true);
    m_toggleButton->setChecked(false);

    m_headerLine->setFrameShape(QFrame::HLine);
    m_headerLine->setFixedHeight(SeparatorLineHeight);
    m_headerLine->setFrameShadow(QFrame::Sunken);
    m_headerLine->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    // TODO: Now it is not possible to use the palette for this widget as long as it has a style in main.qss.
    // If there is a style in qss, Qt does not look at the palette. Fix in the future.
    m_headerLine->setStyleSheet("background-color: #141414");

    m_contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // Start out collapsed
    m_contentArea->setMaximumHeight(0);
    m_contentArea->setMinimumHeight(0);

    // Let the entire widget grow and shrink with its content
    m_toggleAnimation->addAnimation(new QPropertyAnimation(this, "minimumHeight"));
    m_toggleAnimation->addAnimation(new QPropertyAnimation(this, "maximumHeight"));
    m_toggleAnimation->addAnimation(new QPropertyAnimation(m_contentArea, "minimumHeight"));
    m_toggleAnimation->addAnimation(new QPropertyAnimation(m_contentArea, "maximumHeight"));

    m_mainLayout->addWidget(m_contentArea);
    m_mainLayout->addWidget(m_headerLine);

    // TODO: Questionable code. May be worth fixing during redesign work
    m_contentLayout = new NauLayoutVertical();
    m_contentLayout->setSpacing(VerticalItemSpacer);
    m_contentLayout->setContentsMargins(OuterMargin, 0, OuterMargin, OuterMargin - SeparatorLineHeight);
    m_contentArea->setLayout(m_contentLayout);

    connect(m_toggleButton, &QToolButton::clicked, this, &NauSpoiler::handleToggleRequested);
    connect(m_toggleButton, &QObject::destroyed, [this]() { m_toggleButton = nullptr; });
}

void NauSpoiler::setExpanded()
{
    updateAnimation();
    // Now start expanded
    handleToggleRequested(true);
}

void NauSpoiler::addWidget(NauWidget* widget)
{
    m_contentLayout->insertWidget(userWidgetsCount(), widget);
    ++m_userWidgetsCount;
    const int contentHeight = m_contentArea->maximumHeight() + widget->sizeHint().height();

    updateAnimation(0, contentHeight);
    handleToggleRequested(isToggled());
}

void NauSpoiler::removeWidget(const NauWidget* widget)
{
    const int itemIndex = m_contentLayout->indexOf(widget);
    if (itemIndex > -1) {
        auto* item = m_contentLayout->itemAt(itemIndex);
        const int contentHeight = m_contentArea->maximumHeight() - item->widget()->sizeHint().height();
        m_contentLayout->removeItem(item);
        --m_userWidgetsCount;

        updateAnimation(0, contentHeight);
        handleToggleRequested(isToggled());
    }
}

std::vector<NauWidget*> NauSpoiler::removeWidgets()
{
    std::vector<NauWidget*> widgets;
    widgets.reserve(m_contentLayout->count());
    int contentHeight = m_contentArea->maximumHeight();
    for (int index = userWidgetsCount() - 1; index >= 0; --index) {
        auto* item = m_contentLayout->itemAt(index);
        const auto* widget = widgets.emplace_back(static_cast<NauWidget*>(item->widget()));
        contentHeight -= widget->sizeHint().height();
        m_contentLayout->removeItem(item);
    }
    m_userWidgetsCount = 0;
    updateAnimation(0, contentHeight);
    handleToggleRequested(isToggled());

    return widgets;
}

void NauSpoiler::handleToggleRequested(bool expandFlag)
{
    if (m_expanded != expandFlag) {
        emit eventStartedExpanding(expandFlag);
    }
    m_expanded = expandFlag;
    if (m_toggleButton != nullptr) {
        m_toggleButton->setChecked(expandFlag);
    }
    m_toggleAnimation->setDirection(expandFlag ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
    m_toggleAnimation->start();
}

bool NauSpoiler::isToggled() const noexcept
{
    return m_expanded;
}

int NauSpoiler::userWidgetsCount() const noexcept
{
    return m_userWidgetsCount;
}

void NauSpoiler::updateAnimation(int duration, int areaHeight)
{
    constexpr int CONTENT_ANIMATIONS_COUNT = 2;
    const int collapsedHeight = m_headerLayout->sizeHint().height();
    const int contentHeight = m_contentLayout->sizeHint().height();
    const int animationDuration = duration < 0 ? m_animationDuration : duration;

    for (int i = 0; i < m_toggleAnimation->animationCount() - CONTENT_ANIMATIONS_COUNT; ++i) {
        auto* spoilerAnimation = static_cast<QPropertyAnimation*>(m_toggleAnimation->animationAt(i));
        spoilerAnimation->setDuration(animationDuration);
        spoilerAnimation->setStartValue(collapsedHeight);
        spoilerAnimation->setEndValue(collapsedHeight + contentHeight);
    }
    for (int i = CONTENT_ANIMATIONS_COUNT; i < m_toggleAnimation->animationCount(); ++i) {
        auto* contentAnimation = static_cast<QPropertyAnimation*>(m_toggleAnimation->animationAt(i));
        contentAnimation->setDuration(animationDuration);
        contentAnimation->setStartValue(0);
        contentAnimation->setEndValue(contentHeight);
    }
}

bool NauSpoiler::eventFilter(QObject* obj, QEvent* event)
{
    if (event->type() == QEvent::Resize) {
        const auto* resizeEventData = static_cast<QResizeEvent*>(event);
        const int oldHeight = resizeEventData->oldSize().height();
        const int newHeight = resizeEventData->size().height();
        if (resizeEventData->oldSize().isValid() && (oldHeight != newHeight)) {
            const int currentHeight = m_contentArea->maximumHeight();
            const int contentHeight = std::max(0, currentHeight + newHeight - oldHeight);
            updateAnimation(0, contentHeight);
            handleToggleRequested(isToggled());
        }
    }
    return QObject::eventFilter(obj, event);
}


// ** NauSimpleSpoiler

NauSimpleSpoiler::NauSimpleSpoiler(const QString& title, QWidget* parent)
    : NauFrame(parent)
{
    setObjectName("simpleSpoiler");

    const auto& theme = Nau::Theme::current();
    auto mainLayout = new NauLayoutVertical(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(QMargins());

    m_headerArea = new NauFrame(this);
    m_headerArea->setObjectName("simpleSpoilerHeader");
    m_headerLayout = new NauLayoutHorizontal(m_headerArea);

    m_toggleButton = new NauMiscButton(this);
    m_toggleButton->setIcon(theme.iconSpoilerIndicator());
    m_toggleButton->setIconSize(m_toggleIconSize);
    m_toggleButton->setFixedSize(m_toggleButton->iconSize());
    m_toggleButton->setCheckable(true);

    connect(m_toggleButton, &NauMiscButton::toggled, this, &NauSimpleSpoiler::setExpanded);

    m_label = new NauStaticTextLabel(title, m_headerArea);

    connect(m_label, &NauStaticTextLabel::doubleClicked, [this] {
        if (m_editor) {
            const QRect editorRect = m_headerArea->rect() - m_titleEditorOuterMargins;
            m_editor->setGeometry(editorRect);
            m_editor->setFixedHeight(m_headerArea->height());
            m_editor->setText(m_label->text());
            m_editor->selectAll();
            m_editor->show();
            m_editor->setFocusPolicy(Qt::WheelFocus);
            m_editor->setFocus(Qt::FocusReason::MouseFocusReason);
        }
    });

    m_headerLayout->addWidget(m_toggleButton, 0, Qt::AlignLeft | Qt::AlignVCenter);
    m_headerLayout->addWidget(m_label, 1, Qt::AlignLeft | Qt::AlignVCenter);

    m_contentArea = new NauFrame(this);
    m_contentArea->setObjectName("simpleSpoilerContent");
    m_contentLayout = new NauLayoutVertical(m_contentArea);

    mainLayout->addWidget(m_headerArea);
    mainLayout->addWidget(m_contentArea, 1);
}

NauMiscButton* NauSimpleSpoiler::addHeaderButton()
{
    auto button = new NauMiscButton(m_headerArea);
    m_headerLayout->addWidget(button);

    return button;
}

void NauSimpleSpoiler::setExpanded(bool expanded)
{
    m_contentArea->setVisible(expanded);
}

void NauSimpleSpoiler::addWidget(QWidget* widget)
{
    m_contentLayout->addWidget(widget);
}

void NauSimpleSpoiler::setHeaderFixedHeight(int fixedHeight)
{
    m_headerArea->setFixedHeight(fixedHeight);
}

void NauSimpleSpoiler::setHeaderContentMargins(QMargins contentMargins)
{
    m_headerLayout->setContentsMargins(contentMargins);
}

void NauSimpleSpoiler::setHeaderHorizontalSpace(int horizontalSpacing)
{
    m_headerLayout->setSpacing(horizontalSpacing);
}

void NauSimpleSpoiler::setHeaderPalette(NauPalette palette)
{
    m_headerArea->setPalette(std::move(palette));
}

void NauSimpleSpoiler::setToggleIconSize(QSize size)
{
    m_toggleIconSize = size;
}

void NauSimpleSpoiler::setContentAreaMargins(QMargins contentMargins)
{
    m_contentLayout->setContentsMargins(contentMargins);
}

void NauSimpleSpoiler::setContentVerticalSpacing(int vSpacing)
{
    m_contentLayout->setSpacing(vSpacing);
}

void NauSimpleSpoiler::setContentPalette(NauPalette palette)
{
    m_contentArea->setPalette(palette);
}

void NauSimpleSpoiler::setTitleEditable(bool editable)
{
    if (editable && !m_editor) {
        m_label->setToolTip(tr("Double click to rename"));
        m_editor = new NauLineEdit(m_headerArea);
        m_editor->hide();
        connect(m_editor, &NauLineEdit::editingFinished, [this] {
            m_editor->hide();
            emit eventRenameRequested(m_editor->text());
        });
    } else {
        m_label->setToolTip({});
    }
}

void NauSimpleSpoiler::setTitle(const QString& title)
{
    m_label->setText(title);
}

void NauSimpleSpoiler::setTitleEditorOuterMargin(QMargins outerMargins)
{
    m_titleEditorOuterMargins = outerMargins;
}
