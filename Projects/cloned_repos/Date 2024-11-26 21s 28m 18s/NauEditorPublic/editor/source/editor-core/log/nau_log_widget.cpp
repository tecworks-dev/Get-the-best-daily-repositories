// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_widget.hpp"
#include "nau_header_view.hpp"

#include "nau_log_model.hpp"
#include "nau_log_proxy_model.hpp"
#include "nau_log_constants.hpp"
#include "nau_log_tree_view_item_delegate.hpp"
#include "themes/nau_theme.hpp"

#include <QScrollBar>


// ** NauLogWidget

NauLogWidget::NauLogWidget(NauShortcutHub* shortcutHub, QWidget* parent)
    : NauWidget(parent)
    , m_sourceModel(&NauLog::editorLogModel())
    , m_proxyModel(new NauLogProxyModel)
    , m_view(new NauTreeView(this))
{
    m_proxyModel->setSourceModel(m_sourceModel);
    m_view->setModel(m_proxyModel);
    m_view->setObjectName("NauLogWidgetTreeView");
    m_view->setSelectionMode(NauTreeView::ExtendedSelection);
    m_view->setMouseTracking(true);
    m_view->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    m_view->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    m_view->setAlternatingRowColors(true);

    shortcutHub->addWidgetShortcut(NauShortcutOperation::LoggerCopySelectedMessages, *m_view,
        std::bind(&NauLogWidget::eventSelectedMessagesCopyRequested, this));

    auto horizontalHeader = new NauHeaderView(Qt::Horizontal);
    m_view->setHeader(horizontalHeader);
    horizontalHeader->setSectionHideable(+NauLogModel::Column::Message, false);
    horizontalHeader->setSortIndicatorShown(true);
    horizontalHeader->setSortIndicator(+NauLogModel::Column::Time, Qt::SortOrder::AscendingOrder);
    horizontalHeader->setFixedHeight(32);
    horizontalHeader->setStretchLastSection(true);
    horizontalHeader->setFont(Nau::Theme::current().fontLogger());
    horizontalHeader->setPalette(Nau::Theme::current().paletteLogger());

    auto delegate = new NauLogTreeViewItemDelegate(this);
    delegate->setPalette(Nau::Theme::current().paletteLogger());
    delegate->setColumnHighlighted(+NauLogModel::Column::Level, true);

    m_view->setItemDelegate(delegate);

    connect(m_sourceModel, &QAbstractItemModel::rowsAboutToBeInserted, this, [this]
    (const QModelIndex& parent, int first, int last) {
        auto bar = m_view->verticalScrollBar();
        m_scrollAtBottom = bar ? (bar->value() == bar->maximum()) : false;
    });

    connect(m_proxyModel, &QSortFilterProxyModel::rowsRemoved, [this]{
        emit eventMessageCountChanged(m_proxyModel->rowCount());
    });
    connect(m_proxyModel, &QSortFilterProxyModel::rowsInserted, [this]{
        emit eventMessageCountChanged(m_proxyModel->rowCount());
    });

    m_view->setRootIsDecorated(false);

    setLayout(new NauLayoutHorizontal);
    layout()->setContentsMargins(0, 0, 0, 0);
    layout()->addWidget(m_view);

    connect(m_view->selectionModel(), &QItemSelectionModel::currentChanged, this,
        &NauLogWidget::eventCurrentMessageChanged);

    connect(m_view->selectionModel(), &QItemSelectionModel::selectionChanged, [this] {
        QModelIndexList selectedIndices = m_view->selectionModel()->selectedIndexes();
        QModelIndexList selectedFirstColumn;

        for (const QModelIndex& index : selectedIndices) {
            if (index.column() == +NauLogModel::Column::Level) {
                selectedFirstColumn << index;
            }
        }

        emit eventMessagesSelectionChanged(selectedFirstColumn);
    });
}

NauLogWidget::~NauLogWidget()
{
    NauLog::editorLogger().clear();
}

void NauLogWidget::clear()
{
    m_sourceModel->clear();
    emit eventOutputCleared();
}

void NauLogWidget::filterData(const QString& text, bool isRegularExpression, bool isCaseSensitive)
{
    const bool haveTextFilterReset = 
        (!m_proxyModel->filterRegularExpression().pattern().isEmpty()) && text.isEmpty();

    m_proxyModel->setFilterCaseSensitivity(isCaseSensitive ? Qt::CaseSensitive : Qt::CaseInsensitive);

    if (isRegularExpression) {
        QRegularExpression regex(text);

        if (!regex.isValid()) {
            return;
        }

        m_proxyModel->setFilterRegularExpression(text);
    } else {
        m_proxyModel->setFilterFixedString(text);
    }

    if (haveTextFilterReset) {
        // Filter is cleared. Show user newest items.
        m_view->scrollToBottom();
    }
}

void NauLogWidget::writeData(QTextStream& stream)
{
    static const char* const delimiter = "\t";
    const int rowCount = m_proxyModel->rowCount();

    for (int row = 0; row < rowCount; ++row) {
        stream 
            << m_proxyModel->index(row, +NauLogModel::Column::Time).data()
                .toDateTime().toString(NauLogConstants::dateTimeDisplayFormat()) << delimiter
            << m_proxyModel->index(row, +NauLogModel::Column::Level).data().toString() << delimiter
            << m_proxyModel->index(row, +NauLogModel::Column::Message).data().toString()
            << Qt::endl;
    }
}

void NauLogWidget::setAutoScrollPolicy(bool autoScrollEnabled)
{
    QObject::disconnect(m_scrollConnection);

    if (autoScrollEnabled) {
        m_view->scrollToBottom();
        m_scrollConnection = connect(m_sourceModel, &NauLogModel::rowsInserted, this, [ this ]() {
            // We can't check if the scrollbar is at the bottom here because
            // the new rows are already inserted and the position of the
            // scrollbar may not be at the bottom of the widget anymore.
            // That's why the scroll position is checked before actually
            // adding the rows (AKA in the rowsAboutToBeInserted signal).
            if (m_scrollAtBottom) {
                m_view->scrollToBottom();
            }
        });
    }
}

void NauLogWidget::setLogSourceFilter(const QStringList& sourceNames)
{
    m_proxyModel->setLogSourceFilter(sourceNames);
    m_view->scrollToBottom();
}

void NauLogWidget::setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels)
{
    m_proxyModel->setLogLevelFilter(preferredLevels);
}

void NauLogWidget::resizeEvent(QResizeEvent* event)
{
    auto const vScrollWidth = m_view->verticalScrollBar()->isVisible() ? m_view->verticalScrollBar()->width() : 0;
    auto const viewWidth = m_view->width() - vScrollWidth;
    auto viewRemainsWidth = viewWidth;

    const auto applyWidth = [&viewRemainsWidth](NauTreeView* view, NauLogModel::Column column,  double width) {
        view->setColumnWidth(+column, static_cast<int>(width));
        viewRemainsWidth -= width;
    };

    applyWidth(m_view, NauLogModel::Column::Level, std::clamp(viewWidth * 0.15, 40.0, 120.0));
    applyWidth(m_view, NauLogModel::Column::Time, std::clamp(viewWidth * 0.25, 40.0, 150.0));
    applyWidth(m_view, NauLogModel::Column::Tags, std::clamp(viewWidth * 0.10, 40.0, 120.0));
    applyWidth(m_view, NauLogModel::Column::Message, viewRemainsWidth);
}

std::size_t NauLogWidget::itemsCount() const
{
    return static_cast<std::size_t>(m_proxyModel->rowCount());
}

std::size_t NauLogWidget::selectedItemsCount() const
{
    std::size_t result{};
    const auto selection = m_view->selectionModel()->selectedIndexes();
    for (const QModelIndex& index : selection) {
        if (index.column() == +NauLogModel::Column::Message) {
            ++result;
        }
    }

    return result;
}

QString NauLogWidget::messageAt(int index) const
{
    return m_sourceModel->data(m_sourceModel->index(index, +NauLogModel::Column::Message), Qt::DisplayRole).toString();
}

QString NauLogWidget::selectedMessages() const
{
    const auto selection = m_view->selectionModel()->selectedIndexes();
    QString result;
    QTextStream stream(&result);
    for (const QModelIndex& index : selection) {
        if (index.column() == +NauLogModel::Column::Message) {
            stream << index.data(NauLogModel::TimeRole).toDateTime().toString(NauLogConstants::dateTimeDisplayFormat()) << "\t" 
                   << index.data(NauLogModel::MessageRole).toString() << Qt::endl;
        }
    }

    return result;
}

void NauLogWidget::setMaxEntries(std::optional<std::size_t> maxEntries)
{
    m_sourceModel->setMaxEntries(maxEntries);
}

std::optional<std::size_t> NauLogWidget::getMaxEntries() const
{
    return m_sourceModel->getMaxEntries();
}
