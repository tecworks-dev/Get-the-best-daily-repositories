// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Log/console widget 

#pragma once

#include "nau_log_model.hpp"
#include "nau_shortcut_hub.hpp"
#include "nau_log_proxy_model.hpp"
#include "baseWidgets/nau_widget.hpp"


// ** NauLogWidget

class  NauLogWidget : public NauWidget
{
    Q_OBJECT

public:
    explicit NauLogWidget(NauShortcutHub* shortcutHub, QWidget* parent = nullptr);
    ~NauLogWidget();

    // Clear the contents of the model.
    // Emits eventOutputCleared().
    void clear();

    // Get the number of items in the widget.
    std::size_t itemsCount() const;

    // Get the number of selected items in the widget.
    std::size_t selectedItemsCount() const;

    // Get the log message at specific index.
    QString messageAt(int index) const;

    // Get the log message of currently selected row.
    QString selectedMessages() const;

    // Set the maximum number of items in the widget.
    void setMaxEntries(std::optional<std::size_t> maxEntries);

    // Get the maximum number of items in the widget.
    std::optional<std::size_t> getMaxEntries() const;

    // Set the policy of the auto-scrolling feature.
    void setAutoScrollPolicy(bool autoScrollEnabled);

    // Set filter by the specified list of sources.
    void setLogSourceFilter(const QStringList& sourceNames);

    // Set filter log level.
    void setLogLevelFilter(const std::vector<NauLogLevel>& preferredLevels);

    // Perform filter by the specified text. 
    // If regularExpression text is interpreted as regular expression.
    void filterData(const QString& text, bool regularExpression, bool caseSensitive);

    // Write current log output to specified stream.
    void writeData(QTextStream& stream);

signals:
    void eventCurrentMessageChanged(const QModelIndex& current);
    void eventMessagesSelectionChanged(const QModelIndexList& selected);
    void eventMessageCountChanged(std::size_t messageCount);
    void eventSelectedMessagesCopyRequested();
    void eventClearOutputRequested();
    void eventOutputCleared();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    NauLogModel* m_sourceModel = nullptr;
    NauLogProxyModel* m_proxyModel = nullptr;
    NauTreeView* m_view = nullptr;
    bool m_scrollAtBottom = false;
    QMetaObject::Connection m_scrollConnection;
};
