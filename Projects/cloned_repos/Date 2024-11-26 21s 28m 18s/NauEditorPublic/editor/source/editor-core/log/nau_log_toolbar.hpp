// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Implementation of Log/console toolbar

#pragma once

#include "baseWidgets/nau_widget.hpp"
#include "nau_log.hpp"
#include "filter/nau_search_widget.hpp"
#include "filter/nau_filter_widget.hpp"

#include <QActionGroup>


// ** NauLogToolBar

class NauLogToolBar : public NauWidget
{
    Q_OBJECT

public:
    struct FilteringSettings {
        QString text;             // The text to filter by.
        bool isRegularExpression; // Whether the text is a regular expression.
        bool isCaseSensitive;     // Whether the filtering is case sensitive.
    };

public:
    NauLogToolBar(QWidget* parent = nullptr);

    FilteringSettings filteringSettings() const;
    QAction* detailsPanelVisibilityAction() const;
    
    bool autoScrollEnabled() const;
    void setAutoScrollEnabled(bool enabled);

signals:
    void eventFilterChanged(const QString& text, bool regularExpression, bool caseSensitive);
    void eventAutoScrollPolicyChanged(bool autoScrollEnabled);
    void eventClearOutputRequested();
    void eventFilterByLevelChangeRequested(const std::vector<NauLogLevel>& preferredLevels);
    void eventToggleDetailVisibilityRequested(bool visible);
    void eventSaveLogAsRequested();

private:
    void loadCompleterHistory();
    void saveCompleterHistory();
    void emitFilterStateChanged();
    void handleFilterEditFinished();
    void checkInputValidity();
    void clearCompleterHistory();

private:
    NauMenu* m_menu;
    NauFilterWidget* m_filterWidget;
    NauSearchWidget* m_searchWidget;
    QAction* m_caseAction;
    QAction* m_regexAction;
    QAction* m_clearHistoryAction;
    QAction* m_clearOutputAction;
    QAction* m_autoScrollAction;

    NauAction* m_detailsPanelVisibilityAction;

    QAbstractItemModel* m_completerData;
    QCompleter* m_completer;
};
