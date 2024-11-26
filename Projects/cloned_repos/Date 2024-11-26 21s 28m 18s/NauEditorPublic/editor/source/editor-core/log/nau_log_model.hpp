// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Data model of the log/console widget.

#pragma once

#include "baseWidgets/nau_icon.hpp"
#include "nau/nau_editor_engine_log.hpp"

#include <QAbstractListModel>
#include <QDateTime>

#include <deque>
#include <optional>


// ** NauLogModel

class NAU_EDITOR_API NauLogModel : public QAbstractListModel
{
    Q_OBJECT

    struct Entry {
        QDateTime time;

        // TODO this must be replaced the editor wide enum of log levels.
        NauEngineLogLevel level;
        QString logger;
        QString message;
        QStringList tags;
    };

public:
    enum class Column
    {
        Level = 0,
        Time,
        Message,
        Tags,
    };

    enum LogModelRoles
    {
        LevelIconRole = Qt::UserRole,
        LevelDisplayTextRole,
        TimeRole,
        MessageRole,
        TagsRole,
        SourceRole,
        LevelRole,
        LevelDetailsRole,
    };

    NauLogModel(QObject* parent = nullptr);
    ~NauLogModel() = default;

    void addEntry(int64_t time, NauEngineLogLevel level,
        const std::string& logger, const std::string& message);
    void clear();

    void setLevelIcons(std::vector<NauIcon> levelIcons);
    void setMaxEntries(std::optional<std::size_t> maxEntries);
    std::optional<std::size_t> getMaxEntries() const;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

signals:
    void eventCriticalEventOccurred();

private:
    static QString levelDetails(NauEngineLogLevel logLevel);

private:
    std::deque<Entry> m_entries;
    std::optional<std::size_t> m_maxEntries;
    std::vector<NauIcon> m_icons;
};
