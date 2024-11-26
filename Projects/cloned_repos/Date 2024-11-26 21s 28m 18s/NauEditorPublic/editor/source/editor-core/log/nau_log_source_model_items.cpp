// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_log_source_model_items.hpp"
#include "nau_log_source_model.hpp"
#include "nau_log_constants.hpp"
#include "themes/nau_theme.hpp"


namespace NauSourceTreeModelItems
{
    // ** Node
    // Base class of tree data model.

    Node::Node(const QString& text, const NauIcon& icon, Node* parent, int row)
        : m_parent(parent)
        , m_text(text)
        , m_icon(icon)
        , m_row(row)
    {}

    Node* Node::parent() const
    {
        return m_parent;
    }

    QVariant Node::data(int role) const
    {
        if (role == Qt::DisplayRole) {
            return m_text;
        }
        if (role == Qt::DecorationRole) {
            return m_icon;
        }
        if (role == Qt::ToolTipRole) {
            return m_tooltip.isEmpty() ? m_text : m_tooltip;
        }
        return QVariant();
    }

    int Node::childCount() const
    { 
        return 0;
    }
    
    Node* Node::childNode(int row) const
    {
        return nullptr;
    }

    Qt::ItemFlags Node::flags() const
    { 
        return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    }

    int Node::row() const
    {
        return m_row;
    }

    void Node::setRow(int row)
    {
        m_row = row;
    }


    // ** PlayModeSourceItem

    PlayModeSourceItem::PlayModeSourceItem(Node* parent, int row)
        : Node(tr("Play Mode"), Nau::Theme::current().iconLoggerSourcePlayMode(), parent, row)
    {
    }

    QVariant PlayModeSourceItem::data(int role) const
    {
        if (role == Qt::FontRole) {
            return Nau::Theme::current().fontLoggerSource();
        }
        if (role == NauLogSourceModel::SourceNamesRole) {
            return QStringList{NauLogConstants::playModeSourceName()};
        }
        return Node::data(role);
    }


     // ** BuildSourceItem

    BuildSourceItem::BuildSourceItem(Node* parent, int row)
        : Node(tr("Build"), Nau::Theme::current().iconLoggerSourceBuild(), parent, row)
    {
    }

    QVariant BuildSourceItem::data(int role) const
    {
        if (role == Qt::FontRole) {
            return Nau::Theme::current().fontLoggerSource();
        }
        if (role == NauLogSourceModel::SourceNamesRole) {
            return QStringList{NauLogConstants::buildSourceName()};
        }
        return Node::data(role);
    }


     // ** BuildSourceItem

    EditorSourceItem::EditorSourceItem(Node* parent, int row)
        : Node(tr("Editor"), Nau::Theme::current().iconLoggerSourceEditor(), parent, row)
    {
    }

    QVariant EditorSourceItem::data(int role) const
    {
        if (role == Qt::FontRole) {
            return Nau::Theme::current().fontLoggerSource();
        }
        if (role == NauLogSourceModel::SourceNamesRole) {
            return QStringList{NauLogConstants::editorSourceName(), NauLogConstants::engineSourceName()};
        }

        return Node::data(role);
    }


    // ** ExternalSourceItem

    ExternalSourceItem::ExternalSourceItem(Node* parent, int row)
        : Node(tr("External Application"),
            Nau::Theme::current().iconLoggerSourceExternalApplication(), parent, row)
    {
    }

    QVariant ExternalSourceItem::data(int role) const
    {
        if (role == Qt::FontRole) {
            return Nau::Theme::current().fontLoggerSource();
        }
        if (role == NauLogSourceModel::SourceNamesRole) {
            return QStringList{NauLogConstants::externalSourceName()};
        }

        return Node::data(role);
    }


    // ** ImportedSourceItem

    ImportedSourceItem::ImportedSourceItem(Node* parent, int row)
        : Node({}, {}, parent, row)
    {
    }

    QVariant ImportedSourceItem::data(int role) const
    {
        if (role == Qt::FontRole) {
            return Nau::Theme::current().fontLoggerSource();
        }
        if (role == NauLogSourceModel::SourceNamesRole) {
            return QStringList{ NauLogConstants::importSourceName() };
        }

        return Node::data(role);
    }


    // ** RootSourceItem

    RootSourceItem::RootSourceItem()
        : Node({}, {}, nullptr)
    {
        m_children.emplace_back<EditorSourceItem>(this, 0);
        m_children.emplace_back<BuildSourceItem>(this, 1);
        m_children.emplace_back<PlayModeSourceItem>(this, 2);
        m_children.emplace_back<ExternalSourceItem>(this, 3);
    }

    int RootSourceItem::childCount() const
    {
        return static_cast<int>(m_children.size());
    }

    QVariant RootSourceItem::data(int role) const
    { 
        return QVariant();
    }

    Node* RootSourceItem::childNode(int row) const
    {
        return m_children[row];
    }
}
