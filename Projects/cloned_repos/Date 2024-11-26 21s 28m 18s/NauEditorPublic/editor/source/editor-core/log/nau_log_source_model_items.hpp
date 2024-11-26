// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Items and some utils in logger source data model.

#pragma once

#include "baseWidgets/nau_icon.hpp"
#include <QCoreApplication>
#include <QVariant>
#include <QString>
#include <vector>
#include <memory>


namespace NauSourceTreeModelItems
{
    class Node;
    namespace details
    {
        // ** ChildRows
        // Utility class to manipulate children nodes.

        class ChildRows
        {
        public:
            decltype(auto) size() const
            {
                return m_rows.size();
            }

            template<typename T, typename ...Args>
            T* emplace_back(Args&&... args)
            {
                m_rows.push_back(std::make_unique<T>(std::forward<Args>(args)...));
                m_rows.back()->setRow(static_cast<int>(m_rows.size()) - 1);
                return dynamic_cast<T*>(m_rows.back().get());
            }

            Node* operator[](size_t index) const
            {
                if (index < 0 || index >= m_rows.size())
                    return nullptr;

                return m_rows[index].get();
            }

        private:
            std::vector<std::unique_ptr<Node>> m_rows;
        };
    }


    // ** Node
    // Base class on all nodes in the tree data model.

    class Node
    {
    public:
        Node(const QString& text, const NauIcon& icon, Node* parent, int row = -1);
        virtual ~Node() = default;

        virtual Node* parent() const;
        virtual QVariant data(int role) const;

        virtual int childCount() const;
        virtual Node* childNode(int row) const;
        virtual Qt::ItemFlags flags() const;

        int row() const;
        void setRow(int row);

    private:
        Node* m_parent = nullptr;
        QString m_text;
        QString m_tooltip;
        NauIcon m_icon;
        int m_row = -1;
    };


    // ** PlayModeSourceItem

    class PlayModeSourceItem : public Node
    {
        Q_DECLARE_TR_FUNCTIONS(PlayModeSourceItem)
    public:
        PlayModeSourceItem(Node* parent, int row);
        QVariant data(int role) const override;
    };


    // ** BuildSourceItem

    class BuildSourceItem : public Node
    {
        Q_DECLARE_TR_FUNCTIONS(BuildSourceItem)
    public:
        BuildSourceItem(Node* parent, int row);
        QVariant data(int role) const override;
    };


    // ** EditorSourceItem

    class EditorSourceItem : public Node
    {
        Q_DECLARE_TR_FUNCTIONS(EditorSourceItem)
    public:
        EditorSourceItem(Node* parent, int row);
        QVariant data(int role) const override;
    };


    // ** ExternalSourceItem

    class ExternalSourceItem : public Node
    {
        Q_DECLARE_TR_FUNCTIONS(ExternalSourceItem)
    public:
        ExternalSourceItem(Node* parent, int row);
        QVariant data(int role) const override;
    };


    // ** ImportedSourceItem

    class ImportedSourceItem : public Node
    {
        Q_DECLARE_TR_FUNCTIONS(ImportedSourceItem)
    public:
        ImportedSourceItem(Node* parent, int row);
        QVariant data(int role) const override;
    };


    // ** RootSourceItem

    class RootSourceItem : public Node
    {
    public:
        RootSourceItem();

        int childCount() const override;
        QVariant data(int role) const override;
        Node* childNode(int row) const override;

    private:
        // Top level children:
        //    1. m_editorSourceItem
        //    2. m_buildSourceItem
        //    3. m_playModeSourceItem
        //    4. m_externalSourceItems
        // List of imported log sources, e.g. imported from log file from the disk.
        details::ChildRows m_children;
    };
}
