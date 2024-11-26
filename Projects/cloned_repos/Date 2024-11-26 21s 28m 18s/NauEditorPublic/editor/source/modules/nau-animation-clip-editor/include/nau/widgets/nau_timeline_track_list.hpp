// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Animation timeline track list

#pragma once

#include "nau/nau_timeline_utils.hpp"
#include "baseWidgets/nau_widget.hpp"


class NauPrimaryButton;
class NauObjectCreationList;
class NauTimelineTreeWidget;


// ** NauTimelineTrackHandler

class NauTimelineTrackHandler : public QObject
{
    Q_OBJECT

public:
    using Ptr = std::unique_ptr<NauTimelineTrackHandler>;

    NauTimelineTrackHandler(NauTimelineTreeWidget* treeWidget, int propertyIndex);
    virtual ~NauTimelineTrackHandler() noexcept = default;

    void addWidgetToItem(NauTreeWidgetItem* item, int column, QWidget* widget);
    
    [[nodiscard]]
    virtual NauTreeWidgetItem* createTreeItem(NauTreeWidgetItem* parent, const QString& name);

    virtual void setBlockEditingSignals(bool flag) = 0;
    virtual void updateValue(float time) = 0;

signals:
    void eventEditingFinished(int propertyIndex, const NauAnimationPropertyData& data);

protected:
    NauTimelineTreeWidget* m_treeWidget;
    int m_propertyIndex;
};


// ** NauTimelineTrackVec3Handler

class NauTimelineTrackVec3Handler final : public NauTimelineTrackHandler
{
public:
    NauTimelineTrackVec3Handler(NauTimelineTreeWidget* treeWidget, int propertyIndex, const NauAnimationProperty* property);
    ~NauTimelineTrackVec3Handler() noexcept;

    [[nodiscard]]
    NauTreeWidgetItem* createTreeItem(NauTreeWidgetItem* parent, const QString& name) override;

    void setBlockEditingSignals(bool flag) override;
    void updateValue(float time) override;

private:
    const NauAnimationProperty* m_property;
    std::array<NauTreeWidgetItem*, 3> m_components;
    std::array<NauDoubleSpinBox*, 3> m_widgets;
    std::array<QSignalBlocker*, 3> m_blockers;
};


// ** NauTimelineTrackQuatHandler

class NauTimelineTrackQuatHandler final : public NauTimelineTrackHandler
{
public:
    NauTimelineTrackQuatHandler(NauTimelineTreeWidget* treeWidget, int propertyIndex, const NauAnimationProperty* property);
    ~NauTimelineTrackQuatHandler() noexcept;

    [[nodiscard]]
    NauTreeWidgetItem* createTreeItem(NauTreeWidgetItem* parent, const QString& name) override;

    void setBlockEditingSignals(bool flag) override;
    void updateValue(float time) override;

private:
    const NauAnimationProperty* m_property;
    std::array<NauTreeWidgetItem*, 4> m_components;
    std::array<NauDoubleSpinBox*, 4> m_widgets;
    std::array<QSignalBlocker*, 4> m_blockers;
};


// ** NauTimelineTrackListHeader

class NauTimelineTrackListHeader : public NauWidget
{
    Q_OBJECT

public:
    NauTimelineTrackListHeader(NauWidget* parent);

    void setProperties(NauAnimationPropertyListPtr properties);
    void setClipNameList(const NauAnimationNameList& nameList, int currentNameIndex);

    std::string selectedClipName() const;
    NauAnimationPropertyListPtr propertyList() const noexcept { return m_properties; }

signals:
    void eventAddedProperty(int propertyIndex);
    void eventClipSwitched(int clipIndex);
    void eventAddKeyframe();

protected:
    void paintEvent(QPaintEvent* event) override;

    void updatePropertyList();

private:
    NauMenu* m_propertyListMenu;
    NauComboBox* m_animationSelector;
    NauPrimaryButton* m_addPropertyButton;
    NauAnimationPropertyListPtr m_properties;
};


// ** NauTimelineTrackList
// To play the Timeline instance and to control the location of the Timeline Playhead, use the Timeline Playback Controls and the Playhead Location field.

class NauTimelineTrackList : public NauWidget
{
    Q_OBJECT

public:
    NauTimelineTrackList(NauWidget* parent);

    void setCurrentTime(float time);
    void updateTrackList(NauAnimationPropertyListPtr propertyList, float time);

    NauTimelineTrackListHeader& headerWidget() noexcept;

signals:
    void eventAddKeyframe(int propertyIndex);
    void eventItemExpanded(int propertyIndex, bool flag);
    void eventDeleteProperty(int propertyIndex);
    void eventPropertyChanged(int propertyIndex, const NauAnimationPropertyData& data);

protected:
    void addTrack(const NauAnimationProperty* property, int propertyIndex, int index);

    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    NauTimelineTrackListHeader* m_header;
    NauTimelineTreeWidget* m_propertiesTree;

    std::unordered_map<NauTreeWidgetItem*, QWidget*> m_options;
    NauTreeWidgetItem* m_currentOptionItem;

    std::unordered_map<QTreeWidgetItem*, int> m_propertyDictionary;
    QTreeWidgetItem* m_selectedPropertyItem;

    std::vector<NauTimelineTrackHandler::Ptr> m_trackHandlerList;

    QFont m_common;
    QFont m_uncommon;
};