// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Shortcuts event distributor between functional blocks of application.
//
// Usage example:
// <pre>
// // Assign operation ViewportFocus a some key combination, in this case just F.
// shortcutHub->setOperationKeySequence(NauShortcutOperation::ViewportFocus, NauKeySequence(Qt::Key_F));
// // ...
//  // Create a shortcut for operation ViewportFocus(i.e. key F).
//  
//  shortcutHub->addWidgetShortcut(NauShortcutOperation::ViewportFocus, mywidget,
//  [this](NauShortcutOperation){
//     // If user presses F while mywidget has focus, this handler will be called:
//  });
// </pre>

#pragma once

#include "nau_hash.hpp"
#include "baseWidgets/nau_widget.hpp"
#include "baseWidgets/nau_widget_utility.hpp"
#include "nau_shortcut_operation.hpp"

#include <unordered_map>
#include <functional>


// ** NauShortcutHub

class NAU_EDITOR_API NauShortcutHub : public QObject
{
    Q_OBJECT
public:
    using ShortcutEventHandler = std::function<void(NauShortcutOperation)>;

    explicit NauShortcutHub(NauMainWindow& parent);

    // Returns a new valid NauShortcutOperation that can be used for registering a new shortcut.
    // <pre>
    //    NauShortcutOperation myOperation = hub.registerCustomOperation();
    //    hub.setOperationKeySequence(myOperation, NauKeySequence::Delete);
    //    hub.addWidgetShortcut(myOperation, myWidget, [](NauShortcutOperation operation){
    //         // ...
    //    });
    // </pre>
    NauShortcutOperation registerCustomOperation();

    // Returns associated to specified operation key sequence.
    // Result will be empty no key sequence attached to operation.
    NauKeySequence getAssociatedKeySequence(NauShortcutOperation operation) const;

    // Updates key sequence for specified operation.
    void setOperationKeySequence(NauShortcutOperation operation, const NauKeySequence& newKeySequence);

    // Registers specified widget as a listener of shortcut activation(associated to operation).
    // triggerCallback will be called after shortcut activation in case of widget has focus.
    // Returns true on success.
    bool addWidgetShortcut(NauShortcutOperation operation, QWidget& widget, ShortcutEventHandler triggerCallback);

    // Start/stop listening of specified widget for operation.
    void setWidgetShortcutEnabled(NauShortcutOperation operation, QWidget& widget, bool enabled);

    // Set specified callback triggerCallback as a handler of shortcut(associated to operation) activation.
    // Shortcut has an application context, so there is only one handler can be set.
    // Returns false on failure, e.g. there are already attached listener.
    //
    // Widget listener has a priority over application shortcuts. For example:
    // CTRL+O has a global app scope, but some widgets wants to handle its activation too.
    // In this case global handler will be called only if none of these widgets have not focus.
    bool addApplicationShortcut(NauShortcutOperation operation, ShortcutEventHandler triggerCallback);

private:
    struct ShortcutWidgetReceiverData
    {
        bool enabled = false;
        ShortcutEventHandler handler;
    };

    struct ShortcutData
    {
        NauShortcut* shortcut = nullptr;

        // Widget listeners to shortcut.
        std::unordered_map<QWidget*, ShortcutWidgetReceiverData> widgetReceivers;
        
        // Global listeners to shortcut.
        std::optional<ShortcutEventHandler> applicationReceiver;
    };

private:
    void handleShortcutActivation(NauShortcutOperation operation);
    void handleShortcutAmbiguousActivation(NauShortcutOperation operation);

    using ShortcutRepository = std::unordered_map<NauKeySequence, ShortcutData>;
    using KeySequenceRepository = std::unordered_map<NauShortcutOperation, NauKeySequence>;

    ShortcutRepository::iterator findOrCreateShortcut(NauShortcutOperation operation);

private:
    QObject* m_object = nullptr;

    // User specified settings of mapping formalized operation types on concrete key sequences.
    // {NewScene->CTRL+N}, {OpenScene->CTRL+O},... etc.
    KeySequenceRepository m_keySequenceByOperation;

    ShortcutRepository m_shortcutByKeySequence;

    size_t m_lastUserDefinedOperations = magic_enum::enum_count<NauShortcutOperation>();
};
