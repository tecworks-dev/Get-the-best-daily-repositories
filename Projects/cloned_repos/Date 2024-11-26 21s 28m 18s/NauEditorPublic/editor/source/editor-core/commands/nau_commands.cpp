// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_commands.hpp"
#include "nau_assert.hpp"
#include "nau_settings.hpp"


// ** NauAbstractCommand

NauAbstractCommand::NauAbstractCommand()
    : id(NextID++)
{
}


// ** NauCustomCommand

void NauCustomCommand::execute()
{
    if (onExecute) {
        onExecute();
    }
}

void NauCustomCommand::undo()
{
    if (onUndo) {
        onUndo();
    }
}


// ** NauGroupCommand

void NauGroupCommand::push(NauAbstractCommand* command)
{
    m_commands.push(command);
}

NauCommandStack& NauGroupCommand::commands()
{
    return m_commands;
}

void NauGroupCommand::execute()
{
    while (m_commands.canRedo()) {
        m_commands.redo();
    }
}

void NauGroupCommand::undo()
{
    while (m_commands.canUndo()) {
        m_commands.undo();
    }
}

NauCommandDescriptor NauGroupCommand::description() const
{
    // Any child command description will do, since they are all the same
    const auto name = (m_commands.canUndo() ? m_commands.m_commands.front() : m_commands.m_redo.front())->description().name;
    return {
        .id       = id,
        .name     = name,
        .objectId = ""
    };
}


// ** NauCommandStack

NauCommandStack::NauCommandStack()
    : m_sizeLimit(NauSettings::undoRedoStackSize())
{
}

NauCommandStack::~NauCommandStack()
{
    clear();
}

void NauCommandStack::push(NauAbstractCommand* command)
{
    if ((m_sizeLimit > 0) && (m_commands.size() >= m_sizeLimit)) {
        while (m_commands.size() >= m_sizeLimit) {
            m_commands.pop_front();
        }
    }

    m_commands.push_back(command);
    clearStack(m_redo);
    notifyAll();
}

void NauCommandStack::clear()
{
    clearStack(m_commands);
    clearStack(m_redo);
    notifyAll();
}

bool NauCommandStack::empty() const
{
    return m_commands.empty() && m_redo.empty();
}

bool NauCommandStack::canUndo() const
{
    return !m_commands.empty();
}

bool NauCommandStack::canRedo() const
{
    return !m_redo.empty();
}

void NauCommandStack::undo()
{
    if (m_commands.empty()) {
        return;
    }
    auto command = m_commands.back();
    m_commands.pop_back();
    command->undo();
    m_redo.push_back(command);
    notifyAll();
}

void NauCommandStack::undo(NauCommandID id)
{
    NED_ASSERT(id != NauInvalidCommandID);
    m_blockNotifications = true;
    NauAbstractCommand* current = nullptr;
    do {
        current = m_commands.back();
        undo();
    } while (current->id != id);
    m_blockNotifications = false;
    notifyAll();
}

void NauCommandStack::redo()
{
    if (m_redo.empty()) {
        return;
    }
    auto command = m_redo.back();
    m_redo.pop_back();
    command->execute();
    m_commands.push_back(command);
    notifyAll();
}

void NauCommandStack::redo(NauCommandID id)
{
    NED_ASSERT(id != NauInvalidCommandID);
    m_blockNotifications = true;
    NauAbstractCommand* current = nullptr;
    do {
        current = m_redo.back();
        redo();
    } while (current->id != id);
    m_blockNotifications = false;
    notifyAll();
}

void NauCommandStack::markSaved()
{
    if (isSaved()) {
        return;
    }
    m_lastSavedCommand = !m_commands.empty() ? m_commands.back() : nullptr;
    notifyAll();
}

bool NauCommandStack::isSaved() const
{ 
    if (empty() || (m_commands.empty() && !m_lastSavedCommand)) return true;  // Pristine project
    if (!m_lastSavedCommand) return false;   // Already done some actions but never saved
    return !m_commands.empty() && m_lastSavedCommand == m_commands.back();
}

NauCommandStackDescriptor NauCommandStack::describe() const
{
    return {
        .undo = describeStack(m_commands),
        .redo = describeStack(m_redo),
    };
}

void NauCommandStack::addModificationListener(std::function<void()> callback)
{
    m_listeners.push_back(callback);
}

void NauCommandStack::setSizeLimit(size_t size)
{
    m_sizeLimit = size;
}

void NauCommandStack::clearStack(Container& stack)
{
    while (!stack.empty()) {
        auto command = stack.back();
        stack.pop_back();
        delete command;
    }
}

std::vector<NauCommandDescriptor> NauCommandStack::describeStack(Container commands) const
{
    std::vector<NauCommandDescriptor> result;
    while (!commands.empty()) {
        result.push_back(commands.back()->description());
        commands.pop_back();
    }
    return result;
}

void NauCommandStack::notifyAll() const
{
    if (m_blockNotifications) {
        return;
    }

    for (auto& listenerCallback : m_listeners) {
        listenerCallback();
    }
}


// ** NauUndoable

void NauUndoable::groupBegin(size_t commands)
{
    NED_ASSERT(!m_currentGroup);

    m_currentGroup = new NauGroupCommand();
    m_groupCommandsPending = commands;
}

bool NauUndoable::groupIsOpen() const
{ 
    return m_currentGroup != nullptr; 
}

void NauUndoable::groupEnd()
{
    NED_ASSERT(m_currentGroup);
    NED_ASSERT(m_groupCommandsPending == 0);

    m_commands.push(m_currentGroup);
    m_currentGroup = nullptr;
}

bool NauUndoable::undo()
{
    if (m_commands.canUndo()) {
        m_commands.undo();
        return true;
    }
    return false;
}

bool NauUndoable::redo()
{
    if (m_commands.canRedo()) {
        m_commands.redo();
        return true;
    }
    return false;
}
