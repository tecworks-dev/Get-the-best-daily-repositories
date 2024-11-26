// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Command pattern with undo/redo functionality

#pragma once

#include "nau/nau_editor_config.hpp"

#include <deque>
#include <functional>
#include <string>
#include <optional>


// ** NauCommandID

using NauCommandID = uint64_t;
static NauCommandID NauInvalidCommandID = -1;


// ** NauCommandDescriptor

struct NauCommandDescriptor
{
    NauCommandID                id = NauInvalidCommandID;
    std::string                 name;
    std::string                 objectId;
    std::optional<std::string>  value1;
    std::optional<std::string>  value2;
};


// ** NauCommandStackDescriptor

struct NauCommandStackDescriptor
{
    std::vector<NauCommandDescriptor> undo;
    std::vector<NauCommandDescriptor> redo;
};


// ** NauAbstractCommand

class NAU_EDITOR_API NauAbstractCommand
{
public:
    NauAbstractCommand();
    
    virtual void execute() = 0;
    virtual void undo() = 0;
    virtual NauCommandDescriptor description() const = 0;

public:
    NauCommandID id;

private:
    inline static int NextID = 0;
};


// ** NauCustomCommand

using NauCommandClosure = std::function<void()>;

class NAU_EDITOR_API NauCustomCommand : public NauAbstractCommand
{
public:
    NauCustomCommand() = default;

    void execute() override;
    void undo() override;
    NauCommandDescriptor description() const override { return { NauInvalidCommandID }; };

    NauCommandClosure onExecute = nullptr;
    NauCommandClosure onUndo = nullptr;
};


// ** NauCommandStack

class NAU_EDITOR_API NauCommandStack
{
    using Container = std::deque<NauAbstractCommand*>;
    using ModifyCallback = std::function<void()>;
    
    friend class NauGroupCommand;

public:
    NauCommandStack();
    ~NauCommandStack();

    void push(NauAbstractCommand* command);
    void clear();

    bool empty() const;
    bool canUndo() const;
    bool canRedo() const;

    void undo();
    void undo(NauCommandID id);  // Undoes up to (and including) the provided command
    void redo();
    void redo(NauCommandID id);  // Redoes up to (and including) the provided command

    void markSaved();
    bool isSaved() const;

    NauCommandStackDescriptor describe() const;

    void addModificationListener(ModifyCallback callback);
    void setSizeLimit(size_t size);

    // Get stack sizes
    size_t sizeTotal() const { return sizeUndo() + sizeRedo(); }
    size_t sizeUndo() const { return m_commands.size(); }
    size_t sizeRedo() const { return m_redo.size(); }

public:
    inline static constexpr int DefaultStackSize = 0;  // Unlimited by default

protected:
    void clearStack(Container& stack);
    std::vector<NauCommandDescriptor> describeStack(Container stack) const;
    void notifyAll() const;

private:
    Container                     m_commands;
    Container                     m_redo;
    NauAbstractCommand*           m_lastSavedCommand = nullptr;
    std::vector<ModifyCallback>   m_listeners;
    size_t                        m_sizeLimit;
    bool                          m_blockNotifications = false;
};


// ** NauGroupCommand

class NAU_EDITOR_API NauGroupCommand : public NauAbstractCommand
{
public:
    NauGroupCommand() = default;

    void push(NauAbstractCommand* command);
    NauCommandStack& commands();

    void execute() override;
    void undo() override;
    NauCommandDescriptor description() const override;

private:
    NauCommandStack m_commands;
};


// ** NauUndoable

class NAU_EDITOR_API NauUndoable
{
public:
    NauUndoable() = default;

    void groupBegin(size_t commands);
    bool groupIsOpen() const;

    template<std::derived_from<NauAbstractCommand> T, bool execute = true, typename... Args>
    void addCommand(Args&&... args)
    {
        auto command = new T(std::forward<Args>(args)...);
        if (execute) command->execute();

        if (m_currentGroup) {
            m_currentGroup->push(command);
            if (--m_groupCommandsPending == 0) {
                groupEnd();
            }
            return;
        }

        m_commands.push(command);
    }

    bool undo();
    bool redo();

protected:
    void groupEnd();

protected:
    NauCommandStack  m_commands;
    NauGroupCommand* m_currentGroup = nullptr;
    size_t           m_groupCommandsPending = 0;
};
