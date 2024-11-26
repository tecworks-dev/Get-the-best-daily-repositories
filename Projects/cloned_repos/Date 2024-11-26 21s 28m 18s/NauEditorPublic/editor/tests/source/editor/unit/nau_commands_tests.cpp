// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "commands/nau_commands.hpp"

#include <gtest/gtest.h>


namespace
{
    void fillStackWithEmptyCommands(NauCommandStack& stack, int stackSize)
    {
        while (0 < stackSize--) {
            stack.push(new NauCustomCommand());
        }
    };
}
 
TEST(NauCommandStackTests, newCommandStackIsEmpty)
{
    // Arrange
    NauCommandStack stack;

    // Act
    const bool stackIsEmptyChecker = stack.empty();

    // Assert
    EXPECT_TRUE(stackIsEmptyChecker);
}

TEST(NauCommandStackTests, newCommandStackIsReallyEmptyAndHasZeroSize)
{
    // Arrange
    NauCommandStack stack;

    // Act
    const bool stackIsEmptyChecker = stack.sizeUndo();

    // Assert
    EXPECT_EQ(stackIsEmptyChecker, 0);
}

TEST(NauCommandStackTests, pushCommandToEmptyStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const int commandStackSize = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSize, 1);
}

TEST(NauCommandStackTests, redoStackIsEmptyAfterPush)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();
    const int redoStackSizeBeforePush = stack.sizeRedo();

    // Act
    stack.push(someCustomCommand);
    const int redoStackSizeAfterPush = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSizeBeforePush, redoStackSizeAfterPush);
}

TEST(NauCommandStackTests, pushFewCommandsToStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* firstSomeCustomCommand = new NauCustomCommand();
    NauCustomCommand* secondSomeCustomCommand = new NauCustomCommand();
    NauCustomCommand* thirdSomeNauCustomCommand = new NauCustomCommand();

    // Act
    stack.push(firstSomeCustomCommand);
    stack.push(secondSomeCustomCommand);
    stack.push(thirdSomeNauCustomCommand);
    const int commandStackSize = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSize, 3);
}

TEST(NauCommandStackTests, pushToFilledStack)
{
    // Arrange
    NauCommandStack stack;
    fillStackWithEmptyCommands(stack, 10);
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const int commandStackSize = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSize, 11);
}

TEST(NauCommandStackTests, filledCommandStackHasZeroRedoStackAfterPushing)
{
    // Arrange
    NauCommandStack stack;
    fillStackWithEmptyCommands(stack, 10);
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const int redoStackSizeAfterPush = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSizeAfterPush, 0);
}

TEST(NauCommandStackTests, canUndoAfterPushingCommandToStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const bool canUndoChecker = stack.canUndo();

    // Assert
    EXPECT_TRUE(canUndoChecker);
}

TEST(NauCommandStackTests, undoPushedCommandFromStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const int commandStackSizeBeforeUndo = stack.sizeUndo();
    stack.undo();
    const int commandStackSizeAfterUndo = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSizeAfterUndo, commandStackSizeBeforeUndo - 1);
}

TEST(NauCommandStackTests, undoPushedCommandIsFillingRedoStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    stack.undo();
    const int redoStackSize = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSize, 1);
}

TEST(NauCommandStackTests, canRedoAfterUndoCommand)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    stack.undo();
    const bool canRedoChecker = stack.canRedo();
    stack.redo();

    // Assert
    EXPECT_TRUE(canRedoChecker);
}

TEST(NauCommandStackTests, redoPushedCommandFromStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    stack.undo();
    const int redoStackSizeBeforeRedo = stack.sizeRedo();
    stack.redo();
    const int redoStackSizeAfterRedo = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSizeAfterRedo, redoStackSizeBeforeRedo - 1);
}

TEST(NauCommandStackTests, redoPushedCommandFillingCommandStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    stack.undo();
    const int commandStackSizeBeforeRedo = stack.sizeUndo();
    stack.redo();
    const int commandStackSizeAfterRedo = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSizeAfterRedo, commandStackSizeBeforeRedo + 1);
}

TEST(NauCommandStackTests, clearRedoStackAfterPushNewCommand)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();
    fillStackWithEmptyCommands(stack, 10);

    // Act
    stack.undo();
    const int redoStackSizeBefore = stack.sizeRedo();
    stack.push(someCustomCommand);
    const int redoStackSizeAfter = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSizeAfter, redoStackSizeBefore - 1);
}

TEST(NauCommandStackTests, cantUndoEmptyStack)
{
    // Arrange
    NauCommandStack stack;

    // Act
    const bool canUndoChecker = stack.canUndo();

    // Assert
    EXPECT_FALSE(canUndoChecker);
}

TEST(NauCommandStackTests, tryToUndoEmptyStack)
{
    // Arrange
    NauCommandStack stack;
    const int commandStackSizeBeforeUndo = stack.sizeUndo();

    // Act
    stack.undo();
    int commandStackSizeAfterUndo = stack.sizeUndo();

    // Assert
    EXPECT_EQ(commandStackSizeBeforeUndo, commandStackSizeAfterUndo);
}

TEST(NauCommandStackTests, cantRedoEmptyStack)
{
    // Arrange
    NauCommandStack stack;

    // Act
    const bool canRedoChecker = stack.canRedo();

    // Assert
    EXPECT_FALSE(canRedoChecker);
}

TEST(NauCommandStackTests, tryToRedoEmptyStack)
{
    // Arrange
    NauCommandStack stack;
    const int redoStackSizeBeforeRedo = stack.sizeRedo();

    // Act
    stack.redo();
    const int redoStackSizeAfterRedo = stack.sizeRedo();

    // Assert
    EXPECT_EQ(redoStackSizeBeforeRedo, redoStackSizeAfterRedo);
}

TEST(NauCommandStackTests, newCommandIsUnsaved)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const bool isSavedChecker = stack.isSaved();

    // Assert
    EXPECT_FALSE(isSavedChecker);
}

TEST(NauCommandStackTests, emptyStackHasIsSavedStatus)
{
    // Arrange
    NauCommandStack stack;

    // Act
    const bool isSavedChecker = stack.isSaved();

    // Assert
    EXPECT_TRUE(isSavedChecker); // If stack is empty, it means "last command" is saved
}

TEST(NauCommandStackTests, saveNewCommandInStack)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* someCustomCommand = new NauCustomCommand();

    // Act
    stack.push(someCustomCommand);
    const bool isSavedCheckerBeforeSaving = stack.isSaved();
    stack.markSaved();
    const bool isSavedCheckerAfterSaving = stack.isSaved();

    // Assert
    EXPECT_EQ(isSavedCheckerAfterSaving, !isSavedCheckerBeforeSaving);
}

TEST(NauCommandStackTests, isSavedFlagResetsIfNextCommandIsUnsaved)
{
    // Arrange
    NauCommandStack stack;
    NauCustomCommand* firstCustomCommand = new NauCustomCommand();
    NauCustomCommand* secondCustomCommand = new NauCustomCommand();

    // Act
    stack.push(firstCustomCommand);
    stack.markSaved();
    const bool isSavedCheckerAfterSavingCommand = stack.isSaved();
    stack.push(secondCustomCommand);
    const bool isSavedCheckerWithoutSavingCommand = stack.isSaved();

    // Assert
    EXPECT_EQ(isSavedCheckerWithoutSavingCommand, !isSavedCheckerAfterSavingCommand);
}
