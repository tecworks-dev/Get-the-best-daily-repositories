// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "browser/nau_file_operations.hpp"

#include <gtest/gtest.h>


TEST(NauFileOperationsTests, simpleFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(1).txt");
    
    // Act
    const QString& generatedFileName = NauFileOperations::generateFileName("hello.txt");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, simpleWithComplexExtensionFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(1).grp.bin");
    
    // Act
    const QString& generatedFileName = NauFileOperations::generateFileName("hello.grp.bin");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, emptyFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("(1)");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, strangeFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("((1)");
    
    // Act
    const QString& generatedFileName = NauFileOperations::generateFileName("(");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, simpleWithPathFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("text(1).txt");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("/a/c/b/text.txt");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, withEmptyExtensionFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(1).");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("hello.");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, withNoExtensionFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(1)");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("hello");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, dotFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("(1).");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName(".");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, incrementSimpleFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(2).txt");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("hello(1).txt");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, incrementSimpleNoExtensionFileName)
{
    const auto& expectedFileName = QStringLiteral("hello(2)");
    
    // Act
    const QString& generatedFileName =  NauFileOperations::generateFileName("hello(1)");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}

TEST(NauFileOperationsTests, incrementComplexFileName)
{
    // Arrange
    const auto& expectedFileName = QStringLiteral("hello(1)(2).txt");
    
    // Act
    const QString& generatedFileName = NauFileOperations::generateFileName("hello(1)(1).txt");

    // Assert
    EXPECT_EQ(generatedFileName, expectedFileName);
}
