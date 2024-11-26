// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_editor_version.hpp"

#include <string>
#include <QStringList>
#include <gtest/gtest.h>

// test_version.txt file as resource. See nau_editor_unit_tests.qrc

TEST(NauEditorVersionTests, checkCurrentVersion)
{
    // Arrange
    const NauEditorVersion expectedEditorVersion(11, 22, 33);

    // Act
    const NauEditorVersion currentEditorVersion = NauEditorVersion::current();

    // Assert
    EXPECT_EQ(currentEditorVersion, expectedEditorVersion);
}

TEST(NauEditorVersionTests, currentVersionIsValid)
{
    // Arrange
    const NauEditorVersion editorVersion(0, 0, 0);

    // Act
    const NauEditorVersion currentEditorVersion = editorVersion.current();
    const bool currentEditorVersionIsNotInvalid = currentEditorVersion != NauEditorVersion::invalid();

    // Assert
    EXPECT_TRUE(currentEditorVersionIsNotInvalid);
}

TEST(NauEditorVersionTests, checkVersionFromFileAsString)
{
    // Arrange
    const std::string expectedStringEditorVersion = "11.22.33 test";

    // Act
    const NauEditorVersion currentEditorVersion = NauEditorVersion::current();
    const std::string currentEditorVersionAsString = currentEditorVersion.asString();

    // Assert
    EXPECT_EQ(currentEditorVersionAsString, expectedStringEditorVersion);
}

TEST(NauEditorVersionTests, checkVersionFromFileAsQString)
{
    // Arrange
    const QString expectedQStringEditorVersion = "11.22.33 test";

    // Act
    const NauEditorVersion currentEditorVersion = NauEditorVersion::current();
    const QString currentEditorVersionAsString = currentEditorVersion.asQtString();

    // Assert
    EXPECT_EQ(currentEditorVersionAsString, expectedQStringEditorVersion);
}

TEST(NauEditorVersionTests, checkVersionFromText)
{
    // Arrange
    const NauEditorVersion expectedVersion(48, 1516, 2342); //Lost TV Series Easter Egg :)

    // Act
    const NauEditorVersion customEditorVersionFromText = NauEditorVersion::fromText("48.1516.2342");

    // Assert
    EXPECT_EQ(customEditorVersionFromText, expectedVersion);
}

TEST(NauEditorVersionTests, version000isInvalid)
{
    // Arrange
    const NauEditorVersion expectInvalidVersion(0, 0, 0);

    // Act
    const bool invalidEditorVersionChecker = expectInvalidVersion == NauEditorVersion::invalid();

    // Assert
    EXPECT_TRUE(invalidEditorVersionChecker);
}

#pragma region Comparing tests

TEST(NauEditorVersionTests, compareTwoSameVersions)
{
    // Arrange
    const NauEditorVersion firstEditorVersion(1, 1, 1);
    const NauEditorVersion secondEditorVersion(1, 1, 1);

    // Act
    const bool actualResult = firstEditorVersion == secondEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareTwoDifferentPatchVersions)
{
    // Arrange
    const NauEditorVersion firstEditorVersion(1, 1, 1);
    const NauEditorVersion secondEditorVersion(1, 1, 2);

    // Act
    const bool actualResult = firstEditorVersion != secondEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareTwoDifferentMinorVersions)
{
    // Arrange
    const NauEditorVersion firstEditorVersion(1, 1, 1);
    const NauEditorVersion secondEditorVersion(1, 2, 1);

    // Act
    const bool actualResult = firstEditorVersion != secondEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareTwoDifferentMajorVersions)
{
    // Arrange
    const NauEditorVersion firstEditorVersion(1, 1, 1);
    const NauEditorVersion secondEditorVersion(2, 1, 1);

    // Act
    const bool actualResult = firstEditorVersion != secondEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareTwoDifferentFullVersions)
{
    // Arrange
    const NauEditorVersion firstEditorVersion(1, 1, 1);
    const NauEditorVersion secondEditorVersion(2, 2, 2);

    // Act
    const bool actualResult = firstEditorVersion != secondEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareLatestAndHigherPatchVersions)
{
    // Arrange
    const NauEditorVersion latestEditorVersion(1, 1, 1);
    const NauEditorVersion higherEditorVersion(1, 1, 2);

    // Act
    const bool actualResult = latestEditorVersion < higherEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareLatestAndHigherMinorVersions)
{
    // Arrange
    const NauEditorVersion latestEditorVersion(1, 1, 1);
    const NauEditorVersion higherEditorVersion(1, 2, 1);

    // Act
    const bool actualResult = latestEditorVersion < higherEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareLatestAndHigherMajorVersions)
{
    // Arrange
    const NauEditorVersion latestEditorVersion(1, 1, 1);
    const NauEditorVersion higherEditorVersion(2, 1, 1);

    // Act
    const bool actualResult = latestEditorVersion < higherEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareHigherAndLatestPatchVersions)
{
    // Arrange
    const NauEditorVersion higherEditorVersion(1, 1, 2);
    const NauEditorVersion latestEditorVersion(1, 1, 1);

    // Act
    const bool actualResult = higherEditorVersion > latestEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareHigherAndLatestMinorVersions)
{
    // Arrange
    const NauEditorVersion higherEditorVersion(1, 2, 1);
    const NauEditorVersion latestEditorVersion(1, 1, 1);

    // Act
    const bool actualResult = higherEditorVersion > latestEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

TEST(NauEditorVersionTests, compareHigherAndLatestMajorVersions)
{
    // Arrange
    const NauEditorVersion higherEditorVersion(2, 1, 1);
    const NauEditorVersion latestEditorVersion(1, 1, 1);

    // Act
    const bool actualResult = higherEditorVersion > latestEditorVersion;

    // Assert
    EXPECT_TRUE(actualResult);
}

#pragma endregion

// Death tests
TEST(NauEditorVersionDeathTests, checkVersionFromInvalidText)
{
    // Arrange
    NauEditorVersion editorVersion(0, 0, 0);

    // Act
    ASSERT_DEATH(editorVersion.fromText("Invalid Version Here"), "");

    // Assert
    EXPECT_EQ(editorVersion, NauEditorVersion::invalid());
}

TEST(NauEditorVersionDeathTests, checkVersionFromTextInvalidFormat)
{
    // Arrange
    NauEditorVersion editorVersion(0, 0, 0);

    // Act
    ASSERT_DEATH(editorVersion.fromText("11.22"), "");

    // Assert
    EXPECT_EQ(editorVersion, NauEditorVersion::invalid());
}