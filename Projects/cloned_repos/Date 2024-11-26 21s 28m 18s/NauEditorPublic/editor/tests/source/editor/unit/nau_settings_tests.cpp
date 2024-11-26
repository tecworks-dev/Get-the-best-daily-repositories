// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_settings_tests.hpp"
#include "nau_settings.hpp"
#include "nau_gtest_printers.hpp"

#include <QStandardPaths>


using NauSettingsDeathTests = NauSettingsTests;

void NauSettingsTests::SetUp()
{
    // Needed for NauSettings(QSettings)
    QCoreApplication::setOrganizationName("Test Organization");
    QCoreApplication::setApplicationName("Test Application");
}

void NauSettingsTests::TearDown()
{
    // Remove unit-test traces
    NauSettings::clearRecentLauncherOutputDirectory();
    NauSettings::clearRecentProjectDirectory();
    NauSettings::clearRecentProjectPaths();
}

TEST_F(NauSettingsTests, validRecentLauncherOutputDirectory)
{
    // Arrange
    const NauDir validDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::DownloadLocation));

    // Act
    NauSettings::setRecentLauncherOutputDirectory(validDirectory);
    const NauDir actualDirectory = NauSettings::recentLauncherOutputDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, validDirectory);
}

TEST_F(NauSettingsTests, overrideRecentLauncherOutputDirectory)
{
    // Arrange
    const NauDir downloadDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::DownloadLocation));
    const NauDir musicDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::MusicLocation));

    // Act
    NauSettings::setRecentLauncherOutputDirectory(downloadDirectory);
    NauSettings::setRecentLauncherOutputDirectory(musicDirectory);
    const NauDir actualDirectory = NauSettings::recentLauncherOutputDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, musicDirectory);
}

TEST_F(NauSettingsTests, emptyRecentLauncherOutputDirectory)
{
    // Arrange
    const NauDir defaultDirectory = NauSettings::defaultLauncherOutputDirectory();

    // Act
    const NauDir actualDirectory = NauSettings::recentLauncherOutputDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, defaultDirectory);
}

TEST_F(NauSettingsTests, validRecentProjectDirectory)
{
    // Arrange
    const NauDir validDirectory = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);

    // Act
    NauSettings::setRecentProjectDirectory(validDirectory);
    const NauDir actualDirectory = NauSettings::recentProjectDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, validDirectory);
}

TEST_F(NauSettingsTests, overrideRecentProjectDirectory)
{
    // Arrange
    const NauDir downloadDirectory = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    const NauDir musicDirectory = QStandardPaths::writableLocation(QStandardPaths::MusicLocation);

    // Act
    NauSettings::setRecentProjectDirectory(downloadDirectory);
    NauSettings::setRecentProjectDirectory(musicDirectory);
    const NauDir actualDirectory = NauSettings::recentProjectDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, musicDirectory);
}

TEST_F(NauSettingsTests, emptyRecentProjectDirectory)
{
    // Arrange
    const NauDir defaultDirectory = NauSettings::defaultProjectDirectory();

    // Act
    const NauDir actualDirectory = NauSettings::recentProjectDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, defaultDirectory);
}

TEST_F(NauSettingsTests, addTwoUniqueRecentProjects)
{
    // Arrange
    const NauDir documentsDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    const NauProjectPath projectShooter(documentsDirectory.absoluteFilePath("shooter." + NauProjectPath::suffix).toStdString());
    const NauProjectPath projectRacing(documentsDirectory.absoluteFilePath("Racing." + NauProjectPath::suffix).toStdString());
    const NauProjectPathList actualProjectList({ projectShooter, projectRacing });

    // Act
    NauSettings::addRecentProjectPath(projectShooter);
    NauSettings::addRecentProjectPath(projectRacing);
    const NauProjectPathList recentProjects = NauSettings::recentProjectPaths();

    // Assert
    EXPECT_EQ(recentProjects, actualProjectList);
}

TEST_F(NauSettingsTests, addTwoDuplicateRecentProjects)
{
    // Arrange
    const NauDir documentsDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    const NauProjectPath projectShooter(documentsDirectory.absoluteFilePath("shooter." + NauProjectPath::suffix).toStdString());
    const NauProjectPathList actualProjectList({ projectShooter });

    // Act
    NauSettings::addRecentProjectPath(projectShooter);
    NauSettings::addRecentProjectPath(projectShooter);
    const NauProjectPathList recentProjects = NauSettings::recentProjectPaths();

    // Assert
    EXPECT_EQ(recentProjects, actualProjectList);
}

TEST_F(NauSettingsTests, removeOneRecentProject)
{
    // Arrange
    const NauDir documentsDirectory = NauDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    const NauProjectPath projectShooter(documentsDirectory.absoluteFilePath("shooter." + NauProjectPath::suffix).toStdString());
    const NauProjectPath projectRacing(documentsDirectory.absoluteFilePath("Racing." + NauProjectPath::suffix).toStdString());
    const NauProjectPathList actualProjectList({ projectShooter });

    // Act
    NauSettings::addRecentProjectPath(projectShooter);
    NauSettings::addRecentProjectPath(projectRacing);
    NauSettings::tryAndRemoveRecentProjectPath(projectRacing);
    const NauProjectPathList recentProjects = NauSettings::recentProjectPaths();

    // Assert
    EXPECT_EQ(recentProjects, actualProjectList);
}

TEST_F(NauSettingsDeathTests, invalidRecentLauncherOutputDirectory)
{
    // Arrange
    const NauDir invalidDirectory(QString("?"));
    const NauDir defaultDirectory = NauSettings::defaultLauncherOutputDirectory();

    // Act
    ASSERT_DEATH(NauSettings::setRecentLauncherOutputDirectory(invalidDirectory), "");
    const NauDir actualDirectory = NauSettings::recentLauncherOutputDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, defaultDirectory);
}

TEST_F(NauSettingsDeathTests, invalidRecentProjectDirectory)
{
    // Arrange
    const NauDir invalidDirectory(QString("?"));
    const NauDir defaultDirectory = NauSettings::defaultProjectDirectory();

    // Act
    ASSERT_DEATH(NauSettings::setRecentProjectDirectory(invalidDirectory), "");
    const NauDir actualDirectory = NauSettings::recentProjectDirectory();

    // Assert
    EXPECT_EQ(actualDirectory, defaultDirectory);
}
