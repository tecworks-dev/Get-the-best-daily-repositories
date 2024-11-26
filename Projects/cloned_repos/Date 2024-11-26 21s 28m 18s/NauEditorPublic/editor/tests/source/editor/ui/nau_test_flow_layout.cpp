// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_test_flow_layout.hpp"
#include <QTest>
#include <QLayout>
#include <QLayoutItem>

// TODO: Work in Progress. First GUI unit-tests QTest based proof of concept
// TODO: Refactor object creation to heap from stack
// TODO: Add more tests for NauFlowLayout
// TODO: Uncomment addWidgetToLayoutTest() test for checking after QCoreApplication to QApplication change
// TODO: CI (make a separate one job in pipeline)

void NauFlowLayoutUITests::getMarginFromNewLayout()
{
    // Act
    Qt::LayoutDirection direction = Qt::LeftToRight;
    const int margin = 5;
    const int hSpacing = 15;
    const int vSpacing = 10;

    NauFlowLayout nauFlowLayout(direction, margin, hSpacing, vSpacing, nullptr);

    // Arrange
    int marginLeft, marginTop, marginRight, marginBottom;

    nauFlowLayout.getContentsMargins(&marginLeft, &marginTop, &marginRight, &marginBottom);

    // Assert
    QVERIFY2((marginLeft == margin) && (marginTop == margin) && (marginRight == margin) && (marginBottom == margin), "Margin is different!");
}

void NauFlowLayoutUITests::getHorizontalSpacingFromNewLayout()
{
    // Act
    Qt::LayoutDirection direction = Qt::LeftToRight;
    const int margin = 5;
    const int hSpacing = 15;
    const int vSpacing = 10;

    NauFlowLayout nauFlowLayout(direction, margin, hSpacing, vSpacing, nullptr);

    // Arrange
    int actualHorizontalSpacing = nauFlowLayout.horizontalSpacing();

    // Assert
    QCOMPARE(actualHorizontalSpacing, hSpacing);
}

void NauFlowLayoutUITests::getVerticalSpacingFromNewLayout()
{

    // Act
    Qt::LayoutDirection direction = Qt::RightToLeft;
    const int margin = 5;
    const int hSpacing = 15;
    const int vSpacing = 10;

    NauFlowLayout nauFlowLayout(direction, margin, hSpacing, vSpacing, nullptr);

    // Arrange
    int actualVerticalSpacing = nauFlowLayout.verticalSpacing();


    // Assert
    QCOMPARE(actualVerticalSpacing, vSpacing);
}

void NauFlowLayoutUITests::checkExpandingDirectionsFromNewLayout()
{
    // Act
    Qt::LayoutDirection direction = Qt::RightToLeft;
    const int margin = 5;
    const int hSpacing = 15;
    const int vSpacing = 10;

    NauFlowLayout nauFlowLayout(direction, margin, hSpacing, vSpacing, nullptr);

    // Arrange
    Qt::Orientations actualOrientation = nauFlowLayout.expandingDirections();

    // Assert
    QVERIFY2((actualOrientation != Qt::Vertical) && (actualOrientation != Qt::Horizontal), "Expand direction has specific orientation!");
}

void NauFlowLayoutUITests::checkNewLayoutHasNoWidgets()
{
    // Act
    Qt::LayoutDirection direction = Qt::LeftToRight;
    const int margin = 5;
    const int hSpacing = 15;
    const int vSpacing = 10;

    NauFlowLayout nauFlowLayout(direction, margin, hSpacing, vSpacing, nullptr);

    // Arrange
    int actualCountOfWidgets = nauFlowLayout.count();

    // Assert
    QCOMPARE(actualCountOfWidgets, 0);
}

void NauFlowLayoutUITests::addWidgetToLayoutTest()
{
    /*
    NauFlowLayout nauFlowLayout(Qt::LeftToRight, 5, 10, 15, nullptr);
    QWidget* testWidget; //
    QWidget* testWidget = new QWidget(); // Unhandled exception at 0x00007FFD5D75925A (Qt6Cored.dll) in NauEditorUiTests.exe: Fatal program exit requested.

    nauFlowLayout.addWidget(1, testWidget); //Run-Time Check Failure #3 - The variable 'testWidget' is being used without being initialized.

    QVERIFY2(nauFlowLayout.indexOf(testWidget) == 0, "1");
    */
}