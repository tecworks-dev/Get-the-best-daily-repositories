// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include <QString>

#include <gtest/gtest.h>

TEST(NauTestQString, toUpper)
{
    // Arrange.
    const QString str = "Hello World";
    const QString expectation = QStringLiteral("HELLO WORLD");

    // Act.
    const QString upper = str.toUpper();

    // Assert.
    EXPECT_EQ(upper.compare(expectation, Qt::CaseSensitive), 0);
}
