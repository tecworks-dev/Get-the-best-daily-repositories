// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Printers of NAU types for GoogleTest assertions.

#pragma once

#include "nau_widget_utility.hpp"

void PrintTo(const QString& str, std::ostream* os) {
    *os << str.toStdString();
}

void PrintTo(const NauDir& dir, std::ostream* os) {
    *os << dir.absolutePath().toStdString();
}

