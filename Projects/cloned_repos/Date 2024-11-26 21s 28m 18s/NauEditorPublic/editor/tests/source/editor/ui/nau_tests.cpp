// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_tests.hpp"


// ** NauTestManager

void NauTestManager::addTest(NauTestBase* test)
{
    m_tests.push_back(test);
}

int NauTestManager::runAll(const QStringList& arguments, NauTestsRunParameters params)
{
    auto result = NauTestBase::SUCCESS;
    for (auto test : *this) {
        QStringList args = arguments;
        if (params.outputPolicy == NauTestsOutputPolicy::JUnitXML) {
            const QString outputFilename = QString("nau_unittests_output_%1.xml").arg(test->name.c_str());
            args << "-o" << outputFilename;
        }
        result = result > 0 ? result : test->run(args);
        if ((result != NauTestBase::SUCCESS) && (params.executionPolicy == NauTestsExecutionPolicy::RunUntilFailure)) {
            return result;
        }
    }
    return result;
}