// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_tests.hpp"
#include "nau_test_flow_layout.hpp"

#include <QCommandLineParser>


int main(int argc, char* argv[])
{
    // Setup arguments parser
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.setApplicationDescription("NAU Tests");

    // Boolean execution policy argument (see NauTestsExecutionPolicy)
    QCommandLineOption optionExecution("until-failure");
    parser.addOption(optionExecution);

    // Boolean output policy argument (see NauTestsOutputPolicy)
    QCommandLineOption optionXML("xml");
    parser.addOption(optionXML);

    // Parse
    QApplication app(argc, argv);
    if (!parser.parse(app.arguments())) {
        std::fputs(qPrintable(parser.errorText()), stderr);
        return 1;
    }

    // Resolve execution policy
    NauTestsRunParameters params;
    if (parser.isSet(optionExecution)) {
        params.executionPolicy = NauTestsExecutionPolicy::RunUntilFailure;
    }

    // Output policy
    QStringList arguments;
    arguments << app.arguments()[0];  // We need this so the first actual argument we pass won't get ignored or misinterpreted
    if (parser.isSet(optionXML)) {
        // Setup test output to a file
        params.outputPolicy = NauTestsOutputPolicy::JUnitXML;
        const auto outputType = "-junitxml";
        arguments << outputType;
    }

    // Add tests
    NAU_TEST_ADD(NauFlowLayoutUITests);

    // Run all
    return NauTestManager::instance().runAll(arguments, params);
}
