// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Base test suite

#pragma once

#include <QTest>

#include <vector>
#include <string>


// ** NauTestBase

class NauTestBase
{
public:
    enum Result
    {
        SUCCESS = 0,
        ERROR
    };

    NauTestBase(const std::string& name) : name(name) {}

    virtual Result run(const QStringList& arguments) = 0;
    
    const std::string name;
};


// ** NauTest

template <typename Test>
class NauTest : public NauTestBase
{
public:

    NauTest(const std::string& name)
        : NauTestBase(name)
    {
    }

    Result run(const QStringList& arguments) override
    {
        return QTest::qExec(&m_test, arguments) == 0 ? SUCCESS : ERROR;
    }

private:
    Test m_test;
};


// ** NauTestsExecutionPolicy

enum class NauTestsExecutionPolicy
{
    RunAll = 0,        // Runs all tests, regardless of failures
    RunUntilFailure,   // Stops running after the first test failure
};


// ** NauTestsOutputPolicy

enum class NauTestsOutputPolicy
{
    Console = 0,     // Outputs into a console
    JUnitXML,        // Outputs into an xml file
};


// ** NauTestsRunParameters

struct NauTestsRunParameters
{
    NauTestsExecutionPolicy executionPolicy = NauTestsExecutionPolicy::RunAll;
    NauTestsOutputPolicy outputPolicy = NauTestsOutputPolicy::Console;
};


// ** NauTestManager

class NauTestManager
{
public:
    static NauTestManager& instance()
    {
        static const std::unique_ptr<NauTestManager> tests(new NauTestManager());
        return *tests;
    }

    NauTestManager(const NauTestManager&) = delete;
    NauTestManager& operator= (const NauTestManager) = delete;

    void addTest(NauTestBase* test);
    int runAll(const QStringList& arguments, NauTestsRunParameters params);

    auto begin() { return m_tests.begin(); }
    auto end() { return m_tests.end(); }
    auto begin() const { return m_tests.begin(); }
    auto end() const { return m_tests.end(); }

private:
    NauTestManager() {}

private:
    std::vector<NauTestBase*> m_tests;
};


#define NAU_TEST_ADD(X)                            \
    static NauTest<X> test##X(#X);                 \
    NauTestManager::instance().addTest(&test##X);  \
