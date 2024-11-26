// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_build_startup_dialog.hpp"
#include "nau/compiler/nau_source_compiler.hpp"
#include "themes/nau_theme.hpp"
#include "nau_plus_enum.hpp"

#include "QFileDialog"


// Build logger macro

#define NED_BUILD_INFO(...) NauLog::buildLogger().logMessage(NauEngineLogLevel::Info, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)
#define NED_BUILD_CRITICAL(...) NauLog::buildLogger().logMessage(NauEngineLogLevel::Critical, __FUNCTION__, __FILE__, static_cast<unsigned>(__LINE__), __VA_ARGS__)


// ** NauBuildStartupDailog

NauBuildStartupDailog::NauBuildStartupDailog(const NauProject& project, NauMainWindow* parent)
    : NauDialog(parent)
    , m_buildToolPath(project.buildToolPath())
    , m_project(project)
{
    setWindowTitle(tr("Application build"));
    setMinimumSize(560, 480);

    auto mainLayout = new NauLayoutVertical(this);
    mainLayout->setContentsMargins(QMargins(OuterMargin, OuterMargin, OuterMargin, OuterMargin));
    mainLayout->setSpacing(LayoutSpacing);

    // Platforms
    auto platformLayout = new NauLayoutVertical();
    auto platformsTitle = new NauLabel(tr("Platforms"));
    m_platforms = new NauComboBox(this);
    platformLayout->addWidget(platformsTitle);
    platformLayout->addWidget(m_platforms);

    // Configurations
    auto configurationLayout = new NauLayoutVertical();
    auto configurationTitle = new NauLabel(tr("Configuration"));
    m_configuration = new NauComboBox(this);
    configurationLayout->addWidget(configurationTitle);
    configurationLayout->addWidget(m_configuration);

    // Architecture
    auto architectureLayout = new NauLayoutVertical();
    auto architectureTitle = new NauLabel(tr("Architecture"));
    m_architecture = new NauComboBox(this);
    architectureLayout->addWidget(architectureTitle);
    architectureLayout->addWidget(m_architecture);

    // Compression
    auto compressionLayout = new NauLayoutVertical();
    auto compressionTitle = new NauLabel(tr("Compression"));
    m_compression = new NauComboBox(this);
    compressionLayout->addWidget(compressionTitle);
    compressionLayout->addWidget(m_compression);

    // Build directory
    auto buildDirLayout = new NauLayoutVertical;
    buildDirLayout->addWidget(new NauLabel(tr("Build Directory")));

    auto buildDirChooseLayout = new NauLayoutHorizontal;
    m_buildDirLabel = new NauLabel(QString());
    m_buildDirLabel->setWordWrap(true);

    auto chooseDirButton = new NauToolButton();
    
    const auto buildDirectory = getDefaultBuildDirectory();
    chooseDirButton->setText(buildDirectory);
    setBuildDirectory(buildDirectory);
    connect(chooseDirButton, &NauToolButton::clicked, [this](){
        const QString selectedDir = QFileDialog::getExistingDirectory(this,
            tr("Select directory for your build"), m_buildDir.absolutePath());

        if (!selectedDir.isEmpty() && NauDir().exists(selectedDir)) {
            setBuildDirectory(selectedDir);
        }
    });

    buildDirChooseLayout->addWidget(m_buildDirLabel);
    buildDirChooseLayout->addWidget(chooseDirButton);

    buildDirLayout->addLayout(buildDirChooseLayout);
    
    // Build settings layout
    auto platformAndConfigLayout = new NauLayoutHorizontal();
    platformAndConfigLayout->addLayout(platformLayout);
    platformAndConfigLayout->addLayout(configurationLayout);
    platformAndConfigLayout->setSpacing(LayoutSpacing);

    auto architectureAndCompressionLayout = new NauLayoutHorizontal();
    architectureAndCompressionLayout->addLayout(architectureLayout);
    architectureAndCompressionLayout->addLayout(compressionLayout);
    architectureAndCompressionLayout->setSpacing(LayoutSpacing);

    // Post build action
    auto postBuildActionLayout = new NauLayoutVertical();
    auto postBuildActionTitle = new NauLabel(tr("After-Build Action"));
    m_postBuildAction = new NauComboBox(this);
    postBuildActionLayout->addWidget(postBuildActionTitle);
    postBuildActionLayout->addWidget(m_postBuildAction);

    // Build buttons
    m_buildButton = new NauPrimaryButton();
    m_buildButton->setText(tr("Build App"));
    m_buildButton->setIcon(Nau::Theme::current().iconPreferences());
    m_buildButton->setFixedHeight(NauAbstractButton::standardHeight());
    connect(m_buildButton, &NauAbstractButton::clicked, this, &NauBuildStartupDailog::runBuild);

    m_cancelBuildButton = new NauPrimaryButton();
    m_cancelBuildButton->setText(tr("Cancel"));
    m_cancelBuildButton->setIcon(Nau::Theme::current().iconClose());
    m_cancelBuildButton->setFixedHeight(NauAbstractButton::standardHeight());
    m_cancelBuildButton->setEnabled(false);
    connect(m_cancelBuildButton, &NauAbstractButton::clicked, this, &NauBuildStartupDailog::runBuild);

    m_buildStatusLabel = new NauLabel();

    auto buildButtonsLayout = new NauLayoutHorizontal();
    buildButtonsLayout->addWidget(m_buildStatusLabel, Qt::AlignLeft);
    buildButtonsLayout->addStretch(1);
    buildButtonsLayout->addWidget(m_cancelBuildButton, Qt::AlignRight);
    buildButtonsLayout->addWidget(m_buildButton, Qt::AlignRight);

    // Fill main layout
    auto separator = new NauLineWidget(ColorSeparator, SeparatorSize, Qt::Horizontal, this);
    separator->setFixedWidth(width());
    separator->setFixedHeight(SeparatorSize);

    mainLayout->addLayout(platformAndConfigLayout);
    mainLayout->addLayout(architectureAndCompressionLayout);
    mainLayout->addWidget(separator);
    mainLayout->addLayout(buildDirLayout);
    mainLayout->addLayout(postBuildActionLayout);
    mainLayout->addStretch(1);
    mainLayout->addLayout(buildButtonsLayout);

    fillSettings();

    if (!m_project.isSourcesCompiled()) {
        m_buildStatusLabel->setText(tr("<img src=\":/UI/icons/compilation/warning-tri-yellow.svg\">Unable to publish this project for its sources out of a date.<br/>"
            "To recompile the sources, please restart the editor"));
        
        m_buildButton->setEnabled(false);
        m_buildButton->setToolTip(m_buildStatusLabel->text());
    }
}

void NauBuildStartupDailog::setBuildState(BuildState state)
{
    if (m_currentBuildState == state) {
        return;
    }

    if (state == BuildState::Building) {
        m_buildButton->setEnabled(false);
        m_cancelBuildButton->setEnabled(true);
        m_buildStatusLabel->setText(tr("Building..."));
    } else {
        m_buildButton->setEnabled(true);
        m_cancelBuildButton->setEnabled(false);
        m_buildStatusLabel->clear();   
    }

    if (state == BuildState::Ready) {
        m_buildStatusLabel->setText(tr("Build finished successfully"));
    } else if (state == BuildState::Failed) {
        m_buildStatusLabel->setText(tr("Build failed"));
    }

    m_currentBuildState = state;
}

QString NauBuildStartupDailog::getDefaultBuildDirectory()
{
    return m_project.path().root().absoluteFilePath("publish");
}

void NauBuildStartupDailog::fillSettings()
{
    // TODO: Get build settings from engine

    m_platforms->addItems({ "Windows desktop" });
    m_architecture->addItems({ "win_vs2022_x64_dll", "win_vs2022_x64"});
#ifdef QT_NO_DEBUG
    m_configuration->addItems({ "Release" , "Debug" });
#else
    m_configuration->addItems({ "Debug", "Release"});
#endif  // QT_NO_DEBUG
    m_compression->addItems({ "None" });

    m_postBuildAction->addItem(tr("None"), +AfterBuildAction::None);
    m_postBuildAction->addItem(tr("Open After Build"), +AfterBuildAction::OpenDirectory);
    m_postBuildAction->setCurrentIndex(m_postBuildAction->findData(+AfterBuildAction::OpenDirectory));
}

void NauBuildStartupDailog::runBuild()
{
    if (m_currentBuildState == BuildState::Building) {
        NED_DEBUG("Command to build requested while the build-process is proceeding.");
        return;
    }

    NauWinDllCompilerCpp compiler;
    std::vector<std::string> logStrings;
    NauSourceCompiler::NauBuildSettings settings;
    settings.configName = m_configuration->currentText();
    settings.preset = m_architecture->currentText();
    settings.targetDir = NauDir::toNativeSeparators(m_buildDir.absolutePath());
    settings.openAfterBuild = m_postBuildAction->currentData().toInt() == +AfterBuildAction::OpenDirectory;

    if(compiler.buildProject(settings, m_project, logStrings, [](const QString& msg) {})) {
        NED_BUILD_INFO("Build project success");
        if (logStrings.size() > 0) {
            NED_BUILD_CRITICAL("But with errors!");
        }
    } else {
        NED_BUILD_CRITICAL("Build project failed!");
    }
    for (auto& str : logStrings) {
        NED_BUILD_CRITICAL(str);
    }
}

void NauBuildStartupDailog::cancelBuild()
{
    if (m_currentBuildState != BuildState::Building) {
        NED_DEBUG("Command to cancel build requested while the build-process is not proceeding.");
        return;
    }

    m_buildProcess->kill();
    setBuildState(BuildState::None);
}

void NauBuildStartupDailog::setBuildDirectory(const QString& directory)
{
    m_buildDirLabel->setText(directory);
    m_buildDir = directory;

    if (!m_buildDir.exists()) {
        m_buildDir.mkpath(".");
    }
}