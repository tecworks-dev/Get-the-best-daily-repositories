// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.

#include "nau_default_theme.hpp"
#include "nau_color.hpp"
#include "nau_log.hpp"

#include <QFontDatabase>


// ** NauDefaultTheme

NauDefaultTheme::NauDefaultTheme()
{
    NauFont ibmpPlexSansBold = registerFont(":/fonts/fonts/IBMPlexSans-Bold.ttf");
    ibmpPlexSansBold.setPixelSize(12);
    ibmpPlexSansBold.setBold(true);
    ibmpPlexSansBold.setStyleStrategy(NauFont::PreferAntialias);

    NauFont ibmpPlexSansRegular = registerFont(":/fonts/fonts/IBMPlexSans-Regular.ttf");
    ibmpPlexSansRegular.setPixelSize(12);
    ibmpPlexSansRegular.setStyleStrategy(NauFont::PreferAntialias);

    NauFont ibmpPlexSansMedium = registerFont(":/fonts/fonts/IBMPlexSans-Medium.ttf");
    ibmpPlexSansMedium.setPixelSize(12);
    ibmpPlexSansMedium.setStyleStrategy(NauFont::PreferAntialias);

    NauFont ibmpPlexSansSemiBold = registerFont(":/fonts/fonts/IBMPlexSans-SemiBold.ttf");
    ibmpPlexSansSemiBold.setPixelSize(14);
    ibmpPlexSansSemiBold.setStyleStrategy(NauFont::PreferAntialias);

    m_fontTitleBarMenuItem = ibmpPlexSansMedium;

    m_fontDocking = m_fontTitleBarMenuItem;

    m_fontTitleBarTitle = ibmpPlexSansRegular;
    m_fontTitleBarTitle.setPixelSize(12);

    m_fontLogger = ibmpPlexSansRegular;
    m_fontLoggerLevel = ibmpPlexSansBold;
    m_fontLoggerSource = ibmpPlexSansSemiBold;

    m_fontProjectBrowserPrimary = ibmpPlexSansSemiBold;
    m_fontProjectBrowserPrimary.setPixelSize(12);
    m_fontProjectBrowserSecondary = ibmpPlexSansRegular;
    m_fontProjectBrowserSecondary.setPixelSize(12);

    m_fontFilterItem = ibmpPlexSansRegular;

    m_fontWorldOutlineHeaderSemibold = ibmpPlexSansSemiBold;
    m_fontWorldOutlineHeaderSemibold.setPixelSize(12);

    m_fontWorldOutlineHeaderRegular = ibmpPlexSansRegular;
    m_fontWorldOutlineHeaderRegular.setPixelSize(12);

    m_fontWorldOutlineSemibold = ibmpPlexSansSemiBold;
    m_fontWorldOutlineSemibold.setPixelSize(14);

    m_fontWorldOutlineRegular = ibmpPlexSansRegular;
    m_fontWorldOutlineRegular.setPixelSize(12);

    m_fontFeedbackHeaderTitle = ibmpPlexSansBold;
    m_fontFeedbackHeaderTitle.setPixelSize(16);

    m_fontFeedbackCommentInfo = ibmpPlexSansSemiBold;
    m_fontFeedbackCommentInfo.setPixelSize(12);

    m_fontFeedbackErrorInfo = ibmpPlexSansRegular;

    m_fontFeedbackConnectionError = ibmpPlexSansBold;
    m_fontFeedbackConnectionError.setPixelSize(12);

    m_fontFeedbackHeaderThankTitle = ibmpPlexSansBold;
    m_fontFeedbackHeaderThankTitle.setPixelSize(24);

    m_fontInspectorHeaderTitle = ibmpPlexSansSemiBold;
    m_fontInspectorHeaderTitle.setPixelSize(16);

    m_fontInspectorHeaderSubtitle = ibmpPlexSansRegular;
    m_fontInspectorHeaderSubtitle.setPixelSize(12);

    m_fontObjectInspectorSpoiler = ibmpPlexSansSemiBold;
    m_fontObjectInspectorSpoiler.setPixelSize(14);

    m_fontObjectInspector = ibmpPlexSansSemiBold;
    m_fontObjectInspector.setPixelSize(12);

    m_fontPrimaryButton = ibmpPlexSansMedium;
    m_fontPrimaryButton.setPixelSize(14);

    m_fontStaticTextBase = ibmpPlexSansSemiBold;
    m_fontStaticTextBase.setPixelSize(14);

    m_fontTableWidgetBase = ibmpPlexSansSemiBold;
    m_fontTableWidgetBase.setPixelSize(14);

    m_fontProjectManagerMenuButton = ibmpPlexSansBold;
    m_fontProjectManagerMenuButton.setPixelSize(12);

    m_fontProjectManagerLocation = ibmpPlexSansRegular;
    m_fontProjectManagerLocation.setPixelSize(10);

    // Undo history
    m_fontHistoryMain = ibmpPlexSansMedium;
    m_fontHistoryMain.setPixelSize(12);

    m_fontInputHeaderTitle = m_fontInspectorHeaderTitle;
    m_fontInputHeaderTitle.setPixelSize(16);
    m_fontInputHeaderSubtitle = m_fontInspectorHeaderSubtitle;
    m_fontInputHeaderSubtitle.setPixelSize(12);

    m_fontInputLabel = m_fontInputHeaderTitle;
    m_fontInputLabel.setPixelSize(14);

    m_fontInputTab = m_fontPrimaryButton;
    m_fontInputTab.setPixelSize(12);

    m_fontSplashProjectName = ibmpPlexSansSemiBold;
    m_fontSplashProjectName.setPixelSize(16);
    
    m_fontSplashMessage = ibmpPlexSansRegular;
    m_fontSplashMessage.setPixelSize(14);
    
    m_fontSplashCopyright = ibmpPlexSansRegular;
    m_fontSplashCopyright.setPixelSize(12);

    m_paletteTitleBar.setBrush(NauPalette::Role::Background, NauColor(20, 20, 20));
    m_paletteTitleBar.setColor(NauPalette::Role::Foreground, NauColor(128, 128, 128));
    m_paletteTitleBar.setColor(NauPalette::Role::ForegroundBrightText, Qt::white);

    m_fontDataContainers = ibmpPlexSansSemiBold;
    m_fontDataContainers.setPixelSize(12);

    m_fontDataContainersLabel = ibmpPlexSansRegular;
    m_fontDataContainersLabel.setPixelSize(12);

    m_paletteWorldOutline.setBrush(NauPalette::Role::BackgroundHeader, NauColor(0x3F3F3F));
    m_paletteWorldOutline.setColor(NauPalette::Role::ForegroundHeader, NauColor{ 128, 128, 128});
    m_paletteWorldOutline.setColor(NauPalette::Role::ForegroundBrightHeader, NauColor{ 153, 153, 153});
    
    m_paletteWorldOutline.setColor(NauPalette::Role::Foreground, NauColor{ 75, 75, 75}, {}, NauPalette::Category::Disabled);
    m_paletteWorldOutline.setColor(NauPalette::Role::ForegroundBrightText, NauColor{ 75, 75, 75}, {}, NauPalette::Category::Disabled);

    m_paletteWorldOutline.setBrush(NauPalette::Role::Background, NauColor(0x282828));
    m_paletteWorldOutline.setBrush(NauPalette::Role::Background, NauColor(0x282828), NauPalette::Hovered);
    m_paletteWorldOutline.setBrush(NauPalette::Role::AlternateBackground, NauColor(0x242424));
    m_paletteWorldOutline.setColor(NauPalette::Role::Foreground, NauColor(0x808080));
    m_paletteWorldOutline.setColor(NauPalette::Role::Foreground, NauColor(0x808080), NauPalette::Selected);
    m_paletteWorldOutline.setColor(NauPalette::Role::ForegroundBrightText, NauColor(0x999999));
    m_paletteWorldOutline.setColor(NauPalette::Role::ForegroundBrightText, NauColor(0xFFFFFF), NauPalette::Selected);

    m_paletteWorldOutline.setBrush(NauPalette::Role::Background, NauColor{33, 45, 153, 179}, NauPalette::Selected);
    m_paletteWorldOutline.setBrush(NauPalette::Role::Background, NauColor{33, 45, 153, 179}, NauPalette::Selected | NauPalette::Hovered);
    m_paletteWorldOutline.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered);

    m_paletteLogger.setBrush(NauPalette::Role::BackgroundHeader, NauColor(0x3F3F3F));
    m_paletteLogger.setColor(NauPalette::Role::ForegroundHeader, Qt::white);

    m_paletteLogger.setBrush(NauPalette::Role::Background, NauColor(0x282828));
    m_paletteLogger.setBrush(NauPalette::Role::Background, NauColor(0x282828), NauPalette::Hovered);
    m_paletteLogger.setBrush(NauPalette::Role::AlternateBackground, NauColor(0x242424));
    m_paletteLogger.setColor(NauPalette::Role::Foreground, NauColor(0x808080));
    m_paletteLogger.setColor(NauPalette::Role::ForegroundBrightText, Qt::white);
    m_paletteLogger.setColor(NauPalette::Role::Foreground, Qt::white, NauPalette::Selected);
    m_paletteLogger.setColor(NauPalette::Role::ForegroundBrightText, Qt::white, NauPalette::Selected);

    m_paletteLogger.setBrush(NauPalette::Role::Background, NauColor{33, 45, 153, 179}, NauPalette::Selected);
    m_paletteLogger.setBrush(NauPalette::Role::Background, NauColor{33, 45, 153, 179}, NauPalette::Selected | NauPalette::Hovered);
    m_paletteLogger.setColor(NauPalette::Role::Border, NauColor(0x5566FF), NauPalette::Hovered);

    m_paletteProjectBrowser = m_paletteLogger;

    m_paletteObjectInspector.setBrush(NauPalette::Role::Background, NauColor(28, 28, 28));
    m_paletteDataContainers.setBrush(NauPalette::Role::Background, NauColor(22, 22, 22));

    m_palettePrimaryButton.setColor(NauPalette::Role::Background, NauColor(49, 67, 229), NauPalette::Normal, NauPalette::Category::Active);
    m_palettePrimaryButton.setColor(NauPalette::Role::Text, NauColor(255, 255, 255), NauPalette::Normal, NauPalette::Category::Active);
    m_palettePrimaryButton.setColor(NauPalette::Role::Border, NauColor(49, 67, 229), NauPalette::Normal, NauPalette::Category::Active);

    m_palettePrimaryButton.setColor(NauPalette::Role::Background, NauColor(85, 102, 255), NauPalette::Hovered);
    m_palettePrimaryButton.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered);

    m_palettePrimaryButton.setColor(NauPalette::Role::Background, NauColor(38, 52, 178), NauPalette::Pressed);
    m_palettePrimaryButton.setColor(NauPalette::Role::Border, NauColor(38, 52, 178), NauPalette::Pressed);

    m_palettePrimaryButton.setColor(NauPalette::Role::Border, NauColor(255, 255, 255), NauPalette::Selected);

    m_palettePrimaryButton.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Normal, NauPalette::Category::Disabled);
    m_palettePrimaryButton.setColor(NauPalette::Role::Text, NauColor(91, 91, 91), NauPalette::Normal, NauPalette::Category::Disabled);
    m_palettePrimaryButton.setColor(NauPalette::Role::Border, NauColor(0, 0, 0, 0), NauPalette::Normal, NauPalette::Category::Disabled);


    m_paletteSecondaryButton.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Normal, NauPalette::Category::Active);

    m_paletteSecondaryButton.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Hovered);

    m_paletteSecondaryButton.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Pressed);

    m_paletteSecondaryButton.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Selected);


    m_paletteTertiaryButton.setColor(NauPalette::Role::Border, NauColor(0, 0, 0, 0), NauPalette::Normal, NauPalette::Category::Active);
    m_paletteTertiaryButton.setColor(NauPalette::Role::Text, NauColor(91, 91, 91), NauPalette::Normal, NauPalette::Category::Active);

    // Toogle Button
    m_paletteToogleButton.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal, NauPalette::Category::Active);
    m_paletteToogleButton.setColor(NauPalette::Role::Text, NauColor(64, 64, 64), NauPalette::Normal, NauPalette::Category::Active);
    m_paletteToogleButton.setColor(NauPalette::Role::Border, NauColor(64, 64, 64), NauPalette::Normal, NauPalette::Category::Active);

    m_paletteToogleButton.setColor(NauPalette::Role::Background, NauColor(65, 84, 255), NauPalette::Hovered);
    m_paletteToogleButton.setColor(NauPalette::Role::Border, NauColor(65, 84, 255), NauPalette::Hovered);

    m_paletteToogleButton.setColor(NauPalette::Role::Background, NauColor(65, 84, 255), NauPalette::Pressed);
    m_paletteToogleButton.setColor(NauPalette::Role::Text, NauColor(255, 255, 255), NauPalette::Pressed);
    m_paletteToogleButton.setColor(NauPalette::Role::Border, NauColor(65, 84, 255), NauPalette::Pressed);

    m_paletteToogleButton.setColor(NauPalette::Role::Border, NauColor(34, 34, 34), NauPalette::Selected);

    m_paletteToogleButton.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal, NauPalette::Category::Disabled);
    m_paletteToogleButton.setColor(NauPalette::Role::Text, NauColor(63, 63, 63), NauPalette::Normal, NauPalette::Category::Disabled);
    m_paletteToogleButton.setColor(NauPalette::Role::Border, NauColor(63, 63, 63), NauPalette::Normal, NauPalette::Category::Disabled);

    // Search widget
    m_paletteSearchWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal);
    m_paletteSearchWidget.setColor(NauPalette::Role::Border, NauColor(34, 34, 34), NauPalette::Normal);
    m_paletteSearchWidget.setColor(NauPalette::Role::Text, NauColor(100, 100, 100), NauPalette::Normal);

    m_paletteSearchWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Pressed);
    m_paletteSearchWidget.setColor(NauPalette::Role::Border, NauColor(65, 84, 255), NauPalette::Pressed);
    m_paletteSearchWidget.setColor(NauPalette::Role::Text, Qt::white, NauPalette::Pressed);

    m_paletteSearchWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Hovered);
    m_paletteSearchWidget.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered);
    m_paletteSearchWidget.setColor(NauPalette::Role::Text, NauColor(100, 100, 100), NauPalette::Hovered);

    m_paletteSearchWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Selected);
    m_paletteSearchWidget.setColor(NauPalette::Role::Border, Qt::white, NauPalette::Selected);
    m_paletteSearchWidget.setColor(NauPalette::Role::Text, NauColor(100, 100, 100), NauPalette::Selected);

    // Filter widget
    m_paletteFilterWidget.setColor(NauPalette::Role::Border, Qt::white, NauPalette::Selected);
    m_paletteFilterWidget.setColor(NauPalette::Role::Background, NauColor(153, 153, 153), NauPalette::Selected);
    m_paletteFilterWidget.setColor(NauPalette::Role::AlternateBackground, NauColor(153, 153, 153), NauPalette::Selected);

    m_paletteFilterWidget.setColor(NauPalette::Role::Background, NauColor(153, 153, 153), NauPalette::Normal);
    m_paletteFilterWidget.setColor(NauPalette::Role::AlternateBackground, NauColor(153, 153, 153), NauPalette::Normal);

    m_paletteFilterWidget.setColor(NauPalette::Role::Background, NauColor(65, 84, 255 ), NauPalette::Pressed);
    m_paletteFilterWidget.setColor(NauPalette::Role::AlternateBackground, NauColor(153, 153, 153), NauPalette::Pressed);

    m_paletteFilterWidget.setColor(NauPalette::Role::Background, NauColor(255, 255, 255), NauPalette::Hovered);
    m_paletteFilterWidget.setColor(NauPalette::Role::AlternateBackground, NauColor(255, 255, 255), NauPalette::Hovered);

    m_paletteFilterWidget.setColor(NauPalette::Role::Background, NauColor(63, 63, 63), NauPalette::Normal, NauPalette::Category::Disabled);
    m_paletteFilterWidget.setColor(NauPalette::Role::AlternateBackground, NauColor(63, 63, 63), NauPalette::Normal, NauPalette::Category::Disabled);

    // Filter Item widget
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Background, NauColor(44, 60, 204), NauPalette::Normal);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Normal);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Text, NauColor(192, 192, 192), NauPalette::Normal);

    m_paletteFilterItemWidget.setColor(NauPalette::Role::Background, NauColor(44, 60, 204), NauPalette::Hovered);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Border, NauColor(255, 255, 255), NauPalette::Hovered);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 255), NauPalette::Hovered);

    m_paletteFilterItemWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal, NauPalette::Category::Inactive);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Border, NauColor(91, 91, 91), NauPalette::Normal, NauPalette::Category::Inactive);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Text, NauColor(91, 91, 91), NauPalette::Normal, NauPalette::Category::Inactive);

    m_paletteFilterItemWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Hovered, NauPalette::Category::Inactive);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered, NauPalette::Category::Inactive);
    m_paletteFilterItemWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 255), NauPalette::Hovered, NauPalette::Category::Inactive);

    // Feedback Text Edit widget
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Border, NauColor(27, 27, 27), NauPalette::Normal);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 0), NauPalette::Normal);

    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Selected);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Border, NauColor(85, 103, 255), NauPalette::Selected);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 0), NauPalette::Selected);

    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Pressed);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Border, NauColor(85, 103, 255), NauPalette::Pressed);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 0), NauPalette::Pressed);

    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Error);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Border, NauColor(227, 105, 21), NauPalette::Error);
    m_paletteFeedbackTextEditWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 0), NauPalette::Error);

    // Input editor's generic
    m_paletteInputGeneric.setColor(NauPalette::Role::Background, Qt::transparent, NauPalette::Normal);
    m_paletteInputGeneric.setColor(NauPalette::Role::Border, Qt::transparent, NauPalette::Normal);
    m_paletteInputGeneric.setColor(NauPalette::Role::Text, NauColor{ 0x808080 }, NauPalette::Normal);
    m_paletteInputGeneric.setColor(NauPalette::Role::TextHeader, Qt::white, NauPalette::Normal);

    // Input editor's line edit
    m_paletteInputSpoilerLineEdit.setColor(NauPalette::Role::Background, Qt::transparent, NauPalette::Hovered);
    m_paletteInputSpoilerLineEdit.setColor(NauPalette::Role::Border, NauColor{ 0x5566FF }, NauPalette::Hovered);
    m_paletteInputSpoilerLineEdit.setColor(NauPalette::Role::Text, Qt::white, NauPalette::Hovered);

    // Input editor's signal view
    m_paletteInputSignalView = m_paletteInputSpoilerLineEdit;

    // Input editor's bind tab
    m_paletteInputBindTab.setColor(NauPalette::Role::Background, NauColor{ 0x343434 }, NauPalette::Selected);
    m_paletteInputBindTab.setColor(NauPalette::Role::Border, Qt::transparent, NauPalette::Selected);
    m_paletteInputBindTab.setColor(NauPalette::Role::Text, NauColor{ 0xFFFFFF }, NauPalette::Selected);

    m_paletteInputBindTab.setColor(NauPalette::Role::Background, NauColor{ 0x282828 }, NauPalette::Normal);
    m_paletteInputBindTab.setColor(NauPalette::Role::Border, Qt::transparent, NauPalette::Normal);
    m_paletteInputBindTab.setColor(NauPalette::Role::Text, NauColor{ 0x646464 }, NauPalette::Normal);
    
    // VFX editor's generic
    m_paletteVFXGeneric.setColor(NauPalette::Role::Background, Qt::transparent, NauPalette::Normal);
    m_paletteVFXGeneric.setColor(NauPalette::Role::Border, Qt::transparent, NauPalette::Normal);
    m_paletteVFXGeneric.setColor(NauPalette::Role::Text, NauColor(128, 128, 128), NauPalette::Normal);
    m_paletteVFXGeneric.setColor(NauPalette::Role::TextHeader, Qt::white, NauPalette::Normal);

    // Resource Widget
    m_paletteResourceWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal, NauPalette::Category::Active);
    m_paletteResourceWidget.setColor(NauPalette::Role::Text, NauColor(255, 255, 255), NauPalette::Normal, NauPalette::Category::Active);
    m_paletteResourceWidget.setColor(NauPalette::Role::Border, NauColor(34, 34, 34), NauPalette::Normal, NauPalette::Category::Active);

    m_paletteResourceWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Hovered);
    m_paletteResourceWidget.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered);

    m_paletteResourceWidget.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Pressed);
    m_paletteResourceWidget.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Pressed);

    m_paletteResourceWidget.setColor(NauPalette::Role::Border, NauColor(255, 255, 255), NauPalette::Selected);

    m_paletteResourceWidget.setColor(NauPalette::Role::Background, NauColor(0, 0, 0, 0), NauPalette::Normal, NauPalette::Category::Disabled);
    m_paletteResourceWidget.setColor(NauPalette::Role::Text, NauColor(91, 91, 91), NauPalette::Normal, NauPalette::Category::Disabled);
    m_paletteResourceWidget.setColor(NauPalette::Role::Border, NauColor(63, 63, 63), NauPalette::Normal, NauPalette::Category::Disabled);

    // Resource Popup-Widget
    m_paletteResourcePopupWidget.setColor(NauPalette::Role::BackgroundHeader, NauColor(20, 20, 20));
    m_paletteResourcePopupWidget.setColor(NauPalette::Role::Background, NauColor(40, 40, 40));
    m_paletteResourcePopupWidget.setColor(NauPalette::Role::BackgroundFooter, NauColor(20, 20, 20));

    // Numeric Slider
    m_paletteNumericSlider.setColor(NauPalette::Role::Background, NauColor(34, 34, 34));
    m_paletteNumericSlider.setColor(NauPalette::Role::AlternateBackground, NauColor(33, 45, 153));
    m_paletteNumericSlider.setColor(NauPalette::Role::Foreground, NauColor(85, 102, 255));
    m_paletteNumericSlider.setColor(NauPalette::Role::AlternateForeground, Qt::white);
    m_paletteNumericSlider.setColor(NauPalette::Role::Border, NauColor(85, 102, 255));

    // Timeline Track List
    m_paletteTimelineTrackList.setBrush(NauPalette::Role::Background, NauColor(40, 40, 40));
    m_paletteTimelineTrackList.setBrush(NauPalette::Role::Background, NauColor(40, 40, 40), NauPalette::Hovered);
    m_paletteTimelineTrackList.setBrush(NauPalette::Role::AlternateBackground, NauColor(44, 44, 44));

    m_paletteTimelineTrackList.setBrush(NauPalette::Role::Background, NauColor(33, 45, 153, 179), NauPalette::Selected);
    m_paletteTimelineTrackList.setBrush(NauPalette::Role::Background, NauColor(33, 45, 153, 179), NauPalette::Selected | NauPalette::Hovered);
    m_paletteTimelineTrackList.setColor(NauPalette::Role::Border, NauColor(85, 102, 255), NauPalette::Hovered);

    m_paletteTimelineTrackList.setColor(NauPalette::Role::BackgroundHeader, NauColor(27, 27, 27));
    m_paletteTimelineTrackList.setColor(NauPalette::Role::Background, NauColor(40, 40, 40));
    m_paletteTimelineTrackList.setColor(NauPalette::Role::BackgroundFooter, NauColor(40, 40, 40));

    m_paletteTimelineTrackList.setColor(NauPalette::Role::ForegroundBrightText, NauColor(153, 153, 153), NauPalette::Normal);
    m_paletteTimelineTrackList.setColor(NauPalette::Role::ForegroundBrightText, Qt::white, NauPalette::Selected);
    m_paletteTimelineTrackList.setColor(NauPalette::Role::ForegroundBrightText, Qt::white, NauPalette::Hovered);

    m_paletteTimelineMode.setColor(NauPalette::Role::Background, NauColor(52, 52, 52), NauPalette::Selected);
    m_paletteTimelineMode.setColor(NauPalette::Role::Background, NauColor(34, 34, 34), NauPalette::Normal);

    m_paletteTimelineRecord.setColor(NauPalette::Role::Background, NauColor(209, 35, 77));

    // Timeline Content
    m_paletteTimelineKeyframe.setColor(NauPalette::Role::Background, NauColor{ 103, 117, 240 }, NauPalette::Selected);
    m_paletteTimelineKeyframe.setColor(NauPalette::Role::Background, NauColor{ 153, 153, 153 }, NauPalette::Normal);
    m_paletteTimelineKeyframe.setColor(NauPalette::Role::Foreground, NauColor{ 47, 64, 218 }, NauPalette::Selected);
    m_paletteTimelineKeyframe.setColor(NauPalette::Role::Foreground, NauColor{ 128, 128, 128 }, NauPalette::Normal);

    m_paletteTimelineScrollBar.setColor(NauPalette::Role::Background, NauColor{ 63, 63, 63 });
    m_paletteTimelineScrollBar.setColor(NauPalette::Role::Foreground, NauColor{ 40, 40, 40 });

    m_paletteTimelineContentView.setColor(NauPalette::Role::BackgroundFooter, NauColor{ 27, 27, 27 }); // background
    m_paletteTimelineContentView.setColor(NauPalette::Role::BackgroundHeader, NauColor{ 91, 91, 91 }); // timeline section line
    m_paletteTimelineContentView.setColor(NauPalette::Role::Background, NauColor{ 40, 40, 40 }); // sub section line
    m_paletteTimelineContentView.setColor(NauPalette::Role::AlternateBackground, NauColor{ 75, 75, 75 }); // section line
    m_paletteTimelineContentView.setColor(NauPalette::Role::Foreground, NauColor{ 33, 45, 153 }); // animation
    m_paletteTimelineContentView.setColor(NauPalette::Role::AlternateForeground, NauColor{ 247, 63, 107 }); // animation record
    m_paletteTimelineContentView.setColor(NauPalette::Role::ForegroundHeader, NauColor{ 68, 68, 68 }); // safe area
    m_paletteTimelineContentView.setColor(NauPalette::Role::ForegroundBrightHeader, NauColor{ 255, 255, 255, 38 });
    m_paletteTimelineContentView.setColor(NauPalette::Role::Text, NauColor{ 141, 141, 141 });

    m_paletteTimelineFramePointer.setColor(NauPalette::Role::Background, NauColor{ 105, 120, 255 }, NauPalette::Normal);
    m_paletteTimelineFramePointer.setColor(NauPalette::Role::Background, NauColor{ 49, 67, 229 }, NauPalette::Selected);
}

void NauDefaultTheme::applyGlobalStyle(NauApp& app)
{
    app.setStyleSheet(":/Style/stylesheets/main.qss");
}

NauFont NauDefaultTheme::fontTitleBarTitle() const
{
    return m_fontTitleBarTitle;
}

NauFont NauDefaultTheme::fontTitleBarMenuItem() const
{
    return m_fontTitleBarMenuItem;
}

NauFont NauDefaultTheme::fontWorldOutlineSemibold() const
{
    return m_fontWorldOutlineSemibold;
}

NauFont NauDefaultTheme::fontWorldOutlineRegular() const
{
    return m_fontWorldOutlineRegular;
}

NauFont NauDefaultTheme::fontWorldOutlineHeaderSemibold() const
{
    return m_fontWorldOutlineHeaderSemibold;
}

NauFont NauDefaultTheme::fontWorldOutlineHeaderRegular() const
{
    return m_fontWorldOutlineHeaderRegular;
}

NauFont NauDefaultTheme::fontLogger() const
{
    return m_fontLogger;
}

NauFont NauDefaultTheme::fontLoggerLevel() const
{
    return m_fontLoggerLevel;
}

NauFont NauDefaultTheme::fontLoggerSource() const
{
    return m_fontLoggerSource;
}

NauFont NauDefaultTheme::fontProjectBrowserPrimary() const
{
    return m_fontProjectBrowserPrimary;
}

NauFont NauDefaultTheme::fontProjectBrowserSecondary() const
{
    return m_fontProjectBrowserSecondary;
}

NauFont NauDefaultTheme::fontFilterItem() const
{
    return m_fontFilterItem;
}

NauFont NauDefaultTheme::fontFeedbackHeaderTitle() const
{
    return m_fontFeedbackHeaderTitle;
}

NauFont NauDefaultTheme::fontFeedbackHeaderThankTitle() const
{
    return m_fontFeedbackHeaderThankTitle;
}

NauFont NauDefaultTheme::fontFeedbackCommentInfo() const
{
    return m_fontFeedbackCommentInfo;
}

NauFont NauDefaultTheme::fontFeedbackErrorInfo() const
{
    return m_fontFeedbackErrorInfo;
}

NauFont NauDefaultTheme::fontFeedbackConnectionError() const
{
    return m_fontFeedbackConnectionError;
}

NauFont NauDefaultTheme::fontInspectorHeaderTitle() const
{
    return m_fontInspectorHeaderTitle;
}

NauFont NauDefaultTheme::fontInspectorHeaderSubtitle() const
{
    return m_fontInspectorHeaderSubtitle;
}

NauFont NauDefaultTheme::fontPrimaryButton() const
{
    return m_fontPrimaryButton;
}

NauFont NauDefaultTheme::fontInputHeaderTitle() const
{
    return m_fontInputHeaderTitle;
}

NauFont NauDefaultTheme::fontInputHeaderSubtitle() const
{
    return m_fontInputHeaderSubtitle;
}

NauFont NauDefaultTheme::fontInputLabel() const
{
    return m_fontInputLabel;
}

NauFont NauDefaultTheme::fontInputTab() const
{
    return m_fontInputTab;
}

NauFont NauDefaultTheme::fontDocking() const
{
    return m_fontDocking;
}

NauFont NauDefaultTheme::fontStaticTextBase() const
{
    return m_fontStaticTextBase;
}

NauFont NauDefaultTheme::fontTableWidgetBase() const
{
    return m_fontTableWidgetBase;
}

NauFont NauDefaultTheme::fontProjectManagerMenuButton() const
{
    return m_fontProjectManagerMenuButton;
}

NauFont NauDefaultTheme::fontProjectManagerLocation() const
{
    return m_fontProjectManagerLocation;
}

NauFont NauDefaultTheme::fontHistoryMain() const
{
    return m_fontHistoryMain;
}

NauFont NauDefaultTheme::fontObjectInspector() const
{
    return m_fontObjectInspector;
}

NauFont NauDefaultTheme::fontObjectInspectorSpoiler() const
{
    return m_fontObjectInspectorSpoiler;
}

NauFont NauDefaultTheme::fontDataContainers() const
{
    return m_fontDataContainers;
}

NauFont NauDefaultTheme::fontDataContainersLabel() const
{
    return m_fontDataContainersLabel;
}

NauFont NauDefaultTheme::fontSplashProjectName() const
{
    return m_fontSplashProjectName;
}

NauFont NauDefaultTheme::fontSplashMessage() const
{
    return m_fontSplashMessage;
}

NauFont NauDefaultTheme::fontSplashCopyright() const
{
    return m_fontSplashCopyright;
}

NauIcon NauDefaultTheme::iconMinimize() const
{
    return generateTitleBarIcon(QStringLiteral("Minimize"));
}

NauIcon NauDefaultTheme::iconWindowState() const
{
    NauIcon result = generateTitleBarIcon(QStringLiteral("Maximize"));
    const NauIcon restore = generateTitleBarIcon(QStringLiteral("Restore"));
    const QSize size = restore.availableSizes().front();

    result.addPixmap(restore.pixmap(size, NauIcon::Normal), NauIcon::Normal, NauIcon::On);
    result.addPixmap(restore.pixmap(size, NauIcon::Active), NauIcon::Active, NauIcon::On);
    result.addPixmap(restore.pixmap(size, NauIcon::Selected), NauIcon::Selected, NauIcon::On);
    result.addPixmap(restore.pixmap(size, NauIcon::Disabled), NauIcon::Disabled, NauIcon::On);

    return result;
}

NauIcon NauDefaultTheme::iconClose() const
{
    return generateTitleBarIcon(QStringLiteral("Close"));
}

NauIcon NauDefaultTheme::iconTitleBarLogo() const
{
    static const NauIcon icon{ QPixmap(":/UI/icons/titlebar-logo.png") };
    return icon;
}

NauIcon NauDefaultTheme::iconUndock() const
{
    static const auto pixmapUrl = QStringLiteral(":/ads/icons/ads/undock.svg");

    return generatePixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconDockAreaMenu() const
{
    static const auto pixmapUrl = QStringLiteral(":/ads/icons/ads/3-dots.svg");

    return generatePixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconVisibility() const
{
    static const QSize pixmapSize{12, 12};

    NauIcon result;
    result.addPixmap(Nau::paintPixmap(":/UI/icons/iInvisible.png", QColor(128, 128, 128))
            .scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation), NauIcon::Normal, NauIcon::On);
    result.addPixmap(Nau::paintPixmap(":/UI/icons/iVisible.png", QColor(128, 128, 128))
            .scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation), NauIcon::Normal, NauIcon::Off);

    return result;
}

NauIcon NauDefaultTheme::iconAvailability() const
{
    static const QSize pixmapSize{12, 12};

    NauIcon result;
    result.addPixmap(Nau::paintPixmap(":/UI/icons/iLocked.png", QColor(128, 128, 128))
            .scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation), NauIcon::Normal, NauIcon::On);
    result.addPixmap(Nau::paintPixmap(":/UI/icons/iUnlocked.png", QColor(128, 128, 128))
            .scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation), NauIcon::Normal, NauIcon::Off);

    return result;
}

NauIcon NauDefaultTheme::iconCut() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/iCut.png"));
}

NauIcon NauDefaultTheme::iconCopy() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/iCopy.png"));
}

NauIcon NauDefaultTheme::iconLoggerCopy() const
{
    const auto pixmapUrl = QStringLiteral(":/log/icons/log/copy.svg");
    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconLoggerSettings() const
{
    const auto pixmapUrl = QStringLiteral(":/log/icons/log/settings.svg");

    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconPaste() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/iPaste.png"));
}

NauIcon NauDefaultTheme::iconDuplicate() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/iDuplicate.png"));
}

NauIcon NauDefaultTheme::iconDelete() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/iDelete.png"));
}

NauIcon NauDefaultTheme::iconRename() const
{
    return NauIcon(QStringLiteral(":/UI/icons/iRename.png"));
}

NauIcon NauDefaultTheme::iconSave() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/save.svg"));
}

NauIcon NauDefaultTheme::iconViewAssetInShell() const
{
    return NauIcon(":/UI/icons/folder.png");
}

NauIcon NauDefaultTheme::iconMakeNewFolder() const
{
    return NauIcon(":/UI/icons/newdir.png");
}

NauIcon NauDefaultTheme::iconFocusCameraOnObject() const
{
    return NauIcon(":/UI/icons/focus.png");
}

NauIcon NauDefaultTheme::iconResourcePlaceholder() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vResourceIconPlaceholder.svg");
    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconInvisibleChild() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vInvisibleChild.svg");
    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconThreeDotsHorisont() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vThreeDotsHorisont.svg");
    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconPlay() const
{
    const QString resource = ":/UI/icons/toolbar/play.png";
    auto icon = generateViewportPixmapIcon(resource);
    icon.addPixmap(Nau::paintPixmap(resource, QColor(41, 212, 108)), NauIcon::Normal);
    return icon;
}

NauIcon NauDefaultTheme::iconStop() const
{
    const QString resource = ":/UI/icons/toolbar/stop.png";
    auto icon = generateViewportPixmapIcon(resource);
    icon.addPixmap(Nau::paintPixmap(resource, QColor(233, 39, 85)), NauIcon::Normal);
    return icon;
}

NauIcon NauDefaultTheme::iconPause() const
{
    return generateViewportPixmapIcon(":/UI/icons/toolbar/pause.png");
}

NauIcon NauDefaultTheme::iconUndo() const
{
    return generateViewportPixmapIcon(":/UI/icons/toolbar/undo-redo/undo.png");
}

NauIcon NauDefaultTheme::iconRedo() const
{
    return generateViewportPixmapIcon(":/UI/icons/toolbar/undo-redo/redo.png");
}

NauIcon NauDefaultTheme::iconHistory() const
{
    return generateViewportPixmapIcon(":/UI/icons/toolbar/undo-redo/history.png");
}

NauIcon NauDefaultTheme::iconFatDot() const
{
    return NauIcon(":/UI/icons/history/dot.png");
}

NauIcon NauDefaultTheme::iconDottedLine() const
{
    return NauIcon(":/UI/icons/history/line.png");
}

NauIcon NauDefaultTheme::iconDottedLineTail() const
{
    return NauIcon(":/UI/icons/history/line-tail.png");
}

NauIcon NauDefaultTheme::iconSettings() const
{
    return generateViewportPixmapIcon(":/UI/icons/toolbar/settings.png");
}

NauIcon NauDefaultTheme::iconPreferences() const
{
    return generateViewportPixmapIcon(":/UI/icons/iPreferences.png");
}

NauIcon NauDefaultTheme::iconBuildSettings() const
{
    // TODO: Replace with new designed icon
    return generateViewportPixmapIcon(":/UI/icons/toolbar/export.png");
}

NauIcon NauDefaultTheme::iconHamburger() const
{
    return generateViewportIcon(QStringLiteral("iHamburger"));
}

NauIcon NauDefaultTheme::iconUndoAction() const
{
    return NauIcon(":/UI/icons/iUndoAction.png");
}

NauIcon NauDefaultTheme::iconQuestionMark() const
{
    return NauIcon(":/UI/icons/iQuestionMark.png");
}

NauIcon NauDefaultTheme::iconAscendingSortIndicator() const
{
    return NauIcon(":/UI/icons/sortIndicatorAscending.svg");
}

NauIcon NauDefaultTheme::iconDescendingSortIndicator() const
{
    return NauIcon(":/UI/icons/sortIndicatorDescending.svg");
}

std::vector<NauIcon> NauDefaultTheme::iconLogs() const
{
    static const std::vector<NauIcon> icons = {
        NauIcon(":/log/icons/log/debug.svg"),
        NauIcon(":/log/icons/log/info.svg"),
        NauIcon(":/log/icons/log/warn.svg"),
        NauIcon(":/log/icons/log/error.svg"),
        NauIcon(":/log/icons/log/critical.svg"),
        NauIcon(":/log/icons/log/trace.svg"),
    };

    return icons;
}

NauIcon NauDefaultTheme::iconLoggerDetailsToggle() const
{
    NauIcon icon = generatePixmapIcon(":/log/icons/log/details.svg");
    const QSize size = icon.availableSizes().front();
    
    icon.addPixmap(Nau::paintPixmapCopy(icon
        .pixmap(size, NauIcon::Normal), QColor(0x4154FF)), NauIcon::Normal, NauIcon::On);
    icon.addPixmap(icon.pixmap(size, NauIcon::Active), NauIcon::Active, NauIcon::On);

    return icon;
}

NauIcon NauDefaultTheme::iconLoggerSourceEditor() const
{
    return generatePixmapIcon(":/log/icons/log/editor.svg");
}

NauIcon NauDefaultTheme::iconLoggerSourcePlayMode() const
{
    return generatePixmapIcon(":/log/icons/log/playmode.svg");
}

NauIcon NauDefaultTheme::iconLoggerSourceBuild() const
{
    return generatePixmapIcon(":/log/icons/log/build.svg");
}

NauIcon NauDefaultTheme::iconLoggerSourceExternalApplication() const
{
    return generatePixmapIcon(":/log/icons/log/external.svg");
}

NauIcon NauDefaultTheme::iconClean() const
{
    return generatePixmapIcon(":/UI/icons/clean.svg");
}

NauIcon NauDefaultTheme::iconDonePrimaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iCheck.svg");

    return generatePrimaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconClosePrimaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iClose.svg");

    return generatePrimaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconSendPrimaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iSend.svg");

    return generatePrimaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconButtonSettings() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vSettings.svg");

    return generateSecondaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconButtonFolder() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vFolder.svg");

    return generateSecondaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconvResourcePlaceholderSphere() const
{
    static const QSize pixmapSize{ 48, 48 };

    NauIcon result;
    result.addPixmap(QPixmap(QStringLiteral(":/UI/icons/iPlaceholderSphere.png")).scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    return result;
}

NauIcon NauDefaultTheme::iconvResourcePlaceholderSomething() const
{
    static const QSize pixmapSize{ 48, 48 };

    NauIcon result;
    result.addPixmap(QPixmap(QStringLiteral(":/UI/icons/iPlaceholderSomething.png")).scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    return result;
}

NauIcon NauDefaultTheme::iconvResourcePlaceholderVariant() const
{
    static const QSize pixmapSize{ 48, 48 };

    NauIcon result;
    result.addPixmap(QPixmap(QStringLiteral(":/UI/icons/iPlaceholderVariant.png")).scaled(pixmapSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    return result;
}

NauIcon NauDefaultTheme::iconAddPrimaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iPlus.svg");

    return generatePrimaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconAddSecondaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iPlus.svg");

    return generateSecondaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconAddTertiaryStyle() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/iPlus.svg");

    return generateTertiaryButtonPixmapIcon(pixmapUrl);
}

NauIcon NauDefaultTheme::iconLocalCoordinateSpace() const
{
    return generateViewportIcon(QStringLiteral("iLocalTransform"));
}

NauIcon NauDefaultTheme::iconWorldCoordinateSpace() const
{
    return generateViewportIcon(QStringLiteral("iWorldTransform"));
}

NauIcon NauDefaultTheme::iconCameraSettings() const
{
    return generateViewportIcon(QStringLiteral("iCameraSettings"));
}

NauIcon NauDefaultTheme::iconSelectTool() const
{
    return generateViewportIcon(QStringLiteral("iSelectAction"));
}

NauIcon NauDefaultTheme::iconMoveTool() const
{
    return generateViewportIcon(QStringLiteral("iMoveAction"));
}

NauIcon NauDefaultTheme::iconRotateTool() const
{
    return generateViewportIcon(QStringLiteral("iRotateAction"));
}

NauIcon NauDefaultTheme::iconScaleTool() const
{
    return generateViewportIcon(QStringLiteral("iScaleAction"));
}

NauIcon NauDefaultTheme::iconInspectorArrowDown() const
{ 
    return NauIcon(Nau::paintPixmap(":/UI/icons/iArrowDown.png", QColor(75, 75, 75)));
}

NauIcon NauDefaultTheme::iconInspectorArrowRight() const
{
    return NauIcon(Nau::paintPixmap(":/UI/icons/iArrowRight.png", QColor(75, 75, 75)));
}

NauIcon NauDefaultTheme::iconChecked() const
{
    return NauIcon(QStringLiteral(":/UI/icons/iCheck.png"));
}

NauIcon NauDefaultTheme::iconComboBoxDefault() const
{
    return NauIcon(QStringLiteral(":/UI/icons/iEllipse.png"));
}

NauIcon NauDefaultTheme::iconGamePad() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vGamepad.svg"));
}

NauIcon NauDefaultTheme::iconKeyboard() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vKeyboard.svg"));
}

NauIcon NauDefaultTheme::iconMouse() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vMouse.svg"));
}

NauIcon NauDefaultTheme::iconTouchPad() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vTouchpad.svg"));
}

NauIcon NauDefaultTheme::iconVR() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vVR.svg"));
}

NauIcon NauDefaultTheme::iconUnknown() const
{
    return generateInputDeviceIcon(QStringLiteral(":/inputEditor/icons/inputEditor/vUnknown.svg"));
}

NauPalette NauDefaultTheme::paletteTitleBar() const
{
    return m_paletteTitleBar;
}

NauPalette NauDefaultTheme::paletteCompilationStateTitleBar() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::Background, NauBrush(0x141414), NauPalette::Normal);
    pal.setBrush(NauPalette::Role::Background, NauBrush(0x141414), NauPalette::Hovered);
    pal.setBrush(NauPalette::Role::AlternateBackground, NauBrush(0x282828), NauPalette::Normal);
    pal.setBrush(NauPalette::Role::AlternateBackground, NauBrush(0x4B4B4B), NauPalette::Hovered);
    pal.setColor(NauPalette::Role::Foreground, NauColor(255, 255, 255), NauPalette::Normal);
    pal.setColor(NauPalette::Role::Foreground, NauColor(255, 255, 255), NauPalette::Hovered);

    pal.setBrush(NauPalette::Role::BackgroundHeader, NauColor(0x3F3F3F));

    return pal;
}

NauPalette NauDefaultTheme::paletteWorldOutline() const
{
    return m_paletteWorldOutline;
}

NauPalette NauDefaultTheme::paletteLogger() const
{
    return m_paletteLogger;
}

NauPalette NauDefaultTheme::paletteProjectBrowser() const
{
    return m_paletteProjectBrowser;
}

NauPalette NauDefaultTheme::paletteObjectInspector() const
{
    return m_paletteObjectInspector;
}

NauPalette NauDefaultTheme::paletteDataContainers() const
{
    return m_paletteDataContainers;
}

NauPalette NauDefaultTheme::palettePrimaryButton() const
{
    return m_palettePrimaryButton;
}

NauPalette NauDefaultTheme::paletteSecondaryButton() const
{
    return m_paletteSecondaryButton;
}

NauPalette NauDefaultTheme::paletteTertiaryButton() const
{
    return m_paletteTertiaryButton;
}

NauPalette NauDefaultTheme::paletteToogleButton() const
{
    return m_paletteToogleButton;
}

NauPalette NauDefaultTheme::paletteSearchWidget() const
{
    return m_paletteSearchWidget;
}

NauPalette NauDefaultTheme::paletteFilterWidget() const
{
    return m_paletteFilterWidget;
}

NauPalette NauDefaultTheme::paletteFilterItemWidget() const
{
    return m_paletteFilterItemWidget;
}

NauPalette NauDefaultTheme::paletteInputGeneric() const
{
    return m_paletteInputGeneric;
}

NauPalette NauDefaultTheme::paletteVFXGeneric() const
{
    return m_paletteVFXGeneric;
}

NauPalette NauDefaultTheme::paletteInputSpoilerLineEdit() const
{
    return m_paletteInputSpoilerLineEdit;
}

NauPalette NauDefaultTheme::paletteInputSignalView() const
{
    return m_paletteInputSignalView;
}

NauPalette NauDefaultTheme::paletteInputBindTab() const
{
    return m_paletteInputBindTab;
}

NauPalette NauDefaultTheme::paletteSpacerWidget() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::Background, NauBrush(0x141414));
    return pal;
}

NauPalette NauDefaultTheme::paletteFeedbackTextEditWidget() const
{
    return m_paletteFeedbackTextEditWidget;
}

NauPalette NauDefaultTheme::paletteResourceWidget() const
{
    return m_paletteResourceWidget;
}

NauPalette NauDefaultTheme::paletteResourcePopupWidget() const
{
    return m_paletteResourcePopupWidget;
}

NauPalette NauDefaultTheme::paletteNumericSlider() const
{
    return m_paletteNumericSlider;
}

NauPalette NauDefaultTheme::paletteTimelineTrackList() const
{
    return m_paletteTimelineTrackList;
}

NauPalette NauDefaultTheme::paletteTimelineMode() const
{
    return m_paletteTimelineMode;
}

NauPalette NauDefaultTheme::paletteTimelineRecord() const
{
    return m_paletteTimelineRecord;
}

NauPalette NauDefaultTheme::paletteTimelineKeyframe() const
{
    return m_paletteTimelineKeyframe;
}

NauPalette NauDefaultTheme::paletteTimelineScrollBar() const
{
    return m_paletteTimelineScrollBar;
}

NauPalette NauDefaultTheme::paletteTimelineContentView() const
{
    return m_paletteTimelineContentView;
}

NauPalette NauDefaultTheme::paletteTimelineFramePointer() const
{
    return m_paletteTimelineFramePointer;
}

NauPalette NauDefaultTheme::paletteWidgetAppearanceSlider() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::Background, NauBrush(0x5B5B5B));
    pal.setBrush(NauPalette::Role::AlternateBackground, NauBrush(0x999999));
    pal.setBrush(NauPalette::Role::AlternateBackground, NauBrush(0xE8E8E8), NauPalette::State::Hovered);
    return pal;
}

NauPalette NauDefaultTheme::palettePhysicsChannelSettings() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::Background, NauBrush(NauColor(40, 40, 40)));
    pal.setBrush(NauPalette::Role::BackgroundHeader, NauBrush(NauColor(34, 34, 34)));
    pal.setBrush(NauPalette::Role::BackgroundFooter, NauBrush(NauColor(40, 40, 40)));

    return pal;
}

NauPalette NauDefaultTheme::paletteDocking() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::BackgroundHeader, NauBrush(NauColor(40, 40, 40)), NauPalette::Normal, NauPalette::Category::Active);
    pal.setBrush(NauPalette::Role::BackgroundHeader, NauBrush(NauColor(27, 27, 27)), NauPalette::Normal, NauPalette::Category::Inactive);

    pal.setColor(NauPalette::Role::BackgroundHeader, NauColor(247, 63, 107, 0), NauPalette::Flashing);
    pal.setColor(NauPalette::Role::AlternateBackgroundHeader, NauColor(247, 63, 107, 64), NauPalette::Flashing);

    pal.setColor(NauPalette::Role::ForegroundHeader, NauColor(255, 255, 255), NauPalette::Normal, NauPalette::Category::Active);
    pal.setColor(NauPalette::Role::ForegroundHeader, NauColor(100, 100, 100), NauPalette::Normal, NauPalette::Category::Inactive);

    return pal;
}

NauPalette NauDefaultTheme::paletteSplash() const
{
    NauPalette pal;
    pal.setBrush(NauPalette::Role::Background, NauBrush(NauColor(20, 20, 20)));

    return pal;
}

NauWidgetStyle NauDefaultTheme::stylePrimaryButton() const
{
    return NauPrimaryButtonStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleSecondaryButton() const
{
    return NauSecondaryButtonStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleTertiaryButton() const
{
    return NauTertiaryButtonStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleMiscButton() const
{
    return NauMiscButtonStyle();
}

NauWidgetStyle NauDefaultTheme::styleSearchWidget() const
{
    return NauSearchWidgetStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleToogleButton() const
{
    return NauToogleButtonStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleFeedbackTextEditWidget() const
{
    return NauFeedbackTextEditStyle(this);
}

NauWidgetStyle NauDefaultTheme::styleResourceWidget() const
{
    return NauResourceWidgetStyle(this);
}

NauFont NauDefaultTheme::registerFont(const QString& fileName)
{
    const int fontId = QFontDatabase::addApplicationFont(fileName);
    if (fontId > -1) {
        const QStringList families = QFontDatabase::applicationFontFamilies(fontId);
        if (!families.isEmpty()) {
            return NauFont(families.front());
        }
        else
        {
            NED_WARNING("DefaultTheme: Custom font '{}' added successfully, but no data about family generated",
                fileName.toLocal8Bit().constData());
        }
    }
    else
    {
        NED_WARNING("DefaultTheme: Failed to add font from '{}'", fileName.toLocal8Bit().constData());
    }

    return {};
}

NauIcon NauDefaultTheme::generateViewportIcon(const QString& buttonName)
{
    const auto pixmapUrlTemplate = QStringLiteral(":/UI/icons/viewport/%1.png");
    return generateViewportPixmapIcon(pixmapUrlTemplate.arg(buttonName));
}

NauIcon NauDefaultTheme::generateTitleBarIcon(const QString& buttonName)
{
    const auto pixmapUrlTemplate = QStringLiteral(":/titleBar/icons/titleBar/%1.png");
    return generatePixmapIcon(pixmapUrlTemplate.arg(buttonName));
}

NauIcon NauDefaultTheme::generatePixmapIcon(const QString& pixmapTemplate)
{
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(153, 153, 153)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Selected);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(063, 063, 063)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generateViewportPixmapIcon(const QString& pixmapTemplate)
{
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(128, 128, 128)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active, NauIcon::Off);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active, NauIcon::On);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Selected);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(063, 063, 063)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generatePrimaryButtonPixmapIcon(const QString& pixmapTemplate)
{
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(91, 91, 91)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generateSecondaryButtonPixmapIcon(const QString& pixmapTemplate)
{
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(91, 91, 91)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generateTertiaryButtonPixmapIcon(const QString& pixmapTemplate)
{
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(91, 91, 91)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(255, 255, 255)), NauIcon::Active);
    result.addPixmap(Nau::paintPixmap(pixmapTemplate, QColor(91, 91, 91)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generateInputIcon(const QString& iconPath)
{
    QPixmap icon{ iconPath };
    NauIcon result;

    Nau::paintPixmap(icon, QColor{ 0x808080 });
    result.addPixmap(icon, NauIcon::Normal);

    Nau::paintPixmap(icon, Qt::white);
    result.addPixmap(icon, NauIcon::Active);
    result.addPixmap(icon, NauIcon::Selected);

    Nau::paintPixmap(icon, QColor{ 0x3F3F3F });
    result.addPixmap(icon, NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::generateInputDeviceIcon(const QString& iconPath)
{
    return Nau::paintPixmap(iconPath, NauColor{ 0x808080 });
}

NauIcon NauDefaultTheme::iconSearch() const
{
    const auto pixmapUrlTemplate = QStringLiteral(":/UI/icons/search.svg");
    NauIcon result;

    result.addPixmap(Nau::paintPixmap(pixmapUrlTemplate, QColor(91, 91, 91)), NauIcon::Normal);
    result.addPixmap(Nau::paintPixmap(pixmapUrlTemplate, QColor(255, 255, 255)), NauIcon::Active);
    result.addPixmap(Nau::paintPixmap(pixmapUrlTemplate, QColor(91, 91, 91)), NauIcon::Disabled);

    return result;
}

NauIcon NauDefaultTheme::iconSearchClear() const
{
    return { Nau::paintPixmap(":/UI/icons/cross.svg", { 153, 153, 153 } ) };
}

NauIcon NauDefaultTheme::iconSpoilerIndicator() const
{
    const QPixmap down = Nau::paintPixmap(":/UI/icons/iArrowDown.png", QColor(75, 75, 75));
    const QPixmap right = Nau::paintPixmap(":/UI/icons/iArrowRight.png", QColor(75, 75, 75));

    NauIcon icon;

    icon.addPixmap(down, QIcon::Normal, QIcon::On);
    icon.addPixmap(down, QIcon::Disabled, QIcon::On);
    icon.addPixmap(down, QIcon::Disabled, QIcon::On);
    icon.addPixmap(down, QIcon::Selected, QIcon::On);

    icon.addPixmap(right, QIcon::Normal, QIcon::Off);
    icon.addPixmap(right, QIcon::Disabled, QIcon::Off);
    icon.addPixmap(right, QIcon::Disabled, QIcon::Off);
    icon.addPixmap(right, QIcon::Selected, QIcon::Off);

    return icon;
}

NauIcon NauDefaultTheme::iconSpoilerAdd() const
{
    return generateInputIcon(":/inputEditor/icons/inputEditor/vAdd.svg");
}

NauIcon NauDefaultTheme::iconSpoilerClose() const
{
    return generateInputIcon(":/inputEditor/icons/inputEditor/vClose.svg");
}

NauIcon NauDefaultTheme::iconActionEditor() const
{
    return QPixmap(":/inputEditor/icons/inputEditor/vAction.svg").scaledToWidth(48, Qt::SmoothTransformation);
}

NauIcon NauDefaultTheme::iconVFXEditor() const
{
    return QPixmap(":/vfxEditor/icons/vfxEditor/vVFX.svg").scaledToWidth(48, Qt::SmoothTransformation);
}

NauIcon NauDefaultTheme::iconPhysicsMaterialEditor() const
{
    return QPixmap(":/UI/icons/vResourceIconPlaceholder.svg").scaledToWidth(48, Qt::SmoothTransformation);
}

NauIcon NauDefaultTheme::iconThanksForFeedbackCheck() const
{
    return QPixmap(":/UI/icons/iCheck.svg");
}
NauIcon NauDefaultTheme::iconArrowDown() const
{ 
    return { Nau::paintPixmap(":/UI/icons/arrow-down.svg", { 153, 153, 153 } ) };
}


NauIcon NauDefaultTheme::iconArrowRight() const
{
    return { Nau::paintPixmap(":/UI/icons/arrow-right.svg", { 153, 153, 153 } ) };
}

NauIcon NauDefaultTheme::iconBrowserTreeFolder() const
{
    const auto pixmapUrl = QStringLiteral(":/UI/icons/vFolder.svg");

    return {Nau::paintPixmap(pixmapUrl, QColor(91, 91, 91))};
}

NauIcon NauDefaultTheme::iconBreadCrumbsHome() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/browser/home.svg"));
}

NauIcon NauDefaultTheme::iconBreadCrumbsDelimiter() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/browser/nav-delim.svg"));
}

NauIcon NauDefaultTheme::iconTableViewIcon() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/table-view.svg"));
}

NauIcon NauDefaultTheme::iconTileViewIcon() const
{
    return generatePixmapIcon(QStringLiteral(":/UI/icons/tile-view.svg"));
}

std::vector<NauIcon> NauDefaultTheme::iconsTimelinePlayback() const
{
    static const std::vector<NauIcon> icons = {
        NauIcon(":/animationEditor/icons/animationEditor/vStart.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vPrevious.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vPlay.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vPause.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vStop.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vNext.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vEnd.svg"),
    };
    return icons;
}

std::vector<NauIcon> NauDefaultTheme::iconsTimelineTrackListHeader() const
{
    static const std::vector<NauIcon> icons = {
        NauIcon(":/animationEditor/icons/animationEditor/vFrame.svg"),
        NauIcon(":/animationEditor/icons/animationEditor/vEvent.svg"),
    };
    return icons;
}

std::unordered_map<NauSourceState, NauIcon> NauDefaultTheme::iconsScriptsState() const
{
    static const std::unordered_map<NauSourceState, NauIcon> icons = {
        { NauSourceState::NoBuildTools, NauIcon(QStringLiteral(":/UI/icons/compilation/warning-tri-yellow.svg")) },
        { NauSourceState::RecompilationRequired, NauIcon(QStringLiteral(":/UI/icons/compilation/warning-circle-yellow.svg")) },
        { NauSourceState::CompilationError, NauIcon(QStringLiteral(":/UI/icons/compilation/warning-tri-red.svg")) },
        { NauSourceState::FatalError, NauIcon(QStringLiteral(":/UI/icons/compilation/fail.svg")) },
        { NauSourceState::Success, NauIcon(QStringLiteral(":/UI/icons/compilation/success.svg")) },
    };

    return icons;
}

std::vector<NauIcon> NauDefaultTheme::iconsTimelineParameters() const
{
    NauIcon dopeIcon;
    dopeIcon.addPixmap(Nau::paintPixmap(":/animationEditor/icons/animationEditor/vPlaceholder.svg", NauColor(255, 255, 255)), QIcon::Normal, QIcon::On);
    dopeIcon.addPixmap(Nau::paintPixmap(":/animationEditor/icons/animationEditor/vPlaceholder.svg", NauColor(100, 100, 100)), QIcon::Normal, QIcon::Off);

    NauIcon curveIcon = dopeIcon;

    NauIcon stepButtonIcon;
    stepButtonIcon.addPixmap(Nau::paintPixmap(":/animationEditor/icons/animationEditor/vPlaceholder.svg", NauColor(65, 84, 255)), QIcon::Normal, QIcon::On);
    stepButtonIcon.addPixmap(Nau::paintPixmap(":/animationEditor/icons/animationEditor/vPlaceholder.svg", NauColor(153, 153, 153)), QIcon::Normal, QIcon::Off);

    static const std::vector<NauIcon> icons = {
        dopeIcon,
        curveIcon,
        stepButtonIcon,
    };
    return icons;
}


#pragma region BUTTON STYLE LEVEL
// ** NauPrimaryButton

struct NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonActiveStyle : public NauWidgetStyle::NauStyle
{
    NauPrimaryButtonActiveStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->palettePrimaryButton();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Active);
        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Active);

        NauWidgetStyle::NauStyle::iconState = QIcon::Normal;
        NauWidgetStyle::NauStyle::radiusSize = QSize(16, 16);

        auto outline = palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Active);
        auto outlineWidth = 1.0;
        auto penStyle = Qt::SolidLine;
        auto penCapStyle = Qt::SquareCap;
        auto penJoinStyle = Qt::BevelJoin;

        NauWidgetStyle::NauStyle::outlinePen = NauPen(outline, outlineWidth, penStyle, penCapStyle, penJoinStyle);
    }
};

struct NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonHoverStyle : public NauPrimaryButtonActiveStyle
{
    NauPrimaryButtonHoverStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonActiveStyle(theme)
    {
        const auto& palette = theme->palettePrimaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Hovered);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Hovered));

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
    }
};

struct NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonPressedStyle : public NauPrimaryButtonActiveStyle
{
    NauPrimaryButtonPressedStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonActiveStyle(theme)
    {
        const auto& palette = theme->palettePrimaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Pressed));

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
    }
};

struct NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonTabFocusedStyle : public NauPrimaryButtonActiveStyle
{
    NauPrimaryButtonTabFocusedStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonActiveStyle(theme)
    {
        const auto& palette = theme->palettePrimaryButton();

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;

        NauWidgetStyle::NauStyle::outlinePen.setWidth(1);
        NauWidgetStyle::NauStyle::outlinePen.setStyle(Qt::DashLine);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Selected));
    }
};

struct NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonDisableStyle : public NauPrimaryButtonActiveStyle
{
    NauPrimaryButtonDisableStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonActiveStyle(theme)
    {
        const auto& palette = theme->palettePrimaryButton();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Disabled);
        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Disabled);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Disabled));

        NauWidgetStyle::NauStyle::iconState = QIcon::Disabled;
    }
};


// ** NauSecondaryButton

struct NauDefaultTheme::NauSecondaryButtonStyle::NauSecondButtonActiveStyle : public NauPrimaryButtonActiveStyle
{
    NauSecondButtonActiveStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteSecondaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Active);
    }
};

struct NauDefaultTheme::NauSecondaryButtonStyle::NauSecondButtonHoverStyle : public NauPrimaryButtonHoverStyle
{
    NauSecondButtonHoverStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonHoverStyle(theme)
    {
        const auto& palette = theme->paletteSecondaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Hovered);
    }
};

struct NauDefaultTheme::NauSecondaryButtonStyle::NauSecondButtonPressedStyle : public NauPrimaryButtonPressedStyle
{
    NauSecondButtonPressedStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonPressedStyle(theme)
    {
        const auto& palette = theme->paletteSecondaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
    }
};

struct NauDefaultTheme::NauSecondaryButtonStyle::NauSecondButtonTabFocusedStyle : public NauPrimaryButtonTabFocusedStyle
{
    NauSecondButtonTabFocusedStyle(const NauAbstractTheme* theme)
        : NauPrimaryButtonTabFocusedStyle(theme)
    {
        const auto& palette = theme->paletteSecondaryButton();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Selected);
    }
};


// ** NauTertiaryButtonStyle

struct NauDefaultTheme::NauTertiaryButtonStyle::NauTertiaryButtonActiveStyle : public NauSecondButtonActiveStyle
{
    NauTertiaryButtonActiveStyle(const NauAbstractTheme* theme)
        : NauSecondButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteTertiaryButton();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Active);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Active));
    }
};

NauDefaultTheme::NauPrimaryButtonStyle::NauPrimaryButtonStyle(const NauAbstractTheme* theme)
{
    styleByState[NauWidgetState::Active] = NauPrimaryButtonActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauPrimaryButtonHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauPrimaryButtonPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauPrimaryButtonTabFocusedStyle(theme);

    // TODO: It is possible to generate a self style so that the developer doesn't have to fill in the full range of cases.
    styleByState[NauWidgetState::Disabled] = NauPrimaryButtonDisableStyle(theme);
}

NauDefaultTheme::NauSecondaryButtonStyle::NauSecondaryButtonStyle(const NauAbstractTheme* theme)
    : NauPrimaryButtonStyle(theme)
{
    styleByState[NauWidgetState::Active] = NauSecondButtonActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauSecondButtonHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauSecondButtonPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauSecondButtonTabFocusedStyle(theme);
}

NauDefaultTheme::NauTertiaryButtonStyle::NauTertiaryButtonStyle(const NauAbstractTheme* theme)
    : NauSecondaryButtonStyle(theme)
{
    styleByState[NauWidgetState::Active] = NauTertiaryButtonActiveStyle(theme);
}

NauDefaultTheme::NauMiscButtonStyle::NauMiscButtonStyle()
{
    styleByState[NauWidgetState::Active] = NauWidgetStyle::NauStyle{
        NauColor(0x999999),
        NauIcon::Mode::Normal,
        NauBrush(Qt::transparent),
        {},
        NauPen(NauBrush(Qt::transparent), 0.0),
    };

    styleByState[NauWidgetState::Hover] = NauWidgetStyle::NauStyle{
        NauColor(0xFFFFFF),
        NauIcon::Mode::Active,
        NauBrush(Qt::transparent),
        {},
        NauPen(NauBrush(Qt::transparent), 0.0),
    };

    styleByState[NauWidgetState::Pressed] = styleByState[NauWidgetState::Hover];
    styleByState[NauWidgetState::TabFocused] = styleByState[NauWidgetState::Hover];

    styleByState[NauWidgetState::Disabled] = NauWidgetStyle::NauStyle{
        {},
        NauIcon::Mode::Disabled,
        NauBrush(Qt::transparent),
        {},
        NauPen(NauBrush(Qt::transparent), 0.0),
    };
}

#pragma endregion

struct NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetActiveStyle : public NauWidgetStyle::NauStyle
{
    explicit NauSearchWidgetActiveStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteSearchWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Normal), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal);
        background = palette.color(NauPalette::Role::Background, NauPalette::Normal);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetHoverStyle : public NauWidgetStyle::NauStyle
{
    explicit NauSearchWidgetHoverStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteSearchWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Hovered), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Hovered);
        background = palette.color(NauPalette::Role::Background, NauPalette::Hovered);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetPressedStyle : public NauWidgetStyle::NauStyle
{
    explicit NauSearchWidgetPressedStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteSearchWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Pressed), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Pressed);
        background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetTabFocusedStyle : public NauWidgetStyle::NauStyle
{
    explicit NauSearchWidgetTabFocusedStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteSearchWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Selected), 1.0, Qt::DashLine);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Selected);
        background = palette.color(NauPalette::Role::Background, NauPalette::Selected);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetDisableStyle : public NauWidgetStyle::NauStyle
{
    explicit NauSearchWidgetDisableStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteSearchWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Normal), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal);
        background = palette.color(NauPalette::Role::Background, NauPalette::Normal);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

NauDefaultTheme::NauSearchWidgetStyle::NauSearchWidgetStyle(const NauAbstractTheme* theme)
{
    styleByState[NauWidgetState::Active] = NauSearchWidgetActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauSearchWidgetHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauSearchWidgetPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauSearchWidgetTabFocusedStyle(theme);
    styleByState[NauWidgetState::Disabled] = NauSearchWidgetDisableStyle(theme);
}

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditActiveStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditActiveStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();
        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Normal), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal);
        background = palette.color(NauPalette::Role::Background, NauPalette::Normal);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditHoverStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditHoverStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Hovered), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Hovered);
        background = palette.color(NauPalette::Role::Background, NauPalette::Hovered);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditPressedStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditPressedStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Pressed), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Pressed);
        background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditFocusedStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditFocusedStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Selected), 1.0, Qt::DashLine);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Selected);
        background = palette.color(NauPalette::Role::Background, NauPalette::Selected);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditErrorStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditErrorStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Error), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Error);
        background = palette.color(NauPalette::Role::Background, NauPalette::Error);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

struct NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditDisabledStyle : public NauWidgetStyle::NauStyle
{
    explicit NauFeedbackTextEditDisabledStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteFeedbackTextEditWidget();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Normal), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal);
        background = palette.color(NauPalette::Role::Background, NauPalette::Normal);
        iconState = QIcon::Normal;
        radiusSize = QSize(16, 16);
    }
};

NauDefaultTheme::NauFeedbackTextEditStyle::NauFeedbackTextEditStyle(const NauAbstractTheme* theme)
{
    styleByState[NauWidgetState::Active] = NauFeedbackTextEditActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauFeedbackTextEditHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauFeedbackTextEditPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauFeedbackTextEditFocusedStyle(theme);
    styleByState[NauWidgetState::Error] = NauFeedbackTextEditErrorStyle(theme);
    styleByState[NauWidgetState::Disabled] = NauFeedbackTextEditDisabledStyle(theme);
}

// ** NauToogleButtonStyle

struct NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonActiveStyle : public NauWidgetStyle::NauStyle
{
    NauToogleButtonActiveStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteToogleButton();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Active);
        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Active);

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
        NauWidgetStyle::NauStyle::radiusSize = QSize(2, 2);

        auto outline = palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Active);
        auto outlineWidth = 1.0;
        auto penStyle = Qt::SolidLine;
        auto penCapStyle = Qt::SquareCap;
        auto penJoinStyle = Qt::BevelJoin;

        NauWidgetStyle::NauStyle::outlinePen = NauPen(outline, outlineWidth, penStyle, penCapStyle, penJoinStyle);
    }
};

struct NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonHoverStyle : public NauToogleButtonActiveStyle
{
    NauToogleButtonHoverStyle(const NauAbstractTheme* theme)
        : NauToogleButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteToogleButton();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Hovered), 1.0);
    }
};

struct NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonPressedStyle : public NauToogleButtonActiveStyle
{
    explicit NauToogleButtonPressedStyle(const NauAbstractTheme* theme)
        : NauToogleButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteToogleButton();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Pressed), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Pressed);
        background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
    }
};

struct NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonTabFocusedStyle : public NauToogleButtonActiveStyle
{
    explicit NauToogleButtonTabFocusedStyle(const NauAbstractTheme* theme)
        : NauToogleButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteToogleButton();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Selected), 1.0, Qt::DashLine);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Selected);
        background = palette.color(NauPalette::Role::Background, NauPalette::Selected);
    }
};

struct NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonDisableStyle : public NauToogleButtonActiveStyle
{
    explicit NauToogleButtonDisableStyle(const NauAbstractTheme* theme)
        : NauToogleButtonActiveStyle(theme)
    {
        const auto& palette = theme->paletteToogleButton();

        outlinePen = NauPen(palette.color(NauPalette::Role::Border, NauPalette::Normal), 1.0);
        textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal);
        background = palette.color(NauPalette::Role::Background, NauPalette::Normal);
    }
};

NauDefaultTheme::NauToogleButtonStyle::NauToogleButtonStyle(const NauAbstractTheme* theme)
{
    styleByState[NauWidgetState::Active] = NauToogleButtonActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauToogleButtonHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauToogleButtonPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauToogleButtonTabFocusedStyle(theme);
    styleByState[NauWidgetState::Disabled] = NauToogleButtonDisableStyle(theme);
}

// ** NauResourceWidgetStyle

struct NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetActiveStyle : public NauWidgetStyle::NauStyle
{
    NauResourceWidgetActiveStyle(const NauAbstractTheme* theme)
    {
        const auto& palette = theme->paletteResourceWidget();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Active);
        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Active);

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
        NauWidgetStyle::NauStyle::radiusSize = QSize(2, 2);

        auto outline = palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Active);
        auto outlineWidth = 1.0;
        auto penStyle = Qt::SolidLine;
        auto penCapStyle = Qt::SquareCap;
        auto penJoinStyle = Qt::BevelJoin;

        NauWidgetStyle::NauStyle::outlinePen = NauPen(outline, outlineWidth, penStyle, penCapStyle, penJoinStyle);
    }
};

struct NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetHoverStyle : public NauResourceWidgetActiveStyle
{
    NauResourceWidgetHoverStyle(const NauAbstractTheme* theme)
        : NauResourceWidgetActiveStyle(theme)
    {
        const auto& palette = theme->paletteResourceWidget();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Hovered);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Hovered));

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
    }
};

struct NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetPressedStyle : public NauResourceWidgetActiveStyle
{
    NauResourceWidgetPressedStyle(const NauAbstractTheme* theme)
        : NauResourceWidgetActiveStyle(theme)
    {
        const auto& palette = theme->paletteResourceWidget();

        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Pressed);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Pressed));

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;
    }
};

struct NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetTabFocusedStyle : public NauResourceWidgetActiveStyle
{
    NauResourceWidgetTabFocusedStyle(const NauAbstractTheme* theme)
        : NauResourceWidgetActiveStyle(theme)
    {
        const auto& palette = theme->paletteResourceWidget();

        NauWidgetStyle::NauStyle::iconState = QIcon::Active;

        NauWidgetStyle::NauStyle::outlinePen.setWidth(1);
        NauWidgetStyle::NauStyle::outlinePen.setStyle(Qt::DashLine);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Selected));
    }
};

struct NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetDisableStyle : public NauResourceWidgetActiveStyle
{
    NauResourceWidgetDisableStyle(const NauAbstractTheme* theme)
        : NauResourceWidgetActiveStyle(theme)
    {
        const auto& palette = theme->paletteResourceWidget();

        NauWidgetStyle::NauStyle::textColor = palette.color(NauPalette::Role::Text, NauPalette::Normal, NauPalette::Category::Disabled);
        NauWidgetStyle::NauStyle::background = palette.color(NauPalette::Role::Background, NauPalette::Normal, NauPalette::Category::Disabled);
        NauWidgetStyle::NauStyle::outlinePen.setBrush(palette.color(NauPalette::Role::Border, NauPalette::Normal, NauPalette::Category::Disabled));

        NauWidgetStyle::NauStyle::iconState = QIcon::Disabled;
    }
};

NauDefaultTheme::NauResourceWidgetStyle::NauResourceWidgetStyle(const NauAbstractTheme* theme)
{
    styleByState[NauWidgetState::Active] = NauResourceWidgetActiveStyle(theme);
    styleByState[NauWidgetState::Hover] = NauResourceWidgetHoverStyle(theme);
    styleByState[NauWidgetState::Pressed] = NauResourceWidgetPressedStyle(theme);
    styleByState[NauWidgetState::TabFocused] = NauResourceWidgetTabFocusedStyle(theme);

    // TODO: It is possible to generate a self style so that the developer doesn't have to fill in the full range of cases.
    styleByState[NauWidgetState::Disabled] = NauResourceWidgetDisableStyle(theme);
}
