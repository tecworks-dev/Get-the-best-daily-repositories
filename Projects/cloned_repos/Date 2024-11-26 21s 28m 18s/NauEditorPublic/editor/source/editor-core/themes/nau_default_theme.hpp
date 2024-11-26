// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Internal default theme for editor.

#pragma once

#include "nau_abstract_theme.hpp"


// ** NauDefaultTheme

class NauDefaultTheme : public NauAbstractTheme
{
public:
    NauDefaultTheme();

    void applyGlobalStyle(NauApp& app) override;

    NauFont fontTitleBarTitle() const override;
    NauFont fontTitleBarMenuItem() const override;

    NauFont fontWorldOutlineSemibold() const override;
    NauFont fontWorldOutlineRegular() const override;

    NauFont fontWorldOutlineHeaderSemibold() const override;
    NauFont fontWorldOutlineHeaderRegular() const override;

    NauFont fontLogger() const override;
    NauFont fontLoggerLevel() const override;
    NauFont fontLoggerSource() const override;

    NauFont fontProjectBrowserPrimary() const override;
    NauFont fontProjectBrowserSecondary() const override;

    NauFont fontFilterItem() const override;

    NauFont fontFeedbackHeaderTitle() const override;
    NauFont fontFeedbackHeaderThankTitle() const override;

    NauFont fontFeedbackCommentInfo() const override;
    NauFont fontFeedbackErrorInfo() const override;
    NauFont fontFeedbackConnectionError() const override;

    NauFont fontInspectorHeaderTitle() const override;
    NauFont fontInspectorHeaderSubtitle() const override;

    NauFont fontPrimaryButton() const override;

    NauFont fontInputHeaderTitle() const override;
    NauFont fontInputHeaderSubtitle() const override;
    NauFont fontInputLabel() const override;
    NauFont fontInputTab() const override;

    NauFont fontDocking() const override;

    NauFont fontStaticTextBase() const override;
    NauFont fontTableWidgetBase() const override;

    NauFont fontProjectManagerMenuButton() const override;
    NauFont fontProjectManagerLocation() const override;

    // Undo history
    NauFont fontHistoryMain() const override;

    NauIcon iconMinimize() const override;
    NauIcon iconWindowState() const override;
    NauIcon iconClose() const override;
    NauFont fontObjectInspector() const override;
    NauFont fontObjectInspectorSpoiler() const override;

    NauFont fontDataContainers() const override;
    NauFont fontDataContainersLabel() const override;

    NauFont fontSplashProjectName() const override;
    NauFont fontSplashMessage() const override;
    NauFont fontSplashCopyright() const override;

    NauIcon iconTitleBarLogo() const override;

    NauIcon iconUndock() const override;
    NauIcon iconDockAreaMenu() const override;

    NauIcon iconLoggerCopy() const override;
    NauIcon iconLoggerSettings() const override;

    NauIcon iconVisibility() const override;
    NauIcon iconAvailability() const override;
    NauIcon iconCut() const override;
    NauIcon iconCopy() const override;
    NauIcon iconPaste() const override;
    NauIcon iconDuplicate() const override;
    NauIcon iconDelete() const override;
    NauIcon iconRename() const override;
    NauIcon iconSave() const override;

    NauIcon iconViewAssetInShell() const override;
    NauIcon iconMakeNewFolder() const override;
    NauIcon iconFocusCameraOnObject() const override;

    NauIcon iconResourcePlaceholder() const override;
    NauIcon iconInvisibleChild() const override;
    NauIcon iconThreeDotsHorisont() const override;

    NauIcon iconPlay() const override;
    NauIcon iconStop() const override;
    NauIcon iconPause() const override;

    NauIcon iconUndo() const override;
    NauIcon iconRedo() const override;
    NauIcon iconHistory() const override;
    NauIcon iconFatDot() const override;
    NauIcon iconDottedLine() const override;
    NauIcon iconDottedLineTail() const override;

    NauIcon iconSettings() const override;
    NauIcon iconPreferences() const override;

    NauIcon iconBuildSettings() const override;
    NauIcon iconHamburger() const override;
    NauIcon iconCameraSettings() const override;

    NauIcon iconLocalCoordinateSpace() const override;
    NauIcon iconWorldCoordinateSpace() const override;

    NauIcon iconSelectTool() const override;
    NauIcon iconMoveTool() const override;
    NauIcon iconRotateTool() const override;
    NauIcon iconScaleTool() const override;
    NauIcon iconInspectorArrowDown() const override;
    NauIcon iconInspectorArrowRight() const override;
    NauIcon iconChecked() const override;
    NauIcon iconComboBoxDefault() const override;

    NauIcon iconGamePad() const override;
    NauIcon iconKeyboard() const override;
    NauIcon iconMouse() const override;
    NauIcon iconTouchPad() const override;
    NauIcon iconVR() const override;
    NauIcon iconUnknown() const override;

    NauIcon iconUndoAction() const override;
    NauIcon iconQuestionMark() const override;

    NauIcon iconDonePrimaryStyle() const override;
    NauIcon iconClosePrimaryStyle() const override;
    NauIcon iconSendPrimaryStyle() const override;

    NauIcon iconAddPrimaryStyle() const override;
    NauIcon iconAddSecondaryStyle() const override;
    NauIcon iconAddTertiaryStyle() const override;

    NauIcon iconAscendingSortIndicator() const override;
    NauIcon iconDescendingSortIndicator() const override;
    std::vector<NauIcon> iconLogs() const override;

    NauIcon iconButtonSettings() const override;
    NauIcon iconButtonFolder() const override;
    NauIcon iconvResourcePlaceholderSphere() const override;
    NauIcon iconvResourcePlaceholderSomething() const override;
    NauIcon iconvResourcePlaceholderVariant() const override;

    NauIcon iconSearch() const override;
    NauIcon iconSearchClear() const override;

    NauIcon iconLoggerDetailsToggle() const override;
    NauIcon iconLoggerSourceEditor() const override;
    NauIcon iconLoggerSourcePlayMode() const override;
    NauIcon iconLoggerSourceBuild() const override;
    NauIcon iconLoggerSourceExternalApplication() const override;

    NauIcon iconClean() const override;

    NauIcon iconSpoilerIndicator() const override;
    NauIcon iconSpoilerAdd() const override;
    NauIcon iconSpoilerClose() const override;

    NauIcon iconActionEditor() const override;
    NauIcon iconVFXEditor() const override;
    NauIcon iconPhysicsMaterialEditor() const override;

    NauIcon iconThanksForFeedbackCheck() const override;

    NauIcon iconArrowDown() const override;
    NauIcon iconArrowRight() const override;
    NauIcon iconBrowserTreeFolder() const override;
    NauIcon iconBreadCrumbsHome() const override;
    NauIcon iconBreadCrumbsDelimiter() const override;

    NauIcon iconTableViewIcon() const override;
    NauIcon iconTileViewIcon() const override;

    std::vector<NauIcon> iconsTimelinePlayback() const override;
    std::vector<NauIcon> iconsTimelineParameters() const override;
    std::vector<NauIcon> iconsTimelineTrackListHeader() const override;

    std::unordered_map<NauSourceState, NauIcon> iconsScriptsState() const override;

    NauPalette paletteTitleBar() const override;
    NauPalette paletteCompilationStateTitleBar() const override;

    NauPalette paletteWorldOutline() const override;
    NauPalette paletteLogger() const override;
    NauPalette paletteProjectBrowser() const override;
    NauPalette paletteObjectInspector() const override;
    NauPalette paletteDataContainers() const override;

    NauPalette palettePrimaryButton() const override;
    NauPalette paletteSecondaryButton() const override;
    NauPalette paletteTertiaryButton() const override;

    NauPalette paletteToogleButton() const override;

    NauPalette paletteSearchWidget() const override;
    NauPalette paletteFilterWidget() const override;
    NauPalette paletteFilterItemWidget() const override;
    NauPalette paletteSpacerWidget() const override;

    NauPalette paletteFeedbackTextEditWidget() const override;

    NauPalette paletteInputGeneric() const override;
    NauPalette paletteInputSpoilerLineEdit() const override;
    NauPalette paletteInputSignalView() const override;
    NauPalette paletteInputBindTab() const override;

    NauPalette paletteVFXGeneric() const override;
    
    NauPalette paletteResourceWidget() const override;
    NauPalette paletteResourcePopupWidget() const override;

    NauPalette paletteNumericSlider() const override;
    NauPalette paletteTimelineTrackList() const override;
    NauPalette paletteTimelineMode() const override;
    NauPalette paletteTimelineRecord() const override;
    NauPalette paletteTimelineKeyframe() const override;
    NauPalette paletteTimelineScrollBar() const override;
    NauPalette paletteTimelineContentView() const override;
    NauPalette paletteTimelineFramePointer() const override;

    NauPalette paletteWidgetAppearanceSlider() const override;
    NauPalette palettePhysicsChannelSettings() const override;
    NauPalette paletteDocking() const override;

    NauPalette paletteSplash() const override;

    NauWidgetStyle stylePrimaryButton() const override;
    NauWidgetStyle styleSecondaryButton() const override;
    NauWidgetStyle styleTertiaryButton() const override;
    NauWidgetStyle styleMiscButton() const override;
    NauWidgetStyle styleSearchWidget() const override;

    NauWidgetStyle styleFeedbackTextEditWidget() const override;
    NauWidgetStyle styleToogleButton() const override;
    NauWidgetStyle styleResourceWidget() const override;

private:
    static NauFont registerFont(const QString& fileName);
    static NauIcon generateTitleBarIcon(const QString& buttonName);
    static NauIcon generateViewportIcon(const QString& buttonName);
    static NauIcon generatePixmapIcon(const QString& pixmapTemplate);
    static NauIcon generateViewportPixmapIcon(const QString& pixmapTemplate);

    static NauIcon generatePrimaryButtonPixmapIcon(const QString& pixmapTemplate);
    static NauIcon generateSecondaryButtonPixmapIcon(const QString& pixmapTemplate);
    static NauIcon generateTertiaryButtonPixmapIcon(const QString& pixmapTemplate);

    static NauIcon generateInputIcon(const QString& iconPath);
    static NauIcon generateInputDeviceIcon(const QString& iconPath);

private:
    NauFont m_fontTitleBarTitle;
    NauFont m_fontTitleBarMenuItem;
    NauFont m_fontDocking;
    NauFont m_fontLoggerLevel;
    NauFont m_fontLogger;
    NauFont m_fontLoggerSource;
    NauFont m_fontProjectBrowserPrimary;
    NauFont m_fontProjectBrowserSecondary;
    NauFont m_fontFilterItem;

    NauFont m_fontWorldOutlineSemibold;
    NauFont m_fontWorldOutlineRegular;

    NauFont m_fontWorldOutlineHeaderSemibold;
    NauFont m_fontWorldOutlineHeaderRegular;

    NauFont m_fontFeedbackHeaderTitle;
    NauFont m_fontFeedbackHeaderThankTitle;
    NauFont m_fontFeedbackCommentInfo;
    NauFont m_fontFeedbackErrorInfo;
    NauFont m_fontFeedbackConnectionError;

    NauFont m_fontInspectorHeaderTitle;
    NauFont m_fontInspectorHeaderSubtitle;

    NauFont m_fontObjectInspector;
    NauFont m_fontObjectInspectorSpoiler;

    NauFont m_fontDataContainers;
    NauFont m_fontDataContainersLabel;

    NauFont m_fontPrimaryButton;

    NauFont m_fontStaticTextBase;
    NauFont m_fontTableWidgetBase;
    NauFont m_fontProjectManagerMenuButton;
    NauFont m_fontProjectManagerLocation;

    // Undo history
    NauFont m_fontHistoryMain;
    
    NauFont m_fontInputHeaderTitle;
    NauFont m_fontInputHeaderSubtitle;
    NauFont m_fontInputLabel;
    NauFont m_fontInputTab;

    NauFont m_fontSplashProjectName;
    NauFont m_fontSplashMessage;
    NauFont m_fontSplashCopyright;

    NauPalette m_paletteTitleBar;
    NauPalette m_paletteWorldOutline;
    NauPalette m_paletteLogger;
    NauPalette m_paletteProjectBrowser;

    NauPalette m_palettePrimaryButton;
    NauPalette m_paletteSecondaryButton;
    NauPalette m_paletteTertiaryButton;

    NauPalette m_paletteSearchWidget;
    NauPalette m_paletteFilterWidget;
    NauPalette m_paletteFilterItemWidget;

    NauPalette m_paletteFeedbackTextEditWidget;

    NauPalette m_paletteInputGeneric;
    NauPalette m_paletteInputSpoilerLineEdit;
    NauPalette m_paletteInputSignalView;
    NauPalette m_paletteInputBindTab;

    NauPalette m_paletteVFXGeneric;
    
    NauPalette m_paletteResourceWidget;
    NauPalette m_paletteResourcePopupWidget;
    NauPalette m_paletteNumericSlider;
    NauPalette m_paletteTimelineTrackList;
    NauPalette m_paletteTimelineMode;
    NauPalette m_paletteTimelineRecord;
    NauPalette m_paletteTimelineKeyframe;
    NauPalette m_paletteTimelineScrollBar;
    NauPalette m_paletteTimelineContentView;
    NauPalette m_paletteTimelineFramePointer;

    NauPalette m_paletteToogleButton;

    // TODO: Not used now, for the reason that colors from qss are used instead of the palette.
    // To be fixed in the future. 
    NauPalette m_paletteObjectInspector;

    // TODO: Not used now, for the reason that colors from qss are used instead of the palette.
    // To be fixed in the future. 
    NauPalette m_paletteDataContainers;

    struct NauPrimaryButtonStyle : public NauWidgetStyle
    {
        NauPrimaryButtonStyle(const NauAbstractTheme* theme);

        struct NauPrimaryButtonActiveStyle;
        struct NauPrimaryButtonHoverStyle;
        struct NauPrimaryButtonPressedStyle;
        struct NauPrimaryButtonTabFocusedStyle;
        struct NauPrimaryButtonDisableStyle;
    };

    struct NauSecondaryButtonStyle : public NauPrimaryButtonStyle
    {
        NauSecondaryButtonStyle(const NauAbstractTheme* theme);

        struct NauSecondButtonActiveStyle;
        struct NauSecondButtonHoverStyle;
        struct NauSecondButtonPressedStyle;
        struct NauSecondButtonTabFocusedStyle;
    };

    struct NauTertiaryButtonStyle : public NauSecondaryButtonStyle
    {
        NauTertiaryButtonStyle(const NauAbstractTheme* theme);

        struct NauTertiaryButtonActiveStyle;
    };

    struct NauMiscButtonStyle : public NauWidgetStyle
    {
        NauMiscButtonStyle();
    };
    
    struct NauSearchWidgetStyle : public NauWidgetStyle
    {
        explicit NauSearchWidgetStyle(const NauAbstractTheme* theme);

        struct NauSearchWidgetActiveStyle;
        struct NauSearchWidgetHoverStyle;
        struct NauSearchWidgetPressedStyle;
        struct NauSearchWidgetTabFocusedStyle;
        struct NauSearchWidgetDisableStyle;
    };

    struct NauFeedbackTextEditStyle : public NauWidgetStyle
    {
        explicit NauFeedbackTextEditStyle(const NauAbstractTheme* theme);

        struct NauFeedbackTextEditActiveStyle;
        struct NauFeedbackTextEditHoverStyle;
        struct NauFeedbackTextEditPressedStyle;
        struct NauFeedbackTextEditFocusedStyle;
        struct NauFeedbackTextEditErrorStyle;
        struct NauFeedbackTextEditDisabledStyle;
    };

    struct NauToogleButtonStyle : public NauWidgetStyle
    {
        explicit NauToogleButtonStyle(const NauAbstractTheme* theme);

        struct NauToogleButtonActiveStyle;
        struct NauToogleButtonHoverStyle;
        struct NauToogleButtonPressedStyle;
        struct NauToogleButtonTabFocusedStyle;
        struct NauToogleButtonDisableStyle;
    };

    struct NauResourceWidgetStyle : public NauWidgetStyle
    {
        NauResourceWidgetStyle(const NauAbstractTheme* theme);

        struct NauResourceWidgetActiveStyle;
        struct NauResourceWidgetHoverStyle;
        struct NauResourceWidgetPressedStyle;
        struct NauResourceWidgetTabFocusedStyle;
        struct NauResourceWidgetDisableStyle;
    };
};
