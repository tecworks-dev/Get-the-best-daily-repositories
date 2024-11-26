// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// Interface for editor themes.

#pragma once

#include "nau/app/nau_qt_app.hpp"
#include "nau_font.hpp"
#include "baseWidgets/nau_icon.hpp"
#include "nau_palette.hpp"
#include "nau_widget_style.hpp"
#include "nau/compiler/nau_source_state.hpp"


// ** NauAbstractTheme

class NAU_EDITOR_API NauAbstractTheme
{
public:
    virtual ~NauAbstractTheme() = default;

    // Here implementations can apply global stylesheet to whole application.
    virtual void applyGlobalStyle(NauApp& app) = 0;

    virtual NauFont fontTitleBarTitle() const = 0;
    virtual NauFont fontTitleBarMenuItem() const = 0;

    virtual NauFont fontWorldOutlineSemibold() const = 0;
    virtual NauFont fontWorldOutlineRegular() const = 0;

    virtual NauFont fontWorldOutlineHeaderSemibold() const = 0;
    virtual NauFont fontWorldOutlineHeaderRegular() const = 0;

    virtual NauFont fontLogger() const = 0;
    virtual NauFont fontLoggerLevel() const = 0;
    virtual NauFont fontLoggerSource() const = 0;

    virtual NauFont fontProjectBrowserPrimary() const = 0;
    virtual NauFont fontProjectBrowserSecondary() const = 0;

    virtual NauFont fontFilterItem() const = 0;

    virtual NauFont fontDocking() const = 0;

    virtual NauFont fontFeedbackHeaderTitle() const = 0;
    virtual NauFont fontFeedbackHeaderThankTitle() const = 0;

    virtual NauFont fontFeedbackCommentInfo() const = 0;
    virtual NauFont fontFeedbackErrorInfo() const = 0;
    virtual NauFont fontFeedbackConnectionError() const = 0;

    virtual NauFont fontStaticTextBase() const = 0;
    virtual NauFont fontTableWidgetBase() const = 0;

    // Undo history
    virtual NauFont fontHistoryMain() const = 0;

    virtual NauFont fontInspectorHeaderTitle() const = 0;
    virtual NauFont fontInspectorHeaderSubtitle() const = 0;

    virtual NauFont fontObjectInspector() const = 0;
    virtual NauFont fontObjectInspectorSpoiler() const = 0;

    virtual NauFont fontDataContainers() const = 0;
    virtual NauFont fontDataContainersLabel() const = 0;

    virtual NauFont fontPrimaryButton() const = 0;

    virtual NauFont fontInputHeaderTitle() const = 0;
    virtual NauFont fontInputHeaderSubtitle() const = 0;
    virtual NauFont fontInputLabel() const = 0;
    virtual NauFont fontInputTab() const = 0;

    virtual NauFont fontSplashProjectName() const = 0;
    virtual NauFont fontSplashMessage() const = 0;
    virtual NauFont fontSplashCopyright() const = 0;

    virtual NauFont fontProjectManagerMenuButton() const = 0;
    virtual NauFont fontProjectManagerLocation() const = 0;

    // Icons for a title bar's minimize button.
    // Implementations must return filled NauIcon according following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with the min button, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the min button.
    // * NauIcon::Selected - a pixmap when the user clicks on the min button.
    // * NauIcon::Disabled - a pixmap when the min button isn't available.
    virtual NauIcon iconMinimize() const = 0;

    // Icons for a window close button. Used in the title bar and docking system.
    // Implementations must return filled NauIcon according following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with the close button, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Selected - a pixmap when the user clicks on the close button.
    // * NauIcon::Disabled - a pixmap when the close button isn't available.
    virtual NauIcon iconClose() const = 0;

    // Same as for iconMinimize() and iconClose(), but this icon must provide 
    // * pixmap for NauIcon::State::Off(to use for `maximize` button)
    // * pixmap for NauIcon::State::On(to use for `restore` button).
    virtual NauIcon iconWindowState() const = 0;

    // Used in the docking system for undock button.
    // Implementations must return filled NauIcon according following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with the close button, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Selected - a pixmap when the user clicks on the close button.
    // * NauIcon::Disabled - a pixmap when the close button isn't available.
    virtual NauIcon iconUndock() const = 0;

    // Used in the docking system for "more" menu in docking areas.
    // Implementations must return filled NauIcon according following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with the close button, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Selected - a pixmap when the user clicks on the close button.
    // * NauIcon::Disabled - a pixmap when the close button isn't available.
    virtual NauIcon iconDockAreaMenu() const = 0;

    // Logo icon for a title bar.
    virtual NauIcon iconTitleBarLogo() const = 0;

    // Icon for copy text in logger panel.
    virtual NauIcon iconLoggerCopy() const = 0;

    // Icon for settings button in toolbar of logger.
    virtual NauIcon iconLoggerSettings() const = 0;

    // Icon for toggling the visibility of details panel in the logger.
    virtual NauIcon iconLoggerDetailsToggle() const = 0;

    // Icon for Editor source in the logger.
    virtual NauIcon iconLoggerSourceEditor() const = 0;

    // Icon for PlayMode source in the logger.
    virtual NauIcon iconLoggerSourcePlayMode() const = 0;

    // Icon for Build source in the logger.
    virtual NauIcon iconLoggerSourceBuild() const = 0;

    // Icon for External App source in the logger.
    virtual NauIcon iconLoggerSourceExternalApplication() const = 0;

    // Icon to indicate an item's visibility state in data view elements. e.g. world outline.
    virtual NauIcon iconVisibility() const = 0;

    // Icon to indicate the availability(i.e. read only) state in data view elements. e.g. world outline.
    virtual NauIcon iconAvailability() const = 0;

    // Icon for manipulation on items(Cut) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconCut() const = 0;

    // Icon for manipulation on items(Copy) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconCopy() const = 0;

    // Icon for manipulation on items(Paste) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconPaste() const = 0;

    // Icon for manipulation on items(Duplicate) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconDuplicate() const = 0;

    // Icon for manipulation on items(Delete) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconDelete() const = 0;

    // Icon for manipulation on items(Rename) in data view elements. e.g. project browser, world outline.
    virtual NauIcon iconRename() const = 0;

    // Icon for save command. e.g. in the logger.
    virtual NauIcon iconSave() const = 0;

    // Used in the project browser for file operation menu item: "Show in Explorer"(Win).
    virtual NauIcon iconViewAssetInShell() const = 0;

    // Used in the project browser for file operation menu item to create a new folder.
    virtual NauIcon iconMakeNewFolder() const = 0;

    // Used in the scene hierarchy for entity to focus camera on it.
    virtual NauIcon iconFocusCameraOnObject() const = 0;

    // Used temporarily as a replacement for resource icons
    virtual NauIcon iconResourcePlaceholder() const = 0;

    // Used in different widgets, shows that this button can call context menu
    virtual NauIcon iconThreeDotsHorisont() const = 0;

    // Icon used to start simulation
    virtual NauIcon iconPlay() const = 0;

    // Icon used to stop simulation or to stop build
    virtual NauIcon iconStop() const = 0;

    // Icon used to pause simulation
    virtual NauIcon iconPause() const = 0;

    // Undo/redo icons
    virtual NauIcon iconUndo() const = 0;
    virtual NauIcon iconRedo() const = 0;
    virtual NauIcon iconFatDot() const = 0;
    virtual NauIcon iconDottedLine() const = 0;
    virtual NauIcon iconDottedLineTail() const = 0;

    // Generic history icon
    virtual NauIcon iconHistory() const = 0;

    // Generic settings icon
    virtual NauIcon iconSettings() const = 0;    
    
    // Generic preferences icon
    virtual NauIcon iconPreferences() const = 0;

    // Icon used to open build settings
    virtual NauIcon iconBuildSettings() const = 0;

    // Used in viewport toolbar to show additional viewport settings
    virtual NauIcon iconHamburger() const = 0;

    // Icon used to undo action
    virtual NauIcon iconUndoAction() const = 0;

    // Icon used to show help
    virtual NauIcon iconQuestionMark() const = 0;

    // Used in viewport toolbar to show viewport camera settings
    virtual NauIcon iconCameraSettings() const = 0;

    // Used in viewport toolbar to show local coordinate space mode
    virtual NauIcon iconLocalCoordinateSpace() const = 0;    

    // Used in viewport toolbar to show world coordinate space mode
    virtual NauIcon iconWorldCoordinateSpace() const = 0;

    // Used in viewport toolbar to activate Select Tool
    virtual NauIcon iconSelectTool() const = 0;

    // Used in viewport toolbar to activate Move Tool
    virtual NauIcon iconMoveTool() const = 0;

    // Used in viewport toolbar to activate Rotate Tool
    virtual NauIcon iconRotateTool() const = 0;

    // Used in viewport toolbar to activate Scale Tool
    virtual NauIcon iconScaleTool() const = 0;

    // Used in the spoiler to control widget expansion
    virtual NauIcon iconInspectorArrowDown() const = 0;

    // Used in the spoiler to control widget expansion
    virtual NauIcon iconInspectorArrowRight() const = 0;

    // Used in the check box to indicate an enabled state
    virtual NauIcon iconChecked() const = 0;

    // Used in the combo box as a stub for the object icon
    virtual NauIcon iconComboBoxDefault() const = 0;

    // Used in Input Editor for signal source
    virtual NauIcon iconGamePad() const = 0;
    virtual NauIcon iconKeyboard() const = 0;
    virtual NauIcon iconMouse() const = 0;
    virtual NauIcon iconTouchPad() const = 0;
    virtual NauIcon iconVR() const = 0;
    virtual NauIcon iconUnknown() const = 0;

    virtual NauIcon iconArrowDown() const = 0;
    virtual NauIcon iconArrowRight() const = 0;
    virtual NauIcon iconBrowserTreeFolder() const = 0;
    virtual NauIcon iconBreadCrumbsHome() const = 0;
    virtual NauIcon iconBreadCrumbsDelimiter() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconDonePrimaryStyle() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconClosePrimaryStyle() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconSendPrimaryStyle() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconAddPrimaryStyle() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconAddSecondaryStyle() const = 0;

    // Implementations must return filled NauIcon according to the following:
    // * NauIcon::Normal - a pixmap when the user is not interacting with icon, but it is available.
    // * NauIcon::Active - a pixmap when the user moving the mouse over the close button.
    // * NauIcon::Disabled - a pixmap when the icon isn't available.
    virtual NauIcon iconAddTertiaryStyle() const = 0;

    // Sorting in ascending order indicator.Used in header of table views.
    virtual NauIcon iconAscendingSortIndicator() const = 0;

    // Sorting in descending order indicator.Used in header of table views.
    virtual NauIcon iconDescendingSortIndicator() const = 0;

    // Used in NauResourceComboBox
    virtual NauIcon iconButtonSettings() const = 0;

    // Used in NauResourceComboBox
    virtual NauIcon iconButtonFolder() const = 0;

    // Temporarily used in NauResourceComboBox
    virtual NauIcon iconvResourcePlaceholderSphere() const = 0;

    // Temporarily used in NauResourceComboBox
    virtual NauIcon iconvResourcePlaceholderSomething() const = 0;

    // Temporarily used in NauResourceComboBox
    virtual NauIcon iconvResourcePlaceholderVariant() const = 0;

    virtual NauIcon iconInvisibleChild() const = 0;

    virtual NauIcon iconSearch() const = 0;

    virtual NauIcon iconSearchClear() const = 0;

    virtual NauIcon iconSpoilerIndicator() const = 0;

    virtual NauIcon iconSpoilerAdd() const = 0;

    virtual NauIcon iconSpoilerClose() const = 0;

    virtual NauIcon iconActionEditor() const = 0;

    virtual NauIcon iconVFXEditor() const = 0;

    virtual NauIcon iconPhysicsMaterialEditor() const = 0;

    virtual NauIcon iconThanksForFeedbackCheck() const = 0;

    // List of icons representing log level. Index of vector corresponds to log level.
    virtual std::vector<NauIcon> iconLogs() const = 0;

    virtual NauIcon iconClean() const = 0;

    virtual NauIcon iconTableViewIcon() const = 0;
    virtual NauIcon iconTileViewIcon() const = 0;

    virtual std::vector<NauIcon> iconsTimelinePlayback() const = 0;
    virtual std::vector<NauIcon> iconsTimelineParameters() const = 0;
    virtual std::vector<NauIcon> iconsTimelineTrackListHeader() const = 0;

    // Set of icons corresponding to states of source building.
    virtual std::unordered_map<NauSourceState, NauIcon> iconsScriptsState() const = 0;

    // Palette for the title bar.
    // Implementation must return filled palette as follows:
    // * brush NauPalette::Background for background of title bar.
    // * color NauPalette::Foreground for text color of menu items, project name.
    // * color NauPalette::ForegroundBrightText for text color of a scene name.
    virtual NauPalette paletteTitleBar() const = 0;

    // Palette for widget of compilation state in the title bar.
    // Implementation must return filled palette as follows:
    // * brush NauPalette::Background (Hover/Normal) for background.
    // * brush NauPalette::AlternateBackground (Hover/Normal) for background foreground rounded rect.
    virtual NauPalette paletteCompilationStateTitleBar() const = 0;

    // Palette for the world outline.
    // Implementation must return filled palette as follows:
    // * brushes NauPalette::Background NauPalette::AlternateBackground for background rows in table.
    // * color NauPalette::Foreground for text color in cells.
    // * brush NauPalette::BackgroundHeader for background of horizontal header.
    // * color NauPalette::ForegroundHeader for text color of horizontal header.
    // * brush NauPalette::Background(NauPalette::Selected) for background of selected items.
    // * brush NauPalette::Background(NauPalette::Selected | NauPalette::Hovered) for background of selected and hovered items.
    // * color NauPalette::Border(NauPalette::Hovered) for border color of hovered items.
    virtual NauPalette paletteWorldOutline() const = 0;

    // Palette for the logger output window.
    // Implementation must return filled palette as follows:
    // * brushes NauPalette::Background NauPalette::AlternateBackground for background rows in table.
    // * color NauPalette::Foreground for text color in cells.
    // * brush NauPalette::BackgroundHeader for background of horizontal header.
    // * color NauPalette::ForegroundHeader for text color of horizontal header.
    // * brush NauPalette::Background(NauPalette::Selected) for background of selected items.
    // * brush NauPalette::Background(NauPalette::Selected | NauPalette::Hovered) for background of selected and hovered items.
    // * color NauPalette::Border(NauPalette::Hovered) for border color of hovered items.
    virtual NauPalette paletteLogger() const = 0;

    // Palette for the project browser output window.
    // Implementation must return filled palette as follows:
    // * brushes NauPalette::Background NauPalette::AlternateBackground for background rows in table.
    // * color NauPalette::Foreground for text color in cells.
    // * brush NauPalette::BackgroundHeader for background of horizontal header.
    // * color NauPalette::ForegroundHeader for text color of horizontal header.
    // * brush NauPalette::Background(NauPalette::Selected) for background of selected items.
    // * brush NauPalette::Background(NauPalette::Selected | NauPalette::Hovered) for background of selected and hovered items.
    // * color NauPalette::Border(NauPalette::Hovered) for border color of hovered items.
    virtual NauPalette paletteProjectBrowser() const = 0;

    // Palette for the title bar.
    // Implementation must return filled palette as follows:
    // * brush NauPalette::Background for background of widget
    virtual NauPalette paletteObjectInspector() const = 0;

    // Palette for different data containers.
    // Implementation must return filled palette as follows:
    // * brush NauPalette::Background for background of widget
    virtual NauPalette paletteDataContainers() const = 0;

    // Palette for the primary button
    // Implementation must return filled palette as follows:
    // * color NauPalette::Background(NauPalette::Normal, NauPalette::Category::Active) for background of button
    // * color NauPalette::Text(NauPalette::Normal, NauPalette::Category::Active) for text of button
    // * color NauPalette::Border(NauPalette::Normal, NauPalette::Category::Active) for border of button
    // * color NauPalette::Background(NauPalette::Hovered) for background of button
    // * color NauPalette::Border(NauPalette::Hovered) for border of button
    // * color NauPalette::Background(NauPalette::Pressed)  for background of button
    // * color NauPalette::Border(NauPalette::Pressed) for border of button
    // * color NauPalette::Border(NauPalette::Selected) for border of button
    // * color NauPalette::Background(NauPalette::Normal, NauPalette::Category::Disabled)  for background of button
    // * color NauPalette::Text(NauPalette::Normal, NauPalette::Category::Disabled) for text of button
    // * color NauPalette::Border(NauPalette::Normal, NauPalette::Category::Disabled) for border of button
    virtual NauPalette palettePrimaryButton() const = 0;

    // Palette for the second button
    // Implementation must return filled palette as follows:
    // * color NauPalette::Background(NauPalette::Normal, NauPalette::Category::Active) for background of button
    // * color NauPalette::Background(NauPalette::Hovered) for background of button
    // * color NauPalette::Background(NauPalette::Pressed) for background of button
    // * color NauPalette::Background(NauPalette::Selected) for background of button
    virtual NauPalette paletteSecondaryButton() const = 0;

    // Palette for the tertiary button
    // Implementation must return filled palette as follows:
    // * color NauPalette::Background(NauPalette::Normal, NauPalette::Category::Active) for background of button
    // * color NauPalette::Text(NauPalette::Normal, NauPalette::Category::Active) for text of button
    virtual NauPalette paletteTertiaryButton() const = 0;

    virtual NauPalette paletteToogleButton() const = 0;

    virtual NauPalette paletteSearchWidget() const = 0;

    virtual NauPalette paletteFilterWidget() const = 0;

    virtual NauPalette paletteFilterItemWidget() const = 0;

    virtual NauPalette paletteSpacerWidget() const = 0;

    virtual NauPalette paletteFeedbackTextEditWidget() const = 0;

    virtual NauPalette paletteInputGeneric() const = 0;

    virtual NauPalette paletteInputSpoilerLineEdit() const = 0;

    virtual NauPalette paletteInputSignalView() const = 0;

    virtual NauPalette paletteInputBindTab() const = 0;

    virtual NauPalette paletteVFXGeneric() const = 0;
    
    virtual NauPalette paletteResourceWidget() const = 0;

    virtual NauPalette paletteResourcePopupWidget() const = 0;

    virtual NauPalette paletteNumericSlider() const = 0;

    virtual NauPalette paletteTimelineTrackList() const = 0;

    virtual NauPalette paletteTimelineMode() const = 0;

    virtual NauPalette paletteTimelineRecord() const = 0;

    virtual NauPalette paletteTimelineKeyframe() const = 0;

    virtual NauPalette paletteTimelineScrollBar() const = 0;

    virtual NauPalette paletteTimelineContentView() const = 0;

    virtual NauPalette paletteTimelineFramePointer() const = 0;

    // Palette is used to draw a horizontal slider.
    // Background brush for slider's groove.
    // AlternateBackground brush for slider's handle in Normal state.
    // AlternateBackground brush for slider's handle in Hovered state.
    virtual NauPalette paletteWidgetAppearanceSlider() const = 0;

    // Palette for physics channel setting form.
    // Background brush for content background.
    // BackgroundHeader for a header. BackgroundFooter for a footer.
    virtual NauPalette palettePhysicsChannelSettings() const = 0;

    // Palette for docking system:
    // BackgroundHeader/Active - Background brush for active tab header.
    // BackgroundHeader/Inactive - Background brush for inactive tab header. 
    // BackgroundHeader/Flashing - Start color of background of tab header.
    // AlternateBackgroundHeader/Flashing - End color of background of tab header.
    // ForegroundHeader/Active - foreground color of inactive tab header.
    // ForegroundHeader/Inactive - foreground color of active tab header.
    virtual NauPalette paletteDocking() const = 0;

    // Palette for startup splash screen.
    // Background brush for content background.
    virtual NauPalette paletteSplash() const = 0;

    //  Style for the primary button
    virtual NauWidgetStyle stylePrimaryButton() const = 0;

    //  Style for the second button
    virtual NauWidgetStyle styleSecondaryButton() const = 0;

    //  Style for the tertiary button
    virtual NauWidgetStyle styleTertiaryButton() const = 0;

    //  Style for the misc button
    virtual NauWidgetStyle styleMiscButton() const = 0;
    
    virtual NauWidgetStyle styleSearchWidget() const = 0;

    virtual NauWidgetStyle styleToogleButton() const = 0;

    virtual NauWidgetStyle styleFeedbackTextEditWidget() const = 0;

    virtual NauWidgetStyle styleResourceWidget() const = 0;
};
