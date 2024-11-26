!include MUI2.nsh
Unicode true
  
!define ExeDirName ${EXEDIR}
!define OutputFileName ${OUTFILENAME}
!define VcRedistrFile VC_redist.x64.exe

Name "NauEngine"
Caption "NauEngine ${VERSION} Setup"
OutFile ${OutputFileName}
RequestExecutionLevel admin

!define MUI_ABORTWARNING
!define MUI_LANGDLL_ALLLANGUAGES
!define MUI_ICON "../../resources/editor/app/app-icon.ico"
!define MUI_UNICON "../../resources/editor/app/app-icon.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "resources\background.bmp"    # For some reason this requires backslash
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "resources\background.bmp"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE $(LicenseFileName)
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
    !define MUI_FINISHPAGE_NOAUTOCLOSE
    !define MUI_FINISHPAGE_RUN
    !define MUI_FINISHPAGE_RUN_CHECKED
    !define MUI_FINISHPAGE_RUN_TEXT $(StartEditorStr)
    !define MUI_FINISHPAGE_RUN_FUNCTION "LaunchNauEditor"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Russian"
!insertmacro MUI_LANGUAGE "English"

InstallDir $PROGRAMFILES64\NauEngine\NauEditor

InstType "Full"
!define IT_FULL 1

InstType "Minimal"
!define IT_MINIMAL 2

; This key should synchronized with generated one in nau_run_guard.cpp:NauRunGuardPrivate::m_memory.nativeKey().
!define SharedMemoryKey "qipc_sharedmemory_effcdadbbbaefbd084998cdc3415332143ab8d3f167726940b599f5"

ShowInstDetails show

!macro CheckEditorRunningMacro un processType
Function ${un}CheckEditorRunning
    System::Call 'kernel32::OpenFileMapping(i 0x0004, i 0, t "${SharedMemoryKey}")p.R0'
    IntCmp $R0 0 NotRunning
    System::Call 'kernel32::CloseHandle(p $R0)'
    MessageBox MB_OK|MB_ICONEXCLAMATION "Cannot start the ${processType} process because NauEditor is currently running.$\r$\n$\r$\n\
        Please save your data, close NauEditor and try again." /SD IDOK
    Abort
NotRunning:
FunctionEnd
!macroend

; Insert function as an installer and uninstaller function.
!insertmacro CheckEditorRunningMacro "" "installation"
!insertmacro CheckEditorRunningMacro "un." "uninstallation"

Function .onInit
    Call CheckEditorRunning
!insertmacro MUI_LANGDLL_DISPLAY

FunctionEnd

Function un.onInit
    Call un.CheckEditorRunning
!insertmacro MUI_UNGETLANGUAGE    
FunctionEnd

Section $(CoreSectionStr) CoreSection
    SectionIn RO
    SetOutPath $INSTDIR

    File /r /x .* /x *.pdb /x *.ilk /x *.obj /x rcc /x moc /x qmake /x *.tlog /x qt.natvis /x *vcxproj* /x *.log /x *.recipe /x BUILD_INFO.txt ${ExeDirName}\*

    File ${__FILEDIR__}\temp\${VcRedistrFile}

    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NauEngineEditor\" "DisplayName" "NauEditor"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NauEngineEditor\" "DisplayVersion" "${VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NauEngineEditor\" "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NauEngineEditor\" "DisplayIcon" "$INSTDIR\NauEditor.exe"
    WriteRegStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "NAU_EDITOR_DIR" "$INSTDIR"
    WriteRegStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "NAU_ENGINE_SDK_DIR" "$INSTDIR\NauEngineSDK"

    WriteUninstaller uninstall.exe
SectionEnd

Section $(VsSectionStr) VsRedistributable
    SectionIn RO
    DetailPrint "Install MS VC Redistributable"
    ExecWait "${VcRedistrFile} /install /norestart /quiet"

    Delete ${VcRedistrFile}
SectionEnd

Section $(ShortcutSectionStr) ShortcutSection
    SectionIn ${IT_FULL}
    SetShellVarContext all

    CreateDirectory "$SMPROGRAMS\NauEngine"
    CreateDirectory "$SMPROGRAMS\NauEngine\NauEditor"
    CreateShortcut  "$SMPROGRAMS\NauEngine\NauEditor\Uninstall.lnk" "$INSTDIR\uninstall.exe"
    CreateShortcut  "$SMPROGRAMS\NauEngine\NauEditor\NauEditor.lnk" "$INSTDIR\NauEditor.exe"
    CreateShortcut  "$DESKTOP\NauEditor.lnk" "$INSTDIR\NauEditor.exe"
SectionEnd

Section "Uninstall"
    Delete uninstall.exe

    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NauEngineEditor"
    DeleteRegValue HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "NAU_ENGINE_SDK_DIR"
    DeleteRegValue HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "NAU_EDITOR_DIR"

    SetShellVarContext all

    Delete   "$DESKTOP\NauEditor.lnk"
    RMDir /r "$SMPROGRAMS\NauEngine\NauEditor"
    RMDir "$SMPROGRAMS\NauEngine" ; Remove NauEngine dir too if it is empty.

    RMDir /r "$INSTDIR"
    RMDir "$PROGRAMFILES64\NauEngine" ; Remove NauEngine dir too if it is empty.
SectionEnd

Function LaunchNauEditor
    SetOutPath $INSTDIR

    ; Note at this moment we've got elevated priveleges. On other side we cant run editor as admin.
    ShellExecAsUser::ShellExecAsUser "" "$INSTDIR\NauEditor.exe" "" SW_SHOWNORMAL
FunctionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
!insertmacro MUI_DESCRIPTION_TEXT ${CoreSection} $(CoreSectionStr)
!insertmacro MUI_DESCRIPTION_TEXT ${VsRedistributable} $(VsRedistributableStr)
!insertmacro MUI_DESCRIPTION_TEXT ${ShortcutSection} $(ShortcutSectionStr)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

LicenseLangString LicenseFileName ${LANG_RUSSIAN} "license_ru.rtf"
LicenseLangString LicenseFileName ${LANG_ENGLISH} "license_en.rtf"

LangString CoreSectionStr ${LANG_ENGLISH} "Mandatory Engine core files"
LangString CoreSectionStr ${LANG_RUSSIAN} "Обязательные файлы движка"

LangString VsRedistributableStr ${LANG_ENGLISH} "The Visual C++ Redistributable installs Microsoft C and C++ (MSVC) runtime libraries"
LangString VsRedistributableStr ${LANG_RUSSIAN} "Распространяемый компонент Visual C++ устанавливает библиотеки среды выполнения Microsoft C и C++ (MSVC)."

LangString ShortcutSectionStr ${LANG_ENGLISH} "A shortcuts on the desktop and programs menu directories"
LangString ShortcutSectionStr ${LANG_RUSSIAN} "Ярлыки на рабочем столе и в папке Программы"

LangString StartEditorStr ${LANG_ENGLISH} "Start NauEditor"
LangString StartEditorStr ${LANG_RUSSIAN} "Запустить NauEditor"

LangString ShortcutSectionStr ${LANG_ENGLISH} "Create Shortcuts"
LangString ShortcutSectionStr ${LANG_RUSSIAN} "Создать ярлыки"

LangString VsSectionStr ${LANG_ENGLISH} "VS Redistributable"
LangString VsSectionStr ${LANG_RUSSIAN} "Компонент Visual C++"

LangString CoreSectionStr ${LANG_ENGLISH} "!NauEngine Core files"
LangString CoreSectionStr ${LANG_RUSSIAN} "!NauEngine Файлы"
