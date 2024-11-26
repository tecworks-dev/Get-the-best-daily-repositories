// Copyright 2024 N-GINN LLC. All rights reserved.
// Use of this source code is governed by a BSD-3 Clause license that can be found in the LICENSE file.
//
// List of known internal shortcut operations.

#pragma once

#include "nau_plus_enum.hpp"
#include "magic_enum/magic_enum.hpp"
#include "magic_enum/magic_enum_iostream.hpp"

#include <fmt/format.h>
#include <fmt/base.h>

// ** NauShortcutOperation
// Do not change order of this enum, it affects on serialization/deserialization.
// Do not break incremental order too.

enum class NauShortcutOperation
{
    NewScene = 0,
    OpenScene,
    SaveScene,

    Undo,
    Redo,

    ProjectBrowserCopy,
    ProjectBrowserCut,
    ProjectBrowserPaste,
    ProjectBrowserRename,
    ProjectBrowserDelete,
    ProjectBrowserDuplicate,
    ProjectBrowserViewInShell,
    ProjectBrowserCreateDir,
    ProjectBrowserSwitchToTree,
    ProjectBrowserSwitchToContentView,
    ProjectBrowserFindAsset,
    ProjectBrowserAddMaterial,
    ProjectBrowserAddInputAction,
    ProjectBrowserImportAsset,

    WorldOutlineCopy,
    WorldOutlineCut,
    WorldOutlinePaste,
    WorldOutlineRename,
    WorldOutlineDelete,
    WorldOutlineDuplicate,
    WorldOutlineFocusCamera,
    WorldOutlineSelectAll,

    ViewportFocus,
    ViewportCopy,
    ViewportCut,
    ViewportPaste,
    ViewportDelete,
    ViewportDuplicate,
    ViewportSelectTool,
    ViewportTranslateTool,
    ViewportRotateTool,
    ViewportScaleTool,

    LoggerCopySelectedMessages,
    LoggerCopyTextSelection,
};

template<>
struct fmt::formatter<NauShortcutOperation, char> : fmt::formatter<const char*, char>
{
    template <typename FormatContext>
    auto format(const NauShortcutOperation& input, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(input));
    }
};