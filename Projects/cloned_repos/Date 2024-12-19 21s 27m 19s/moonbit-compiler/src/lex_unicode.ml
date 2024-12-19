(*
   Copyright (C) 2024 International Digital Economy Academy.
   This program is licensed under the MoonBit Public Source
   License as published by the International Digital Economy Academy,
   either version 1 of the License, or (at your option) any later
   version. This program is distributed in the hope that it will be
   useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MoonBit
   Public Source License for more details. You should have received a
   copy of the MoonBit Public Source License along with this program. If
   not, see
   <https://www.moonbitlang.com/licenses/moonbit-public-source-license-v1>.
*)


let blocks : int array =
  [|
    0x30;
    0x39;
    0x41;
    0x5a;
    0x5f;
    0x5f;
    0x61;
    0x7a;
    0xa1;
    0xac;
    0xae;
    0x2af;
    0x1100;
    0x11ff;
    0x1e00;
    0x1eff;
    0x2070;
    0x209f;
    0x2150;
    0x218f;
    0x2e80;
    0x2eff;
    0x2ff0;
    0x2fff;
    0x3001;
    0x30ff;
    0x31c0;
    0x9fff;
    0xac00;
    0xd7ff;
    0xf900;
    0xfaff;
    0xfe00;
    0xfe0f;
    0xfe30;
    0xfe4f;
    0x1f000;
    0x1fbff;
    0x20000;
    0x2a6df;
    0x2a700;
    0x2ebef;
    0x2f800;
    0x2fa1f;
    0x30000;
    0x323af;
    0xe0100;
    0xe01ef;
  |]

let is_valid_unicode_codepoint c =
  Basic_binary_search.search_range c blocks <> -1
