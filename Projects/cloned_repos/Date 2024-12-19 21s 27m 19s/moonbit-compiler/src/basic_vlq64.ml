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


module Unsafe_external = Basic_unsafe_external
module Vec_int = Basic_vec_int
open Unsafe_external

let base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
let base64_len = String.length base64

let rec encode_vlq buf vlq =
  let digit = vlq land 31 in
  let vlq = vlq lsr 5 in
  if vlq = 0 then Buffer.add_char buf base64.![digit]
  else (
    Buffer.add_char buf base64.![digit lor 32];
    encode_vlq buf vlq)

let buffer_add_vlq64 buf (value : int) =
  let vlq = if value < 0 then (-value lsl 1) + 1 else (value lsl 1) + 0 in
  encode_vlq buf vlq

let shift_const = 5
let vlq_base = 1 lsl shift_const
let vlq_base_mask = vlq_base - 1
let vlq_continuation_bit = vlq_base

let rec index_rec_opt s ~lim i c =
  if i >= lim then -1
  else if s.![i] = c then i
  else index_rec_opt s ~lim (i + 1) c

let rec helper ~lim ~acc ~shift ~offset ~vec (str : string) =
  let digit =
    if offset >= lim || offset < 0 then -1
    else index_rec_opt base64 ~lim:base64_len 0 str.![offset]
  in
  if digit < 0 then -1
  else
    let continued = digit land vlq_continuation_bit <> 0 in
    let acc = acc + ((digit land vlq_base_mask) lsl shift) in
    if continued then
      helper ~lim ~acc ~shift:(shift + shift_const) ~offset:(offset + 1) str
        ~vec
    else
      let abs = acc / 2 in
      let result = if acc land 1 = 0 then abs else -abs in
      Vec_int.push vec result;
      offset + 1

let decode_string string vec =
  let len = String.length string in
  let offset = ref 0 in
  let no_error = ref true in
  while !offset < len && !no_error do
    let next = helper ~lim:len ~acc:0 ~shift:0 ~offset:!offset ~vec string in
    if next < 0 then no_error := false else offset := next
  done;
  !offset
