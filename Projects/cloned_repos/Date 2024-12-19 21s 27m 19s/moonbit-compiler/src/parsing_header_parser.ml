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


module Vec = Basic_vec
module Token_triple = Lex_token_triple
module Vec_token = Lex_vec_token

type state = { mutable cur : int; tokens : Token_triple.t array; len : int }

let bump_any state : unit = state.cur <- state.cur + 1

let rec peek state : Token_triple.t =
  if state.cur >= state.len then
    let _, posa, posb = state.tokens.(state.cur - 1) in
    (EOF, posa, posb)
  else
    let ((t, _, _) as res) = state.tokens.(state.cur) in
    if t = NEWLINE then (
      bump_any state;
      peek state)
    else res

type directive = (string * string) list

let parse_directive state : directive =
  let parse_entry start s =
    Basic_strutil.split_on_first ':'
      (String.sub s start (String.length s - start))
  in
  let rec go acc =
    match peek state with
    | COMMENT s, _, _ when String.starts_with s.content ~prefix:"//!" ->
        bump_any state;
        go
          (let k, v = parse_entry 3 s.content in
           (k, v) :: acc)
    | _ -> acc
  in
  List.rev (go [])

let parse (tokens : Vec_token.t) : directive * int =
  let state : state =
    {
      cur = 0;
      tokens = Vec.unsafe_internal_array tokens;
      len = Vec.length tokens;
    }
  in
  let directive = parse_directive state in
  (directive, state.cur)
