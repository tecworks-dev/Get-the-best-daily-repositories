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
module Sexp_parser = Dwarfsm_sexp_parser
module Parse_error = Dwarfsm_sexp_parse_error

let parse s =
  let lexbuf = Lexing.from_string s in
  try Sexp_parser.sexps Wasm_lex.token lexbuf
  with Sexp_parser.Error ->
    (* must be unmatched right parenthesis *)
    let pos = (Lexing.lexeme_start_p lexbuf, Lexing.lexeme_end_p lexbuf) in
    raise (Parse_error.Unmatched_parenthesis pos)
