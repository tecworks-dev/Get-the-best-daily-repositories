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


module Literal = Lex_literal
module Unicode_lex = Lex_unicode_lex
module Segment = Parsing_segment
module Syntax = Parsing_syntax
module Menhir_token = Lex_menhir_token
module Parser = Parsing_parser

let menhir_parse ~loc_ ~diagnostics ~base segment =
  let lexbuf : Lexing.lexbuf = Lexing.from_string "" in
  let lexer (lexbuf : Lexing.lexbuf) : Menhir_token.token =
    Segment.next_with_lexbuf_update segment lexbuf
  in
  Parsing_menhir_state.initialize_base_pos base;
  try Parser.expression lexer lexbuf
  with Parser.Error ->
    Diagnostics.add_error diagnostics
      (Errors.parse_error ~loc_start:lexbuf.lex_start_p
         ~loc_end:lexbuf.lex_curr_p "parse error");
    Syntax.Pexpr_unit { loc_; faked = true }

let expr_of_interp ~diagnostics ~base ({ source; loc_ } : Literal.interp_source)
    =
  let start_pos = Loc.get_start loc_ in
  let tokens =
    Unicode_lex.tokens_of_string ~start_pos ~is_interpolation:true
      ~name:start_pos.pos_fname source ~comment:false ~diagnostics
  in
  let segment =
    match Segment.toplevel_segments tokens with
    | segment :: [] -> segment
    | _ -> assert false
  in
  let loc_ = Rloc.of_loc ~base loc_ in
  menhir_parse ~loc_ ~diagnostics ~base segment
