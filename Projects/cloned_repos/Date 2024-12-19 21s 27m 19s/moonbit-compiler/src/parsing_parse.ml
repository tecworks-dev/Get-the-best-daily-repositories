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


module Syntax = Parsing_syntax
module Segment = Parsing_segment
module Parser = Parsing_parser
module Header_parser = Parsing_header_parser
module Ast_lint = Parsing_ast_lint
module Vec_token = Lex_vec_token
module Menhir_token = Lex_menhir_token
module Vec_comment = Lex_vec_comment
module Unicode_lex = Lex_unicode_lex

type output = {
  ast : Syntax.impls;
  tokens : Vec_token.t;
  directive : (string * string) list;
  name : string;
}

let sexp_of_output (x : output) : S.t =
  let impls = Basic_lst.map x.ast Syntax.sexp_of_impl in
  let directive =
    match Basic_lst.assoc_str x.directive "build" with
    | None -> []
    | Some d ->
        let msg = "//!build:" ^ d in
        ([ List (List.cons (Atom "directive" : S.t) ([ Atom msg ] : S.t list)) ]
          : S.t list)
  in
  (List (List.append (directive : S.t list) (impls : S.t list)) : S.t)

let menhir_parse_toplevel ~diagnostics segment : Syntax.impl list =
  let lexbuf : Lexing.lexbuf = Lexing.from_string "" in
  let lexer (lexbuf : Lexing.lexbuf) : Menhir_token.token =
    Segment.next_with_lexbuf_update segment lexbuf
  in
  Parsing_menhir_state.initialize_state segment;
  try Parser.structure lexer lexbuf
  with Parser.Error ->
    Diagnostics.add_error diagnostics
      (Errors.parse_error ~loc_start:lexbuf.lex_start_p
         ~loc_end:lexbuf.lex_curr_p "parse error");
    []

let doc_search ~diagnostics (docstrings : Vec_comment.t) (last_pos : Loc.t ref)
    (ast : Syntax.impl) =
  match ast with
  | Ptop_typedef td ->
      let comments =
        Vec_comment.search ~last:!last_pos ~loc_:td.loc_ docstrings
      in
      td.doc_ <- Docstring.of_comments ~diagnostics comments;
      last_pos := td.loc_
  | Ptop_funcdef fd ->
      let comments =
        Vec_comment.search ~last:!last_pos ~loc_:fd.loc_ docstrings
      in
      fd.fun_decl.doc_ <- Docstring.of_comments ~diagnostics comments;
      last_pos := fd.loc_
  | Ptop_letdef ld ->
      let comments =
        Vec_comment.search ~last:!last_pos ~loc_:ld.loc_ docstrings
      in
      ld.doc_ <- Docstring.of_comments ~diagnostics comments;
      last_pos := ld.loc_
  | Ptop_expr _ | Ptop_test _ | Ptop_impl_relation _ -> ()
  | Ptop_trait trait ->
      let comments =
        Vec_comment.search ~last:!last_pos ~loc_:trait.trait_loc_ docstrings
      in
      trait.trait_doc_ <- Docstring.of_comments ~diagnostics comments;
      last_pos := trait.trait_loc_
  | Ptop_impl impl ->
      let comments =
        Vec_comment.search ~last:!last_pos ~loc_:impl.loc_ docstrings
      in
      impl.doc_ <- Docstring.of_comments ~diagnostics comments;
      last_pos := impl.loc_

let debug_tokens_info name tokens =
  match name with
  | Some name ->
      Basic_io.write (name ^ ".tokens") (tokens |> Vec_token.string_of_tokens)
  | None -> S.print (Vec_token.sexp_of_t tokens)

let parse_by_menhir ~diagnostics segments : Syntax.impls =
  Basic_lst.concat_map segments (fun x -> menhir_parse_toplevel ~diagnostics x)

let parse_segment ~diagnostics segment : Syntax.impls =
  menhir_parse_toplevel ~diagnostics segment

let impl_of_string ~diagnostics ?name ?(debug_tokens = false) ?directive_handler
    ~transform source =
  let docstrings = Basic_vec.empty () in
  let tokens =
    Unicode_lex.tokens_of_string ?name ~docstrings source ~comment:true
      ~diagnostics
  in
  if debug_tokens then debug_tokens_info name tokens;
  let directive, start_position = Header_parser.parse tokens in
  (match directive_handler with None -> () | Some f -> f directive);
  let segments = Segment.toplevel_segments ~start:start_position tokens in
  let ast =
    parse_by_menhir ~diagnostics segments
  in
  let last_pos = ref Loc.no_location in
  Basic_lst.iter ast (doc_search ~diagnostics docstrings last_pos);
  let ast = if transform then Ast_lint.post_process ~diagnostics ast else ast in
  { ast; tokens; directive; name = Option.value ~default:"" name }

let parse ~diagnostics ?(debug_tokens = false) ?directive_handler ~transform
    path =
  In_channel.with_open_bin path In_channel.input_all
  |> impl_of_string ~diagnostics ~debug_tokens ~transform ?directive_handler
       ~name:(Filename.basename path)
