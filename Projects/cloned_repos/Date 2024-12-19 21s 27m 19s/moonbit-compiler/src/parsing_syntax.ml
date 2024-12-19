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

type absolute_loc = Loc.t

include struct
  let _ = fun (_ : absolute_loc) -> ()
  let sexp_of_absolute_loc = (Loc.sexp_of_t : absolute_loc -> S.t)
  let _ = sexp_of_absolute_loc
end

type location = Rloc.t

include struct
  let _ = fun (_ : location) -> ()
  let sexp_of_location = (Rloc.sexp_of_t : location -> S.t)
  let _ = sexp_of_location
end

let hide_loc _ = not !Basic_config.show_loc

type string_literal = Literal.string_literal

include struct
  let _ = fun (_ : string_literal) -> ()

  let sexp_of_string_literal =
    (Literal.sexp_of_string_literal : string_literal -> S.t)

  let _ = sexp_of_string_literal
end

type char_literal = Literal.char_literal

include struct
  let _ = fun (_ : char_literal) -> ()

  let sexp_of_char_literal =
    (Literal.sexp_of_char_literal : char_literal -> S.t)

  let _ = sexp_of_char_literal
end

type byte_literal = Literal.byte_literal

include struct
  let _ = fun (_ : byte_literal) -> ()

  let sexp_of_byte_literal =
    (Literal.sexp_of_byte_literal : byte_literal -> S.t)

  let _ = sexp_of_byte_literal
end

type bytes_literal = Literal.bytes_literal

include struct
  let _ = fun (_ : bytes_literal) -> ()

  let sexp_of_bytes_literal =
    (Literal.sexp_of_bytes_literal : bytes_literal -> S.t)

  let _ = sexp_of_bytes_literal
end

type constant =
  | Const_bool of bool
  | Const_byte of byte_literal
  | Const_bytes of bytes_literal
  | Const_char of char_literal
  | Const_int of string
  | Const_int64 of string
  | Const_uint of string
  | Const_uint64 of string
  | Const_double of string
  | Const_string of string_literal
  | Const_bigint of string

include struct
  let _ = fun (_ : constant) -> ()

  let sexp_of_constant =
    (function
     | Const_bool arg0__001_ ->
         let res0__002_ = Moon_sexp_conv.sexp_of_bool arg0__001_ in
         S.List [ S.Atom "Const_bool"; res0__002_ ]
     | Const_byte arg0__003_ ->
         let res0__004_ = sexp_of_byte_literal arg0__003_ in
         S.List [ S.Atom "Const_byte"; res0__004_ ]
     | Const_bytes arg0__005_ ->
         let res0__006_ = sexp_of_bytes_literal arg0__005_ in
         S.List [ S.Atom "Const_bytes"; res0__006_ ]
     | Const_char arg0__007_ ->
         let res0__008_ = sexp_of_char_literal arg0__007_ in
         S.List [ S.Atom "Const_char"; res0__008_ ]
     | Const_int arg0__009_ ->
         let res0__010_ = Moon_sexp_conv.sexp_of_string arg0__009_ in
         S.List [ S.Atom "Const_int"; res0__010_ ]
     | Const_int64 arg0__011_ ->
         let res0__012_ = Moon_sexp_conv.sexp_of_string arg0__011_ in
         S.List [ S.Atom "Const_int64"; res0__012_ ]
     | Const_uint arg0__013_ ->
         let res0__014_ = Moon_sexp_conv.sexp_of_string arg0__013_ in
         S.List [ S.Atom "Const_uint"; res0__014_ ]
     | Const_uint64 arg0__015_ ->
         let res0__016_ = Moon_sexp_conv.sexp_of_string arg0__015_ in
         S.List [ S.Atom "Const_uint64"; res0__016_ ]
     | Const_double arg0__017_ ->
         let res0__018_ = Moon_sexp_conv.sexp_of_string arg0__017_ in
         S.List [ S.Atom "Const_double"; res0__018_ ]
     | Const_string arg0__019_ ->
         let res0__020_ = sexp_of_string_literal arg0__019_ in
         S.List [ S.Atom "Const_string"; res0__020_ ]
     | Const_bigint arg0__021_ ->
         let res0__022_ = Moon_sexp_conv.sexp_of_string arg0__021_ in
         S.List [ S.Atom "Const_bigint"; res0__022_ ]
      : constant -> S.t)

  let _ = sexp_of_constant
end

type longident = Basic_longident.t

include struct
  let _ = fun (_ : longident) -> ()
  let sexp_of_longident = (Basic_longident.sexp_of_t : longident -> S.t)
  let _ = sexp_of_longident
end

type docstring = Docstring.t

include struct
  let _ = fun (_ : docstring) -> ()
  let sexp_of_docstring = (Docstring.sexp_of_t : docstring -> S.t)
  let _ = sexp_of_docstring
end

type constrid_loc = {
  lid : Basic_longident.t;
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : constrid_loc) -> ()

  let sexp_of_constrid_loc =
    (let (drop_if__027_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { lid = lid__024_; loc_ = loc___028_ } ->
       let bnds__023_ = ([] : _ Stdlib.List.t) in
       let bnds__023_ =
         if drop_if__027_ loc___028_ then bnds__023_
         else
           let arg__030_ = sexp_of_location loc___028_ in
           let bnd__029_ = S.List [ S.Atom "loc_"; arg__030_ ] in
           (bnd__029_ :: bnds__023_ : _ Stdlib.List.t)
       in
       let bnds__023_ =
         let arg__025_ = Basic_longident.sexp_of_t lid__024_ in
         (S.List [ S.Atom "lid"; arg__025_ ] :: bnds__023_ : _ Stdlib.List.t)
       in
       S.List bnds__023_
      : constrid_loc -> S.t)

  let _ = sexp_of_constrid_loc
end

type label = {
  label_name : string;
  loc_ : location; [@sexp_drop_if hide_loc] [@ceh.ignore]
}

include struct
  let _ = fun (_ : label) -> ()

  let equal_label =
    (fun a__031_ b__032_ ->
       if Stdlib.( == ) a__031_ b__032_ then true
       else Stdlib.( = ) (a__031_.label_name : string) b__032_.label_name
      : label -> label -> bool)

  let _ = equal_label

  let sexp_of_label =
    (let (drop_if__037_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { label_name = label_name__034_; loc_ = loc___038_ } ->
       let bnds__033_ = ([] : _ Stdlib.List.t) in
       let bnds__033_ =
         if drop_if__037_ loc___038_ then bnds__033_
         else
           let arg__040_ = sexp_of_location loc___038_ in
           let bnd__039_ = S.List [ S.Atom "loc_"; arg__040_ ] in
           (bnd__039_ :: bnds__033_ : _ Stdlib.List.t)
       in
       let bnds__033_ =
         let arg__035_ = Moon_sexp_conv.sexp_of_string label_name__034_ in
         (S.List [ S.Atom "label_name"; arg__035_ ] :: bnds__033_
           : _ Stdlib.List.t)
       in
       S.List bnds__033_
      : label -> S.t)

  let _ = sexp_of_label
end

type accessor =
  | Label of label
  | Index of { tuple_index : int; loc_ : location [@sexp_drop_if hide_loc] }
  | Newtype

include struct
  let _ = fun (_ : accessor) -> ()

  let sexp_of_accessor =
    (let (drop_if__047_ : location -> Stdlib.Bool.t) = hide_loc in
     function
     | Label arg0__041_ ->
         let res0__042_ = sexp_of_label arg0__041_ in
         S.List [ S.Atom "Label"; res0__042_ ]
     | Index { tuple_index = tuple_index__044_; loc_ = loc___048_ } ->
         let bnds__043_ = ([] : _ Stdlib.List.t) in
         let bnds__043_ =
           if drop_if__047_ loc___048_ then bnds__043_
           else
             let arg__050_ = sexp_of_location loc___048_ in
             let bnd__049_ = S.List [ S.Atom "loc_"; arg__050_ ] in
             (bnd__049_ :: bnds__043_ : _ Stdlib.List.t)
         in
         let bnds__043_ =
           let arg__045_ = Moon_sexp_conv.sexp_of_int tuple_index__044_ in
           (S.List [ S.Atom "tuple_index"; arg__045_ ] :: bnds__043_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Index" :: bnds__043_)
     | Newtype -> S.Atom "Newtype"
      : accessor -> S.t)

  let _ = sexp_of_accessor
end

type constr_name = { name : string; loc_ : location [@sexp_drop_if hide_loc] }

include struct
  let _ = fun (_ : constr_name) -> ()

  let sexp_of_constr_name =
    (let (drop_if__055_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { name = name__052_; loc_ = loc___056_ } ->
       let bnds__051_ = ([] : _ Stdlib.List.t) in
       let bnds__051_ =
         if drop_if__055_ loc___056_ then bnds__051_
         else
           let arg__058_ = sexp_of_location loc___056_ in
           let bnd__057_ = S.List [ S.Atom "loc_"; arg__058_ ] in
           (bnd__057_ :: bnds__051_ : _ Stdlib.List.t)
       in
       let bnds__051_ =
         let arg__053_ = Moon_sexp_conv.sexp_of_string name__052_ in
         (S.List [ S.Atom "name"; arg__053_ ] :: bnds__051_ : _ Stdlib.List.t)
       in
       S.List bnds__051_
      : constr_name -> S.t)

  let _ = sexp_of_constr_name
end

type type_name = { name : longident; loc_ : location [@sexp_drop_if hide_loc] }

include struct
  let _ = fun (_ : type_name) -> ()

  let sexp_of_type_name =
    (let (drop_if__063_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { name = name__060_; loc_ = loc___064_ } ->
       let bnds__059_ = ([] : _ Stdlib.List.t) in
       let bnds__059_ =
         if drop_if__063_ loc___064_ then bnds__059_
         else
           let arg__066_ = sexp_of_location loc___064_ in
           let bnd__065_ = S.List [ S.Atom "loc_"; arg__066_ ] in
           (bnd__065_ :: bnds__059_ : _ Stdlib.List.t)
       in
       let bnds__059_ =
         let arg__061_ = sexp_of_longident name__060_ in
         (S.List [ S.Atom "name"; arg__061_ ] :: bnds__059_ : _ Stdlib.List.t)
       in
       S.List bnds__059_
      : type_name -> S.t)

  let _ = sexp_of_type_name
end

type constructor_extra_info =
  | Type_name of type_name
  | Package of string
  | No_extra_info

include struct
  let _ = fun (_ : constructor_extra_info) -> ()

  let sexp_of_constructor_extra_info =
    (function
     | Type_name arg0__067_ ->
         let res0__068_ = sexp_of_type_name arg0__067_ in
         S.List [ S.Atom "Type_name"; res0__068_ ]
     | Package arg0__069_ ->
         let res0__070_ = Moon_sexp_conv.sexp_of_string arg0__069_ in
         S.List [ S.Atom "Package"; res0__070_ ]
     | No_extra_info -> S.Atom "No_extra_info"
      : constructor_extra_info -> S.t)

  let _ = sexp_of_constructor_extra_info
end

type constructor = {
  constr_name : constr_name;
  extra_info : constructor_extra_info;
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : constructor) -> ()

  let sexp_of_constructor =
    (let (drop_if__077_ : location -> Stdlib.Bool.t) = hide_loc in
     fun {
           constr_name = constr_name__072_;
           extra_info = extra_info__074_;
           loc_ = loc___078_;
         } ->
       let bnds__071_ = ([] : _ Stdlib.List.t) in
       let bnds__071_ =
         if drop_if__077_ loc___078_ then bnds__071_
         else
           let arg__080_ = sexp_of_location loc___078_ in
           let bnd__079_ = S.List [ S.Atom "loc_"; arg__080_ ] in
           (bnd__079_ :: bnds__071_ : _ Stdlib.List.t)
       in
       let bnds__071_ =
         let arg__075_ = sexp_of_constructor_extra_info extra_info__074_ in
         (S.List [ S.Atom "extra_info"; arg__075_ ] :: bnds__071_
           : _ Stdlib.List.t)
       in
       let bnds__071_ =
         let arg__073_ = sexp_of_constr_name constr_name__072_ in
         (S.List [ S.Atom "constr_name"; arg__073_ ] :: bnds__071_
           : _ Stdlib.List.t)
       in
       S.List bnds__071_
      : constructor -> S.t)

  let _ = sexp_of_constructor
end

type binder = { binder_name : string; loc_ : location [@sexp_drop_if hide_loc] }

include struct
  let _ = fun (_ : binder) -> ()

  let sexp_of_binder =
    (let (drop_if__085_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { binder_name = binder_name__082_; loc_ = loc___086_ } ->
       let bnds__081_ = ([] : _ Stdlib.List.t) in
       let bnds__081_ =
         if drop_if__085_ loc___086_ then bnds__081_
         else
           let arg__088_ = sexp_of_location loc___086_ in
           let bnd__087_ = S.List [ S.Atom "loc_"; arg__088_ ] in
           (bnd__087_ :: bnds__081_ : _ Stdlib.List.t)
       in
       let bnds__081_ =
         let arg__083_ = Moon_sexp_conv.sexp_of_string binder_name__082_ in
         (S.List [ S.Atom "binder_name"; arg__083_ ] :: bnds__081_
           : _ Stdlib.List.t)
       in
       S.List bnds__081_
      : binder -> S.t)

  let _ = sexp_of_binder
end

type tvar_constraint = {
  tvc_trait : longident;
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : tvar_constraint) -> ()

  let sexp_of_tvar_constraint =
    (let (drop_if__093_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { tvc_trait = tvc_trait__090_; loc_ = loc___094_ } ->
       let bnds__089_ = ([] : _ Stdlib.List.t) in
       let bnds__089_ =
         if drop_if__093_ loc___094_ then bnds__089_
         else
           let arg__096_ = sexp_of_location loc___094_ in
           let bnd__095_ = S.List [ S.Atom "loc_"; arg__096_ ] in
           (bnd__095_ :: bnds__089_ : _ Stdlib.List.t)
       in
       let bnds__089_ =
         let arg__091_ = sexp_of_longident tvc_trait__090_ in
         (S.List [ S.Atom "tvc_trait"; arg__091_ ] :: bnds__089_
           : _ Stdlib.List.t)
       in
       S.List bnds__089_
      : tvar_constraint -> S.t)

  let _ = sexp_of_tvar_constraint
end

type type_decl_binder = {
  tvar_name : string option;
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : type_decl_binder) -> ()

  let sexp_of_type_decl_binder =
    (let (drop_if__101_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { tvar_name = tvar_name__098_; loc_ = loc___102_ } ->
       let bnds__097_ = ([] : _ Stdlib.List.t) in
       let bnds__097_ =
         if drop_if__101_ loc___102_ then bnds__097_
         else
           let arg__104_ = sexp_of_location loc___102_ in
           let bnd__103_ = S.List [ S.Atom "loc_"; arg__104_ ] in
           (bnd__103_ :: bnds__097_ : _ Stdlib.List.t)
       in
       let bnds__097_ =
         let arg__099_ =
           Moon_sexp_conv.sexp_of_option Moon_sexp_conv.sexp_of_string
             tvar_name__098_
         in
         (S.List [ S.Atom "tvar_name"; arg__099_ ] :: bnds__097_
           : _ Stdlib.List.t)
       in
       S.List bnds__097_
      : type_decl_binder -> S.t)

  let _ = sexp_of_type_decl_binder
end

type tvar_binder = {
  tvar_name : string;
  tvar_constraints : tvar_constraint list; [@list]
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : tvar_binder) -> ()

  let sexp_of_tvar_binder =
    (let (drop_if__113_ : location -> Stdlib.Bool.t) = hide_loc in
     fun {
           tvar_name = tvar_name__106_;
           tvar_constraints = tvar_constraints__109_;
           loc_ = loc___114_;
         } ->
       let bnds__105_ = ([] : _ Stdlib.List.t) in
       let bnds__105_ =
         if drop_if__113_ loc___114_ then bnds__105_
         else
           let arg__116_ = sexp_of_location loc___114_ in
           let bnd__115_ = S.List [ S.Atom "loc_"; arg__116_ ] in
           (bnd__115_ :: bnds__105_ : _ Stdlib.List.t)
       in
       let bnds__105_ =
         if match tvar_constraints__109_ with [] -> true | _ -> false then
           bnds__105_
         else
           let arg__111_ =
             (Moon_sexp_conv.sexp_of_list sexp_of_tvar_constraint)
               tvar_constraints__109_
           in
           let bnd__110_ = S.List [ S.Atom "tvar_constraints"; arg__111_ ] in
           (bnd__110_ :: bnds__105_ : _ Stdlib.List.t)
       in
       let bnds__105_ =
         let arg__107_ = Moon_sexp_conv.sexp_of_string tvar_name__106_ in
         (S.List [ S.Atom "tvar_name"; arg__107_ ] :: bnds__105_
           : _ Stdlib.List.t)
       in
       S.List bnds__105_
      : tvar_binder -> S.t)

  let _ = sexp_of_tvar_binder
end

type test_name = string_literal Rloc.loced option

include struct
  let _ = fun (_ : test_name) -> ()

  let sexp_of_test_name =
    (fun x__117_ ->
       Moon_sexp_conv.sexp_of_option
         (Rloc.sexp_of_loced sexp_of_string_literal)
         x__117_
      : test_name -> S.t)

  let _ = sexp_of_test_name
end

type hole = Synthesized | Incomplete | Todo

include struct
  let _ = fun (_ : hole) -> ()

  let sexp_of_hole =
    (function
     | Synthesized -> S.Atom "Synthesized"
     | Incomplete -> S.Atom "Incomplete"
     | Todo -> S.Atom "Todo"
      : hole -> S.t)

  let _ = sexp_of_hole
end

let sexp_of_tvar_binder tvb =
  if tvb.tvar_constraints = [] then Moon_sexp_conv.sexp_of_string tvb.tvar_name
  else sexp_of_tvar_binder tvb

let sexp_of_type_decl_binder (tvb : type_decl_binder) =
  match tvb.tvar_name with
  | Some name -> Moon_sexp_conv.sexp_of_string name
  | None -> S.Atom "_"

type var = { var_name : longident; loc_ : location [@sexp_drop_if hide_loc] }

include struct
  let _ = fun (_ : var) -> ()

  let sexp_of_var =
    (let (drop_if__122_ : location -> Stdlib.Bool.t) = hide_loc in
     fun { var_name = var_name__119_; loc_ = loc___123_ } ->
       let bnds__118_ = ([] : _ Stdlib.List.t) in
       let bnds__118_ =
         if drop_if__122_ loc___123_ then bnds__118_
         else
           let arg__125_ = sexp_of_location loc___123_ in
           let bnd__124_ = S.List [ S.Atom "loc_"; arg__125_ ] in
           (bnd__124_ :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__120_ = sexp_of_longident var_name__119_ in
         (S.List [ S.Atom "var_name"; arg__120_ ] :: bnds__118_
           : _ Stdlib.List.t)
       in
       S.List bnds__118_
      : var -> S.t)

  let _ = sexp_of_var
end

let sexp_of_constant (c : constant) =
  match c with
  | Const_bool b -> Moon_sexp_conv.sexp_of_bool b
  | Const_byte b -> S.Atom b.byte_repr
  | Const_bytes lit -> S.Atom lit.bytes_repr
  | Const_char c -> Basic_uchar_utils.sexp_of_uchar c.char_val
  | Const_double f -> S.Atom f
  | Const_int s | Const_int64 s | Const_uint s | Const_uint64 s -> S.Atom s
  | Const_string s -> Moon_sexp_conv.sexp_of_string s.string_val
  | Const_bigint s -> Moon_sexp_conv.sexp_of_string s

type argument_kind =
  | Positional
  | Labelled of label
  | Labelled_pun of label
  | Labelled_option of { label : label; question_loc : location }
  | Labelled_option_pun of { label : label; question_loc : location }

include struct
  let _ = fun (_ : argument_kind) -> ()

  let sexp_of_argument_kind =
    (function
     | Positional -> S.Atom "Positional"
     | Labelled arg0__126_ ->
         let res0__127_ = sexp_of_label arg0__126_ in
         S.List [ S.Atom "Labelled"; res0__127_ ]
     | Labelled_pun arg0__128_ ->
         let res0__129_ = sexp_of_label arg0__128_ in
         S.List [ S.Atom "Labelled_pun"; res0__129_ ]
     | Labelled_option
         { label = label__131_; question_loc = question_loc__133_ } ->
         let bnds__130_ = ([] : _ Stdlib.List.t) in
         let bnds__130_ =
           let arg__134_ = sexp_of_location question_loc__133_ in
           (S.List [ S.Atom "question_loc"; arg__134_ ] :: bnds__130_
             : _ Stdlib.List.t)
         in
         let bnds__130_ =
           let arg__132_ = sexp_of_label label__131_ in
           (S.List [ S.Atom "label"; arg__132_ ] :: bnds__130_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Labelled_option" :: bnds__130_)
     | Labelled_option_pun
         { label = label__136_; question_loc = question_loc__138_ } ->
         let bnds__135_ = ([] : _ Stdlib.List.t) in
         let bnds__135_ =
           let arg__139_ = sexp_of_location question_loc__138_ in
           (S.List [ S.Atom "question_loc"; arg__139_ ] :: bnds__135_
             : _ Stdlib.List.t)
         in
         let bnds__135_ =
           let arg__137_ = sexp_of_label label__136_ in
           (S.List [ S.Atom "label"; arg__137_ ] :: bnds__135_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Labelled_option_pun" :: bnds__135_)
      : argument_kind -> S.t)

  let _ = sexp_of_argument_kind
end

type fn_kind = Lambda | Matrix

include struct
  let _ = fun (_ : fn_kind) -> ()

  let sexp_of_fn_kind =
    (function Lambda -> S.Atom "Lambda" | Matrix -> S.Atom "Matrix"
      : fn_kind -> S.t)

  let _ = sexp_of_fn_kind
end

type group = Group_brace | Group_paren

include struct
  let _ = fun (_ : group) -> ()

  let sexp_of_group =
    (function
     | Group_brace -> S.Atom "Group_brace" | Group_paren -> S.Atom "Group_paren"
      : group -> S.t)

  let _ = sexp_of_group
end

type trailing_mark = Trailing_comma | Trailing_semi | Trailing_none

include struct
  let _ = fun (_ : trailing_mark) -> ()

  let sexp_of_trailing_mark =
    (function
     | Trailing_comma -> S.Atom "Trailing_comma"
     | Trailing_semi -> S.Atom "Trailing_semi"
     | Trailing_none -> S.Atom "Trailing_none"
      : trailing_mark -> S.t)

  let _ = sexp_of_trailing_mark
end

type apply_attr = No_attr | Exclamation | Question

include struct
  let _ = fun (_ : apply_attr) -> ()

  let sexp_of_apply_attr =
    (function
     | No_attr -> S.Atom "No_attr"
     | Exclamation -> S.Atom "Exclamation"
     | Question -> S.Atom "Question"
      : apply_attr -> S.t)

  let _ = sexp_of_apply_attr
end

include struct
  class ['a] iterbase =
    object
      method visit_longident : 'a -> longident -> unit = fun _ _ -> ()
      method visit_label : 'a -> label -> unit = fun _ _ -> ()
      method visit_accessor : 'a -> accessor -> unit = fun _ _ -> ()
      method visit_constructor : 'a -> constructor -> unit = fun _ _ -> ()
      method visit_constr_name : 'a -> constr_name -> unit = fun _ _ -> ()
      method visit_var : 'a -> var -> unit = fun _ _ -> ()
      method visit_binder : 'a -> binder -> unit = fun _ _ -> ()

      method visit_tvar_constraint : 'a -> tvar_constraint -> unit =
        fun _ _ -> ()

      method visit_tvar_binder : 'a -> tvar_binder -> unit = fun _ _ -> ()

      method visit_type_decl_binder : 'a -> type_decl_binder -> unit =
        fun _ _ -> ()

      method visit_constrid_loc : 'a -> constrid_loc -> unit = fun _ _ -> ()
      method visit_argument_kind : 'a -> argument_kind -> unit = fun _ _ -> ()
      method visit_fn_kind : 'a -> fn_kind -> unit = fun _ _ -> ()
      method visit_type_name : 'a -> type_name -> unit = fun _ _ -> ()

      method private visit_string_literal : 'a -> string_literal -> unit =
        fun _ _ -> ()

      method private visit_docstring : 'a -> docstring -> unit = fun _ _ -> ()

      method private visit_interp_source : 'a -> Literal.interp_source -> unit =
        fun _ _ -> ()

      method private visit_hole : 'a -> hole -> unit = fun _ _ -> ()
    end

  class ['a] mapbase =
    object
      method visit_longident : 'a -> longident -> longident = fun _ e -> e
      method visit_label : 'a -> label -> label = fun _ e -> e
      method visit_accessor : 'a -> accessor -> accessor = fun _ e -> e
      method visit_constructor : 'a -> constructor -> constructor = fun _ e -> e
      method visit_constr_name : 'a -> constr_name -> constr_name = fun _ e -> e
      method visit_var : 'a -> var -> var = fun _ e -> e
      method visit_binder : 'a -> binder -> binder = fun _ e -> e

      method visit_tvar_constraint : 'a -> tvar_constraint -> tvar_constraint =
        fun _ e -> e

      method visit_type_decl_binder : 'a -> type_decl_binder -> type_decl_binder
          =
        fun _ e -> e

      method visit_tvar_binder : 'a -> tvar_binder -> tvar_binder = fun _ e -> e

      method visit_constrid_loc : 'a -> constrid_loc -> constrid_loc =
        fun _ e -> e

      method visit_argument_kind : 'a -> argument_kind -> argument_kind =
        fun _ e -> e

      method visit_fn_kind : 'a -> fn_kind -> fn_kind = fun _ e -> e
      method visit_type_name : 'a -> type_name -> type_name = fun _ e -> e

      method private visit_string_literal
          : 'a -> string_literal -> string_literal =
        fun _ e -> e

      method private visit_docstring : 'a -> docstring -> docstring =
        fun _ e -> e

      method private visit_interp_source
          : 'a -> Literal.interp_source -> Literal.interp_source =
        fun _ e -> e

      method private visit_hole : 'a -> hole -> hole = fun _ e -> e
    end

  class ['a] sexpbase =
    object
      inherit [_] Sexp_visitors.sexp

      method visit_location : 'a -> location -> S.t =
        fun _ x -> sexp_of_location x

      method visit_absolute_loc : 'a -> absolute_loc -> S.t =
        fun _ x -> sexp_of_absolute_loc x

      method visit_constant : 'a -> constant -> S.t =
        fun _ x -> sexp_of_constant x

      method visit_longident : 'a -> longident -> S.t =
        fun _ x -> sexp_of_longident x

      method visit_label : 'a -> label -> S.t = fun _ x -> sexp_of_label x

      method visit_accessor : 'a -> accessor -> S.t =
        fun _ x -> sexp_of_accessor x

      method visit_constructor : 'a -> constructor -> S.t =
        fun _ x -> sexp_of_constructor x

      method visit_constr_name : 'a -> constr_name -> S.t =
        fun _ x -> sexp_of_constr_name x

      method visit_var : 'a -> var -> S.t = fun _ x -> sexp_of_var x
      method visit_binder : 'a -> binder -> S.t = fun _ x -> sexp_of_binder x

      method visit_tvar_constraint : 'a -> tvar_constraint -> S.t =
        fun _ x -> sexp_of_tvar_constraint x

      method visit_tvar_binder : 'a -> tvar_binder -> S.t =
        fun _ x -> sexp_of_tvar_binder x

      method visit_type_decl_binder : 'a -> type_decl_binder -> S.t =
        fun _ x -> sexp_of_type_decl_binder x

      method visit_constrid_loc : 'a -> constrid_loc -> S.t =
        fun _ x -> sexp_of_constrid_loc x

      method visit_argument_kind : 'a -> argument_kind -> S.t =
        fun _ x -> sexp_of_argument_kind x

      method visit_fn_kind : 'a -> fn_kind -> S.t = fun _ x -> sexp_of_fn_kind x

      method visit_type_name : 'a -> type_name -> S.t =
        fun _ x -> sexp_of_type_name x

      method visit_docstring : 'a -> docstring -> S.t =
        fun _ x -> sexp_of_docstring x

      method visit_string_literal : 'a -> string_literal -> S.t =
        fun _ x -> sexp_of_string_literal x

      method visit_apply_attr : 'a -> apply_attr -> S.t =
        fun _ x -> sexp_of_apply_attr x

      method private visit_test_name : 'a -> test_name -> S.t =
        fun _ x -> sexp_of_test_name x

      method private visit_group : 'a -> group -> S.t =
        fun _ x -> sexp_of_group x

      method private visit_trailing_mark : 'a -> trailing_mark -> S.t =
        fun _ x -> sexp_of_trailing_mark x

      method private visit_interp_source : 'a -> Literal.interp_source -> S.t =
        fun _ x -> Literal.sexp_of_interp_source x

      method private visit_hole : 'a -> hole -> S.t = fun _ x -> sexp_of_hole x
    end

  type expr =
    | Pexpr_apply of {
        func : expr;
        args : argument list;
        attr : apply_attr;
        loc_ : location;
      }
    | Pexpr_infix of { op : var; lhs : expr; rhs : expr; loc_ : location }
    | Pexpr_unary of { op : var; expr : expr; loc_ : location }
    | Pexpr_array of { exprs : expr list; loc_ : location }
    | Pexpr_array_spread of { elems : spreadable_elem list; loc_ : location }
    | Pexpr_array_get of { array : expr; index : expr; loc_ : location }
    | Pexpr_array_get_slice of {
        array : expr;
        start_index : expr option;
        end_index : expr option;
        index_loc_ : location;
        loc_ : location;
      }
    | Pexpr_array_set of {
        array : expr;
        index : expr;
        value : expr;
        loc_ : location;
      }
    | Pexpr_array_augmented_set of {
        op : var;
        array : expr;
        index : expr;
        value : expr;
        loc_ : location;
      }
    | Pexpr_constant of { c : constant; loc_ : location }
    | Pexpr_multiline_string of {
        elems : multiline_string_elem list;
        loc_ : location;
      }
    | Pexpr_interp of { elems : interp_elem list; loc_ : location }
    | Pexpr_constraint of { expr : expr; ty : typ; loc_ : location }
    | Pexpr_constr of { constr : constructor; loc_ : location }
    | Pexpr_while of {
        loop_cond : expr;
        loop_body : expr;
        while_else : expr option;
        loc_ : location;
      }
    | Pexpr_function of { func : func; loc_ : location }
    | Pexpr_ident of { id : var; loc_ : location }
    | Pexpr_if of {
        cond : expr;
        ifso : expr;
        ifnot : expr option;
        loc_ : location;
      }
    | Pexpr_guard of {
        cond : expr;
        otherwise : expr option;
        body : expr;
        loc_ : location;
      }
    | Pexpr_guard_let of {
        pat : pattern;
        expr : expr;
        otherwise : (pattern * expr) list option;
        body : expr;
        loc_ : location;
      }
    | Pexpr_letfn of {
        name : binder;
        func : func;
        body : expr;
        loc_ : location;
      }
    | Pexpr_letrec of {
        bindings : (binder * func) list;
        body : expr;
        loc_ : location;
      }
    | Pexpr_let of {
        pattern : pattern;
        expr : expr;
        body : expr;
        loc_ : location;
      }
    | Pexpr_sequence of { expr1 : expr; expr2 : expr; loc_ : location }
    | Pexpr_tuple of { exprs : expr list; loc_ : location }
    | Pexpr_record of {
        type_name : type_name option;
        fields : field_def list;
        trailing : trailing_mark;
        loc_ : location;
      }
    | Pexpr_record_update of {
        type_name : type_name option;
        record : expr;
        fields : field_def list;
        loc_ : location;
      }
    | Pexpr_field of { record : expr; accessor : accessor; loc_ : location }
    | Pexpr_method of {
        type_name : type_name;
        method_name : label;
        loc_ : location;
      }
    | Pexpr_dot_apply of {
        self : expr;
        method_name : label;
        args : argument list;
        return_self : bool;
        attr : apply_attr;
        loc_ : location;
      }
    | Pexpr_as of { expr : expr; trait : type_name; loc_ : location }
    | Pexpr_mutate of {
        record : expr;
        accessor : accessor;
        field : expr;
        augmented_by : var option;
        loc_ : location;
      }
    | Pexpr_match of {
        expr : expr;
        cases : (pattern * expr) list;
        match_loc_ : location;
        loc_ : location;
      }
    | Pexpr_letmut of {
        binder : binder;
        ty : typ option;
        expr : expr;
        body : expr;
        loc_ : location;
      }
    | Pexpr_pipe of { lhs : expr; rhs : expr; loc_ : location }
    | Pexpr_assign of {
        var : var;
        expr : expr;
        augmented_by : var option;
        loc_ : location;
      }
    | Pexpr_hole of { loc_ : location; kind : hole }
    | Pexpr_return of { return_value : expr option; loc_ : location }
    | Pexpr_raise of { err_value : expr; loc_ : location }
    | Pexpr_unit of { loc_ : location; faked : bool }
    | Pexpr_break of { arg : expr option; loc_ : location }
    | Pexpr_continue of { args : expr list; loc_ : location }
    | Pexpr_loop of {
        args : expr list;
        body : (pattern list * expr) list;
        loop_loc_ : location;
        loc_ : location;
      }
    | Pexpr_for of {
        binders : (binder * expr) list;
        condition : expr option;
        continue_block : (binder * expr) list;
        body : expr;
        for_else : expr option;
        loc_ : location;
      }
    | Pexpr_foreach of {
        binders : binder option list;
        expr : expr;
        body : expr;
        else_block : expr option;
        loc_ : location;
      }
    | Pexpr_try of {
        body : expr;
        catch : (pattern * expr) list;
        catch_all : bool;
        try_else : (pattern * expr) list option;
        try_loc_ : location;
        catch_loc_ : location;
        else_loc_ : location;
        loc_ : location;
      }
    | Pexpr_map of { elems : map_expr_elem list; loc_ : location }
    | Pexpr_group of { expr : expr; group : group; loc_ : location }
    | Pexpr_static_assert of { asserts : static_assertion list; body : expr }

  and static_assertion = {
    assert_type : typ;
    assert_trait : longident;
    assert_loc : location;
    assert_msg : string;
  }

  and argument = { arg_value : expr; arg_kind : argument_kind }
  and parameters = parameter list

  and parameter = {
    param_binder : binder;
    param_annot : typ option;
    param_kind : parameter_kind;
  }

  and parameter_kind =
    | Positional
    | Labelled
    | Optional of { default : expr }
    | Question_optional

  and func =
    | Lambda of {
        parameters : parameters;
        params_loc_ : location;
        body : expr;
        return_type : (typ * error_typ) option;
        kind_ : fn_kind;
        has_error : bool;
      }
    | Match of {
        cases : (pattern list * expr) list;
        has_error : bool;
        fn_loc_ : location;
        loc_ : location;
      }

  and spreadable_elem =
    | Elem_regular of expr
    | Elem_spread of { expr : expr; loc_ : location }

  and map_expr_elem =
    | Map_expr_elem of {
        key : constant;
        expr : expr;
        key_loc_ : location;
        loc_ : location;
      }

  and error_typ =
    | Error_typ of { ty : typ }
    | Default_error_typ of { loc_ : location }
    | No_error_typ

  and typ =
    | Ptype_any of { loc_ : location }
    | Ptype_arrow of {
        ty_arg : typ list;
        ty_res : typ;
        ty_err : error_typ;
        loc_ : location;
      }
    | Ptype_tuple of { tys : typ list; loc_ : location }
    | Ptype_name of {
        constr_id : constrid_loc;
        tys : typ list;
        loc_ : location;
      }
    | Ptype_option of { ty : typ; loc_ : location; question_loc : location }

  and pattern =
    | Ppat_alias of { pat : pattern; alias : binder; loc_ : location }
    | Ppat_any of { loc_ : location }
    | Ppat_array of { pats : array_pattern; loc_ : location }
    | Ppat_constant of { c : constant; loc_ : location }
    | Ppat_constraint of { pat : pattern; ty : typ; loc_ : location }
    | Ppat_constr of {
        constr : constructor;
        args : constr_pat_arg list option;
        is_open : bool;
        loc_ : location;
      }
    | Ppat_or of { pat1 : pattern; pat2 : pattern; loc_ : location }
    | Ppat_tuple of { pats : pattern list; loc_ : location }
    | Ppat_var of binder
    | Ppat_record of {
        fields : field_pat list;
        is_closed : bool;
        loc_ : location;
      }
    | Ppat_map of { elems : map_pat_elem list; loc_ : location }
    | Ppat_range of {
        lhs : pattern;
        rhs : pattern;
        inclusive : bool;
        loc_ : location;
      }

  and array_pattern =
    | Closed of pattern list
    | Open of pattern list * pattern list * binder option

  and field_def =
    | Field_def of {
        label : label;
        expr : expr;
        is_pun : bool;
        loc_ : location;
      }

  and field_pat =
    | Field_pat of {
        label : label;
        pattern : pattern;
        is_pun : bool;
        loc_ : location;
      }

  and constr_pat_arg =
    | Constr_pat_arg of { pat : pattern; kind : argument_kind }

  and map_pat_elem =
    | Map_pat_elem of {
        key : constant;
        pat : pattern;
        match_absent : bool;
        key_loc_ : location;
        loc_ : location;
      }

  and constr_param = {
    cparam_typ : typ;
    cparam_mut : bool;
    cparam_label : label option;
  }

  and constr_decl = {
    constr_name : constr_name;
    constr_args : constr_param list option;
    constr_loc_ : location;
  }

  and field_name = { label : string; loc_ : location }

  and field_decl = {
    field_name : field_name;
    field_ty : typ;
    field_mut : bool;
    field_vis : visibility;
    field_loc_ : location;
  }

  and exception_decl =
    | No_payload
    | Single_payload of typ
    | Enum_payload of constr_decl list

  and type_desc =
    | Ptd_abstract
    | Ptd_newtype of typ
    | Ptd_error of exception_decl
    | Ptd_variant of constr_decl list
    | Ptd_record of field_decl list
    | Ptd_alias of typ

  and type_decl = {
    tycon : string;
    tycon_loc_ : location;
    params : type_decl_binder list;
    components : type_desc;
    mutable doc_ : docstring;
    type_vis : visibility;
    deriving_ : deriving_directive list;
    loc_ : absolute_loc;
  }

  and local_type_decl = {
    local_tycon : string;
    local_tycon_loc_ : location; [@dead "local_type_decl.local_tycon_loc_"]
    local_components : type_desc;
  }

  and deriving_directive = {
    type_name_ : type_name;
    args : argument list;
    loc_ : location;
  }

  and visibility =
    | Vis_default
    | Vis_pub of { attr : string option; loc_ : location }
    | Vis_priv of { loc_ : location }

  and func_stubs =
    | Import of { module_name : string_literal; func_name : string_literal }
    | Embedded of { language : string_literal option; code : embedded_code }

  and embedded_code =
    | Code_string of string_literal
    | Code_multiline_string of string list

  and decl_body =
    | Decl_body of { local_types : local_type_decl list; expr : expr }
    | Decl_stubs of func_stubs

  and fun_decl = {
    type_name : type_name option;
    name : binder;
    has_error : bool;
    decl_params : parameters option;
    params_loc_ : location;
    quantifiers : tvar_binder list;
    return_type : (typ * error_typ) option;
    is_pub : bool;
    mutable doc_ : docstring;
  }

  and trait_method_param = { tmparam_typ : typ; tmparam_label : label option }

  and trait_method_decl =
    | Trait_method of {
        name : binder;
        has_error : bool;
        quantifiers : tvar_binder list;
        params : trait_method_param list;
        return_type : (typ * error_typ) option;
        loc_ : location;
      }

  and trait_decl = {
    trait_name : binder;
    trait_supers : tvar_constraint list;
    trait_methods : trait_method_decl list;
    trait_vis : visibility;
    trait_loc_ : absolute_loc;
    mutable trait_doc_ : docstring;
  }

  and impl =
    | Ptop_expr of {
        expr : expr;
        is_main : bool;
        local_types : local_type_decl list;
        loc_ : absolute_loc;
      }
    | Ptop_test of {
        expr : expr;
        name : test_name;
        params : parameters option;
        local_types : local_type_decl list;
        loc_ : absolute_loc;
      }
    | Ptop_typedef of type_decl
    | Ptop_funcdef of {
        fun_decl : fun_decl;
        decl_body : decl_body;
        loc_ : absolute_loc;
      }
    | Ptop_letdef of {
        binder : binder;
        ty : typ option;
        expr : expr;
        is_pub : bool;
        is_constant : bool;
        loc_ : absolute_loc;
        mutable doc_ : docstring;
      }
    | Ptop_trait of trait_decl
    | Ptop_impl of {
        self_ty : typ option;
        trait : type_name;
        method_name : binder;
        has_error : bool;
        quantifiers : tvar_binder list;
        params : parameters;
        ret_ty : (typ * error_typ) option;
        body : expr;
        is_pub : bool;
        local_types : local_type_decl list;
        loc_ : absolute_loc;
        header_loc_ : location;
        mutable doc_ : docstring;
      }
    | Ptop_impl_relation of {
        self_ty : typ;
        trait : type_name;
        quantifiers : tvar_binder list;
        is_pub : bool;
        loc_ : absolute_loc;
      } [@dead "impl.Ptop_impl_relation"]

  and interp_elem =
    | Interp_lit of { str : string; repr : string; loc_ : location }
    | Interp_expr of { expr : expr; loc_ : location }
    | Interp_source of Literal.interp_source

  and multiline_string_elem =
    | Multiline_string of string
    | Multiline_interp of interp_elem list

  and impls = impl list

  include struct
    [@@@ocaml.warning "-4-26-27"]
    [@@@VISITORS.BEGIN]

    class virtual ['self] sexp =
      object (self : 'self)
        inherit [_] sexpbase

        method visit_Pexpr_apply
            : _ -> expr -> argument list -> apply_attr -> location -> S.t =
          fun env _visitors_ffunc _visitors_fargs _visitors_fattr
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ffunc in
            let _visitors_r1 =
              self#visit_list self#visit_argument env _visitors_fargs
            in
            let _visitors_r2 = self#visit_apply_attr env _visitors_fattr in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_apply"
              [
                ("func", _visitors_r0);
                ("args", _visitors_r1);
                ("attr", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_infix : _ -> var -> expr -> expr -> location -> S.t =
          fun env _visitors_fop _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_flhs in
            let _visitors_r2 = self#visit_expr env _visitors_frhs in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_infix"
              [
                ("op", _visitors_r0);
                ("lhs", _visitors_r1);
                ("rhs", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_unary : _ -> var -> expr -> location -> S.t =
          fun env _visitors_fop _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_unary"
              [
                ("op", _visitors_r0);
                ("expr", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_array : _ -> expr list -> location -> S.t =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_expr env _visitors_fexprs
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array"
              [ ("exprs", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_array_spread
            : _ -> spreadable_elem list -> location -> S.t =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_spreadable_elem env _visitors_felems
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array_spread"
              [ ("elems", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_array_get : _ -> expr -> expr -> location -> S.t =
          fun env _visitors_farray _visitors_findex _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array_get"
              [
                ("array", _visitors_r0);
                ("index", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_array_get_slice
            : _ ->
              expr ->
              expr option ->
              expr option ->
              location ->
              location ->
              S.t =
          fun env _visitors_farray _visitors_fstart_index _visitors_fend_index
              _visitors_findex_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 =
              self#visit_option self#visit_expr env _visitors_fstart_index
            in
            let _visitors_r2 =
              self#visit_option self#visit_expr env _visitors_fend_index
            in
            let _visitors_r3 = self#visit_location env _visitors_findex_loc_ in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array_get_slice"
              [
                ("array", _visitors_r0);
                ("start_index", _visitors_r1);
                ("end_index", _visitors_r2);
                ("index_loc_", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_array_set
            : _ -> expr -> expr -> expr -> location -> S.t =
          fun env _visitors_farray _visitors_findex _visitors_fvalue
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 = self#visit_expr env _visitors_fvalue in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array_set"
              [
                ("array", _visitors_r0);
                ("index", _visitors_r1);
                ("value", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_array_augmented_set
            : _ -> var -> expr -> expr -> expr -> location -> S.t =
          fun env _visitors_fop _visitors_farray _visitors_findex
              _visitors_fvalue _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_farray in
            let _visitors_r2 = self#visit_expr env _visitors_findex in
            let _visitors_r3 = self#visit_expr env _visitors_fvalue in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_array_augmented_set"
              [
                ("op", _visitors_r0);
                ("array", _visitors_r1);
                ("index", _visitors_r2);
                ("value", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_constant : _ -> constant -> location -> S.t =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 = self#visit_constant env _visitors_fc in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_constant"
              [ ("c", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_multiline_string
            : _ -> multiline_string_elem list -> location -> S.t =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_multiline_string_elem env
                _visitors_felems
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_multiline_string"
              [ ("elems", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_interp : _ -> interp_elem list -> location -> S.t =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_interp_elem env _visitors_felems
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_interp"
              [ ("elems", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_constraint : _ -> expr -> typ -> location -> S.t =
          fun env _visitors_fexpr _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_constraint"
              [
                ("expr", _visitors_r0);
                ("ty", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_constr : _ -> constructor -> location -> S.t =
          fun env _visitors_fconstr _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_constr"
              [ ("constr", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_while
            : _ -> expr -> expr -> expr option -> location -> S.t =
          fun env _visitors_floop_cond _visitors_floop_body
              _visitors_fwhile_else _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_floop_cond in
            let _visitors_r1 = self#visit_expr env _visitors_floop_body in
            let _visitors_r2 =
              self#visit_option self#visit_expr env _visitors_fwhile_else
            in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_while"
              [
                ("loop_cond", _visitors_r0);
                ("loop_body", _visitors_r1);
                ("while_else", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_function : _ -> func -> location -> S.t =
          fun env _visitors_ffunc _visitors_floc_ ->
            let _visitors_r0 = self#visit_func env _visitors_ffunc in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_function"
              [ ("func", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_ident : _ -> var -> location -> S.t =
          fun env _visitors_fid _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fid in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_ident"
              [ ("id", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_if
            : _ -> expr -> expr -> expr option -> location -> S.t =
          fun env _visitors_fcond _visitors_fifso _visitors_fifnot
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 = self#visit_expr env _visitors_fifso in
            let _visitors_r2 =
              self#visit_option self#visit_expr env _visitors_fifnot
            in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_if"
              [
                ("cond", _visitors_r0);
                ("ifso", _visitors_r1);
                ("ifnot", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_guard
            : _ -> expr -> expr option -> expr -> location -> S.t =
          fun env _visitors_fcond _visitors_fotherwise _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 =
              self#visit_option self#visit_expr env _visitors_fotherwise
            in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_guard"
              [
                ("cond", _visitors_r0);
                ("otherwise", _visitors_r1);
                ("body", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_guard_let
            : _ ->
              pattern ->
              expr ->
              (pattern * expr) list option ->
              expr ->
              location ->
              S.t =
          fun env _visitors_fpat _visitors_fexpr _visitors_fotherwise
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              self#visit_option
                (self#visit_list (fun env (_visitors_c0, _visitors_c1) ->
                     let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                     let _visitors_r1 = self#visit_expr env _visitors_c1 in
                     self#visit_tuple env [ _visitors_r0; _visitors_r1 ]))
                env _visitors_fotherwise
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_guard_let"
              [
                ("pat", _visitors_r0);
                ("expr", _visitors_r1);
                ("otherwise", _visitors_r2);
                ("body", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_letfn
            : _ -> binder -> func -> expr -> location -> S.t =
          fun env _visitors_fname _visitors_ffunc _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 = self#visit_func env _visitors_ffunc in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_letfn"
              [
                ("name", _visitors_r0);
                ("func", _visitors_r1);
                ("body", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_letrec
            : _ -> (binder * func) list -> expr -> location -> S.t =
          fun env _visitors_fbindings _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_binder env _visitors_c0 in
                  let _visitors_r1 = self#visit_func env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fbindings
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_letrec"
              [
                ("bindings", _visitors_r0);
                ("body", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_let : _ -> pattern -> expr -> expr -> location -> S.t
            =
          fun env _visitors_fpattern _visitors_fexpr _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_let"
              [
                ("pattern", _visitors_r0);
                ("expr", _visitors_r1);
                ("body", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_sequence : _ -> expr -> expr -> location -> S.t =
          fun env _visitors_fexpr1 _visitors_fexpr2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr1 in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr2 in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_sequence"
              [
                ("expr1", _visitors_r0);
                ("expr2", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_tuple : _ -> expr list -> location -> S.t =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_expr env _visitors_fexprs
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_tuple"
              [ ("exprs", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_record
            : _ ->
              type_name option ->
              field_def list ->
              trailing_mark ->
              location ->
              S.t =
          fun env _visitors_ftype_name _visitors_ffields _visitors_ftrailing
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_type_name env _visitors_ftype_name
            in
            let _visitors_r1 =
              self#visit_list self#visit_field_def env _visitors_ffields
            in
            let _visitors_r2 =
              self#visit_trailing_mark env _visitors_ftrailing
            in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_record"
              [
                ("type_name", _visitors_r0);
                ("fields", _visitors_r1);
                ("trailing", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_record_update
            : _ -> type_name option -> expr -> field_def list -> location -> S.t
            =
          fun env _visitors_ftype_name _visitors_frecord _visitors_ffields
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_type_name env _visitors_ftype_name
            in
            let _visitors_r1 = self#visit_expr env _visitors_frecord in
            let _visitors_r2 =
              self#visit_list self#visit_field_def env _visitors_ffields
            in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_record_update"
              [
                ("type_name", _visitors_r0);
                ("record", _visitors_r1);
                ("fields", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_field : _ -> expr -> accessor -> location -> S.t =
          fun env _visitors_frecord _visitors_faccessor _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_field"
              [
                ("record", _visitors_r0);
                ("accessor", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_method : _ -> type_name -> label -> location -> S.t =
          fun env _visitors_ftype_name _visitors_fmethod_name _visitors_floc_ ->
            let _visitors_r0 = self#visit_type_name env _visitors_ftype_name in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_method"
              [
                ("type_name", _visitors_r0);
                ("method_name", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_dot_apply
            : _ ->
              expr ->
              label ->
              argument list ->
              bool ->
              apply_attr ->
              location ->
              S.t =
          fun env _visitors_fself _visitors_fmethod_name _visitors_fargs
              _visitors_freturn_self _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fself in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 =
              self#visit_list self#visit_argument env _visitors_fargs
            in
            let _visitors_r3 = self#visit_bool env _visitors_freturn_self in
            let _visitors_r4 = self#visit_apply_attr env _visitors_fattr in
            let _visitors_r5 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_dot_apply"
              [
                ("self", _visitors_r0);
                ("method_name", _visitors_r1);
                ("args", _visitors_r2);
                ("return_self", _visitors_r3);
                ("attr", _visitors_r4);
                ("loc_", _visitors_r5);
              ]

        method visit_Pexpr_as : _ -> expr -> type_name -> location -> S.t =
          fun env _visitors_fexpr _visitors_ftrait _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_as"
              [
                ("expr", _visitors_r0);
                ("trait", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_mutate
            : _ -> expr -> accessor -> expr -> var option -> location -> S.t =
          fun env _visitors_frecord _visitors_faccessor _visitors_ffield
              _visitors_faugmented_by _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 = self#visit_expr env _visitors_ffield in
            let _visitors_r3 =
              self#visit_option self#visit_var env _visitors_faugmented_by
            in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_mutate"
              [
                ("record", _visitors_r0);
                ("accessor", _visitors_r1);
                ("field", _visitors_r2);
                ("augmented_by", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_match
            : _ -> expr -> (pattern * expr) list -> location -> location -> S.t
            =
          fun env _visitors_fexpr _visitors_fcases _visitors_fmatch_loc_
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fcases
            in
            let _visitors_r2 = self#visit_location env _visitors_fmatch_loc_ in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_match"
              [
                ("expr", _visitors_r0);
                ("cases", _visitors_r1);
                ("match_loc_", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_letmut
            : _ -> binder -> typ option -> expr -> expr -> location -> S.t =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              self#visit_option self#visit_typ env _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_letmut"
              [
                ("binder", _visitors_r0);
                ("ty", _visitors_r1);
                ("expr", _visitors_r2);
                ("body", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_pipe : _ -> expr -> expr -> location -> S.t =
          fun env _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_flhs in
            let _visitors_r1 = self#visit_expr env _visitors_frhs in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_pipe"
              [
                ("lhs", _visitors_r0);
                ("rhs", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_assign
            : _ -> var -> expr -> var option -> location -> S.t =
          fun env _visitors_fvar _visitors_fexpr _visitors_faugmented_by
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fvar in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              self#visit_option self#visit_var env _visitors_faugmented_by
            in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_assign"
              [
                ("var", _visitors_r0);
                ("expr", _visitors_r1);
                ("augmented_by", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_hole : _ -> location -> hole -> S.t =
          fun env _visitors_floc_ _visitors_fkind ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            let _visitors_r1 = self#visit_hole env _visitors_fkind in
            self#visit_inline_record env "Pexpr_hole"
              [ ("loc_", _visitors_r0); ("kind", _visitors_r1) ]

        method visit_Pexpr_return : _ -> expr option -> location -> S.t =
          fun env _visitors_freturn_value _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_expr env _visitors_freturn_value
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_return"
              [ ("return_value", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_raise : _ -> expr -> location -> S.t =
          fun env _visitors_ferr_value _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ferr_value in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_raise"
              [ ("err_value", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_unit : _ -> location -> bool -> S.t =
          fun env _visitors_floc_ _visitors_ffaked ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            let _visitors_r1 = self#visit_bool env _visitors_ffaked in
            self#visit_inline_record env "Pexpr_unit"
              [ ("loc_", _visitors_r0); ("faked", _visitors_r1) ]

        method visit_Pexpr_break : _ -> expr option -> location -> S.t =
          fun env _visitors_farg _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_expr env _visitors_farg
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_break"
              [ ("arg", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_continue : _ -> expr list -> location -> S.t =
          fun env _visitors_fargs _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_expr env _visitors_fargs
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_continue"
              [ ("args", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_loop
            : _ ->
              expr list ->
              (pattern list * expr) list ->
              location ->
              location ->
              S.t =
          fun env _visitors_fargs _visitors_fbody _visitors_floop_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_expr env _visitors_fargs
            in
            let _visitors_r1 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 =
                    self#visit_list self#visit_pattern env _visitors_c0
                  in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fbody
            in
            let _visitors_r2 = self#visit_location env _visitors_floop_loc_ in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_loop"
              [
                ("args", _visitors_r0);
                ("body", _visitors_r1);
                ("loop_loc_", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Pexpr_for
            : _ ->
              (binder * expr) list ->
              expr option ->
              (binder * expr) list ->
              expr ->
              expr option ->
              location ->
              S.t =
          fun env _visitors_fbinders _visitors_fcondition
              _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_binder env _visitors_c0 in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fbinders
            in
            let _visitors_r1 =
              self#visit_option self#visit_expr env _visitors_fcondition
            in
            let _visitors_r2 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_binder env _visitors_c0 in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fcontinue_block
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 =
              self#visit_option self#visit_expr env _visitors_ffor_else
            in
            let _visitors_r5 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_for"
              [
                ("binders", _visitors_r0);
                ("condition", _visitors_r1);
                ("continue_block", _visitors_r2);
                ("body", _visitors_r3);
                ("for_else", _visitors_r4);
                ("loc_", _visitors_r5);
              ]

        method visit_Pexpr_foreach
            : _ ->
              binder option list ->
              expr ->
              expr ->
              expr option ->
              location ->
              S.t =
          fun env _visitors_fbinders _visitors_fexpr _visitors_fbody
              _visitors_felse_block _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list
                (self#visit_option self#visit_binder)
                env _visitors_fbinders
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              self#visit_option self#visit_expr env _visitors_felse_block
            in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_foreach"
              [
                ("binders", _visitors_r0);
                ("expr", _visitors_r1);
                ("body", _visitors_r2);
                ("else_block", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Pexpr_try
            : _ ->
              expr ->
              (pattern * expr) list ->
              bool ->
              (pattern * expr) list option ->
              location ->
              location ->
              location ->
              location ->
              S.t =
          fun env _visitors_fbody _visitors_fcatch _visitors_fcatch_all
              _visitors_ftry_else _visitors_ftry_loc_ _visitors_fcatch_loc_
              _visitors_felse_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fbody in
            let _visitors_r1 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fcatch
            in
            let _visitors_r2 = self#visit_bool env _visitors_fcatch_all in
            let _visitors_r3 =
              self#visit_option
                (self#visit_list (fun env (_visitors_c0, _visitors_c1) ->
                     let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                     let _visitors_r1 = self#visit_expr env _visitors_c1 in
                     self#visit_tuple env [ _visitors_r0; _visitors_r1 ]))
                env _visitors_ftry_else
            in
            let _visitors_r4 = self#visit_location env _visitors_ftry_loc_ in
            let _visitors_r5 = self#visit_location env _visitors_fcatch_loc_ in
            let _visitors_r6 = self#visit_location env _visitors_felse_loc_ in
            let _visitors_r7 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_try"
              [
                ("body", _visitors_r0);
                ("catch", _visitors_r1);
                ("catch_all", _visitors_r2);
                ("try_else", _visitors_r3);
                ("try_loc_", _visitors_r4);
                ("catch_loc_", _visitors_r5);
                ("else_loc_", _visitors_r6);
                ("loc_", _visitors_r7);
              ]

        method visit_Pexpr_map : _ -> map_expr_elem list -> location -> S.t =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_map_expr_elem env _visitors_felems
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_map"
              [ ("elems", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Pexpr_group : _ -> expr -> group -> location -> S.t =
          fun env _visitors_fexpr _visitors_fgroup _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_group env _visitors_fgroup in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Pexpr_group"
              [
                ("expr", _visitors_r0);
                ("group", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Pexpr_static_assert
            : _ -> static_assertion list -> expr -> S.t =
          fun env _visitors_fasserts _visitors_fbody ->
            let _visitors_r0 =
              self#visit_list self#visit_static_assertion env _visitors_fasserts
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            self#visit_inline_record env "Pexpr_static_assert"
              [ ("asserts", _visitors_r0); ("body", _visitors_r1) ]

        method visit_expr : _ -> expr -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Pexpr_apply
                {
                  func = _visitors_ffunc;
                  args = _visitors_fargs;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_apply env _visitors_ffunc _visitors_fargs
                  _visitors_fattr _visitors_floc_
            | Pexpr_infix
                {
                  op = _visitors_fop;
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_infix env _visitors_fop _visitors_flhs
                  _visitors_frhs _visitors_floc_
            | Pexpr_unary
                {
                  op = _visitors_fop;
                  expr = _visitors_fexpr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_unary env _visitors_fop _visitors_fexpr
                  _visitors_floc_
            | Pexpr_array { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_array env _visitors_fexprs _visitors_floc_
            | Pexpr_array_spread
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_array_spread env _visitors_felems
                  _visitors_floc_
            | Pexpr_array_get
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get env _visitors_farray _visitors_findex
                  _visitors_floc_
            | Pexpr_array_get_slice
                {
                  array = _visitors_farray;
                  start_index = _visitors_fstart_index;
                  end_index = _visitors_fend_index;
                  index_loc_ = _visitors_findex_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get_slice env _visitors_farray
                  _visitors_fstart_index _visitors_fend_index
                  _visitors_findex_loc_ _visitors_floc_
            | Pexpr_array_set
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_set env _visitors_farray _visitors_findex
                  _visitors_fvalue _visitors_floc_
            | Pexpr_array_augmented_set
                {
                  op = _visitors_fop;
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_augmented_set env _visitors_fop
                  _visitors_farray _visitors_findex _visitors_fvalue
                  _visitors_floc_
            | Pexpr_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constant env _visitors_fc _visitors_floc_
            | Pexpr_multiline_string
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_multiline_string env _visitors_felems
                  _visitors_floc_
            | Pexpr_interp { elems = _visitors_felems; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_interp env _visitors_felems _visitors_floc_
            | Pexpr_constraint
                {
                  expr = _visitors_fexpr;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_constraint env _visitors_fexpr _visitors_fty
                  _visitors_floc_
            | Pexpr_constr
                { constr = _visitors_fconstr; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constr env _visitors_fconstr _visitors_floc_
            | Pexpr_while
                {
                  loop_cond = _visitors_floop_cond;
                  loop_body = _visitors_floop_body;
                  while_else = _visitors_fwhile_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_while env _visitors_floop_cond
                  _visitors_floop_body _visitors_fwhile_else _visitors_floc_
            | Pexpr_function { func = _visitors_ffunc; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_function env _visitors_ffunc _visitors_floc_
            | Pexpr_ident { id = _visitors_fid; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_ident env _visitors_fid _visitors_floc_
            | Pexpr_if
                {
                  cond = _visitors_fcond;
                  ifso = _visitors_fifso;
                  ifnot = _visitors_fifnot;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_if env _visitors_fcond _visitors_fifso
                  _visitors_fifnot _visitors_floc_
            | Pexpr_guard
                {
                  cond = _visitors_fcond;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard env _visitors_fcond _visitors_fotherwise
                  _visitors_fbody _visitors_floc_
            | Pexpr_guard_let
                {
                  pat = _visitors_fpat;
                  expr = _visitors_fexpr;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard_let env _visitors_fpat _visitors_fexpr
                  _visitors_fotherwise _visitors_fbody _visitors_floc_
            | Pexpr_letfn
                {
                  name = _visitors_fname;
                  func = _visitors_ffunc;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letfn env _visitors_fname _visitors_ffunc
                  _visitors_fbody _visitors_floc_
            | Pexpr_letrec
                {
                  bindings = _visitors_fbindings;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letrec env _visitors_fbindings _visitors_fbody
                  _visitors_floc_
            | Pexpr_let
                {
                  pattern = _visitors_fpattern;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_let env _visitors_fpattern _visitors_fexpr
                  _visitors_fbody _visitors_floc_
            | Pexpr_sequence
                {
                  expr1 = _visitors_fexpr1;
                  expr2 = _visitors_fexpr2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_sequence env _visitors_fexpr1 _visitors_fexpr2
                  _visitors_floc_
            | Pexpr_tuple { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_tuple env _visitors_fexprs _visitors_floc_
            | Pexpr_record
                {
                  type_name = _visitors_ftype_name;
                  fields = _visitors_ffields;
                  trailing = _visitors_ftrailing;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record env _visitors_ftype_name
                  _visitors_ffields _visitors_ftrailing _visitors_floc_
            | Pexpr_record_update
                {
                  type_name = _visitors_ftype_name;
                  record = _visitors_frecord;
                  fields = _visitors_ffields;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record_update env _visitors_ftype_name
                  _visitors_frecord _visitors_ffields _visitors_floc_
            | Pexpr_field
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_field env _visitors_frecord _visitors_faccessor
                  _visitors_floc_
            | Pexpr_method
                {
                  type_name = _visitors_ftype_name;
                  method_name = _visitors_fmethod_name;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_method env _visitors_ftype_name
                  _visitors_fmethod_name _visitors_floc_
            | Pexpr_dot_apply
                {
                  self = _visitors_fself;
                  method_name = _visitors_fmethod_name;
                  args = _visitors_fargs;
                  return_self = _visitors_freturn_self;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_dot_apply env _visitors_fself
                  _visitors_fmethod_name _visitors_fargs _visitors_freturn_self
                  _visitors_fattr _visitors_floc_
            | Pexpr_as
                {
                  expr = _visitors_fexpr;
                  trait = _visitors_ftrait;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_as env _visitors_fexpr _visitors_ftrait
                  _visitors_floc_
            | Pexpr_mutate
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  field = _visitors_ffield;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_mutate env _visitors_frecord
                  _visitors_faccessor _visitors_ffield _visitors_faugmented_by
                  _visitors_floc_
            | Pexpr_match
                {
                  expr = _visitors_fexpr;
                  cases = _visitors_fcases;
                  match_loc_ = _visitors_fmatch_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_match env _visitors_fexpr _visitors_fcases
                  _visitors_fmatch_loc_ _visitors_floc_
            | Pexpr_letmut
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letmut env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fbody _visitors_floc_
            | Pexpr_pipe
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_pipe env _visitors_flhs _visitors_frhs
                  _visitors_floc_
            | Pexpr_assign
                {
                  var = _visitors_fvar;
                  expr = _visitors_fexpr;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_assign env _visitors_fvar _visitors_fexpr
                  _visitors_faugmented_by _visitors_floc_
            | Pexpr_hole { loc_ = _visitors_floc_; kind = _visitors_fkind } ->
                self#visit_Pexpr_hole env _visitors_floc_ _visitors_fkind
            | Pexpr_return
                {
                  return_value = _visitors_freturn_value;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_return env _visitors_freturn_value
                  _visitors_floc_
            | Pexpr_raise
                { err_value = _visitors_ferr_value; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_raise env _visitors_ferr_value _visitors_floc_
            | Pexpr_unit { loc_ = _visitors_floc_; faked = _visitors_ffaked } ->
                self#visit_Pexpr_unit env _visitors_floc_ _visitors_ffaked
            | Pexpr_break { arg = _visitors_farg; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_break env _visitors_farg _visitors_floc_
            | Pexpr_continue { args = _visitors_fargs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_continue env _visitors_fargs _visitors_floc_
            | Pexpr_loop
                {
                  args = _visitors_fargs;
                  body = _visitors_fbody;
                  loop_loc_ = _visitors_floop_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_loop env _visitors_fargs _visitors_fbody
                  _visitors_floop_loc_ _visitors_floc_
            | Pexpr_for
                {
                  binders = _visitors_fbinders;
                  condition = _visitors_fcondition;
                  continue_block = _visitors_fcontinue_block;
                  body = _visitors_fbody;
                  for_else = _visitors_ffor_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_for env _visitors_fbinders _visitors_fcondition
                  _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
                  _visitors_floc_
            | Pexpr_foreach
                {
                  binders = _visitors_fbinders;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  else_block = _visitors_felse_block;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_foreach env _visitors_fbinders _visitors_fexpr
                  _visitors_fbody _visitors_felse_block _visitors_floc_
            | Pexpr_try
                {
                  body = _visitors_fbody;
                  catch = _visitors_fcatch;
                  catch_all = _visitors_fcatch_all;
                  try_else = _visitors_ftry_else;
                  try_loc_ = _visitors_ftry_loc_;
                  catch_loc_ = _visitors_fcatch_loc_;
                  else_loc_ = _visitors_felse_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_try env _visitors_fbody _visitors_fcatch
                  _visitors_fcatch_all _visitors_ftry_else _visitors_ftry_loc_
                  _visitors_fcatch_loc_ _visitors_felse_loc_ _visitors_floc_
            | Pexpr_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_map env _visitors_felems _visitors_floc_
            | Pexpr_group
                {
                  expr = _visitors_fexpr;
                  group = _visitors_fgroup;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_group env _visitors_fexpr _visitors_fgroup
                  _visitors_floc_
            | Pexpr_static_assert
                { asserts = _visitors_fasserts; body = _visitors_fbody } ->
                self#visit_Pexpr_static_assert env _visitors_fasserts
                  _visitors_fbody

        method visit_static_assertion : _ -> static_assertion -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.assert_type in
            let _visitors_r1 =
              self#visit_longident env _visitors_this.assert_trait
            in
            let _visitors_r2 =
              self#visit_location env _visitors_this.assert_loc
            in
            let _visitors_r3 =
              self#visit_string env _visitors_this.assert_msg
            in
            self#visit_record env
              [
                ("assert_type", _visitors_r0);
                ("assert_trait", _visitors_r1);
                ("assert_loc", _visitors_r2);
                ("assert_msg", _visitors_r3);
              ]

        method visit_argument : _ -> argument -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_expr env _visitors_this.arg_value in
            let _visitors_r1 =
              self#visit_argument_kind env _visitors_this.arg_kind
            in
            self#visit_record env
              [ ("arg_value", _visitors_r0); ("arg_kind", _visitors_r1) ]

        method visit_parameters : _ -> parameters -> S.t =
          fun env -> self#visit_list self#visit_parameter env

        method visit_parameter : _ -> parameter -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.param_binder
            in
            let _visitors_r1 =
              self#visit_option self#visit_typ env _visitors_this.param_annot
            in
            let _visitors_r2 =
              self#visit_parameter_kind env _visitors_this.param_kind
            in
            self#visit_record env
              [
                ("param_binder", _visitors_r0);
                ("param_annot", _visitors_r1);
                ("param_kind", _visitors_r2);
              ]

        method visit_Positional : _ -> S.t =
          fun env -> self#visit_inline_tuple env "Positional" []

        method visit_Labelled : _ -> S.t =
          fun env -> self#visit_inline_tuple env "Labelled" []

        method visit_Optional : _ -> expr -> S.t =
          fun env _visitors_fdefault ->
            let _visitors_r0 = self#visit_expr env _visitors_fdefault in
            self#visit_inline_record env "Optional"
              [ ("default", _visitors_r0) ]

        method visit_Question_optional : _ -> S.t =
          fun env -> self#visit_inline_tuple env "Question_optional" []

        method visit_parameter_kind : _ -> parameter_kind -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Positional -> self#visit_Positional env
            | Labelled -> self#visit_Labelled env
            | Optional { default = _visitors_fdefault } ->
                self#visit_Optional env _visitors_fdefault
            | Question_optional -> self#visit_Question_optional env

        method visit_Lambda
            : _ ->
              parameters ->
              location ->
              expr ->
              (typ * error_typ) option ->
              fn_kind ->
              bool ->
              S.t =
          fun env _visitors_fparameters _visitors_fparams_loc_ _visitors_fbody
              _visitors_freturn_type _visitors_fkind_ _visitors_fhas_error ->
            let _visitors_r0 =
              self#visit_parameters env _visitors_fparameters
            in
            let _visitors_r1 = self#visit_location env _visitors_fparams_loc_ in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              self#visit_option
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_typ env _visitors_c0 in
                  let _visitors_r1 = self#visit_error_typ env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_freturn_type
            in
            let _visitors_r4 = self#visit_fn_kind env _visitors_fkind_ in
            let _visitors_r5 = self#visit_bool env _visitors_fhas_error in
            self#visit_inline_record env "Lambda"
              [
                ("parameters", _visitors_r0);
                ("params_loc_", _visitors_r1);
                ("body", _visitors_r2);
                ("return_type", _visitors_r3);
                ("kind_", _visitors_r4);
                ("has_error", _visitors_r5);
              ]

        method visit_Match
            : _ ->
              (pattern list * expr) list ->
              bool ->
              location ->
              location ->
              S.t =
          fun env _visitors_fcases _visitors_fhas_error _visitors_ffn_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 =
                    self#visit_list self#visit_pattern env _visitors_c0
                  in
                  let _visitors_r1 = self#visit_expr env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fcases
            in
            let _visitors_r1 = self#visit_bool env _visitors_fhas_error in
            let _visitors_r2 = self#visit_location env _visitors_ffn_loc_ in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Match"
              [
                ("cases", _visitors_r0);
                ("has_error", _visitors_r1);
                ("fn_loc_", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_func : _ -> func -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Lambda
                {
                  parameters = _visitors_fparameters;
                  params_loc_ = _visitors_fparams_loc_;
                  body = _visitors_fbody;
                  return_type = _visitors_freturn_type;
                  kind_ = _visitors_fkind_;
                  has_error = _visitors_fhas_error;
                } ->
                self#visit_Lambda env _visitors_fparameters
                  _visitors_fparams_loc_ _visitors_fbody _visitors_freturn_type
                  _visitors_fkind_ _visitors_fhas_error
            | Match
                {
                  cases = _visitors_fcases;
                  has_error = _visitors_fhas_error;
                  fn_loc_ = _visitors_ffn_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Match env _visitors_fcases _visitors_fhas_error
                  _visitors_ffn_loc_ _visitors_floc_

        method visit_Elem_regular : _ -> expr -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_expr env _visitors_c0 in
            self#visit_inline_tuple env "Elem_regular" [ _visitors_r0 ]

        method visit_Elem_spread : _ -> expr -> location -> S.t =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Elem_spread"
              [ ("expr", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_spreadable_elem : _ -> spreadable_elem -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Elem_regular _visitors_c0 ->
                self#visit_Elem_regular env _visitors_c0
            | Elem_spread { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Elem_spread env _visitors_fexpr _visitors_floc_

        method visit_Map_expr_elem
            : _ -> constant -> expr -> location -> location -> S.t =
          fun env _visitors_fkey _visitors_fexpr _visitors_fkey_loc_
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_constant env _visitors_fkey in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_location env _visitors_fkey_loc_ in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Map_expr_elem"
              [
                ("key", _visitors_r0);
                ("expr", _visitors_r1);
                ("key_loc_", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_map_expr_elem : _ -> map_expr_elem -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_expr_elem
                {
                  key = _visitors_fkey;
                  expr = _visitors_fexpr;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_expr_elem env _visitors_fkey _visitors_fexpr
                  _visitors_fkey_loc_ _visitors_floc_

        method visit_Error_typ : _ -> typ -> S.t =
          fun env _visitors_fty ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            self#visit_inline_record env "Error_typ" [ ("ty", _visitors_r0) ]

        method visit_Default_error_typ : _ -> location -> S.t =
          fun env _visitors_floc_ ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Default_error_typ"
              [ ("loc_", _visitors_r0) ]

        method visit_No_error_typ : _ -> S.t =
          fun env -> self#visit_inline_tuple env "No_error_typ" []

        method visit_error_typ : _ -> error_typ -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Error_typ { ty = _visitors_fty } ->
                self#visit_Error_typ env _visitors_fty
            | Default_error_typ { loc_ = _visitors_floc_ } ->
                self#visit_Default_error_typ env _visitors_floc_
            | No_error_typ -> self#visit_No_error_typ env

        method visit_Ptype_any : _ -> location -> S.t =
          fun env _visitors_floc_ ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ptype_any" [ ("loc_", _visitors_r0) ]

        method visit_Ptype_arrow
            : _ -> typ list -> typ -> error_typ -> location -> S.t =
          fun env _visitors_fty_arg _visitors_fty_res _visitors_fty_err
              _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_typ env _visitors_fty_arg
            in
            let _visitors_r1 = self#visit_typ env _visitors_fty_res in
            let _visitors_r2 = self#visit_error_typ env _visitors_fty_err in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ptype_arrow"
              [
                ("ty_arg", _visitors_r0);
                ("ty_res", _visitors_r1);
                ("ty_err", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Ptype_tuple : _ -> typ list -> location -> S.t =
          fun env _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_typ env _visitors_ftys
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ptype_tuple"
              [ ("tys", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Ptype_name
            : _ -> constrid_loc -> typ list -> location -> S.t =
          fun env _visitors_fconstr_id _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_constrid_loc env _visitors_fconstr_id
            in
            let _visitors_r1 =
              self#visit_list self#visit_typ env _visitors_ftys
            in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ptype_name"
              [
                ("constr_id", _visitors_r0);
                ("tys", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ptype_option : _ -> typ -> location -> location -> S.t =
          fun env _visitors_fty _visitors_floc_ _visitors_fquestion_loc ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            let _visitors_r2 =
              self#visit_location env _visitors_fquestion_loc
            in
            self#visit_inline_record env "Ptype_option"
              [
                ("ty", _visitors_r0);
                ("loc_", _visitors_r1);
                ("question_loc", _visitors_r2);
              ]

        method visit_typ : _ -> typ -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptype_any { loc_ = _visitors_floc_ } ->
                self#visit_Ptype_any env _visitors_floc_
            | Ptype_arrow
                {
                  ty_arg = _visitors_fty_arg;
                  ty_res = _visitors_fty_res;
                  ty_err = _visitors_fty_err;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_arrow env _visitors_fty_arg _visitors_fty_res
                  _visitors_fty_err _visitors_floc_
            | Ptype_tuple { tys = _visitors_ftys; loc_ = _visitors_floc_ } ->
                self#visit_Ptype_tuple env _visitors_ftys _visitors_floc_
            | Ptype_name
                {
                  constr_id = _visitors_fconstr_id;
                  tys = _visitors_ftys;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_name env _visitors_fconstr_id _visitors_ftys
                  _visitors_floc_
            | Ptype_option
                {
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                  question_loc = _visitors_fquestion_loc;
                } ->
                self#visit_Ptype_option env _visitors_fty _visitors_floc_
                  _visitors_fquestion_loc

        method visit_Ppat_alias : _ -> pattern -> binder -> location -> S.t =
          fun env _visitors_fpat _visitors_falias _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_binder env _visitors_falias in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_alias"
              [
                ("pat", _visitors_r0);
                ("alias", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ppat_any : _ -> location -> S.t =
          fun env _visitors_floc_ ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_any" [ ("loc_", _visitors_r0) ]

        method visit_Ppat_array : _ -> array_pattern -> location -> S.t =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 = self#visit_array_pattern env _visitors_fpats in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_array"
              [ ("pats", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Ppat_constant : _ -> constant -> location -> S.t =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 = self#visit_constant env _visitors_fc in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_constant"
              [ ("c", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Ppat_constraint : _ -> pattern -> typ -> location -> S.t =
          fun env _visitors_fpat _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_constraint"
              [
                ("pat", _visitors_r0);
                ("ty", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ppat_constr
            : _ ->
              constructor ->
              constr_pat_arg list option ->
              bool ->
              location ->
              S.t =
          fun env _visitors_fconstr _visitors_fargs _visitors_fis_open
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 =
              self#visit_option
                (self#visit_list self#visit_constr_pat_arg)
                env _visitors_fargs
            in
            let _visitors_r2 = self#visit_bool env _visitors_fis_open in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_constr"
              [
                ("constr", _visitors_r0);
                ("args", _visitors_r1);
                ("is_open", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Ppat_or : _ -> pattern -> pattern -> location -> S.t =
          fun env _visitors_fpat1 _visitors_fpat2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat1 in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat2 in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_or"
              [
                ("pat1", _visitors_r0);
                ("pat2", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ppat_tuple : _ -> pattern list -> location -> S.t =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_pattern env _visitors_fpats
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_tuple"
              [ ("pats", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Ppat_var : _ -> binder -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_binder env _visitors_c0 in
            self#visit_inline_tuple env "Ppat_var" [ _visitors_r0 ]

        method visit_Ppat_record
            : _ -> field_pat list -> bool -> location -> S.t =
          fun env _visitors_ffields _visitors_fis_closed _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_field_pat env _visitors_ffields
            in
            let _visitors_r1 = self#visit_bool env _visitors_fis_closed in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_record"
              [
                ("fields", _visitors_r0);
                ("is_closed", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ppat_map : _ -> map_pat_elem list -> location -> S.t =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_list self#visit_map_pat_elem env _visitors_felems
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_map"
              [ ("elems", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Ppat_range
            : _ -> pattern -> pattern -> bool -> location -> S.t =
          fun env _visitors_flhs _visitors_frhs _visitors_finclusive
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_flhs in
            let _visitors_r1 = self#visit_pattern env _visitors_frhs in
            let _visitors_r2 = self#visit_bool env _visitors_finclusive in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Ppat_range"
              [
                ("lhs", _visitors_r0);
                ("rhs", _visitors_r1);
                ("inclusive", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_pattern : _ -> pattern -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Ppat_alias
                {
                  pat = _visitors_fpat;
                  alias = _visitors_falias;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_alias env _visitors_fpat _visitors_falias
                  _visitors_floc_
            | Ppat_any { loc_ = _visitors_floc_ } ->
                self#visit_Ppat_any env _visitors_floc_
            | Ppat_array { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_array env _visitors_fpats _visitors_floc_
            | Ppat_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_constant env _visitors_fc _visitors_floc_
            | Ppat_constraint
                {
                  pat = _visitors_fpat;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constraint env _visitors_fpat _visitors_fty
                  _visitors_floc_
            | Ppat_constr
                {
                  constr = _visitors_fconstr;
                  args = _visitors_fargs;
                  is_open = _visitors_fis_open;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constr env _visitors_fconstr _visitors_fargs
                  _visitors_fis_open _visitors_floc_
            | Ppat_or
                {
                  pat1 = _visitors_fpat1;
                  pat2 = _visitors_fpat2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_or env _visitors_fpat1 _visitors_fpat2
                  _visitors_floc_
            | Ppat_tuple { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_tuple env _visitors_fpats _visitors_floc_
            | Ppat_var _visitors_c0 -> self#visit_Ppat_var env _visitors_c0
            | Ppat_record
                {
                  fields = _visitors_ffields;
                  is_closed = _visitors_fis_closed;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_record env _visitors_ffields
                  _visitors_fis_closed _visitors_floc_
            | Ppat_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_map env _visitors_felems _visitors_floc_
            | Ppat_range
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  inclusive = _visitors_finclusive;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_range env _visitors_flhs _visitors_frhs
                  _visitors_finclusive _visitors_floc_

        method visit_Closed : _ -> pattern list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_pattern env _visitors_c0
            in
            self#visit_inline_tuple env "Closed" [ _visitors_r0 ]

        method visit_Open
            : _ -> pattern list -> pattern list -> binder option -> S.t =
          fun env _visitors_c0 _visitors_c1 _visitors_c2 ->
            let _visitors_r0 =
              self#visit_list self#visit_pattern env _visitors_c0
            in
            let _visitors_r1 =
              self#visit_list self#visit_pattern env _visitors_c1
            in
            let _visitors_r2 =
              self#visit_option self#visit_binder env _visitors_c2
            in
            self#visit_inline_tuple env "Open"
              [ _visitors_r0; _visitors_r1; _visitors_r2 ]

        method visit_array_pattern : _ -> array_pattern -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Closed _visitors_c0 -> self#visit_Closed env _visitors_c0
            | Open (_visitors_c0, _visitors_c1, _visitors_c2) ->
                self#visit_Open env _visitors_c0 _visitors_c1 _visitors_c2

        method visit_Field_def : _ -> label -> expr -> bool -> location -> S.t =
          fun env _visitors_flabel _visitors_fexpr _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_bool env _visitors_fis_pun in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Field_def"
              [
                ("label", _visitors_r0);
                ("expr", _visitors_r1);
                ("is_pun", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_field_def : _ -> field_def -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_def
                {
                  label = _visitors_flabel;
                  expr = _visitors_fexpr;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_def env _visitors_flabel _visitors_fexpr
                  _visitors_fis_pun _visitors_floc_

        method visit_Field_pat
            : _ -> label -> pattern -> bool -> location -> S.t =
          fun env _visitors_flabel _visitors_fpattern _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r2 = self#visit_bool env _visitors_fis_pun in
            let _visitors_r3 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Field_pat"
              [
                ("label", _visitors_r0);
                ("pattern", _visitors_r1);
                ("is_pun", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_field_pat : _ -> field_pat -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_pat
                {
                  label = _visitors_flabel;
                  pattern = _visitors_fpattern;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_pat env _visitors_flabel _visitors_fpattern
                  _visitors_fis_pun _visitors_floc_

        method visit_Constr_pat_arg : _ -> pattern -> argument_kind -> S.t =
          fun env _visitors_fpat _visitors_fkind ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_argument_kind env _visitors_fkind in
            self#visit_inline_record env "Constr_pat_arg"
              [ ("pat", _visitors_r0); ("kind", _visitors_r1) ]

        method visit_constr_pat_arg : _ -> constr_pat_arg -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Constr_pat_arg { pat = _visitors_fpat; kind = _visitors_fkind } ->
                self#visit_Constr_pat_arg env _visitors_fpat _visitors_fkind

        method visit_Map_pat_elem
            : _ -> constant -> pattern -> bool -> location -> location -> S.t =
          fun env _visitors_fkey _visitors_fpat _visitors_fmatch_absent
              _visitors_fkey_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_constant env _visitors_fkey in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat in
            let _visitors_r2 = self#visit_bool env _visitors_fmatch_absent in
            let _visitors_r3 = self#visit_location env _visitors_fkey_loc_ in
            let _visitors_r4 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Map_pat_elem"
              [
                ("key", _visitors_r0);
                ("pat", _visitors_r1);
                ("match_absent", _visitors_r2);
                ("key_loc_", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_map_pat_elem : _ -> map_pat_elem -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_pat_elem
                {
                  key = _visitors_fkey;
                  pat = _visitors_fpat;
                  match_absent = _visitors_fmatch_absent;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_pat_elem env _visitors_fkey _visitors_fpat
                  _visitors_fmatch_absent _visitors_fkey_loc_ _visitors_floc_

        method visit_constr_param : _ -> constr_param -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.cparam_typ in
            let _visitors_r1 = self#visit_bool env _visitors_this.cparam_mut in
            let _visitors_r2 =
              self#visit_option self#visit_label env _visitors_this.cparam_label
            in
            self#visit_record env
              [
                ("cparam_typ", _visitors_r0);
                ("cparam_mut", _visitors_r1);
                ("cparam_label", _visitors_r2);
              ]

        method visit_constr_decl : _ -> constr_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_constr_name env _visitors_this.constr_name
            in
            let _visitors_r1 =
              self#visit_option
                (self#visit_list self#visit_constr_param)
                env _visitors_this.constr_args
            in
            let _visitors_r2 =
              self#visit_location env _visitors_this.constr_loc_
            in
            self#visit_record env
              [
                ("constr_name", _visitors_r0);
                ("constr_args", _visitors_r1);
                ("constr_loc_", _visitors_r2);
              ]

        method visit_field_name : _ -> field_name -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_string env _visitors_this.label in
            let _visitors_r1 = self#visit_location env _visitors_this.loc_ in
            self#visit_record env
              [ ("label", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_field_decl : _ -> field_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_field_name env _visitors_this.field_name
            in
            let _visitors_r1 = self#visit_typ env _visitors_this.field_ty in
            let _visitors_r2 = self#visit_bool env _visitors_this.field_mut in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.field_vis
            in
            let _visitors_r4 =
              self#visit_location env _visitors_this.field_loc_
            in
            self#visit_record env
              [
                ("field_name", _visitors_r0);
                ("field_ty", _visitors_r1);
                ("field_mut", _visitors_r2);
                ("field_vis", _visitors_r3);
                ("field_loc_", _visitors_r4);
              ]

        method visit_No_payload : _ -> S.t =
          fun env -> self#visit_inline_tuple env "No_payload" []

        method visit_Single_payload : _ -> typ -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            self#visit_inline_tuple env "Single_payload" [ _visitors_r0 ]

        method visit_Enum_payload : _ -> constr_decl list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_constr_decl env _visitors_c0
            in
            self#visit_inline_tuple env "Enum_payload" [ _visitors_r0 ]

        method visit_exception_decl : _ -> exception_decl -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | No_payload -> self#visit_No_payload env
            | Single_payload _visitors_c0 ->
                self#visit_Single_payload env _visitors_c0
            | Enum_payload _visitors_c0 ->
                self#visit_Enum_payload env _visitors_c0

        method visit_Ptd_abstract : _ -> S.t =
          fun env -> self#visit_inline_tuple env "Ptd_abstract" []

        method visit_Ptd_newtype : _ -> typ -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            self#visit_inline_tuple env "Ptd_newtype" [ _visitors_r0 ]

        method visit_Ptd_error : _ -> exception_decl -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_exception_decl env _visitors_c0 in
            self#visit_inline_tuple env "Ptd_error" [ _visitors_r0 ]

        method visit_Ptd_variant : _ -> constr_decl list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_constr_decl env _visitors_c0
            in
            self#visit_inline_tuple env "Ptd_variant" [ _visitors_r0 ]

        method visit_Ptd_record : _ -> field_decl list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_field_decl env _visitors_c0
            in
            self#visit_inline_tuple env "Ptd_record" [ _visitors_r0 ]

        method visit_Ptd_alias : _ -> typ -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            self#visit_inline_tuple env "Ptd_alias" [ _visitors_r0 ]

        method visit_type_desc : _ -> type_desc -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptd_abstract -> self#visit_Ptd_abstract env
            | Ptd_newtype _visitors_c0 ->
                self#visit_Ptd_newtype env _visitors_c0
            | Ptd_error _visitors_c0 -> self#visit_Ptd_error env _visitors_c0
            | Ptd_variant _visitors_c0 ->
                self#visit_Ptd_variant env _visitors_c0
            | Ptd_record _visitors_c0 -> self#visit_Ptd_record env _visitors_c0
            | Ptd_alias _visitors_c0 -> self#visit_Ptd_alias env _visitors_c0

        method visit_type_decl : _ -> type_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_string env _visitors_this.tycon in
            let _visitors_r1 =
              self#visit_location env _visitors_this.tycon_loc_
            in
            let _visitors_r2 =
              self#visit_list self#visit_type_decl_binder env
                _visitors_this.params
            in
            let _visitors_r3 =
              self#visit_type_desc env _visitors_this.components
            in
            let _visitors_r4 = self#visit_docstring env _visitors_this.doc_ in
            let _visitors_r5 =
              self#visit_visibility env _visitors_this.type_vis
            in
            let _visitors_r6 =
              self#visit_list self#visit_deriving_directive env
                _visitors_this.deriving_
            in
            let _visitors_r7 =
              self#visit_absolute_loc env _visitors_this.loc_
            in
            self#visit_record env
              [
                ("tycon", _visitors_r0);
                ("tycon_loc_", _visitors_r1);
                ("params", _visitors_r2);
                ("components", _visitors_r3);
                ("doc_", _visitors_r4);
                ("type_vis", _visitors_r5);
                ("deriving_", _visitors_r6);
                ("loc_", _visitors_r7);
              ]

        method visit_local_type_decl : _ -> local_type_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_string env _visitors_this.local_tycon
            in
            let _visitors_r1 =
              self#visit_location env _visitors_this.local_tycon_loc_
            in
            let _visitors_r2 =
              self#visit_type_desc env _visitors_this.local_components
            in
            self#visit_record env
              [
                ("local_tycon", _visitors_r0);
                ("local_tycon_loc_", _visitors_r1);
                ("local_components", _visitors_r2);
              ]

        method visit_deriving_directive : _ -> deriving_directive -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_type_name env _visitors_this.type_name_
            in
            let _visitors_r1 =
              self#visit_list self#visit_argument env _visitors_this.args
            in
            let _visitors_r2 = self#visit_location env _visitors_this.loc_ in
            self#visit_record env
              [
                ("type_name_", _visitors_r0);
                ("args", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Vis_default : _ -> S.t =
          fun env -> self#visit_inline_tuple env "Vis_default" []

        method visit_Vis_pub : _ -> string option -> location -> S.t =
          fun env _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_string env _visitors_fattr
            in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Vis_pub"
              [ ("attr", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Vis_priv : _ -> location -> S.t =
          fun env _visitors_floc_ ->
            let _visitors_r0 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Vis_priv" [ ("loc_", _visitors_r0) ]

        method visit_visibility : _ -> visibility -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Vis_default -> self#visit_Vis_default env
            | Vis_pub { attr = _visitors_fattr; loc_ = _visitors_floc_ } ->
                self#visit_Vis_pub env _visitors_fattr _visitors_floc_
            | Vis_priv { loc_ = _visitors_floc_ } ->
                self#visit_Vis_priv env _visitors_floc_

        method visit_Import : _ -> string_literal -> string_literal -> S.t =
          fun env _visitors_fmodule_name _visitors_ffunc_name ->
            let _visitors_r0 =
              self#visit_string_literal env _visitors_fmodule_name
            in
            let _visitors_r1 =
              self#visit_string_literal env _visitors_ffunc_name
            in
            self#visit_inline_record env "Import"
              [ ("module_name", _visitors_r0); ("func_name", _visitors_r1) ]

        method visit_Embedded
            : _ -> string_literal option -> embedded_code -> S.t =
          fun env _visitors_flanguage _visitors_fcode ->
            let _visitors_r0 =
              self#visit_option self#visit_string_literal env
                _visitors_flanguage
            in
            let _visitors_r1 = self#visit_embedded_code env _visitors_fcode in
            self#visit_inline_record env "Embedded"
              [ ("language", _visitors_r0); ("code", _visitors_r1) ]

        method visit_func_stubs : _ -> func_stubs -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Import
                {
                  module_name = _visitors_fmodule_name;
                  func_name = _visitors_ffunc_name;
                } ->
                self#visit_Import env _visitors_fmodule_name
                  _visitors_ffunc_name
            | Embedded
                { language = _visitors_flanguage; code = _visitors_fcode } ->
                self#visit_Embedded env _visitors_flanguage _visitors_fcode

        method visit_Code_string : _ -> string_literal -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_string_literal env _visitors_c0 in
            self#visit_inline_tuple env "Code_string" [ _visitors_r0 ]

        method visit_Code_multiline_string : _ -> string list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_string env _visitors_c0
            in
            self#visit_inline_tuple env "Code_multiline_string" [ _visitors_r0 ]

        method visit_embedded_code : _ -> embedded_code -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Code_string _visitors_c0 ->
                self#visit_Code_string env _visitors_c0
            | Code_multiline_string _visitors_c0 ->
                self#visit_Code_multiline_string env _visitors_c0

        method visit_Decl_body : _ -> local_type_decl list -> expr -> S.t =
          fun env _visitors_flocal_types _visitors_fexpr ->
            let _visitors_r0 =
              self#visit_list self#visit_local_type_decl env
                _visitors_flocal_types
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            self#visit_inline_record env "Decl_body"
              [ ("local_types", _visitors_r0); ("expr", _visitors_r1) ]

        method visit_Decl_stubs : _ -> func_stubs -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_func_stubs env _visitors_c0 in
            self#visit_inline_tuple env "Decl_stubs" [ _visitors_r0 ]

        method visit_decl_body : _ -> decl_body -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Decl_body
                { local_types = _visitors_flocal_types; expr = _visitors_fexpr }
              ->
                self#visit_Decl_body env _visitors_flocal_types _visitors_fexpr
            | Decl_stubs _visitors_c0 -> self#visit_Decl_stubs env _visitors_c0

        method visit_fun_decl : _ -> fun_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_option self#visit_type_name env
                _visitors_this.type_name
            in
            let _visitors_r1 = self#visit_binder env _visitors_this.name in
            let _visitors_r2 = self#visit_bool env _visitors_this.has_error in
            let _visitors_r3 =
              self#visit_option self#visit_parameters env
                _visitors_this.decl_params
            in
            let _visitors_r4 =
              self#visit_location env _visitors_this.params_loc_
            in
            let _visitors_r5 =
              self#visit_list self#visit_tvar_binder env
                _visitors_this.quantifiers
            in
            let _visitors_r6 =
              self#visit_option
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_typ env _visitors_c0 in
                  let _visitors_r1 = self#visit_error_typ env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_this.return_type
            in
            let _visitors_r7 = self#visit_bool env _visitors_this.is_pub in
            let _visitors_r8 = self#visit_docstring env _visitors_this.doc_ in
            self#visit_record env
              [
                ("type_name", _visitors_r0);
                ("name", _visitors_r1);
                ("has_error", _visitors_r2);
                ("decl_params", _visitors_r3);
                ("params_loc_", _visitors_r4);
                ("quantifiers", _visitors_r5);
                ("return_type", _visitors_r6);
                ("is_pub", _visitors_r7);
                ("doc_", _visitors_r8);
              ]

        method visit_trait_method_param : _ -> trait_method_param -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.tmparam_typ in
            let _visitors_r1 =
              self#visit_option self#visit_label env
                _visitors_this.tmparam_label
            in
            self#visit_record env
              [ ("tmparam_typ", _visitors_r0); ("tmparam_label", _visitors_r1) ]

        method visit_Trait_method
            : _ ->
              binder ->
              bool ->
              tvar_binder list ->
              trait_method_param list ->
              (typ * error_typ) option ->
              location ->
              S.t =
          fun env _visitors_fname _visitors_fhas_error _visitors_fquantifiers
              _visitors_fparams _visitors_freturn_type _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 = self#visit_bool env _visitors_fhas_error in
            let _visitors_r2 =
              self#visit_list self#visit_tvar_binder env _visitors_fquantifiers
            in
            let _visitors_r3 =
              self#visit_list self#visit_trait_method_param env
                _visitors_fparams
            in
            let _visitors_r4 =
              self#visit_option
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_typ env _visitors_c0 in
                  let _visitors_r1 = self#visit_error_typ env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_freturn_type
            in
            let _visitors_r5 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Trait_method"
              [
                ("name", _visitors_r0);
                ("has_error", _visitors_r1);
                ("quantifiers", _visitors_r2);
                ("params", _visitors_r3);
                ("return_type", _visitors_r4);
                ("loc_", _visitors_r5);
              ]

        method visit_trait_method_decl : _ -> trait_method_decl -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Trait_method
                {
                  name = _visitors_fname;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  return_type = _visitors_freturn_type;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Trait_method env _visitors_fname _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams
                  _visitors_freturn_type _visitors_floc_

        method visit_trait_decl : _ -> trait_decl -> S.t =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.trait_name
            in
            let _visitors_r1 =
              self#visit_list self#visit_tvar_constraint env
                _visitors_this.trait_supers
            in
            let _visitors_r2 =
              self#visit_list self#visit_trait_method_decl env
                _visitors_this.trait_methods
            in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.trait_vis
            in
            let _visitors_r4 =
              self#visit_absolute_loc env _visitors_this.trait_loc_
            in
            let _visitors_r5 =
              self#visit_docstring env _visitors_this.trait_doc_
            in
            self#visit_record env
              [
                ("trait_name", _visitors_r0);
                ("trait_supers", _visitors_r1);
                ("trait_methods", _visitors_r2);
                ("trait_vis", _visitors_r3);
                ("trait_loc_", _visitors_r4);
                ("trait_doc_", _visitors_r5);
              ]

        method visit_Ptop_expr
            : _ -> expr -> bool -> local_type_decl list -> absolute_loc -> S.t =
          fun env _visitors_fexpr _visitors_fis_main _visitors_flocal_types
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_bool env _visitors_fis_main in
            let _visitors_r2 =
              self#visit_list self#visit_local_type_decl env
                _visitors_flocal_types
            in
            let _visitors_r3 = self#visit_absolute_loc env _visitors_floc_ in
            self#visit_inline_record env "Ptop_expr"
              [
                ("expr", _visitors_r0);
                ("is_main", _visitors_r1);
                ("local_types", _visitors_r2);
                ("loc_", _visitors_r3);
              ]

        method visit_Ptop_test
            : _ ->
              expr ->
              test_name ->
              parameters option ->
              local_type_decl list ->
              absolute_loc ->
              S.t =
          fun env _visitors_fexpr _visitors_fname _visitors_fparams
              _visitors_flocal_types _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_test_name env _visitors_fname in
            let _visitors_r2 =
              self#visit_option self#visit_parameters env _visitors_fparams
            in
            let _visitors_r3 =
              self#visit_list self#visit_local_type_decl env
                _visitors_flocal_types
            in
            let _visitors_r4 = self#visit_absolute_loc env _visitors_floc_ in
            self#visit_inline_record env "Ptop_test"
              [
                ("expr", _visitors_r0);
                ("name", _visitors_r1);
                ("params", _visitors_r2);
                ("local_types", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_Ptop_typedef : _ -> type_decl -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_type_decl env _visitors_c0 in
            self#visit_inline_tuple env "Ptop_typedef" [ _visitors_r0 ]

        method visit_Ptop_funcdef
            : _ -> fun_decl -> decl_body -> absolute_loc -> S.t =
          fun env _visitors_ffun_decl _visitors_fdecl_body _visitors_floc_ ->
            let _visitors_r0 = self#visit_fun_decl env _visitors_ffun_decl in
            let _visitors_r1 = self#visit_decl_body env _visitors_fdecl_body in
            let _visitors_r2 = self#visit_absolute_loc env _visitors_floc_ in
            self#visit_inline_record env "Ptop_funcdef"
              [
                ("fun_decl", _visitors_r0);
                ("decl_body", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Ptop_letdef
            : _ ->
              binder ->
              typ option ->
              expr ->
              bool ->
              bool ->
              absolute_loc ->
              docstring ->
              S.t =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fis_pub _visitors_fis_constant _visitors_floc_
              _visitors_fdoc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              self#visit_option self#visit_typ env _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 = self#visit_bool env _visitors_fis_pub in
            let _visitors_r4 = self#visit_bool env _visitors_fis_constant in
            let _visitors_r5 = self#visit_absolute_loc env _visitors_floc_ in
            let _visitors_r6 = self#visit_docstring env _visitors_fdoc_ in
            self#visit_inline_record env "Ptop_letdef"
              [
                ("binder", _visitors_r0);
                ("ty", _visitors_r1);
                ("expr", _visitors_r2);
                ("is_pub", _visitors_r3);
                ("is_constant", _visitors_r4);
                ("loc_", _visitors_r5);
                ("doc_", _visitors_r6);
              ]

        method visit_Ptop_trait : _ -> trait_decl -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_trait_decl env _visitors_c0 in
            self#visit_inline_tuple env "Ptop_trait" [ _visitors_r0 ]

        method visit_Ptop_impl
            : _ ->
              typ option ->
              type_name ->
              binder ->
              bool ->
              tvar_binder list ->
              parameters ->
              (typ * error_typ) option ->
              expr ->
              bool ->
              local_type_decl list ->
              absolute_loc ->
              location ->
              docstring ->
              S.t =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fmethod_name
              _visitors_fhas_error _visitors_fquantifiers _visitors_fparams
              _visitors_fret_ty _visitors_fbody _visitors_fis_pub
              _visitors_flocal_types _visitors_floc_ _visitors_fheader_loc_
              _visitors_fdoc_ ->
            let _visitors_r0 =
              self#visit_option self#visit_typ env _visitors_fself_ty
            in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 = self#visit_binder env _visitors_fmethod_name in
            let _visitors_r3 = self#visit_bool env _visitors_fhas_error in
            let _visitors_r4 =
              self#visit_list self#visit_tvar_binder env _visitors_fquantifiers
            in
            let _visitors_r5 = self#visit_parameters env _visitors_fparams in
            let _visitors_r6 =
              self#visit_option
                (fun env (_visitors_c0, _visitors_c1) ->
                  let _visitors_r0 = self#visit_typ env _visitors_c0 in
                  let _visitors_r1 = self#visit_error_typ env _visitors_c1 in
                  self#visit_tuple env [ _visitors_r0; _visitors_r1 ])
                env _visitors_fret_ty
            in
            let _visitors_r7 = self#visit_expr env _visitors_fbody in
            let _visitors_r8 = self#visit_bool env _visitors_fis_pub in
            let _visitors_r9 =
              self#visit_list self#visit_local_type_decl env
                _visitors_flocal_types
            in
            let _visitors_r10 = self#visit_absolute_loc env _visitors_floc_ in
            let _visitors_r11 =
              self#visit_location env _visitors_fheader_loc_
            in
            let _visitors_r12 = self#visit_docstring env _visitors_fdoc_ in
            self#visit_inline_record env "Ptop_impl"
              [
                ("self_ty", _visitors_r0);
                ("trait", _visitors_r1);
                ("method_name", _visitors_r2);
                ("has_error", _visitors_r3);
                ("quantifiers", _visitors_r4);
                ("params", _visitors_r5);
                ("ret_ty", _visitors_r6);
                ("body", _visitors_r7);
                ("is_pub", _visitors_r8);
                ("local_types", _visitors_r9);
                ("loc_", _visitors_r10);
                ("header_loc_", _visitors_r11);
                ("doc_", _visitors_r12);
              ]

        method visit_Ptop_impl_relation
            : _ ->
              typ ->
              type_name ->
              tvar_binder list ->
              bool ->
              absolute_loc ->
              S.t =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fquantifiers
              _visitors_fis_pub _visitors_floc_ ->
            let _visitors_r0 = self#visit_typ env _visitors_fself_ty in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 =
              self#visit_list self#visit_tvar_binder env _visitors_fquantifiers
            in
            let _visitors_r3 = self#visit_bool env _visitors_fis_pub in
            let _visitors_r4 = self#visit_absolute_loc env _visitors_floc_ in
            self#visit_inline_record env "Ptop_impl_relation"
              [
                ("self_ty", _visitors_r0);
                ("trait", _visitors_r1);
                ("quantifiers", _visitors_r2);
                ("is_pub", _visitors_r3);
                ("loc_", _visitors_r4);
              ]

        method visit_impl : _ -> impl -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptop_expr
                {
                  expr = _visitors_fexpr;
                  is_main = _visitors_fis_main;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_expr env _visitors_fexpr _visitors_fis_main
                  _visitors_flocal_types _visitors_floc_
            | Ptop_test
                {
                  expr = _visitors_fexpr;
                  name = _visitors_fname;
                  params = _visitors_fparams;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_test env _visitors_fexpr _visitors_fname
                  _visitors_fparams _visitors_flocal_types _visitors_floc_
            | Ptop_typedef _visitors_c0 ->
                self#visit_Ptop_typedef env _visitors_c0
            | Ptop_funcdef
                {
                  fun_decl = _visitors_ffun_decl;
                  decl_body = _visitors_fdecl_body;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_funcdef env _visitors_ffun_decl
                  _visitors_fdecl_body _visitors_floc_
            | Ptop_letdef
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  is_pub = _visitors_fis_pub;
                  is_constant = _visitors_fis_constant;
                  loc_ = _visitors_floc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_letdef env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fis_pub _visitors_fis_constant
                  _visitors_floc_ _visitors_fdoc_
            | Ptop_trait _visitors_c0 -> self#visit_Ptop_trait env _visitors_c0
            | Ptop_impl
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  method_name = _visitors_fmethod_name;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  ret_ty = _visitors_fret_ty;
                  body = _visitors_fbody;
                  is_pub = _visitors_fis_pub;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                  header_loc_ = _visitors_fheader_loc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_impl env _visitors_fself_ty _visitors_ftrait
                  _visitors_fmethod_name _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams _visitors_fret_ty
                  _visitors_fbody _visitors_fis_pub _visitors_flocal_types
                  _visitors_floc_ _visitors_fheader_loc_ _visitors_fdoc_
            | Ptop_impl_relation
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  quantifiers = _visitors_fquantifiers;
                  is_pub = _visitors_fis_pub;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_impl_relation env _visitors_fself_ty
                  _visitors_ftrait _visitors_fquantifiers _visitors_fis_pub
                  _visitors_floc_

        method visit_Interp_lit : _ -> string -> string -> location -> S.t =
          fun env _visitors_fstr _visitors_frepr _visitors_floc_ ->
            let _visitors_r0 = self#visit_string env _visitors_fstr in
            let _visitors_r1 = self#visit_string env _visitors_frepr in
            let _visitors_r2 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Interp_lit"
              [
                ("str", _visitors_r0);
                ("repr", _visitors_r1);
                ("loc_", _visitors_r2);
              ]

        method visit_Interp_expr : _ -> expr -> location -> S.t =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_location env _visitors_floc_ in
            self#visit_inline_record env "Interp_expr"
              [ ("expr", _visitors_r0); ("loc_", _visitors_r1) ]

        method visit_Interp_source : _ -> Literal.interp_source -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_interp_source env _visitors_c0 in
            self#visit_inline_tuple env "Interp_source" [ _visitors_r0 ]

        method visit_interp_elem : _ -> interp_elem -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Interp_lit
                {
                  str = _visitors_fstr;
                  repr = _visitors_frepr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Interp_lit env _visitors_fstr _visitors_frepr
                  _visitors_floc_
            | Interp_expr { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Interp_expr env _visitors_fexpr _visitors_floc_
            | Interp_source _visitors_c0 ->
                self#visit_Interp_source env _visitors_c0

        method visit_Multiline_string : _ -> string -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_string env _visitors_c0 in
            self#visit_inline_tuple env "Multiline_string" [ _visitors_r0 ]

        method visit_Multiline_interp : _ -> interp_elem list -> S.t =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              self#visit_list self#visit_interp_elem env _visitors_c0
            in
            self#visit_inline_tuple env "Multiline_interp" [ _visitors_r0 ]

        method visit_multiline_string_elem : _ -> multiline_string_elem -> S.t =
          fun env _visitors_this ->
            match _visitors_this with
            | Multiline_string _visitors_c0 ->
                self#visit_Multiline_string env _visitors_c0
            | Multiline_interp _visitors_c0 ->
                self#visit_Multiline_interp env _visitors_c0

        method visit_impls : _ -> impls -> S.t =
          fun env -> self#visit_list self#visit_impl env
      end

    [@@@VISITORS.END]
  end

  include struct
    [@@@ocaml.warning "-4-26-27"]
    [@@@VISITORS.BEGIN]

    class virtual ['self] iter =
      object (self : 'self)
        inherit [_] iterbase

        method visit_Pexpr_apply
            : _ -> expr -> argument list -> apply_attr -> location -> unit =
          fun env _visitors_ffunc _visitors_fargs _visitors_fattr
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ffunc in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_argument env))
                _visitors_fargs
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_fattr in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_infix : _ -> var -> expr -> expr -> location -> unit
            =
          fun env _visitors_fop _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_flhs in
            let _visitors_r2 = self#visit_expr env _visitors_frhs in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_unary : _ -> var -> expr -> location -> unit =
          fun env _visitors_fop _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array : _ -> expr list -> location -> unit =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_expr env))
                _visitors_fexprs
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array_spread
            : _ -> spreadable_elem list -> location -> unit =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_spreadable_elem env))
                _visitors_felems
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array_get : _ -> expr -> expr -> location -> unit =
          fun env _visitors_farray _visitors_findex _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array_get_slice
            : _ ->
              expr ->
              expr option ->
              expr option ->
              location ->
              location ->
              unit =
          fun env _visitors_farray _visitors_fstart_index _visitors_fend_index
              _visitors_findex_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fstart_index
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fend_index
            in
            let _visitors_r3 =
              (fun _visitors_this -> ()) _visitors_findex_loc_
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array_set
            : _ -> expr -> expr -> expr -> location -> unit =
          fun env _visitors_farray _visitors_findex _visitors_fvalue
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 = self#visit_expr env _visitors_fvalue in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_array_augmented_set
            : _ -> var -> expr -> expr -> expr -> location -> unit =
          fun env _visitors_fop _visitors_farray _visitors_findex
              _visitors_fvalue _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_farray in
            let _visitors_r2 = self#visit_expr env _visitors_findex in
            let _visitors_r3 = self#visit_expr env _visitors_fvalue in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_constant : _ -> constant -> location -> unit =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_fc in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_multiline_string
            : _ -> multiline_string_elem list -> location -> unit =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (self#visit_multiline_string_elem env))
                _visitors_felems
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_interp : _ -> interp_elem list -> location -> unit =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_interp_elem env))
                _visitors_felems
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_constraint : _ -> expr -> typ -> location -> unit =
          fun env _visitors_fexpr _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_constr : _ -> constructor -> location -> unit =
          fun env _visitors_fconstr _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_while
            : _ -> expr -> expr -> expr option -> location -> unit =
          fun env _visitors_floop_cond _visitors_floop_body
              _visitors_fwhile_else _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_floop_cond in
            let _visitors_r1 = self#visit_expr env _visitors_floop_body in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fwhile_else
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_function : _ -> func -> location -> unit =
          fun env _visitors_ffunc _visitors_floc_ ->
            let _visitors_r0 = self#visit_func env _visitors_ffunc in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_ident : _ -> var -> location -> unit =
          fun env _visitors_fid _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fid in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_if
            : _ -> expr -> expr -> expr option -> location -> unit =
          fun env _visitors_fcond _visitors_fifso _visitors_fifnot
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 = self#visit_expr env _visitors_fifso in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fifnot
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_guard
            : _ -> expr -> expr option -> expr -> location -> unit =
          fun env _visitors_fcond _visitors_fotherwise _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fotherwise
            in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_guard_let
            : _ ->
              pattern ->
              expr ->
              (pattern * expr) list option ->
              expr ->
              location ->
              unit =
          fun env _visitors_fpat _visitors_fexpr _visitors_fotherwise
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun _visitors_this ->
                      Basic_lst.iter _visitors_this
                        (fun (_visitors_c0, _visitors_c1) ->
                          let _visitors_r0 =
                            self#visit_pattern env _visitors_c0
                          in
                          let _visitors_r1 = self#visit_expr env _visitors_c1 in
                          ()))
                      t
                | None -> ())
                _visitors_fotherwise
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_letfn
            : _ -> binder -> func -> expr -> location -> unit =
          fun env _visitors_fname _visitors_ffunc _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 = self#visit_func env _visitors_ffunc in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_letrec
            : _ -> (binder * func) list -> expr -> location -> unit =
          fun env _visitors_fbindings _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_func env _visitors_c1 in
                    ()))
                _visitors_fbindings
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_let
            : _ -> pattern -> expr -> expr -> location -> unit =
          fun env _visitors_fpattern _visitors_fexpr _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_sequence : _ -> expr -> expr -> location -> unit =
          fun env _visitors_fexpr1 _visitors_fexpr2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr1 in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr2 in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_tuple : _ -> expr list -> location -> unit =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_expr env))
                _visitors_fexprs
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_record
            : _ ->
              type_name option ->
              field_def list ->
              trailing_mark ->
              location ->
              unit =
          fun env _visitors_ftype_name _visitors_ffields _visitors_ftrailing
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_type_name env) t
                | None -> ())
                _visitors_ftype_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_field_def env))
                _visitors_ffields
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_ftrailing in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_record_update
            : _ ->
              type_name option ->
              expr ->
              field_def list ->
              location ->
              unit =
          fun env _visitors_ftype_name _visitors_frecord _visitors_ffields
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_type_name env) t
                | None -> ())
                _visitors_ftype_name
            in
            let _visitors_r1 = self#visit_expr env _visitors_frecord in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_field_def env))
                _visitors_ffields
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_field : _ -> expr -> accessor -> location -> unit =
          fun env _visitors_frecord _visitors_faccessor _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_method : _ -> type_name -> label -> location -> unit
            =
          fun env _visitors_ftype_name _visitors_fmethod_name _visitors_floc_ ->
            let _visitors_r0 = self#visit_type_name env _visitors_ftype_name in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_dot_apply
            : _ ->
              expr ->
              label ->
              argument list ->
              bool ->
              apply_attr ->
              location ->
              unit =
          fun env _visitors_fself _visitors_fmethod_name _visitors_fargs
              _visitors_freturn_self _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fself in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_argument env))
                _visitors_fargs
            in
            let _visitors_r3 =
              (fun _visitors_this -> ()) _visitors_freturn_self
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_fattr in
            let _visitors_r5 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_as : _ -> expr -> type_name -> location -> unit =
          fun env _visitors_fexpr _visitors_ftrait _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_mutate
            : _ -> expr -> accessor -> expr -> var option -> location -> unit =
          fun env _visitors_frecord _visitors_faccessor _visitors_ffield
              _visitors_faugmented_by _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 = self#visit_expr env _visitors_ffield in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_var env) t
                | None -> ())
                _visitors_faugmented_by
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_match
            : _ -> expr -> (pattern * expr) list -> location -> location -> unit
            =
          fun env _visitors_fexpr _visitors_fcases _visitors_fmatch_loc_
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fcases
            in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_fmatch_loc_
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_letmut
            : _ -> binder -> typ option -> expr -> expr -> location -> unit =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_typ env) t
                | None -> ())
                _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_pipe : _ -> expr -> expr -> location -> unit =
          fun env _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_flhs in
            let _visitors_r1 = self#visit_expr env _visitors_frhs in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_assign
            : _ -> var -> expr -> var option -> location -> unit =
          fun env _visitors_fvar _visitors_fexpr _visitors_faugmented_by
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fvar in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_var env) t
                | None -> ())
                _visitors_faugmented_by
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_hole : _ -> location -> hole -> unit =
          fun env _visitors_floc_ _visitors_fkind ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            let _visitors_r1 = self#visit_hole env _visitors_fkind in
            ()

        method visit_Pexpr_return : _ -> expr option -> location -> unit =
          fun env _visitors_freturn_value _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_freturn_value
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_raise : _ -> expr -> location -> unit =
          fun env _visitors_ferr_value _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ferr_value in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_unit : _ -> location -> bool -> unit =
          fun env _visitors_floc_ _visitors_ffaked ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_ffaked in
            ()

        method visit_Pexpr_break : _ -> expr option -> location -> unit =
          fun env _visitors_farg _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_farg
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_continue : _ -> expr list -> location -> unit =
          fun env _visitors_fargs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_expr env))
                _visitors_fargs
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_loop
            : _ ->
              expr list ->
              (pattern list * expr) list ->
              location ->
              location ->
              unit =
          fun env _visitors_fargs _visitors_fbody _visitors_floop_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_expr env))
                _visitors_fargs
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 =
                      (fun _visitors_this ->
                        Basic_lst.iter _visitors_this (self#visit_pattern env))
                        _visitors_c0
                    in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fbody
            in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_floop_loc_
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_for
            : _ ->
              (binder * expr) list ->
              expr option ->
              (binder * expr) list ->
              expr ->
              expr option ->
              location ->
              unit =
          fun env _visitors_fbinders _visitors_fcondition
              _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fbinders
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_fcondition
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fcontinue_block
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_ffor_else
            in
            let _visitors_r5 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_foreach
            : _ ->
              binder option list ->
              expr ->
              expr ->
              expr option ->
              location ->
              unit =
          fun env _visitors_fbinders _visitors_fexpr _visitors_fbody
              _visitors_felse_block _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (fun _visitors_this ->
                    match _visitors_this with
                    | Some t -> (self#visit_binder env) t
                    | None -> ()))
                _visitors_fbinders
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_expr env) t
                | None -> ())
                _visitors_felse_block
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_try
            : _ ->
              expr ->
              (pattern * expr) list ->
              bool ->
              (pattern * expr) list option ->
              location ->
              location ->
              location ->
              location ->
              unit =
          fun env _visitors_fbody _visitors_fcatch _visitors_fcatch_all
              _visitors_ftry_else _visitors_ftry_loc_ _visitors_fcatch_loc_
              _visitors_felse_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fbody in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fcatch
            in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_fcatch_all
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun _visitors_this ->
                      Basic_lst.iter _visitors_this
                        (fun (_visitors_c0, _visitors_c1) ->
                          let _visitors_r0 =
                            self#visit_pattern env _visitors_c0
                          in
                          let _visitors_r1 = self#visit_expr env _visitors_c1 in
                          ()))
                      t
                | None -> ())
                _visitors_ftry_else
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_ftry_loc_ in
            let _visitors_r5 =
              (fun _visitors_this -> ()) _visitors_fcatch_loc_
            in
            let _visitors_r6 =
              (fun _visitors_this -> ()) _visitors_felse_loc_
            in
            let _visitors_r7 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_map : _ -> map_expr_elem list -> location -> unit =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_map_expr_elem env))
                _visitors_felems
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_group : _ -> expr -> group -> location -> unit =
          fun env _visitors_fexpr _visitors_fgroup _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_fgroup in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Pexpr_static_assert
            : _ -> static_assertion list -> expr -> unit =
          fun env _visitors_fasserts _visitors_fbody ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_static_assertion env))
                _visitors_fasserts
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            ()

        method visit_expr : _ -> expr -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Pexpr_apply
                {
                  func = _visitors_ffunc;
                  args = _visitors_fargs;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_apply env _visitors_ffunc _visitors_fargs
                  _visitors_fattr _visitors_floc_
            | Pexpr_infix
                {
                  op = _visitors_fop;
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_infix env _visitors_fop _visitors_flhs
                  _visitors_frhs _visitors_floc_
            | Pexpr_unary
                {
                  op = _visitors_fop;
                  expr = _visitors_fexpr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_unary env _visitors_fop _visitors_fexpr
                  _visitors_floc_
            | Pexpr_array { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_array env _visitors_fexprs _visitors_floc_
            | Pexpr_array_spread
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_array_spread env _visitors_felems
                  _visitors_floc_
            | Pexpr_array_get
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get env _visitors_farray _visitors_findex
                  _visitors_floc_
            | Pexpr_array_get_slice
                {
                  array = _visitors_farray;
                  start_index = _visitors_fstart_index;
                  end_index = _visitors_fend_index;
                  index_loc_ = _visitors_findex_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get_slice env _visitors_farray
                  _visitors_fstart_index _visitors_fend_index
                  _visitors_findex_loc_ _visitors_floc_
            | Pexpr_array_set
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_set env _visitors_farray _visitors_findex
                  _visitors_fvalue _visitors_floc_
            | Pexpr_array_augmented_set
                {
                  op = _visitors_fop;
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_augmented_set env _visitors_fop
                  _visitors_farray _visitors_findex _visitors_fvalue
                  _visitors_floc_
            | Pexpr_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constant env _visitors_fc _visitors_floc_
            | Pexpr_multiline_string
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_multiline_string env _visitors_felems
                  _visitors_floc_
            | Pexpr_interp { elems = _visitors_felems; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_interp env _visitors_felems _visitors_floc_
            | Pexpr_constraint
                {
                  expr = _visitors_fexpr;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_constraint env _visitors_fexpr _visitors_fty
                  _visitors_floc_
            | Pexpr_constr
                { constr = _visitors_fconstr; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constr env _visitors_fconstr _visitors_floc_
            | Pexpr_while
                {
                  loop_cond = _visitors_floop_cond;
                  loop_body = _visitors_floop_body;
                  while_else = _visitors_fwhile_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_while env _visitors_floop_cond
                  _visitors_floop_body _visitors_fwhile_else _visitors_floc_
            | Pexpr_function { func = _visitors_ffunc; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_function env _visitors_ffunc _visitors_floc_
            | Pexpr_ident { id = _visitors_fid; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_ident env _visitors_fid _visitors_floc_
            | Pexpr_if
                {
                  cond = _visitors_fcond;
                  ifso = _visitors_fifso;
                  ifnot = _visitors_fifnot;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_if env _visitors_fcond _visitors_fifso
                  _visitors_fifnot _visitors_floc_
            | Pexpr_guard
                {
                  cond = _visitors_fcond;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard env _visitors_fcond _visitors_fotherwise
                  _visitors_fbody _visitors_floc_
            | Pexpr_guard_let
                {
                  pat = _visitors_fpat;
                  expr = _visitors_fexpr;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard_let env _visitors_fpat _visitors_fexpr
                  _visitors_fotherwise _visitors_fbody _visitors_floc_
            | Pexpr_letfn
                {
                  name = _visitors_fname;
                  func = _visitors_ffunc;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letfn env _visitors_fname _visitors_ffunc
                  _visitors_fbody _visitors_floc_
            | Pexpr_letrec
                {
                  bindings = _visitors_fbindings;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letrec env _visitors_fbindings _visitors_fbody
                  _visitors_floc_
            | Pexpr_let
                {
                  pattern = _visitors_fpattern;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_let env _visitors_fpattern _visitors_fexpr
                  _visitors_fbody _visitors_floc_
            | Pexpr_sequence
                {
                  expr1 = _visitors_fexpr1;
                  expr2 = _visitors_fexpr2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_sequence env _visitors_fexpr1 _visitors_fexpr2
                  _visitors_floc_
            | Pexpr_tuple { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_tuple env _visitors_fexprs _visitors_floc_
            | Pexpr_record
                {
                  type_name = _visitors_ftype_name;
                  fields = _visitors_ffields;
                  trailing = _visitors_ftrailing;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record env _visitors_ftype_name
                  _visitors_ffields _visitors_ftrailing _visitors_floc_
            | Pexpr_record_update
                {
                  type_name = _visitors_ftype_name;
                  record = _visitors_frecord;
                  fields = _visitors_ffields;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record_update env _visitors_ftype_name
                  _visitors_frecord _visitors_ffields _visitors_floc_
            | Pexpr_field
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_field env _visitors_frecord _visitors_faccessor
                  _visitors_floc_
            | Pexpr_method
                {
                  type_name = _visitors_ftype_name;
                  method_name = _visitors_fmethod_name;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_method env _visitors_ftype_name
                  _visitors_fmethod_name _visitors_floc_
            | Pexpr_dot_apply
                {
                  self = _visitors_fself;
                  method_name = _visitors_fmethod_name;
                  args = _visitors_fargs;
                  return_self = _visitors_freturn_self;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_dot_apply env _visitors_fself
                  _visitors_fmethod_name _visitors_fargs _visitors_freturn_self
                  _visitors_fattr _visitors_floc_
            | Pexpr_as
                {
                  expr = _visitors_fexpr;
                  trait = _visitors_ftrait;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_as env _visitors_fexpr _visitors_ftrait
                  _visitors_floc_
            | Pexpr_mutate
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  field = _visitors_ffield;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_mutate env _visitors_frecord
                  _visitors_faccessor _visitors_ffield _visitors_faugmented_by
                  _visitors_floc_
            | Pexpr_match
                {
                  expr = _visitors_fexpr;
                  cases = _visitors_fcases;
                  match_loc_ = _visitors_fmatch_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_match env _visitors_fexpr _visitors_fcases
                  _visitors_fmatch_loc_ _visitors_floc_
            | Pexpr_letmut
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letmut env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fbody _visitors_floc_
            | Pexpr_pipe
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_pipe env _visitors_flhs _visitors_frhs
                  _visitors_floc_
            | Pexpr_assign
                {
                  var = _visitors_fvar;
                  expr = _visitors_fexpr;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_assign env _visitors_fvar _visitors_fexpr
                  _visitors_faugmented_by _visitors_floc_
            | Pexpr_hole { loc_ = _visitors_floc_; kind = _visitors_fkind } ->
                self#visit_Pexpr_hole env _visitors_floc_ _visitors_fkind
            | Pexpr_return
                {
                  return_value = _visitors_freturn_value;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_return env _visitors_freturn_value
                  _visitors_floc_
            | Pexpr_raise
                { err_value = _visitors_ferr_value; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_raise env _visitors_ferr_value _visitors_floc_
            | Pexpr_unit { loc_ = _visitors_floc_; faked = _visitors_ffaked } ->
                self#visit_Pexpr_unit env _visitors_floc_ _visitors_ffaked
            | Pexpr_break { arg = _visitors_farg; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_break env _visitors_farg _visitors_floc_
            | Pexpr_continue { args = _visitors_fargs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_continue env _visitors_fargs _visitors_floc_
            | Pexpr_loop
                {
                  args = _visitors_fargs;
                  body = _visitors_fbody;
                  loop_loc_ = _visitors_floop_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_loop env _visitors_fargs _visitors_fbody
                  _visitors_floop_loc_ _visitors_floc_
            | Pexpr_for
                {
                  binders = _visitors_fbinders;
                  condition = _visitors_fcondition;
                  continue_block = _visitors_fcontinue_block;
                  body = _visitors_fbody;
                  for_else = _visitors_ffor_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_for env _visitors_fbinders _visitors_fcondition
                  _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
                  _visitors_floc_
            | Pexpr_foreach
                {
                  binders = _visitors_fbinders;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  else_block = _visitors_felse_block;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_foreach env _visitors_fbinders _visitors_fexpr
                  _visitors_fbody _visitors_felse_block _visitors_floc_
            | Pexpr_try
                {
                  body = _visitors_fbody;
                  catch = _visitors_fcatch;
                  catch_all = _visitors_fcatch_all;
                  try_else = _visitors_ftry_else;
                  try_loc_ = _visitors_ftry_loc_;
                  catch_loc_ = _visitors_fcatch_loc_;
                  else_loc_ = _visitors_felse_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_try env _visitors_fbody _visitors_fcatch
                  _visitors_fcatch_all _visitors_ftry_else _visitors_ftry_loc_
                  _visitors_fcatch_loc_ _visitors_felse_loc_ _visitors_floc_
            | Pexpr_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_map env _visitors_felems _visitors_floc_
            | Pexpr_group
                {
                  expr = _visitors_fexpr;
                  group = _visitors_fgroup;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_group env _visitors_fexpr _visitors_fgroup
                  _visitors_floc_
            | Pexpr_static_assert
                { asserts = _visitors_fasserts; body = _visitors_fbody } ->
                self#visit_Pexpr_static_assert env _visitors_fasserts
                  _visitors_fbody

        method visit_static_assertion : _ -> static_assertion -> unit =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.assert_type in
            let _visitors_r1 =
              self#visit_longident env _visitors_this.assert_trait
            in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_this.assert_loc
            in
            let _visitors_r3 =
              (fun _visitors_this -> ()) _visitors_this.assert_msg
            in
            ()

        method visit_argument : _ -> argument -> unit =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_expr env _visitors_this.arg_value in
            let _visitors_r1 =
              self#visit_argument_kind env _visitors_this.arg_kind
            in
            ()

        method visit_parameters : _ -> parameters -> unit =
          fun env _visitors_this ->
            Basic_lst.iter _visitors_this (self#visit_parameter env)

        method visit_parameter : _ -> parameter -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.param_binder
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_typ env) t
                | None -> ())
                _visitors_this.param_annot
            in
            let _visitors_r2 =
              self#visit_parameter_kind env _visitors_this.param_kind
            in
            ()

        method visit_Positional : _ -> unit = fun env -> ()
        method visit_Labelled : _ -> unit = fun env -> ()

        method visit_Optional : _ -> expr -> unit =
          fun env _visitors_fdefault ->
            let _visitors_r0 = self#visit_expr env _visitors_fdefault in
            ()

        method visit_Question_optional : _ -> unit = fun env -> ()

        method visit_parameter_kind : _ -> parameter_kind -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Positional -> self#visit_Positional env
            | Labelled -> self#visit_Labelled env
            | Optional { default = _visitors_fdefault } ->
                self#visit_Optional env _visitors_fdefault
            | Question_optional -> self#visit_Question_optional env

        method visit_Lambda
            : _ ->
              parameters ->
              location ->
              expr ->
              (typ * error_typ) option ->
              fn_kind ->
              bool ->
              unit =
          fun env _visitors_fparameters _visitors_fparams_loc_ _visitors_fbody
              _visitors_freturn_type _visitors_fkind_ _visitors_fhas_error ->
            let _visitors_r0 =
              self#visit_parameters env _visitors_fparameters
            in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_fparams_loc_
            in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun (_visitors_c0, _visitors_c1) ->
                      let _visitors_r0 = self#visit_typ env _visitors_c0 in
                      let _visitors_r1 =
                        self#visit_error_typ env _visitors_c1
                      in
                      ())
                      t
                | None -> ())
                _visitors_freturn_type
            in
            let _visitors_r4 = self#visit_fn_kind env _visitors_fkind_ in
            let _visitors_r5 =
              (fun _visitors_this -> ()) _visitors_fhas_error
            in
            ()

        method visit_Match
            : _ ->
              (pattern list * expr) list ->
              bool ->
              location ->
              location ->
              unit =
          fun env _visitors_fcases _visitors_fhas_error _visitors_ffn_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 =
                      (fun _visitors_this ->
                        Basic_lst.iter _visitors_this (self#visit_pattern env))
                        _visitors_c0
                    in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    ()))
                _visitors_fcases
            in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_fhas_error
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_ffn_loc_ in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_func : _ -> func -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Lambda
                {
                  parameters = _visitors_fparameters;
                  params_loc_ = _visitors_fparams_loc_;
                  body = _visitors_fbody;
                  return_type = _visitors_freturn_type;
                  kind_ = _visitors_fkind_;
                  has_error = _visitors_fhas_error;
                } ->
                self#visit_Lambda env _visitors_fparameters
                  _visitors_fparams_loc_ _visitors_fbody _visitors_freturn_type
                  _visitors_fkind_ _visitors_fhas_error
            | Match
                {
                  cases = _visitors_fcases;
                  has_error = _visitors_fhas_error;
                  fn_loc_ = _visitors_ffn_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Match env _visitors_fcases _visitors_fhas_error
                  _visitors_ffn_loc_ _visitors_floc_

        method visit_Elem_regular : _ -> expr -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_expr env _visitors_c0 in
            ()

        method visit_Elem_spread : _ -> expr -> location -> unit =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_spreadable_elem : _ -> spreadable_elem -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Elem_regular _visitors_c0 ->
                self#visit_Elem_regular env _visitors_c0
            | Elem_spread { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Elem_spread env _visitors_fexpr _visitors_floc_

        method visit_Map_expr_elem
            : _ -> constant -> expr -> location -> location -> unit =
          fun env _visitors_fkey _visitors_fexpr _visitors_fkey_loc_
              _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_fkey in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_fkey_loc_ in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_map_expr_elem : _ -> map_expr_elem -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_expr_elem
                {
                  key = _visitors_fkey;
                  expr = _visitors_fexpr;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_expr_elem env _visitors_fkey _visitors_fexpr
                  _visitors_fkey_loc_ _visitors_floc_

        method visit_Error_typ : _ -> typ -> unit =
          fun env _visitors_fty ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            ()

        method visit_Default_error_typ : _ -> location -> unit =
          fun env _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_No_error_typ : _ -> unit = fun env -> ()

        method visit_error_typ : _ -> error_typ -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Error_typ { ty = _visitors_fty } ->
                self#visit_Error_typ env _visitors_fty
            | Default_error_typ { loc_ = _visitors_floc_ } ->
                self#visit_Default_error_typ env _visitors_floc_
            | No_error_typ -> self#visit_No_error_typ env

        method visit_Ptype_any : _ -> location -> unit =
          fun env _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptype_arrow
            : _ -> typ list -> typ -> error_typ -> location -> unit =
          fun env _visitors_fty_arg _visitors_fty_res _visitors_fty_err
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_typ env))
                _visitors_fty_arg
            in
            let _visitors_r1 = self#visit_typ env _visitors_fty_res in
            let _visitors_r2 = self#visit_error_typ env _visitors_fty_err in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptype_tuple : _ -> typ list -> location -> unit =
          fun env _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_typ env))
                _visitors_ftys
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptype_name
            : _ -> constrid_loc -> typ list -> location -> unit =
          fun env _visitors_fconstr_id _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_constrid_loc env _visitors_fconstr_id
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_typ env))
                _visitors_ftys
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptype_option : _ -> typ -> location -> location -> unit =
          fun env _visitors_fty _visitors_floc_ _visitors_fquestion_loc ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_fquestion_loc
            in
            ()

        method visit_typ : _ -> typ -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptype_any { loc_ = _visitors_floc_ } ->
                self#visit_Ptype_any env _visitors_floc_
            | Ptype_arrow
                {
                  ty_arg = _visitors_fty_arg;
                  ty_res = _visitors_fty_res;
                  ty_err = _visitors_fty_err;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_arrow env _visitors_fty_arg _visitors_fty_res
                  _visitors_fty_err _visitors_floc_
            | Ptype_tuple { tys = _visitors_ftys; loc_ = _visitors_floc_ } ->
                self#visit_Ptype_tuple env _visitors_ftys _visitors_floc_
            | Ptype_name
                {
                  constr_id = _visitors_fconstr_id;
                  tys = _visitors_ftys;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_name env _visitors_fconstr_id _visitors_ftys
                  _visitors_floc_
            | Ptype_option
                {
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                  question_loc = _visitors_fquestion_loc;
                } ->
                self#visit_Ptype_option env _visitors_fty _visitors_floc_
                  _visitors_fquestion_loc

        method visit_Ppat_alias : _ -> pattern -> binder -> location -> unit =
          fun env _visitors_fpat _visitors_falias _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_binder env _visitors_falias in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_any : _ -> location -> unit =
          fun env _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_array : _ -> array_pattern -> location -> unit =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 = self#visit_array_pattern env _visitors_fpats in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_constant : _ -> constant -> location -> unit =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_fc in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_constraint : _ -> pattern -> typ -> location -> unit =
          fun env _visitors_fpat _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_constr
            : _ ->
              constructor ->
              constr_pat_arg list option ->
              bool ->
              location ->
              unit =
          fun env _visitors_fconstr _visitors_fargs _visitors_fis_open
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun _visitors_this ->
                      Basic_lst.iter _visitors_this
                        (self#visit_constr_pat_arg env))
                      t
                | None -> ())
                _visitors_fargs
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_fis_open in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_or : _ -> pattern -> pattern -> location -> unit =
          fun env _visitors_fpat1 _visitors_fpat2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat1 in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat2 in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_tuple : _ -> pattern list -> location -> unit =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_pattern env))
                _visitors_fpats
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_var : _ -> binder -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_binder env _visitors_c0 in
            ()

        method visit_Ppat_record
            : _ -> field_pat list -> bool -> location -> unit =
          fun env _visitors_ffields _visitors_fis_closed _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_field_pat env))
                _visitors_ffields
            in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_fis_closed
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_map : _ -> map_pat_elem list -> location -> unit =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_map_pat_elem env))
                _visitors_felems
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ppat_range
            : _ -> pattern -> pattern -> bool -> location -> unit =
          fun env _visitors_flhs _visitors_frhs _visitors_finclusive
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_flhs in
            let _visitors_r1 = self#visit_pattern env _visitors_frhs in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_finclusive
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_pattern : _ -> pattern -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Ppat_alias
                {
                  pat = _visitors_fpat;
                  alias = _visitors_falias;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_alias env _visitors_fpat _visitors_falias
                  _visitors_floc_
            | Ppat_any { loc_ = _visitors_floc_ } ->
                self#visit_Ppat_any env _visitors_floc_
            | Ppat_array { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_array env _visitors_fpats _visitors_floc_
            | Ppat_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_constant env _visitors_fc _visitors_floc_
            | Ppat_constraint
                {
                  pat = _visitors_fpat;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constraint env _visitors_fpat _visitors_fty
                  _visitors_floc_
            | Ppat_constr
                {
                  constr = _visitors_fconstr;
                  args = _visitors_fargs;
                  is_open = _visitors_fis_open;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constr env _visitors_fconstr _visitors_fargs
                  _visitors_fis_open _visitors_floc_
            | Ppat_or
                {
                  pat1 = _visitors_fpat1;
                  pat2 = _visitors_fpat2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_or env _visitors_fpat1 _visitors_fpat2
                  _visitors_floc_
            | Ppat_tuple { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_tuple env _visitors_fpats _visitors_floc_
            | Ppat_var _visitors_c0 -> self#visit_Ppat_var env _visitors_c0
            | Ppat_record
                {
                  fields = _visitors_ffields;
                  is_closed = _visitors_fis_closed;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_record env _visitors_ffields
                  _visitors_fis_closed _visitors_floc_
            | Ppat_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_map env _visitors_felems _visitors_floc_
            | Ppat_range
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  inclusive = _visitors_finclusive;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_range env _visitors_flhs _visitors_frhs
                  _visitors_finclusive _visitors_floc_

        method visit_Closed : _ -> pattern list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_pattern env))
                _visitors_c0
            in
            ()

        method visit_Open
            : _ -> pattern list -> pattern list -> binder option -> unit =
          fun env _visitors_c0 _visitors_c1 _visitors_c2 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_pattern env))
                _visitors_c0
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_pattern env))
                _visitors_c1
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_binder env) t
                | None -> ())
                _visitors_c2
            in
            ()

        method visit_array_pattern : _ -> array_pattern -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Closed _visitors_c0 -> self#visit_Closed env _visitors_c0
            | Open (_visitors_c0, _visitors_c1, _visitors_c2) ->
                self#visit_Open env _visitors_c0 _visitors_c1 _visitors_c2

        method visit_Field_def : _ -> label -> expr -> bool -> location -> unit
            =
          fun env _visitors_flabel _visitors_fexpr _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_fis_pun in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_field_def : _ -> field_def -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_def
                {
                  label = _visitors_flabel;
                  expr = _visitors_fexpr;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_def env _visitors_flabel _visitors_fexpr
                  _visitors_fis_pun _visitors_floc_

        method visit_Field_pat
            : _ -> label -> pattern -> bool -> location -> unit =
          fun env _visitors_flabel _visitors_fpattern _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_fis_pun in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_field_pat : _ -> field_pat -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_pat
                {
                  label = _visitors_flabel;
                  pattern = _visitors_fpattern;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_pat env _visitors_flabel _visitors_fpattern
                  _visitors_fis_pun _visitors_floc_

        method visit_Constr_pat_arg : _ -> pattern -> argument_kind -> unit =
          fun env _visitors_fpat _visitors_fkind ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_argument_kind env _visitors_fkind in
            ()

        method visit_constr_pat_arg : _ -> constr_pat_arg -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Constr_pat_arg { pat = _visitors_fpat; kind = _visitors_fkind } ->
                self#visit_Constr_pat_arg env _visitors_fpat _visitors_fkind

        method visit_Map_pat_elem
            : _ -> constant -> pattern -> bool -> location -> location -> unit =
          fun env _visitors_fkey _visitors_fpat _visitors_fmatch_absent
              _visitors_fkey_loc_ _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_fkey in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_fmatch_absent
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_fkey_loc_ in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_map_pat_elem : _ -> map_pat_elem -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_pat_elem
                {
                  key = _visitors_fkey;
                  pat = _visitors_fpat;
                  match_absent = _visitors_fmatch_absent;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_pat_elem env _visitors_fkey _visitors_fpat
                  _visitors_fmatch_absent _visitors_fkey_loc_ _visitors_floc_

        method visit_constr_param : _ -> constr_param -> unit =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.cparam_typ in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_this.cparam_mut
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_label env) t
                | None -> ())
                _visitors_this.cparam_label
            in
            ()

        method visit_constr_decl : _ -> constr_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_constr_name env _visitors_this.constr_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun _visitors_this ->
                      Basic_lst.iter _visitors_this
                        (self#visit_constr_param env))
                      t
                | None -> ())
                _visitors_this.constr_args
            in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_this.constr_loc_
            in
            ()

        method visit_field_name : _ -> field_name -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> ()) _visitors_this.label
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_this.loc_ in
            ()

        method visit_field_decl : _ -> field_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_field_name env _visitors_this.field_name
            in
            let _visitors_r1 = self#visit_typ env _visitors_this.field_ty in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_this.field_mut
            in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.field_vis
            in
            let _visitors_r4 =
              (fun _visitors_this -> ()) _visitors_this.field_loc_
            in
            ()

        method visit_No_payload : _ -> unit = fun env -> ()

        method visit_Single_payload : _ -> typ -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            ()

        method visit_Enum_payload : _ -> constr_decl list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_constr_decl env))
                _visitors_c0
            in
            ()

        method visit_exception_decl : _ -> exception_decl -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | No_payload -> self#visit_No_payload env
            | Single_payload _visitors_c0 ->
                self#visit_Single_payload env _visitors_c0
            | Enum_payload _visitors_c0 ->
                self#visit_Enum_payload env _visitors_c0

        method visit_Ptd_abstract : _ -> unit = fun env -> ()

        method visit_Ptd_newtype : _ -> typ -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            ()

        method visit_Ptd_error : _ -> exception_decl -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_exception_decl env _visitors_c0 in
            ()

        method visit_Ptd_variant : _ -> constr_decl list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_constr_decl env))
                _visitors_c0
            in
            ()

        method visit_Ptd_record : _ -> field_decl list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_field_decl env))
                _visitors_c0
            in
            ()

        method visit_Ptd_alias : _ -> typ -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            ()

        method visit_type_desc : _ -> type_desc -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptd_abstract -> self#visit_Ptd_abstract env
            | Ptd_newtype _visitors_c0 ->
                self#visit_Ptd_newtype env _visitors_c0
            | Ptd_error _visitors_c0 -> self#visit_Ptd_error env _visitors_c0
            | Ptd_variant _visitors_c0 ->
                self#visit_Ptd_variant env _visitors_c0
            | Ptd_record _visitors_c0 -> self#visit_Ptd_record env _visitors_c0
            | Ptd_alias _visitors_c0 -> self#visit_Ptd_alias env _visitors_c0

        method visit_type_decl : _ -> type_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> ()) _visitors_this.tycon
            in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_this.tycon_loc_
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_type_decl_binder env))
                _visitors_this.params
            in
            let _visitors_r3 =
              self#visit_type_desc env _visitors_this.components
            in
            let _visitors_r4 = self#visit_docstring env _visitors_this.doc_ in
            let _visitors_r5 =
              self#visit_visibility env _visitors_this.type_vis
            in
            let _visitors_r6 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (self#visit_deriving_directive env))
                _visitors_this.deriving_
            in
            let _visitors_r7 = (fun _visitors_this -> ()) _visitors_this.loc_ in
            ()

        method visit_local_type_decl : _ -> local_type_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> ()) _visitors_this.local_tycon
            in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_this.local_tycon_loc_
            in
            let _visitors_r2 =
              self#visit_type_desc env _visitors_this.local_components
            in
            ()

        method visit_deriving_directive : _ -> deriving_directive -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_type_name env _visitors_this.type_name_
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_argument env))
                _visitors_this.args
            in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_this.loc_ in
            ()

        method visit_Vis_default : _ -> unit = fun env -> ()

        method visit_Vis_pub : _ -> string option -> location -> unit =
          fun env _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (fun _visitors_this -> ()) t
                | None -> ())
                _visitors_fattr
            in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Vis_priv : _ -> location -> unit =
          fun env _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_visibility : _ -> visibility -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Vis_default -> self#visit_Vis_default env
            | Vis_pub { attr = _visitors_fattr; loc_ = _visitors_floc_ } ->
                self#visit_Vis_pub env _visitors_fattr _visitors_floc_
            | Vis_priv { loc_ = _visitors_floc_ } ->
                self#visit_Vis_priv env _visitors_floc_

        method visit_Import : _ -> string_literal -> string_literal -> unit =
          fun env _visitors_fmodule_name _visitors_ffunc_name ->
            let _visitors_r0 =
              self#visit_string_literal env _visitors_fmodule_name
            in
            let _visitors_r1 =
              self#visit_string_literal env _visitors_ffunc_name
            in
            ()

        method visit_Embedded
            : _ -> string_literal option -> embedded_code -> unit =
          fun env _visitors_flanguage _visitors_fcode ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_string_literal env) t
                | None -> ())
                _visitors_flanguage
            in
            let _visitors_r1 = self#visit_embedded_code env _visitors_fcode in
            ()

        method visit_func_stubs : _ -> func_stubs -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Import
                {
                  module_name = _visitors_fmodule_name;
                  func_name = _visitors_ffunc_name;
                } ->
                self#visit_Import env _visitors_fmodule_name
                  _visitors_ffunc_name
            | Embedded
                { language = _visitors_flanguage; code = _visitors_fcode } ->
                self#visit_Embedded env _visitors_flanguage _visitors_fcode

        method visit_Code_string : _ -> string_literal -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_string_literal env _visitors_c0 in
            ()

        method visit_Code_multiline_string : _ -> string list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (fun _visitors_this -> ()))
                _visitors_c0
            in
            ()

        method visit_embedded_code : _ -> embedded_code -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Code_string _visitors_c0 ->
                self#visit_Code_string env _visitors_c0
            | Code_multiline_string _visitors_c0 ->
                self#visit_Code_multiline_string env _visitors_c0

        method visit_Decl_body : _ -> local_type_decl list -> expr -> unit =
          fun env _visitors_flocal_types _visitors_fexpr ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            ()

        method visit_Decl_stubs : _ -> func_stubs -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_func_stubs env _visitors_c0 in
            ()

        method visit_decl_body : _ -> decl_body -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Decl_body
                { local_types = _visitors_flocal_types; expr = _visitors_fexpr }
              ->
                self#visit_Decl_body env _visitors_flocal_types _visitors_fexpr
            | Decl_stubs _visitors_c0 -> self#visit_Decl_stubs env _visitors_c0

        method visit_fun_decl : _ -> fun_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_type_name env) t
                | None -> ())
                _visitors_this.type_name
            in
            let _visitors_r1 = self#visit_binder env _visitors_this.name in
            let _visitors_r2 =
              (fun _visitors_this -> ()) _visitors_this.has_error
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_parameters env) t
                | None -> ())
                _visitors_this.decl_params
            in
            let _visitors_r4 =
              (fun _visitors_this -> ()) _visitors_this.params_loc_
            in
            let _visitors_r5 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_tvar_binder env))
                _visitors_this.quantifiers
            in
            let _visitors_r6 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun (_visitors_c0, _visitors_c1) ->
                      let _visitors_r0 = self#visit_typ env _visitors_c0 in
                      let _visitors_r1 =
                        self#visit_error_typ env _visitors_c1
                      in
                      ())
                      t
                | None -> ())
                _visitors_this.return_type
            in
            let _visitors_r7 =
              (fun _visitors_this -> ()) _visitors_this.is_pub
            in
            let _visitors_r8 = self#visit_docstring env _visitors_this.doc_ in
            ()

        method visit_trait_method_param : _ -> trait_method_param -> unit =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.tmparam_typ in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_label env) t
                | None -> ())
                _visitors_this.tmparam_label
            in
            ()

        method visit_Trait_method
            : _ ->
              binder ->
              bool ->
              tvar_binder list ->
              trait_method_param list ->
              (typ * error_typ) option ->
              location ->
              unit =
          fun env _visitors_fname _visitors_fhas_error _visitors_fquantifiers
              _visitors_fparams _visitors_freturn_type _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 =
              (fun _visitors_this -> ()) _visitors_fhas_error
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this
                  (self#visit_trait_method_param env))
                _visitors_fparams
            in
            let _visitors_r4 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun (_visitors_c0, _visitors_c1) ->
                      let _visitors_r0 = self#visit_typ env _visitors_c0 in
                      let _visitors_r1 =
                        self#visit_error_typ env _visitors_c1
                      in
                      ())
                      t
                | None -> ())
                _visitors_freturn_type
            in
            let _visitors_r5 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_trait_method_decl : _ -> trait_method_decl -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Trait_method
                {
                  name = _visitors_fname;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  return_type = _visitors_freturn_type;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Trait_method env _visitors_fname _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams
                  _visitors_freturn_type _visitors_floc_

        method visit_trait_decl : _ -> trait_decl -> unit =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.trait_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_tvar_constraint env))
                _visitors_this.trait_supers
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_trait_method_decl env))
                _visitors_this.trait_methods
            in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.trait_vis
            in
            let _visitors_r4 =
              (fun _visitors_this -> ()) _visitors_this.trait_loc_
            in
            let _visitors_r5 =
              self#visit_docstring env _visitors_this.trait_doc_
            in
            ()

        method visit_Ptop_expr
            : _ -> expr -> bool -> local_type_decl list -> absolute_loc -> unit
            =
          fun env _visitors_fexpr _visitors_fis_main _visitors_flocal_types
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_fis_main in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptop_test
            : _ ->
              expr ->
              test_name ->
              parameters option ->
              local_type_decl list ->
              absolute_loc ->
              unit =
          fun env _visitors_fexpr _visitors_fname _visitors_fparams
              _visitors_flocal_types _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_fname in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_parameters env) t
                | None -> ())
                _visitors_fparams
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptop_typedef : _ -> type_decl -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_type_decl env _visitors_c0 in
            ()

        method visit_Ptop_funcdef
            : _ -> fun_decl -> decl_body -> absolute_loc -> unit =
          fun env _visitors_ffun_decl _visitors_fdecl_body _visitors_floc_ ->
            let _visitors_r0 = self#visit_fun_decl env _visitors_ffun_decl in
            let _visitors_r1 = self#visit_decl_body env _visitors_fdecl_body in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Ptop_letdef
            : _ ->
              binder ->
              typ option ->
              expr ->
              bool ->
              bool ->
              absolute_loc ->
              docstring ->
              unit =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fis_pub _visitors_fis_constant _visitors_floc_
              _visitors_fdoc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_typ env) t
                | None -> ())
                _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_fis_pub in
            let _visitors_r4 =
              (fun _visitors_this -> ()) _visitors_fis_constant
            in
            let _visitors_r5 = (fun _visitors_this -> ()) _visitors_floc_ in
            let _visitors_r6 = self#visit_docstring env _visitors_fdoc_ in
            ()

        method visit_Ptop_trait : _ -> trait_decl -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_trait_decl env _visitors_c0 in
            ()

        method visit_Ptop_impl
            : _ ->
              typ option ->
              type_name ->
              binder ->
              bool ->
              tvar_binder list ->
              parameters ->
              (typ * error_typ) option ->
              expr ->
              bool ->
              local_type_decl list ->
              absolute_loc ->
              location ->
              docstring ->
              unit =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fmethod_name
              _visitors_fhas_error _visitors_fquantifiers _visitors_fparams
              _visitors_fret_ty _visitors_fbody _visitors_fis_pub
              _visitors_flocal_types _visitors_floc_ _visitors_fheader_loc_
              _visitors_fdoc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> (self#visit_typ env) t
                | None -> ())
                _visitors_fself_ty
            in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 = self#visit_binder env _visitors_fmethod_name in
            let _visitors_r3 =
              (fun _visitors_this -> ()) _visitors_fhas_error
            in
            let _visitors_r4 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r5 = self#visit_parameters env _visitors_fparams in
            let _visitors_r6 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    (fun (_visitors_c0, _visitors_c1) ->
                      let _visitors_r0 = self#visit_typ env _visitors_c0 in
                      let _visitors_r1 =
                        self#visit_error_typ env _visitors_c1
                      in
                      ())
                      t
                | None -> ())
                _visitors_fret_ty
            in
            let _visitors_r7 = self#visit_expr env _visitors_fbody in
            let _visitors_r8 = (fun _visitors_this -> ()) _visitors_fis_pub in
            let _visitors_r9 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r10 = (fun _visitors_this -> ()) _visitors_floc_ in
            let _visitors_r11 =
              (fun _visitors_this -> ()) _visitors_fheader_loc_
            in
            let _visitors_r12 = self#visit_docstring env _visitors_fdoc_ in
            ()

        method visit_Ptop_impl_relation
            : _ ->
              typ ->
              type_name ->
              tvar_binder list ->
              bool ->
              absolute_loc ->
              unit =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fquantifiers
              _visitors_fis_pub _visitors_floc_ ->
            let _visitors_r0 = self#visit_typ env _visitors_fself_ty in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r3 = (fun _visitors_this -> ()) _visitors_fis_pub in
            let _visitors_r4 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_impl : _ -> impl -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptop_expr
                {
                  expr = _visitors_fexpr;
                  is_main = _visitors_fis_main;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_expr env _visitors_fexpr _visitors_fis_main
                  _visitors_flocal_types _visitors_floc_
            | Ptop_test
                {
                  expr = _visitors_fexpr;
                  name = _visitors_fname;
                  params = _visitors_fparams;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_test env _visitors_fexpr _visitors_fname
                  _visitors_fparams _visitors_flocal_types _visitors_floc_
            | Ptop_typedef _visitors_c0 ->
                self#visit_Ptop_typedef env _visitors_c0
            | Ptop_funcdef
                {
                  fun_decl = _visitors_ffun_decl;
                  decl_body = _visitors_fdecl_body;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_funcdef env _visitors_ffun_decl
                  _visitors_fdecl_body _visitors_floc_
            | Ptop_letdef
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  is_pub = _visitors_fis_pub;
                  is_constant = _visitors_fis_constant;
                  loc_ = _visitors_floc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_letdef env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fis_pub _visitors_fis_constant
                  _visitors_floc_ _visitors_fdoc_
            | Ptop_trait _visitors_c0 -> self#visit_Ptop_trait env _visitors_c0
            | Ptop_impl
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  method_name = _visitors_fmethod_name;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  ret_ty = _visitors_fret_ty;
                  body = _visitors_fbody;
                  is_pub = _visitors_fis_pub;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                  header_loc_ = _visitors_fheader_loc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_impl env _visitors_fself_ty _visitors_ftrait
                  _visitors_fmethod_name _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams _visitors_fret_ty
                  _visitors_fbody _visitors_fis_pub _visitors_flocal_types
                  _visitors_floc_ _visitors_fheader_loc_ _visitors_fdoc_
            | Ptop_impl_relation
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  quantifiers = _visitors_fquantifiers;
                  is_pub = _visitors_fis_pub;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_impl_relation env _visitors_fself_ty
                  _visitors_ftrait _visitors_fquantifiers _visitors_fis_pub
                  _visitors_floc_

        method visit_Interp_lit : _ -> string -> string -> location -> unit =
          fun env _visitors_fstr _visitors_frepr _visitors_floc_ ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_fstr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_frepr in
            let _visitors_r2 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Interp_expr : _ -> expr -> location -> unit =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = (fun _visitors_this -> ()) _visitors_floc_ in
            ()

        method visit_Interp_source : _ -> Literal.interp_source -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_interp_source env _visitors_c0 in
            ()

        method visit_interp_elem : _ -> interp_elem -> unit =
          fun env _visitors_this ->
            match _visitors_this with
            | Interp_lit
                {
                  str = _visitors_fstr;
                  repr = _visitors_frepr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Interp_lit env _visitors_fstr _visitors_frepr
                  _visitors_floc_
            | Interp_expr { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Interp_expr env _visitors_fexpr _visitors_floc_
            | Interp_source _visitors_c0 ->
                self#visit_Interp_source env _visitors_c0

        method visit_Multiline_string : _ -> string -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 = (fun _visitors_this -> ()) _visitors_c0 in
            ()

        method visit_Multiline_interp : _ -> interp_elem list -> unit =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.iter _visitors_this (self#visit_interp_elem env))
                _visitors_c0
            in
            ()

        method visit_multiline_string_elem : _ -> multiline_string_elem -> unit
            =
          fun env _visitors_this ->
            match _visitors_this with
            | Multiline_string _visitors_c0 ->
                self#visit_Multiline_string env _visitors_c0
            | Multiline_interp _visitors_c0 ->
                self#visit_Multiline_interp env _visitors_c0

        method visit_impls : _ -> impls -> unit =
          fun env _visitors_this ->
            Basic_lst.iter _visitors_this (self#visit_impl env)
      end

    [@@@VISITORS.END]
  end

  include struct
    [@@@ocaml.warning "-4-26-27"]
    [@@@VISITORS.BEGIN]

    class virtual ['self] map =
      object (self : 'self)
        inherit [_] mapbase

        method visit_Pexpr_apply
            : _ -> expr -> argument list -> apply_attr -> location -> expr =
          fun env _visitors_ffunc _visitors_fargs _visitors_fattr
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ffunc in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_argument env))
                _visitors_fargs
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fattr
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_apply
              {
                func = _visitors_r0;
                args = _visitors_r1;
                attr = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_infix : _ -> var -> expr -> expr -> location -> expr
            =
          fun env _visitors_fop _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_flhs in
            let _visitors_r2 = self#visit_expr env _visitors_frhs in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_infix
              {
                op = _visitors_r0;
                lhs = _visitors_r1;
                rhs = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_unary : _ -> var -> expr -> location -> expr =
          fun env _visitors_fop _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_unary
              { op = _visitors_r0; expr = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Pexpr_array : _ -> expr list -> location -> expr =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_expr env))
                _visitors_fexprs
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array { exprs = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_array_spread
            : _ -> spreadable_elem list -> location -> expr =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_spreadable_elem env))
                _visitors_felems
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array_spread { elems = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_array_get : _ -> expr -> expr -> location -> expr =
          fun env _visitors_farray _visitors_findex _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array_get
              {
                array = _visitors_r0;
                index = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Pexpr_array_get_slice
            : _ ->
              expr ->
              expr option ->
              expr option ->
              location ->
              location ->
              expr =
          fun env _visitors_farray _visitors_fstart_index _visitors_fend_index
              _visitors_findex_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fstart_index
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fend_index
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_findex_loc_
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array_get_slice
              {
                array = _visitors_r0;
                start_index = _visitors_r1;
                end_index = _visitors_r2;
                index_loc_ = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_array_set
            : _ -> expr -> expr -> expr -> location -> expr =
          fun env _visitors_farray _visitors_findex _visitors_fvalue
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_farray in
            let _visitors_r1 = self#visit_expr env _visitors_findex in
            let _visitors_r2 = self#visit_expr env _visitors_fvalue in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array_set
              {
                array = _visitors_r0;
                index = _visitors_r1;
                value = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_array_augmented_set
            : _ -> var -> expr -> expr -> expr -> location -> expr =
          fun env _visitors_fop _visitors_farray _visitors_findex
              _visitors_fvalue _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fop in
            let _visitors_r1 = self#visit_expr env _visitors_farray in
            let _visitors_r2 = self#visit_expr env _visitors_findex in
            let _visitors_r3 = self#visit_expr env _visitors_fvalue in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_array_augmented_set
              {
                op = _visitors_r0;
                array = _visitors_r1;
                index = _visitors_r2;
                value = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_constant : _ -> constant -> location -> expr =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_fc
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_constant { c = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_multiline_string
            : _ -> multiline_string_elem list -> location -> expr =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (self#visit_multiline_string_elem env))
                _visitors_felems
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_multiline_string { elems = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_interp : _ -> interp_elem list -> location -> expr =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_interp_elem env))
                _visitors_felems
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_interp { elems = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_constraint : _ -> expr -> typ -> location -> expr =
          fun env _visitors_fexpr _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_constraint
              { expr = _visitors_r0; ty = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Pexpr_constr : _ -> constructor -> location -> expr =
          fun env _visitors_fconstr _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_constr { constr = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_while
            : _ -> expr -> expr -> expr option -> location -> expr =
          fun env _visitors_floop_cond _visitors_floop_body
              _visitors_fwhile_else _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_floop_cond in
            let _visitors_r1 = self#visit_expr env _visitors_floop_body in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fwhile_else
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_while
              {
                loop_cond = _visitors_r0;
                loop_body = _visitors_r1;
                while_else = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_function : _ -> func -> location -> expr =
          fun env _visitors_ffunc _visitors_floc_ ->
            let _visitors_r0 = self#visit_func env _visitors_ffunc in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_function { func = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_ident : _ -> var -> location -> expr =
          fun env _visitors_fid _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fid in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_ident { id = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_if
            : _ -> expr -> expr -> expr option -> location -> expr =
          fun env _visitors_fcond _visitors_fifso _visitors_fifnot
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 = self#visit_expr env _visitors_fifso in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fifnot
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_if
              {
                cond = _visitors_r0;
                ifso = _visitors_r1;
                ifnot = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_guard
            : _ -> expr -> expr option -> expr -> location -> expr =
          fun env _visitors_fcond _visitors_fotherwise _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fcond in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fotherwise
            in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_guard
              {
                cond = _visitors_r0;
                otherwise = _visitors_r1;
                body = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_guard_let
            : _ ->
              pattern ->
              expr ->
              (pattern * expr) list option ->
              expr ->
              location ->
              expr =
          fun env _visitors_fpat _visitors_fexpr _visitors_fotherwise
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun _visitors_this ->
                         Basic_lst.map _visitors_this
                           (fun (_visitors_c0, _visitors_c1) ->
                             let _visitors_r0 =
                               self#visit_pattern env _visitors_c0
                             in
                             let _visitors_r1 =
                               self#visit_expr env _visitors_c1
                             in
                             (_visitors_r0, _visitors_r1)))
                         t)
                | None -> None)
                _visitors_fotherwise
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_guard_let
              {
                pat = _visitors_r0;
                expr = _visitors_r1;
                otherwise = _visitors_r2;
                body = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_letfn
            : _ -> binder -> func -> expr -> location -> expr =
          fun env _visitors_fname _visitors_ffunc _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 = self#visit_func env _visitors_ffunc in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_letfn
              {
                name = _visitors_r0;
                func = _visitors_r1;
                body = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_letrec
            : _ -> (binder * func) list -> expr -> location -> expr =
          fun env _visitors_fbindings _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_func env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fbindings
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_letrec
              {
                bindings = _visitors_r0;
                body = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Pexpr_let
            : _ -> pattern -> expr -> expr -> location -> expr =
          fun env _visitors_fpattern _visitors_fexpr _visitors_fbody
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_let
              {
                pattern = _visitors_r0;
                expr = _visitors_r1;
                body = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_sequence : _ -> expr -> expr -> location -> expr =
          fun env _visitors_fexpr1 _visitors_fexpr2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr1 in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr2 in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_sequence
              {
                expr1 = _visitors_r0;
                expr2 = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Pexpr_tuple : _ -> expr list -> location -> expr =
          fun env _visitors_fexprs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_expr env))
                _visitors_fexprs
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_tuple { exprs = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_record
            : _ ->
              type_name option ->
              field_def list ->
              trailing_mark ->
              location ->
              expr =
          fun env _visitors_ftype_name _visitors_ffields _visitors_ftrailing
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_type_name env) t)
                | None -> None)
                _visitors_ftype_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_field_def env))
                _visitors_ffields
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_ftrailing
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_record
              {
                type_name = _visitors_r0;
                fields = _visitors_r1;
                trailing = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_record_update
            : _ ->
              type_name option ->
              expr ->
              field_def list ->
              location ->
              expr =
          fun env _visitors_ftype_name _visitors_frecord _visitors_ffields
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_type_name env) t)
                | None -> None)
                _visitors_ftype_name
            in
            let _visitors_r1 = self#visit_expr env _visitors_frecord in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_field_def env))
                _visitors_ffields
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_record_update
              {
                type_name = _visitors_r0;
                record = _visitors_r1;
                fields = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_field : _ -> expr -> accessor -> location -> expr =
          fun env _visitors_frecord _visitors_faccessor _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_field
              {
                record = _visitors_r0;
                accessor = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Pexpr_method : _ -> type_name -> label -> location -> expr
            =
          fun env _visitors_ftype_name _visitors_fmethod_name _visitors_floc_ ->
            let _visitors_r0 = self#visit_type_name env _visitors_ftype_name in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_method
              {
                type_name = _visitors_r0;
                method_name = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Pexpr_dot_apply
            : _ ->
              expr ->
              label ->
              argument list ->
              bool ->
              apply_attr ->
              location ->
              expr =
          fun env _visitors_fself _visitors_fmethod_name _visitors_fargs
              _visitors_freturn_self _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fself in
            let _visitors_r1 = self#visit_label env _visitors_fmethod_name in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_argument env))
                _visitors_fargs
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_freturn_self
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_fattr
            in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_dot_apply
              {
                self = _visitors_r0;
                method_name = _visitors_r1;
                args = _visitors_r2;
                return_self = _visitors_r3;
                attr = _visitors_r4;
                loc_ = _visitors_r5;
              }

        method visit_Pexpr_as : _ -> expr -> type_name -> location -> expr =
          fun env _visitors_fexpr _visitors_ftrait _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_as
              { expr = _visitors_r0; trait = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Pexpr_mutate
            : _ -> expr -> accessor -> expr -> var option -> location -> expr =
          fun env _visitors_frecord _visitors_faccessor _visitors_ffield
              _visitors_faugmented_by _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_frecord in
            let _visitors_r1 = self#visit_accessor env _visitors_faccessor in
            let _visitors_r2 = self#visit_expr env _visitors_ffield in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_var env) t)
                | None -> None)
                _visitors_faugmented_by
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_mutate
              {
                record = _visitors_r0;
                accessor = _visitors_r1;
                field = _visitors_r2;
                augmented_by = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_match
            : _ -> expr -> (pattern * expr) list -> location -> location -> expr
            =
          fun env _visitors_fexpr _visitors_fcases _visitors_fmatch_loc_
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fcases
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fmatch_loc_
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_match
              {
                expr = _visitors_r0;
                cases = _visitors_r1;
                match_loc_ = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_letmut
            : _ -> binder -> typ option -> expr -> expr -> location -> expr =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fbody _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_typ env) t)
                | None -> None)
                _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_letmut
              {
                binder = _visitors_r0;
                ty = _visitors_r1;
                expr = _visitors_r2;
                body = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_pipe : _ -> expr -> expr -> location -> expr =
          fun env _visitors_flhs _visitors_frhs _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_flhs in
            let _visitors_r1 = self#visit_expr env _visitors_frhs in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_pipe
              { lhs = _visitors_r0; rhs = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Pexpr_assign
            : _ -> var -> expr -> var option -> location -> expr =
          fun env _visitors_fvar _visitors_fexpr _visitors_faugmented_by
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_var env _visitors_fvar in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_var env) t)
                | None -> None)
                _visitors_faugmented_by
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_assign
              {
                var = _visitors_r0;
                expr = _visitors_r1;
                augmented_by = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_hole : _ -> location -> hole -> expr =
          fun env _visitors_floc_ _visitors_fkind ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            let _visitors_r1 = self#visit_hole env _visitors_fkind in
            Pexpr_hole { loc_ = _visitors_r0; kind = _visitors_r1 }

        method visit_Pexpr_return : _ -> expr option -> location -> expr =
          fun env _visitors_freturn_value _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_freturn_value
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_return { return_value = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_raise : _ -> expr -> location -> expr =
          fun env _visitors_ferr_value _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_ferr_value in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_raise { err_value = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_unit : _ -> location -> bool -> expr =
          fun env _visitors_floc_ _visitors_ffaked ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_ffaked
            in
            Pexpr_unit { loc_ = _visitors_r0; faked = _visitors_r1 }

        method visit_Pexpr_break : _ -> expr option -> location -> expr =
          fun env _visitors_farg _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_farg
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_break { arg = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_continue : _ -> expr list -> location -> expr =
          fun env _visitors_fargs _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_expr env))
                _visitors_fargs
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_continue { args = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_loop
            : _ ->
              expr list ->
              (pattern list * expr) list ->
              location ->
              location ->
              expr =
          fun env _visitors_fargs _visitors_fbody _visitors_floop_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_expr env))
                _visitors_fargs
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 =
                      (fun _visitors_this ->
                        Basic_lst.map _visitors_this (self#visit_pattern env))
                        _visitors_c0
                    in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fbody
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floop_loc_
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_loop
              {
                args = _visitors_r0;
                body = _visitors_r1;
                loop_loc_ = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Pexpr_for
            : _ ->
              (binder * expr) list ->
              expr option ->
              (binder * expr) list ->
              expr ->
              expr option ->
              location ->
              expr =
          fun env _visitors_fbinders _visitors_fcondition
              _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fbinders
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_fcondition
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_binder env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fcontinue_block
            in
            let _visitors_r3 = self#visit_expr env _visitors_fbody in
            let _visitors_r4 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_ffor_else
            in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_for
              {
                binders = _visitors_r0;
                condition = _visitors_r1;
                continue_block = _visitors_r2;
                body = _visitors_r3;
                for_else = _visitors_r4;
                loc_ = _visitors_r5;
              }

        method visit_Pexpr_foreach
            : _ ->
              binder option list ->
              expr ->
              expr ->
              expr option ->
              location ->
              expr =
          fun env _visitors_fbinders _visitors_fexpr _visitors_fbody
              _visitors_felse_block _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (fun _visitors_this ->
                    match _visitors_this with
                    | Some t -> Some ((self#visit_binder env) t)
                    | None -> None))
                _visitors_fbinders
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_expr env) t)
                | None -> None)
                _visitors_felse_block
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_foreach
              {
                binders = _visitors_r0;
                expr = _visitors_r1;
                body = _visitors_r2;
                else_block = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Pexpr_try
            : _ ->
              expr ->
              (pattern * expr) list ->
              bool ->
              (pattern * expr) list option ->
              location ->
              location ->
              location ->
              location ->
              expr =
          fun env _visitors_fbody _visitors_fcatch _visitors_fcatch_all
              _visitors_ftry_else _visitors_ftry_loc_ _visitors_fcatch_loc_
              _visitors_felse_loc_ _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fbody in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 = self#visit_pattern env _visitors_c0 in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fcatch
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fcatch_all
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun _visitors_this ->
                         Basic_lst.map _visitors_this
                           (fun (_visitors_c0, _visitors_c1) ->
                             let _visitors_r0 =
                               self#visit_pattern env _visitors_c0
                             in
                             let _visitors_r1 =
                               self#visit_expr env _visitors_c1
                             in
                             (_visitors_r0, _visitors_r1)))
                         t)
                | None -> None)
                _visitors_ftry_else
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_ftry_loc_
            in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_fcatch_loc_
            in
            let _visitors_r6 =
              (fun _visitors_this -> _visitors_this) _visitors_felse_loc_
            in
            let _visitors_r7 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_try
              {
                body = _visitors_r0;
                catch = _visitors_r1;
                catch_all = _visitors_r2;
                try_else = _visitors_r3;
                try_loc_ = _visitors_r4;
                catch_loc_ = _visitors_r5;
                else_loc_ = _visitors_r6;
                loc_ = _visitors_r7;
              }

        method visit_Pexpr_map : _ -> map_expr_elem list -> location -> expr =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_map_expr_elem env))
                _visitors_felems
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_map { elems = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Pexpr_group : _ -> expr -> group -> location -> expr =
          fun env _visitors_fexpr _visitors_fgroup _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fgroup
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Pexpr_group
              { expr = _visitors_r0; group = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Pexpr_static_assert
            : _ -> static_assertion list -> expr -> expr =
          fun env _visitors_fasserts _visitors_fbody ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_static_assertion env))
                _visitors_fasserts
            in
            let _visitors_r1 = self#visit_expr env _visitors_fbody in
            Pexpr_static_assert { asserts = _visitors_r0; body = _visitors_r1 }

        method visit_expr : _ -> expr -> expr =
          fun env _visitors_this ->
            match _visitors_this with
            | Pexpr_apply
                {
                  func = _visitors_ffunc;
                  args = _visitors_fargs;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_apply env _visitors_ffunc _visitors_fargs
                  _visitors_fattr _visitors_floc_
            | Pexpr_infix
                {
                  op = _visitors_fop;
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_infix env _visitors_fop _visitors_flhs
                  _visitors_frhs _visitors_floc_
            | Pexpr_unary
                {
                  op = _visitors_fop;
                  expr = _visitors_fexpr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_unary env _visitors_fop _visitors_fexpr
                  _visitors_floc_
            | Pexpr_array { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_array env _visitors_fexprs _visitors_floc_
            | Pexpr_array_spread
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_array_spread env _visitors_felems
                  _visitors_floc_
            | Pexpr_array_get
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get env _visitors_farray _visitors_findex
                  _visitors_floc_
            | Pexpr_array_get_slice
                {
                  array = _visitors_farray;
                  start_index = _visitors_fstart_index;
                  end_index = _visitors_fend_index;
                  index_loc_ = _visitors_findex_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_get_slice env _visitors_farray
                  _visitors_fstart_index _visitors_fend_index
                  _visitors_findex_loc_ _visitors_floc_
            | Pexpr_array_set
                {
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_set env _visitors_farray _visitors_findex
                  _visitors_fvalue _visitors_floc_
            | Pexpr_array_augmented_set
                {
                  op = _visitors_fop;
                  array = _visitors_farray;
                  index = _visitors_findex;
                  value = _visitors_fvalue;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_array_augmented_set env _visitors_fop
                  _visitors_farray _visitors_findex _visitors_fvalue
                  _visitors_floc_
            | Pexpr_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constant env _visitors_fc _visitors_floc_
            | Pexpr_multiline_string
                { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_multiline_string env _visitors_felems
                  _visitors_floc_
            | Pexpr_interp { elems = _visitors_felems; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_interp env _visitors_felems _visitors_floc_
            | Pexpr_constraint
                {
                  expr = _visitors_fexpr;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_constraint env _visitors_fexpr _visitors_fty
                  _visitors_floc_
            | Pexpr_constr
                { constr = _visitors_fconstr; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_constr env _visitors_fconstr _visitors_floc_
            | Pexpr_while
                {
                  loop_cond = _visitors_floop_cond;
                  loop_body = _visitors_floop_body;
                  while_else = _visitors_fwhile_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_while env _visitors_floop_cond
                  _visitors_floop_body _visitors_fwhile_else _visitors_floc_
            | Pexpr_function { func = _visitors_ffunc; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_function env _visitors_ffunc _visitors_floc_
            | Pexpr_ident { id = _visitors_fid; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_ident env _visitors_fid _visitors_floc_
            | Pexpr_if
                {
                  cond = _visitors_fcond;
                  ifso = _visitors_fifso;
                  ifnot = _visitors_fifnot;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_if env _visitors_fcond _visitors_fifso
                  _visitors_fifnot _visitors_floc_
            | Pexpr_guard
                {
                  cond = _visitors_fcond;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard env _visitors_fcond _visitors_fotherwise
                  _visitors_fbody _visitors_floc_
            | Pexpr_guard_let
                {
                  pat = _visitors_fpat;
                  expr = _visitors_fexpr;
                  otherwise = _visitors_fotherwise;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_guard_let env _visitors_fpat _visitors_fexpr
                  _visitors_fotherwise _visitors_fbody _visitors_floc_
            | Pexpr_letfn
                {
                  name = _visitors_fname;
                  func = _visitors_ffunc;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letfn env _visitors_fname _visitors_ffunc
                  _visitors_fbody _visitors_floc_
            | Pexpr_letrec
                {
                  bindings = _visitors_fbindings;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letrec env _visitors_fbindings _visitors_fbody
                  _visitors_floc_
            | Pexpr_let
                {
                  pattern = _visitors_fpattern;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_let env _visitors_fpattern _visitors_fexpr
                  _visitors_fbody _visitors_floc_
            | Pexpr_sequence
                {
                  expr1 = _visitors_fexpr1;
                  expr2 = _visitors_fexpr2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_sequence env _visitors_fexpr1 _visitors_fexpr2
                  _visitors_floc_
            | Pexpr_tuple { exprs = _visitors_fexprs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_tuple env _visitors_fexprs _visitors_floc_
            | Pexpr_record
                {
                  type_name = _visitors_ftype_name;
                  fields = _visitors_ffields;
                  trailing = _visitors_ftrailing;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record env _visitors_ftype_name
                  _visitors_ffields _visitors_ftrailing _visitors_floc_
            | Pexpr_record_update
                {
                  type_name = _visitors_ftype_name;
                  record = _visitors_frecord;
                  fields = _visitors_ffields;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_record_update env _visitors_ftype_name
                  _visitors_frecord _visitors_ffields _visitors_floc_
            | Pexpr_field
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_field env _visitors_frecord _visitors_faccessor
                  _visitors_floc_
            | Pexpr_method
                {
                  type_name = _visitors_ftype_name;
                  method_name = _visitors_fmethod_name;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_method env _visitors_ftype_name
                  _visitors_fmethod_name _visitors_floc_
            | Pexpr_dot_apply
                {
                  self = _visitors_fself;
                  method_name = _visitors_fmethod_name;
                  args = _visitors_fargs;
                  return_self = _visitors_freturn_self;
                  attr = _visitors_fattr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_dot_apply env _visitors_fself
                  _visitors_fmethod_name _visitors_fargs _visitors_freturn_self
                  _visitors_fattr _visitors_floc_
            | Pexpr_as
                {
                  expr = _visitors_fexpr;
                  trait = _visitors_ftrait;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_as env _visitors_fexpr _visitors_ftrait
                  _visitors_floc_
            | Pexpr_mutate
                {
                  record = _visitors_frecord;
                  accessor = _visitors_faccessor;
                  field = _visitors_ffield;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_mutate env _visitors_frecord
                  _visitors_faccessor _visitors_ffield _visitors_faugmented_by
                  _visitors_floc_
            | Pexpr_match
                {
                  expr = _visitors_fexpr;
                  cases = _visitors_fcases;
                  match_loc_ = _visitors_fmatch_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_match env _visitors_fexpr _visitors_fcases
                  _visitors_fmatch_loc_ _visitors_floc_
            | Pexpr_letmut
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_letmut env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fbody _visitors_floc_
            | Pexpr_pipe
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_pipe env _visitors_flhs _visitors_frhs
                  _visitors_floc_
            | Pexpr_assign
                {
                  var = _visitors_fvar;
                  expr = _visitors_fexpr;
                  augmented_by = _visitors_faugmented_by;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_assign env _visitors_fvar _visitors_fexpr
                  _visitors_faugmented_by _visitors_floc_
            | Pexpr_hole { loc_ = _visitors_floc_; kind = _visitors_fkind } ->
                self#visit_Pexpr_hole env _visitors_floc_ _visitors_fkind
            | Pexpr_return
                {
                  return_value = _visitors_freturn_value;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_return env _visitors_freturn_value
                  _visitors_floc_
            | Pexpr_raise
                { err_value = _visitors_ferr_value; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_raise env _visitors_ferr_value _visitors_floc_
            | Pexpr_unit { loc_ = _visitors_floc_; faked = _visitors_ffaked } ->
                self#visit_Pexpr_unit env _visitors_floc_ _visitors_ffaked
            | Pexpr_break { arg = _visitors_farg; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_break env _visitors_farg _visitors_floc_
            | Pexpr_continue { args = _visitors_fargs; loc_ = _visitors_floc_ }
              ->
                self#visit_Pexpr_continue env _visitors_fargs _visitors_floc_
            | Pexpr_loop
                {
                  args = _visitors_fargs;
                  body = _visitors_fbody;
                  loop_loc_ = _visitors_floop_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_loop env _visitors_fargs _visitors_fbody
                  _visitors_floop_loc_ _visitors_floc_
            | Pexpr_for
                {
                  binders = _visitors_fbinders;
                  condition = _visitors_fcondition;
                  continue_block = _visitors_fcontinue_block;
                  body = _visitors_fbody;
                  for_else = _visitors_ffor_else;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_for env _visitors_fbinders _visitors_fcondition
                  _visitors_fcontinue_block _visitors_fbody _visitors_ffor_else
                  _visitors_floc_
            | Pexpr_foreach
                {
                  binders = _visitors_fbinders;
                  expr = _visitors_fexpr;
                  body = _visitors_fbody;
                  else_block = _visitors_felse_block;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_foreach env _visitors_fbinders _visitors_fexpr
                  _visitors_fbody _visitors_felse_block _visitors_floc_
            | Pexpr_try
                {
                  body = _visitors_fbody;
                  catch = _visitors_fcatch;
                  catch_all = _visitors_fcatch_all;
                  try_else = _visitors_ftry_else;
                  try_loc_ = _visitors_ftry_loc_;
                  catch_loc_ = _visitors_fcatch_loc_;
                  else_loc_ = _visitors_felse_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_try env _visitors_fbody _visitors_fcatch
                  _visitors_fcatch_all _visitors_ftry_else _visitors_ftry_loc_
                  _visitors_fcatch_loc_ _visitors_felse_loc_ _visitors_floc_
            | Pexpr_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Pexpr_map env _visitors_felems _visitors_floc_
            | Pexpr_group
                {
                  expr = _visitors_fexpr;
                  group = _visitors_fgroup;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Pexpr_group env _visitors_fexpr _visitors_fgroup
                  _visitors_floc_
            | Pexpr_static_assert
                { asserts = _visitors_fasserts; body = _visitors_fbody } ->
                self#visit_Pexpr_static_assert env _visitors_fasserts
                  _visitors_fbody

        method visit_static_assertion
            : _ -> static_assertion -> static_assertion =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.assert_type in
            let _visitors_r1 =
              self#visit_longident env _visitors_this.assert_trait
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_this.assert_loc
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_this.assert_msg
            in
            {
              assert_type = _visitors_r0;
              assert_trait = _visitors_r1;
              assert_loc = _visitors_r2;
              assert_msg = _visitors_r3;
            }

        method visit_argument : _ -> argument -> argument =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_expr env _visitors_this.arg_value in
            let _visitors_r1 =
              self#visit_argument_kind env _visitors_this.arg_kind
            in
            { arg_value = _visitors_r0; arg_kind = _visitors_r1 }

        method visit_parameters : _ -> parameters -> parameters =
          fun env _visitors_this ->
            Basic_lst.map _visitors_this (self#visit_parameter env)

        method visit_parameter : _ -> parameter -> parameter =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.param_binder
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_typ env) t)
                | None -> None)
                _visitors_this.param_annot
            in
            let _visitors_r2 =
              self#visit_parameter_kind env _visitors_this.param_kind
            in
            {
              param_binder = _visitors_r0;
              param_annot = _visitors_r1;
              param_kind = _visitors_r2;
            }

        method visit_Positional : _ -> parameter_kind = fun env -> Positional
        method visit_Labelled : _ -> parameter_kind = fun env -> Labelled

        method visit_Optional : _ -> expr -> parameter_kind =
          fun env _visitors_fdefault ->
            let _visitors_r0 = self#visit_expr env _visitors_fdefault in
            Optional { default = _visitors_r0 }

        method visit_Question_optional : _ -> parameter_kind =
          fun env -> Question_optional

        method visit_parameter_kind : _ -> parameter_kind -> parameter_kind =
          fun env _visitors_this ->
            match _visitors_this with
            | Positional -> self#visit_Positional env
            | Labelled -> self#visit_Labelled env
            | Optional { default = _visitors_fdefault } ->
                self#visit_Optional env _visitors_fdefault
            | Question_optional -> self#visit_Question_optional env

        method visit_Lambda
            : _ ->
              parameters ->
              location ->
              expr ->
              (typ * error_typ) option ->
              fn_kind ->
              bool ->
              func =
          fun env _visitors_fparameters _visitors_fparams_loc_ _visitors_fbody
              _visitors_freturn_type _visitors_fkind_ _visitors_fhas_error ->
            let _visitors_r0 =
              self#visit_parameters env _visitors_fparameters
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fparams_loc_
            in
            let _visitors_r2 = self#visit_expr env _visitors_fbody in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun (_visitors_c0, _visitors_c1) ->
                         let _visitors_r0 = self#visit_typ env _visitors_c0 in
                         let _visitors_r1 =
                           self#visit_error_typ env _visitors_c1
                         in
                         (_visitors_r0, _visitors_r1))
                         t)
                | None -> None)
                _visitors_freturn_type
            in
            let _visitors_r4 = self#visit_fn_kind env _visitors_fkind_ in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_fhas_error
            in
            Lambda
              {
                parameters = _visitors_r0;
                params_loc_ = _visitors_r1;
                body = _visitors_r2;
                return_type = _visitors_r3;
                kind_ = _visitors_r4;
                has_error = _visitors_r5;
              }

        method visit_Match
            : _ ->
              (pattern list * expr) list ->
              bool ->
              location ->
              location ->
              func =
          fun env _visitors_fcases _visitors_fhas_error _visitors_ffn_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this
                  (fun (_visitors_c0, _visitors_c1) ->
                    let _visitors_r0 =
                      (fun _visitors_this ->
                        Basic_lst.map _visitors_this (self#visit_pattern env))
                        _visitors_c0
                    in
                    let _visitors_r1 = self#visit_expr env _visitors_c1 in
                    (_visitors_r0, _visitors_r1)))
                _visitors_fcases
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fhas_error
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_ffn_loc_
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Match
              {
                cases = _visitors_r0;
                has_error = _visitors_r1;
                fn_loc_ = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_func : _ -> func -> func =
          fun env _visitors_this ->
            match _visitors_this with
            | Lambda
                {
                  parameters = _visitors_fparameters;
                  params_loc_ = _visitors_fparams_loc_;
                  body = _visitors_fbody;
                  return_type = _visitors_freturn_type;
                  kind_ = _visitors_fkind_;
                  has_error = _visitors_fhas_error;
                } ->
                self#visit_Lambda env _visitors_fparameters
                  _visitors_fparams_loc_ _visitors_fbody _visitors_freturn_type
                  _visitors_fkind_ _visitors_fhas_error
            | Match
                {
                  cases = _visitors_fcases;
                  has_error = _visitors_fhas_error;
                  fn_loc_ = _visitors_ffn_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Match env _visitors_fcases _visitors_fhas_error
                  _visitors_ffn_loc_ _visitors_floc_

        method visit_Elem_regular : _ -> expr -> spreadable_elem =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_expr env _visitors_c0 in
            Elem_regular _visitors_r0

        method visit_Elem_spread : _ -> expr -> location -> spreadable_elem =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Elem_spread { expr = _visitors_r0; loc_ = _visitors_r1 }

        method visit_spreadable_elem : _ -> spreadable_elem -> spreadable_elem =
          fun env _visitors_this ->
            match _visitors_this with
            | Elem_regular _visitors_c0 ->
                self#visit_Elem_regular env _visitors_c0
            | Elem_spread { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Elem_spread env _visitors_fexpr _visitors_floc_

        method visit_Map_expr_elem
            : _ -> constant -> expr -> location -> location -> map_expr_elem =
          fun env _visitors_fkey _visitors_fexpr _visitors_fkey_loc_
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_fkey
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fkey_loc_
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Map_expr_elem
              {
                key = _visitors_r0;
                expr = _visitors_r1;
                key_loc_ = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_map_expr_elem : _ -> map_expr_elem -> map_expr_elem =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_expr_elem
                {
                  key = _visitors_fkey;
                  expr = _visitors_fexpr;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_expr_elem env _visitors_fkey _visitors_fexpr
                  _visitors_fkey_loc_ _visitors_floc_

        method visit_Error_typ : _ -> typ -> error_typ =
          fun env _visitors_fty ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            Error_typ { ty = _visitors_r0 }

        method visit_Default_error_typ : _ -> location -> error_typ =
          fun env _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Default_error_typ { loc_ = _visitors_r0 }

        method visit_No_error_typ : _ -> error_typ = fun env -> No_error_typ

        method visit_error_typ : _ -> error_typ -> error_typ =
          fun env _visitors_this ->
            match _visitors_this with
            | Error_typ { ty = _visitors_fty } ->
                self#visit_Error_typ env _visitors_fty
            | Default_error_typ { loc_ = _visitors_floc_ } ->
                self#visit_Default_error_typ env _visitors_floc_
            | No_error_typ -> self#visit_No_error_typ env

        method visit_Ptype_any : _ -> location -> typ =
          fun env _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptype_any { loc_ = _visitors_r0 }

        method visit_Ptype_arrow
            : _ -> typ list -> typ -> error_typ -> location -> typ =
          fun env _visitors_fty_arg _visitors_fty_res _visitors_fty_err
              _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_typ env))
                _visitors_fty_arg
            in
            let _visitors_r1 = self#visit_typ env _visitors_fty_res in
            let _visitors_r2 = self#visit_error_typ env _visitors_fty_err in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptype_arrow
              {
                ty_arg = _visitors_r0;
                ty_res = _visitors_r1;
                ty_err = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Ptype_tuple : _ -> typ list -> location -> typ =
          fun env _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_typ env))
                _visitors_ftys
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptype_tuple { tys = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Ptype_name
            : _ -> constrid_loc -> typ list -> location -> typ =
          fun env _visitors_fconstr_id _visitors_ftys _visitors_floc_ ->
            let _visitors_r0 =
              self#visit_constrid_loc env _visitors_fconstr_id
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_typ env))
                _visitors_ftys
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptype_name
              {
                constr_id = _visitors_r0;
                tys = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Ptype_option : _ -> typ -> location -> location -> typ =
          fun env _visitors_fty _visitors_floc_ _visitors_fquestion_loc ->
            let _visitors_r0 = self#visit_typ env _visitors_fty in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fquestion_loc
            in
            Ptype_option
              {
                ty = _visitors_r0;
                loc_ = _visitors_r1;
                question_loc = _visitors_r2;
              }

        method visit_typ : _ -> typ -> typ =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptype_any { loc_ = _visitors_floc_ } ->
                self#visit_Ptype_any env _visitors_floc_
            | Ptype_arrow
                {
                  ty_arg = _visitors_fty_arg;
                  ty_res = _visitors_fty_res;
                  ty_err = _visitors_fty_err;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_arrow env _visitors_fty_arg _visitors_fty_res
                  _visitors_fty_err _visitors_floc_
            | Ptype_tuple { tys = _visitors_ftys; loc_ = _visitors_floc_ } ->
                self#visit_Ptype_tuple env _visitors_ftys _visitors_floc_
            | Ptype_name
                {
                  constr_id = _visitors_fconstr_id;
                  tys = _visitors_ftys;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptype_name env _visitors_fconstr_id _visitors_ftys
                  _visitors_floc_
            | Ptype_option
                {
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                  question_loc = _visitors_fquestion_loc;
                } ->
                self#visit_Ptype_option env _visitors_fty _visitors_floc_
                  _visitors_fquestion_loc

        method visit_Ppat_alias : _ -> pattern -> binder -> location -> pattern
            =
          fun env _visitors_fpat _visitors_falias _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_binder env _visitors_falias in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_alias
              { pat = _visitors_r0; alias = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Ppat_any : _ -> location -> pattern =
          fun env _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_any { loc_ = _visitors_r0 }

        method visit_Ppat_array : _ -> array_pattern -> location -> pattern =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 = self#visit_array_pattern env _visitors_fpats in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_array { pats = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Ppat_constant : _ -> constant -> location -> pattern =
          fun env _visitors_fc _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_fc
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_constant { c = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Ppat_constraint
            : _ -> pattern -> typ -> location -> pattern =
          fun env _visitors_fpat _visitors_fty _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_typ env _visitors_fty in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_constraint
              { pat = _visitors_r0; ty = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Ppat_constr
            : _ ->
              constructor ->
              constr_pat_arg list option ->
              bool ->
              location ->
              pattern =
          fun env _visitors_fconstr _visitors_fargs _visitors_fis_open
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_constructor env _visitors_fconstr in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun _visitors_this ->
                         Basic_lst.map _visitors_this
                           (self#visit_constr_pat_arg env))
                         t)
                | None -> None)
                _visitors_fargs
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_open
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_constr
              {
                constr = _visitors_r0;
                args = _visitors_r1;
                is_open = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Ppat_or : _ -> pattern -> pattern -> location -> pattern =
          fun env _visitors_fpat1 _visitors_fpat2 _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat1 in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat2 in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_or
              { pat1 = _visitors_r0; pat2 = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Ppat_tuple : _ -> pattern list -> location -> pattern =
          fun env _visitors_fpats _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_pattern env))
                _visitors_fpats
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_tuple { pats = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Ppat_var : _ -> binder -> pattern =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_binder env _visitors_c0 in
            Ppat_var _visitors_r0

        method visit_Ppat_record
            : _ -> field_pat list -> bool -> location -> pattern =
          fun env _visitors_ffields _visitors_fis_closed _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_field_pat env))
                _visitors_ffields
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_closed
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_record
              {
                fields = _visitors_r0;
                is_closed = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Ppat_map : _ -> map_pat_elem list -> location -> pattern =
          fun env _visitors_felems _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_map_pat_elem env))
                _visitors_felems
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_map { elems = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Ppat_range
            : _ -> pattern -> pattern -> bool -> location -> pattern =
          fun env _visitors_flhs _visitors_frhs _visitors_finclusive
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_pattern env _visitors_flhs in
            let _visitors_r1 = self#visit_pattern env _visitors_frhs in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_finclusive
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ppat_range
              {
                lhs = _visitors_r0;
                rhs = _visitors_r1;
                inclusive = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_pattern : _ -> pattern -> pattern =
          fun env _visitors_this ->
            match _visitors_this with
            | Ppat_alias
                {
                  pat = _visitors_fpat;
                  alias = _visitors_falias;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_alias env _visitors_fpat _visitors_falias
                  _visitors_floc_
            | Ppat_any { loc_ = _visitors_floc_ } ->
                self#visit_Ppat_any env _visitors_floc_
            | Ppat_array { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_array env _visitors_fpats _visitors_floc_
            | Ppat_constant { c = _visitors_fc; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_constant env _visitors_fc _visitors_floc_
            | Ppat_constraint
                {
                  pat = _visitors_fpat;
                  ty = _visitors_fty;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constraint env _visitors_fpat _visitors_fty
                  _visitors_floc_
            | Ppat_constr
                {
                  constr = _visitors_fconstr;
                  args = _visitors_fargs;
                  is_open = _visitors_fis_open;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_constr env _visitors_fconstr _visitors_fargs
                  _visitors_fis_open _visitors_floc_
            | Ppat_or
                {
                  pat1 = _visitors_fpat1;
                  pat2 = _visitors_fpat2;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_or env _visitors_fpat1 _visitors_fpat2
                  _visitors_floc_
            | Ppat_tuple { pats = _visitors_fpats; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_tuple env _visitors_fpats _visitors_floc_
            | Ppat_var _visitors_c0 -> self#visit_Ppat_var env _visitors_c0
            | Ppat_record
                {
                  fields = _visitors_ffields;
                  is_closed = _visitors_fis_closed;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_record env _visitors_ffields
                  _visitors_fis_closed _visitors_floc_
            | Ppat_map { elems = _visitors_felems; loc_ = _visitors_floc_ } ->
                self#visit_Ppat_map env _visitors_felems _visitors_floc_
            | Ppat_range
                {
                  lhs = _visitors_flhs;
                  rhs = _visitors_frhs;
                  inclusive = _visitors_finclusive;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ppat_range env _visitors_flhs _visitors_frhs
                  _visitors_finclusive _visitors_floc_

        method visit_Closed : _ -> pattern list -> array_pattern =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_pattern env))
                _visitors_c0
            in
            Closed _visitors_r0

        method visit_Open
            : _ ->
              pattern list ->
              pattern list ->
              binder option ->
              array_pattern =
          fun env _visitors_c0 _visitors_c1 _visitors_c2 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_pattern env))
                _visitors_c0
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_pattern env))
                _visitors_c1
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_binder env) t)
                | None -> None)
                _visitors_c2
            in
            Open (_visitors_r0, _visitors_r1, _visitors_r2)

        method visit_array_pattern : _ -> array_pattern -> array_pattern =
          fun env _visitors_this ->
            match _visitors_this with
            | Closed _visitors_c0 -> self#visit_Closed env _visitors_c0
            | Open (_visitors_c0, _visitors_c1, _visitors_c2) ->
                self#visit_Open env _visitors_c0 _visitors_c1 _visitors_c2

        method visit_Field_def
            : _ -> label -> expr -> bool -> location -> field_def =
          fun env _visitors_flabel _visitors_fexpr _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_pun
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Field_def
              {
                label = _visitors_r0;
                expr = _visitors_r1;
                is_pun = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_field_def : _ -> field_def -> field_def =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_def
                {
                  label = _visitors_flabel;
                  expr = _visitors_fexpr;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_def env _visitors_flabel _visitors_fexpr
                  _visitors_fis_pun _visitors_floc_

        method visit_Field_pat
            : _ -> label -> pattern -> bool -> location -> field_pat =
          fun env _visitors_flabel _visitors_fpattern _visitors_fis_pun
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_label env _visitors_flabel in
            let _visitors_r1 = self#visit_pattern env _visitors_fpattern in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_pun
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Field_pat
              {
                label = _visitors_r0;
                pattern = _visitors_r1;
                is_pun = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_field_pat : _ -> field_pat -> field_pat =
          fun env _visitors_this ->
            match _visitors_this with
            | Field_pat
                {
                  label = _visitors_flabel;
                  pattern = _visitors_fpattern;
                  is_pun = _visitors_fis_pun;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Field_pat env _visitors_flabel _visitors_fpattern
                  _visitors_fis_pun _visitors_floc_

        method visit_Constr_pat_arg
            : _ -> pattern -> argument_kind -> constr_pat_arg =
          fun env _visitors_fpat _visitors_fkind ->
            let _visitors_r0 = self#visit_pattern env _visitors_fpat in
            let _visitors_r1 = self#visit_argument_kind env _visitors_fkind in
            Constr_pat_arg { pat = _visitors_r0; kind = _visitors_r1 }

        method visit_constr_pat_arg : _ -> constr_pat_arg -> constr_pat_arg =
          fun env _visitors_this ->
            match _visitors_this with
            | Constr_pat_arg { pat = _visitors_fpat; kind = _visitors_fkind } ->
                self#visit_Constr_pat_arg env _visitors_fpat _visitors_fkind

        method visit_Map_pat_elem
            : _ ->
              constant ->
              pattern ->
              bool ->
              location ->
              location ->
              map_pat_elem =
          fun env _visitors_fkey _visitors_fpat _visitors_fmatch_absent
              _visitors_fkey_loc_ _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_fkey
            in
            let _visitors_r1 = self#visit_pattern env _visitors_fpat in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_fmatch_absent
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_fkey_loc_
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Map_pat_elem
              {
                key = _visitors_r0;
                pat = _visitors_r1;
                match_absent = _visitors_r2;
                key_loc_ = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_map_pat_elem : _ -> map_pat_elem -> map_pat_elem =
          fun env _visitors_this ->
            match _visitors_this with
            | Map_pat_elem
                {
                  key = _visitors_fkey;
                  pat = _visitors_fpat;
                  match_absent = _visitors_fmatch_absent;
                  key_loc_ = _visitors_fkey_loc_;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Map_pat_elem env _visitors_fkey _visitors_fpat
                  _visitors_fmatch_absent _visitors_fkey_loc_ _visitors_floc_

        method visit_constr_param : _ -> constr_param -> constr_param =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.cparam_typ in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_this.cparam_mut
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_label env) t)
                | None -> None)
                _visitors_this.cparam_label
            in
            {
              cparam_typ = _visitors_r0;
              cparam_mut = _visitors_r1;
              cparam_label = _visitors_r2;
            }

        method visit_constr_decl : _ -> constr_decl -> constr_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_constr_name env _visitors_this.constr_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun _visitors_this ->
                         Basic_lst.map _visitors_this
                           (self#visit_constr_param env))
                         t)
                | None -> None)
                _visitors_this.constr_args
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_this.constr_loc_
            in
            {
              constr_name = _visitors_r0;
              constr_args = _visitors_r1;
              constr_loc_ = _visitors_r2;
            }

        method visit_field_name : _ -> field_name -> field_name =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_this.label
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_this.loc_
            in
            { label = _visitors_r0; loc_ = _visitors_r1 }

        method visit_field_decl : _ -> field_decl -> field_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_field_name env _visitors_this.field_name
            in
            let _visitors_r1 = self#visit_typ env _visitors_this.field_ty in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_this.field_mut
            in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.field_vis
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_this.field_loc_
            in
            {
              field_name = _visitors_r0;
              field_ty = _visitors_r1;
              field_mut = _visitors_r2;
              field_vis = _visitors_r3;
              field_loc_ = _visitors_r4;
            }

        method visit_No_payload : _ -> exception_decl = fun env -> No_payload

        method visit_Single_payload : _ -> typ -> exception_decl =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            Single_payload _visitors_r0

        method visit_Enum_payload : _ -> constr_decl list -> exception_decl =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_constr_decl env))
                _visitors_c0
            in
            Enum_payload _visitors_r0

        method visit_exception_decl : _ -> exception_decl -> exception_decl =
          fun env _visitors_this ->
            match _visitors_this with
            | No_payload -> self#visit_No_payload env
            | Single_payload _visitors_c0 ->
                self#visit_Single_payload env _visitors_c0
            | Enum_payload _visitors_c0 ->
                self#visit_Enum_payload env _visitors_c0

        method visit_Ptd_abstract : _ -> type_desc = fun env -> Ptd_abstract

        method visit_Ptd_newtype : _ -> typ -> type_desc =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            Ptd_newtype _visitors_r0

        method visit_Ptd_error : _ -> exception_decl -> type_desc =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_exception_decl env _visitors_c0 in
            Ptd_error _visitors_r0

        method visit_Ptd_variant : _ -> constr_decl list -> type_desc =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_constr_decl env))
                _visitors_c0
            in
            Ptd_variant _visitors_r0

        method visit_Ptd_record : _ -> field_decl list -> type_desc =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_field_decl env))
                _visitors_c0
            in
            Ptd_record _visitors_r0

        method visit_Ptd_alias : _ -> typ -> type_desc =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_typ env _visitors_c0 in
            Ptd_alias _visitors_r0

        method visit_type_desc : _ -> type_desc -> type_desc =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptd_abstract -> self#visit_Ptd_abstract env
            | Ptd_newtype _visitors_c0 ->
                self#visit_Ptd_newtype env _visitors_c0
            | Ptd_error _visitors_c0 -> self#visit_Ptd_error env _visitors_c0
            | Ptd_variant _visitors_c0 ->
                self#visit_Ptd_variant env _visitors_c0
            | Ptd_record _visitors_c0 -> self#visit_Ptd_record env _visitors_c0
            | Ptd_alias _visitors_c0 -> self#visit_Ptd_alias env _visitors_c0

        method visit_type_decl : _ -> type_decl -> type_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_this.tycon
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_this.tycon_loc_
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_type_decl_binder env))
                _visitors_this.params
            in
            let _visitors_r3 =
              self#visit_type_desc env _visitors_this.components
            in
            let _visitors_r4 = self#visit_docstring env _visitors_this.doc_ in
            let _visitors_r5 =
              self#visit_visibility env _visitors_this.type_vis
            in
            let _visitors_r6 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_deriving_directive env))
                _visitors_this.deriving_
            in
            let _visitors_r7 =
              (fun _visitors_this -> _visitors_this) _visitors_this.loc_
            in
            {
              tycon = _visitors_r0;
              tycon_loc_ = _visitors_r1;
              params = _visitors_r2;
              components = _visitors_r3;
              doc_ = _visitors_r4;
              type_vis = _visitors_r5;
              deriving_ = _visitors_r6;
              loc_ = _visitors_r7;
            }

        method visit_local_type_decl : _ -> local_type_decl -> local_type_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_this.local_tycon
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this)
                _visitors_this.local_tycon_loc_
            in
            let _visitors_r2 =
              self#visit_type_desc env _visitors_this.local_components
            in
            {
              local_tycon = _visitors_r0;
              local_tycon_loc_ = _visitors_r1;
              local_components = _visitors_r2;
            }

        method visit_deriving_directive
            : _ -> deriving_directive -> deriving_directive =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_type_name env _visitors_this.type_name_
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_argument env))
                _visitors_this.args
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_this.loc_
            in
            {
              type_name_ = _visitors_r0;
              args = _visitors_r1;
              loc_ = _visitors_r2;
            }

        method visit_Vis_default : _ -> visibility = fun env -> Vis_default

        method visit_Vis_pub : _ -> string option -> location -> visibility =
          fun env _visitors_fattr _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((fun _visitors_this -> _visitors_this) t)
                | None -> None)
                _visitors_fattr
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Vis_pub { attr = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Vis_priv : _ -> location -> visibility =
          fun env _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Vis_priv { loc_ = _visitors_r0 }

        method visit_visibility : _ -> visibility -> visibility =
          fun env _visitors_this ->
            match _visitors_this with
            | Vis_default -> self#visit_Vis_default env
            | Vis_pub { attr = _visitors_fattr; loc_ = _visitors_floc_ } ->
                self#visit_Vis_pub env _visitors_fattr _visitors_floc_
            | Vis_priv { loc_ = _visitors_floc_ } ->
                self#visit_Vis_priv env _visitors_floc_

        method visit_Import
            : _ -> string_literal -> string_literal -> func_stubs =
          fun env _visitors_fmodule_name _visitors_ffunc_name ->
            let _visitors_r0 =
              self#visit_string_literal env _visitors_fmodule_name
            in
            let _visitors_r1 =
              self#visit_string_literal env _visitors_ffunc_name
            in
            Import { module_name = _visitors_r0; func_name = _visitors_r1 }

        method visit_Embedded
            : _ -> string_literal option -> embedded_code -> func_stubs =
          fun env _visitors_flanguage _visitors_fcode ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_string_literal env) t)
                | None -> None)
                _visitors_flanguage
            in
            let _visitors_r1 = self#visit_embedded_code env _visitors_fcode in
            Embedded { language = _visitors_r0; code = _visitors_r1 }

        method visit_func_stubs : _ -> func_stubs -> func_stubs =
          fun env _visitors_this ->
            match _visitors_this with
            | Import
                {
                  module_name = _visitors_fmodule_name;
                  func_name = _visitors_ffunc_name;
                } ->
                self#visit_Import env _visitors_fmodule_name
                  _visitors_ffunc_name
            | Embedded
                { language = _visitors_flanguage; code = _visitors_fcode } ->
                self#visit_Embedded env _visitors_flanguage _visitors_fcode

        method visit_Code_string : _ -> string_literal -> embedded_code =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_string_literal env _visitors_c0 in
            Code_string _visitors_r0

        method visit_Code_multiline_string : _ -> string list -> embedded_code =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (fun _visitors_this ->
                    _visitors_this))
                _visitors_c0
            in
            Code_multiline_string _visitors_r0

        method visit_embedded_code : _ -> embedded_code -> embedded_code =
          fun env _visitors_this ->
            match _visitors_this with
            | Code_string _visitors_c0 ->
                self#visit_Code_string env _visitors_c0
            | Code_multiline_string _visitors_c0 ->
                self#visit_Code_multiline_string env _visitors_c0

        method visit_Decl_body : _ -> local_type_decl list -> expr -> decl_body
            =
          fun env _visitors_flocal_types _visitors_fexpr ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r1 = self#visit_expr env _visitors_fexpr in
            Decl_body { local_types = _visitors_r0; expr = _visitors_r1 }

        method visit_Decl_stubs : _ -> func_stubs -> decl_body =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_func_stubs env _visitors_c0 in
            Decl_stubs _visitors_r0

        method visit_decl_body : _ -> decl_body -> decl_body =
          fun env _visitors_this ->
            match _visitors_this with
            | Decl_body
                { local_types = _visitors_flocal_types; expr = _visitors_fexpr }
              ->
                self#visit_Decl_body env _visitors_flocal_types _visitors_fexpr
            | Decl_stubs _visitors_c0 -> self#visit_Decl_stubs env _visitors_c0

        method visit_fun_decl : _ -> fun_decl -> fun_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_type_name env) t)
                | None -> None)
                _visitors_this.type_name
            in
            let _visitors_r1 = self#visit_binder env _visitors_this.name in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_this.has_error
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_parameters env) t)
                | None -> None)
                _visitors_this.decl_params
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_this.params_loc_
            in
            let _visitors_r5 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_tvar_binder env))
                _visitors_this.quantifiers
            in
            let _visitors_r6 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun (_visitors_c0, _visitors_c1) ->
                         let _visitors_r0 = self#visit_typ env _visitors_c0 in
                         let _visitors_r1 =
                           self#visit_error_typ env _visitors_c1
                         in
                         (_visitors_r0, _visitors_r1))
                         t)
                | None -> None)
                _visitors_this.return_type
            in
            let _visitors_r7 =
              (fun _visitors_this -> _visitors_this) _visitors_this.is_pub
            in
            let _visitors_r8 = self#visit_docstring env _visitors_this.doc_ in
            {
              type_name = _visitors_r0;
              name = _visitors_r1;
              has_error = _visitors_r2;
              decl_params = _visitors_r3;
              params_loc_ = _visitors_r4;
              quantifiers = _visitors_r5;
              return_type = _visitors_r6;
              is_pub = _visitors_r7;
              doc_ = _visitors_r8;
            }

        method visit_trait_method_param
            : _ -> trait_method_param -> trait_method_param =
          fun env _visitors_this ->
            let _visitors_r0 = self#visit_typ env _visitors_this.tmparam_typ in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_label env) t)
                | None -> None)
                _visitors_this.tmparam_label
            in
            { tmparam_typ = _visitors_r0; tmparam_label = _visitors_r1 }

        method visit_Trait_method
            : _ ->
              binder ->
              bool ->
              tvar_binder list ->
              trait_method_param list ->
              (typ * error_typ) option ->
              location ->
              trait_method_decl =
          fun env _visitors_fname _visitors_fhas_error _visitors_fquantifiers
              _visitors_fparams _visitors_freturn_type _visitors_floc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fname in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fhas_error
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_trait_method_param env))
                _visitors_fparams
            in
            let _visitors_r4 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun (_visitors_c0, _visitors_c1) ->
                         let _visitors_r0 = self#visit_typ env _visitors_c0 in
                         let _visitors_r1 =
                           self#visit_error_typ env _visitors_c1
                         in
                         (_visitors_r0, _visitors_r1))
                         t)
                | None -> None)
                _visitors_freturn_type
            in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Trait_method
              {
                name = _visitors_r0;
                has_error = _visitors_r1;
                quantifiers = _visitors_r2;
                params = _visitors_r3;
                return_type = _visitors_r4;
                loc_ = _visitors_r5;
              }

        method visit_trait_method_decl
            : _ -> trait_method_decl -> trait_method_decl =
          fun env _visitors_this ->
            match _visitors_this with
            | Trait_method
                {
                  name = _visitors_fname;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  return_type = _visitors_freturn_type;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Trait_method env _visitors_fname _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams
                  _visitors_freturn_type _visitors_floc_

        method visit_trait_decl : _ -> trait_decl -> trait_decl =
          fun env _visitors_this ->
            let _visitors_r0 =
              self#visit_binder env _visitors_this.trait_name
            in
            let _visitors_r1 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_tvar_constraint env))
                _visitors_this.trait_supers
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_trait_method_decl env))
                _visitors_this.trait_methods
            in
            let _visitors_r3 =
              self#visit_visibility env _visitors_this.trait_vis
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_this.trait_loc_
            in
            let _visitors_r5 =
              self#visit_docstring env _visitors_this.trait_doc_
            in
            {
              trait_name = _visitors_r0;
              trait_supers = _visitors_r1;
              trait_methods = _visitors_r2;
              trait_vis = _visitors_r3;
              trait_loc_ = _visitors_r4;
              trait_doc_ = _visitors_r5;
            }

        method visit_Ptop_expr
            : _ -> expr -> bool -> local_type_decl list -> absolute_loc -> impl
            =
          fun env _visitors_fexpr _visitors_fis_main _visitors_flocal_types
              _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_main
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptop_expr
              {
                expr = _visitors_r0;
                is_main = _visitors_r1;
                local_types = _visitors_r2;
                loc_ = _visitors_r3;
              }

        method visit_Ptop_test
            : _ ->
              expr ->
              test_name ->
              parameters option ->
              local_type_decl list ->
              absolute_loc ->
              impl =
          fun env _visitors_fexpr _visitors_fname _visitors_fparams
              _visitors_flocal_types _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_fname
            in
            let _visitors_r2 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_parameters env) t)
                | None -> None)
                _visitors_fparams
            in
            let _visitors_r3 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptop_test
              {
                expr = _visitors_r0;
                name = _visitors_r1;
                params = _visitors_r2;
                local_types = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_Ptop_typedef : _ -> type_decl -> impl =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_type_decl env _visitors_c0 in
            Ptop_typedef _visitors_r0

        method visit_Ptop_funcdef
            : _ -> fun_decl -> decl_body -> absolute_loc -> impl =
          fun env _visitors_ffun_decl _visitors_fdecl_body _visitors_floc_ ->
            let _visitors_r0 = self#visit_fun_decl env _visitors_ffun_decl in
            let _visitors_r1 = self#visit_decl_body env _visitors_fdecl_body in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptop_funcdef
              {
                fun_decl = _visitors_r0;
                decl_body = _visitors_r1;
                loc_ = _visitors_r2;
              }

        method visit_Ptop_letdef
            : _ ->
              binder ->
              typ option ->
              expr ->
              bool ->
              bool ->
              absolute_loc ->
              docstring ->
              impl =
          fun env _visitors_fbinder _visitors_fty _visitors_fexpr
              _visitors_fis_pub _visitors_fis_constant _visitors_floc_
              _visitors_fdoc_ ->
            let _visitors_r0 = self#visit_binder env _visitors_fbinder in
            let _visitors_r1 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_typ env) t)
                | None -> None)
                _visitors_fty
            in
            let _visitors_r2 = self#visit_expr env _visitors_fexpr in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_pub
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_constant
            in
            let _visitors_r5 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            let _visitors_r6 = self#visit_docstring env _visitors_fdoc_ in
            Ptop_letdef
              {
                binder = _visitors_r0;
                ty = _visitors_r1;
                expr = _visitors_r2;
                is_pub = _visitors_r3;
                is_constant = _visitors_r4;
                loc_ = _visitors_r5;
                doc_ = _visitors_r6;
              }

        method visit_Ptop_trait : _ -> trait_decl -> impl =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_trait_decl env _visitors_c0 in
            Ptop_trait _visitors_r0

        method visit_Ptop_impl
            : _ ->
              typ option ->
              type_name ->
              binder ->
              bool ->
              tvar_binder list ->
              parameters ->
              (typ * error_typ) option ->
              expr ->
              bool ->
              local_type_decl list ->
              absolute_loc ->
              location ->
              docstring ->
              impl =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fmethod_name
              _visitors_fhas_error _visitors_fquantifiers _visitors_fparams
              _visitors_fret_ty _visitors_fbody _visitors_fis_pub
              _visitors_flocal_types _visitors_floc_ _visitors_fheader_loc_
              _visitors_fdoc_ ->
            let _visitors_r0 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t -> Some ((self#visit_typ env) t)
                | None -> None)
                _visitors_fself_ty
            in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 = self#visit_binder env _visitors_fmethod_name in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_fhas_error
            in
            let _visitors_r4 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r5 = self#visit_parameters env _visitors_fparams in
            let _visitors_r6 =
              (fun _visitors_this ->
                match _visitors_this with
                | Some t ->
                    Some
                      ((fun (_visitors_c0, _visitors_c1) ->
                         let _visitors_r0 = self#visit_typ env _visitors_c0 in
                         let _visitors_r1 =
                           self#visit_error_typ env _visitors_c1
                         in
                         (_visitors_r0, _visitors_r1))
                         t)
                | None -> None)
                _visitors_fret_ty
            in
            let _visitors_r7 = self#visit_expr env _visitors_fbody in
            let _visitors_r8 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_pub
            in
            let _visitors_r9 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_local_type_decl env))
                _visitors_flocal_types
            in
            let _visitors_r10 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            let _visitors_r11 =
              (fun _visitors_this -> _visitors_this) _visitors_fheader_loc_
            in
            let _visitors_r12 = self#visit_docstring env _visitors_fdoc_ in
            Ptop_impl
              {
                self_ty = _visitors_r0;
                trait = _visitors_r1;
                method_name = _visitors_r2;
                has_error = _visitors_r3;
                quantifiers = _visitors_r4;
                params = _visitors_r5;
                ret_ty = _visitors_r6;
                body = _visitors_r7;
                is_pub = _visitors_r8;
                local_types = _visitors_r9;
                loc_ = _visitors_r10;
                header_loc_ = _visitors_r11;
                doc_ = _visitors_r12;
              }

        method visit_Ptop_impl_relation
            : _ ->
              typ ->
              type_name ->
              tvar_binder list ->
              bool ->
              absolute_loc ->
              impl =
          fun env _visitors_fself_ty _visitors_ftrait _visitors_fquantifiers
              _visitors_fis_pub _visitors_floc_ ->
            let _visitors_r0 = self#visit_typ env _visitors_fself_ty in
            let _visitors_r1 = self#visit_type_name env _visitors_ftrait in
            let _visitors_r2 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_tvar_binder env))
                _visitors_fquantifiers
            in
            let _visitors_r3 =
              (fun _visitors_this -> _visitors_this) _visitors_fis_pub
            in
            let _visitors_r4 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Ptop_impl_relation
              {
                self_ty = _visitors_r0;
                trait = _visitors_r1;
                quantifiers = _visitors_r2;
                is_pub = _visitors_r3;
                loc_ = _visitors_r4;
              }

        method visit_impl : _ -> impl -> impl =
          fun env _visitors_this ->
            match _visitors_this with
            | Ptop_expr
                {
                  expr = _visitors_fexpr;
                  is_main = _visitors_fis_main;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_expr env _visitors_fexpr _visitors_fis_main
                  _visitors_flocal_types _visitors_floc_
            | Ptop_test
                {
                  expr = _visitors_fexpr;
                  name = _visitors_fname;
                  params = _visitors_fparams;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_test env _visitors_fexpr _visitors_fname
                  _visitors_fparams _visitors_flocal_types _visitors_floc_
            | Ptop_typedef _visitors_c0 ->
                self#visit_Ptop_typedef env _visitors_c0
            | Ptop_funcdef
                {
                  fun_decl = _visitors_ffun_decl;
                  decl_body = _visitors_fdecl_body;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_funcdef env _visitors_ffun_decl
                  _visitors_fdecl_body _visitors_floc_
            | Ptop_letdef
                {
                  binder = _visitors_fbinder;
                  ty = _visitors_fty;
                  expr = _visitors_fexpr;
                  is_pub = _visitors_fis_pub;
                  is_constant = _visitors_fis_constant;
                  loc_ = _visitors_floc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_letdef env _visitors_fbinder _visitors_fty
                  _visitors_fexpr _visitors_fis_pub _visitors_fis_constant
                  _visitors_floc_ _visitors_fdoc_
            | Ptop_trait _visitors_c0 -> self#visit_Ptop_trait env _visitors_c0
            | Ptop_impl
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  method_name = _visitors_fmethod_name;
                  has_error = _visitors_fhas_error;
                  quantifiers = _visitors_fquantifiers;
                  params = _visitors_fparams;
                  ret_ty = _visitors_fret_ty;
                  body = _visitors_fbody;
                  is_pub = _visitors_fis_pub;
                  local_types = _visitors_flocal_types;
                  loc_ = _visitors_floc_;
                  header_loc_ = _visitors_fheader_loc_;
                  doc_ = _visitors_fdoc_;
                } ->
                self#visit_Ptop_impl env _visitors_fself_ty _visitors_ftrait
                  _visitors_fmethod_name _visitors_fhas_error
                  _visitors_fquantifiers _visitors_fparams _visitors_fret_ty
                  _visitors_fbody _visitors_fis_pub _visitors_flocal_types
                  _visitors_floc_ _visitors_fheader_loc_ _visitors_fdoc_
            | Ptop_impl_relation
                {
                  self_ty = _visitors_fself_ty;
                  trait = _visitors_ftrait;
                  quantifiers = _visitors_fquantifiers;
                  is_pub = _visitors_fis_pub;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Ptop_impl_relation env _visitors_fself_ty
                  _visitors_ftrait _visitors_fquantifiers _visitors_fis_pub
                  _visitors_floc_

        method visit_Interp_lit
            : _ -> string -> string -> location -> interp_elem =
          fun env _visitors_fstr _visitors_frepr _visitors_floc_ ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_fstr
            in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_frepr
            in
            let _visitors_r2 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Interp_lit
              { str = _visitors_r0; repr = _visitors_r1; loc_ = _visitors_r2 }

        method visit_Interp_expr : _ -> expr -> location -> interp_elem =
          fun env _visitors_fexpr _visitors_floc_ ->
            let _visitors_r0 = self#visit_expr env _visitors_fexpr in
            let _visitors_r1 =
              (fun _visitors_this -> _visitors_this) _visitors_floc_
            in
            Interp_expr { expr = _visitors_r0; loc_ = _visitors_r1 }

        method visit_Interp_source : _ -> Literal.interp_source -> interp_elem =
          fun env _visitors_c0 ->
            let _visitors_r0 = self#visit_interp_source env _visitors_c0 in
            Interp_source _visitors_r0

        method visit_interp_elem : _ -> interp_elem -> interp_elem =
          fun env _visitors_this ->
            match _visitors_this with
            | Interp_lit
                {
                  str = _visitors_fstr;
                  repr = _visitors_frepr;
                  loc_ = _visitors_floc_;
                } ->
                self#visit_Interp_lit env _visitors_fstr _visitors_frepr
                  _visitors_floc_
            | Interp_expr { expr = _visitors_fexpr; loc_ = _visitors_floc_ } ->
                self#visit_Interp_expr env _visitors_fexpr _visitors_floc_
            | Interp_source _visitors_c0 ->
                self#visit_Interp_source env _visitors_c0

        method visit_Multiline_string : _ -> string -> multiline_string_elem =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this -> _visitors_this) _visitors_c0
            in
            Multiline_string _visitors_r0

        method visit_Multiline_interp
            : _ -> interp_elem list -> multiline_string_elem =
          fun env _visitors_c0 ->
            let _visitors_r0 =
              (fun _visitors_this ->
                Basic_lst.map _visitors_this (self#visit_interp_elem env))
                _visitors_c0
            in
            Multiline_interp _visitors_r0

        method visit_multiline_string_elem
            : _ -> multiline_string_elem -> multiline_string_elem =
          fun env _visitors_this ->
            match _visitors_this with
            | Multiline_string _visitors_c0 ->
                self#visit_Multiline_string env _visitors_c0
            | Multiline_interp _visitors_c0 ->
                self#visit_Multiline_interp env _visitors_c0

        method visit_impls : _ -> impls -> impls =
          fun env _visitors_this ->
            Basic_lst.map _visitors_this (self#visit_impl env)
      end

    [@@@VISITORS.END]
  end

  include struct
    let _ = fun (_ : expr) -> ()
    let _ = fun (_ : static_assertion) -> ()
    let _ = fun (_ : argument) -> ()
    let _ = fun (_ : parameters) -> ()
    let _ = fun (_ : parameter) -> ()
    let _ = fun (_ : parameter_kind) -> ()
    let _ = fun (_ : func) -> ()
    let _ = fun (_ : spreadable_elem) -> ()
    let _ = fun (_ : map_expr_elem) -> ()
    let _ = fun (_ : error_typ) -> ()
    let _ = fun (_ : typ) -> ()
    let _ = fun (_ : pattern) -> ()
    let _ = fun (_ : array_pattern) -> ()
    let _ = fun (_ : field_def) -> ()
    let _ = fun (_ : field_pat) -> ()
    let _ = fun (_ : constr_pat_arg) -> ()
    let _ = fun (_ : map_pat_elem) -> ()
    let _ = fun (_ : constr_param) -> ()
    let _ = fun (_ : constr_decl) -> ()
    let _ = fun (_ : field_name) -> ()
    let _ = fun (_ : field_decl) -> ()
    let _ = fun (_ : exception_decl) -> ()
    let _ = fun (_ : type_desc) -> ()
    let _ = fun (_ : type_decl) -> ()
    let _ = fun (_ : local_type_decl) -> ()
    let _ = fun (_ : deriving_directive) -> ()
    let _ = fun (_ : visibility) -> ()
    let _ = fun (_ : func_stubs) -> ()
    let _ = fun (_ : embedded_code) -> ()
    let _ = fun (_ : decl_body) -> ()
    let _ = fun (_ : fun_decl) -> ()
    let _ = fun (_ : trait_method_param) -> ()
    let _ = fun (_ : trait_method_decl) -> ()
    let _ = fun (_ : trait_decl) -> ()
    let _ = fun (_ : impl) -> ()
    let _ = fun (_ : interp_elem) -> ()
    let _ = fun (_ : multiline_string_elem) -> ()
    let _ = fun (_ : impls) -> ()
  end
end

let filter_fields ctor fields =
  Basic_lst.fold_right fields [] (fun field acc ->
      match field with
      | ( ( "params_loc_" | "loc_" | "index_loc_" | "try_loc_" | "catch_loc_"
          | "constr_loc" | "tycon_loc_" | "local_tycon_loc_" | "constr_loc_"
          | "field_loc_" | "header_loc_" | "trait_loc_" | "question_loc"
          | "match_loc_" | "loop_loc_" | "fn_loc_" | "key_loc_" | "else_loc_" ),
          _ )
        when not !Basic_config.show_loc ->
          acc
      | "args_loc_", _ -> acc
      | "repr", _ -> acc
      | ("doc_" | "trait_doc_"), _ when not !Basic_config.show_doc -> acc
      | ("is_pub" | "abbreviation" | "faked"), S.Atom "false" -> acc
      | "faked", S.Atom "true" -> ("faked", S.Atom "faked") :: acc
      | ("field_vis" | "type_vis"), S.Atom "Vis_default" -> acc
      | "deriving_", List [] -> acc
      | ("ty_params_" | "type_name"), List [] -> acc
      | "is_closed", S.Atom "true" -> acc
      | "continue_block", List [] -> acc
      | "args", List [] when ctor = "Pexpr_continue" -> acc
      | "arg", List [] when ctor = "Pexpr_break" -> acc
      | "trait_supers", List [] -> acc
      | "param_kind", Atom "Positional" -> acc
      | "is_main", S.Atom "false" -> acc
      | "arg_is_pun_", Atom "false" -> acc
      | ("doc_" | "intf_doc_"), List [] -> acc
      | "pragmas", List [] -> acc
      | "kind_", Atom "Lambda" -> acc
      | "is_open", Atom "false" when ctor = "Ppat_constr" -> acc
      | "augmented_by", List [] -> acc
      | "return_self", Atom "false" -> acc
      | "return_self", Atom "true" ->
          ("return_self", List [ Atom "return_self" ]) :: acc
      | "has_error", Atom "false" -> acc
      | "extend_error", Atom "false" -> acc
      | "attr", Atom "No_attr" -> acc
      | "local_types", List [] -> acc
      | "is_constant", Atom "false" when ctor = "Ptop_letdef" -> acc
      | "is_constant", Atom "true" when ctor = "Ptop_letdef" ->
          ("is_constant", List [ Atom "is_constant" ]) :: acc
      | _ -> field :: acc)

let loc_of_impl i =
  match i with
  | Ptop_expr { loc_; _ }
  | Ptop_test { loc_; _ }
  | Ptop_typedef { loc_; _ }
  | Ptop_funcdef { loc_; _ }
  | Ptop_letdef { loc_; _ }
  | Ptop_trait { trait_loc_ = loc_; _ }
  | Ptop_impl { loc_; _ }
  | Ptop_impl_relation { loc_; _ } ->
      loc_

type loc_ctx = Use_absolute_loc of absolute_loc | Use_relative_loc

let sexp =
  object (self)
    inherit [_] sexp as super

    method! visit_inline_record env ctor fields =
      super#visit_inline_record env ctor (filter_fields ctor fields)

    method! visit_record env fields =
      super#visit_record env (filter_fields "" fields)

    method! visit_docstring env docstring =
      let comment = Docstring.comment_string docstring in
      if comment = "" && Docstring.pragmas docstring = [] then S.List []
      else if Docstring.pragmas docstring = [] then S.Atom comment
      else super#visit_docstring env docstring

    method! visit_parameter env p =
      match p.param_kind with
      | Positional ->
          List
            [
              self#visit_binder env p.param_binder;
              Moon_sexp_conv.sexp_of_option (self#visit_typ env) p.param_annot;
            ]
      | _ ->
          List
            [
              self#visit_binder env p.param_binder;
              Moon_sexp_conv.sexp_of_option (self#visit_typ env) p.param_annot;
              self#visit_parameter_kind env p.param_kind;
            ]

    method! visit_argument env arg =
      match arg.arg_kind with
      | Positional -> self#visit_expr env arg.arg_value
      | kind ->
          S.List
            [ self#visit_expr env arg.arg_value; sexp_of_argument_kind kind ]

    method! visit_constr_param env cparam =
      let typ = self#visit_typ env cparam.cparam_typ in
      match cparam.cparam_label with
      | None ->
          if cparam.cparam_mut then S.List [ typ; List [ Atom "mut" ] ] else typ
      | Some label ->
          if cparam.cparam_mut then
            List
              [
                Atom "Labelled";
                self#visit_label env label;
                typ;
                List [ Atom "mut" ];
              ]
          else List [ Atom "Labelled"; self#visit_label env label; typ ]

    method! visit_Constr_pat_arg env pat kind =
      match kind with
      | Positional -> self#visit_pattern env pat
      | _ -> S.List [ self#visit_pattern env pat; sexp_of_argument_kind kind ]

    method! visit_location env loc =
      match env with
      | Use_absolute_loc base -> Rloc.to_loc ~base loc |> sexp_of_absolute_loc
      | Use_relative_loc -> super#visit_location env loc

    method! visit_impl env impl =
      match env with
      | Use_absolute_loc _ ->
          let base = loc_of_impl impl in
          super#visit_impl (Use_absolute_loc base) impl
      | Use_relative_loc -> super#visit_impl Use_relative_loc impl
  end

let sexp_of_impl impl = sexp#visit_impl Use_relative_loc impl

let sexp_of_impls ?(use_absolute_loc = false) impls =
  let ctx =
    if use_absolute_loc then Use_absolute_loc Loc.no_location
    else Use_relative_loc
  in
  sexp#visit_impls ctx impls

let sexp_of_visibility vis = sexp#visit_visibility Use_relative_loc vis

let sexp_of_type_decl type_decl =
  sexp#visit_type_decl Use_relative_loc type_decl

let sexp_of_trait_decl trait_decl =
  sexp#visit_trait_decl Use_relative_loc trait_decl

let sexp_of_argument argument = sexp#visit_argument Use_relative_loc argument

let sexp_of_deriving_directive deriving =
  sexp#visit_deriving_directive Use_relative_loc deriving

let string_of_vis = function
  | Vis_default -> "abstract"
  | Vis_priv _ -> "private"
  | Vis_pub { attr = None } -> "public"
  | Vis_pub { attr = Some attr } -> "public " ^ attr

let loc_of_expression e =
  match e with
  | Pexpr_apply { loc_; _ }
  | Pexpr_array { loc_; _ }
  | Pexpr_array_spread { loc_; _ }
  | Pexpr_array_get_slice { loc_; _ }
  | Pexpr_array_get { loc_; _ }
  | Pexpr_array_set { loc_; _ }
  | Pexpr_array_augmented_set { loc_; _ }
  | Pexpr_constant { loc_; _ }
  | Pexpr_interp { loc_; _ }
  | Pexpr_constraint { loc_; _ }
  | Pexpr_constr { loc_; _ }
  | Pexpr_while { loc_; _ }
  | Pexpr_function { loc_; _ }
  | Pexpr_ident { loc_; _ }
  | Pexpr_if { loc_; _ }
  | Pexpr_letrec { loc_; _ }
  | Pexpr_let { loc_; _ }
  | Pexpr_letfn { loc_; _ }
  | Pexpr_sequence { loc_; _ }
  | Pexpr_tuple { loc_; _ }
  | Pexpr_match { loc_; _ }
  | Pexpr_record { loc_; _ }
  | Pexpr_record_update { loc_; _ }
  | Pexpr_mutate { loc_; _ }
  | Pexpr_field { loc_; _ }
  | Pexpr_method { loc_; _ }
  | Pexpr_dot_apply { loc_; _ }
  | Pexpr_as { loc_; _ }
  | Pexpr_letmut { loc_; _ }
  | Pexpr_assign { loc_; _ }
  | Pexpr_unit { loc_; _ }
  | Pexpr_break { loc_; _ }
  | Pexpr_continue { loc_; _ }
  | Pexpr_loop { loc_; _ }
  | Pexpr_for { loc_; _ }
  | Pexpr_foreach { loc_; _ }
  | Pexpr_try { loc_; _ }
  | Pexpr_return { loc_; _ }
  | Pexpr_raise { loc_; _ }
  | Pexpr_hole { loc_; _ }
  | Pexpr_infix { loc_; _ }
  | Pexpr_unary { loc_; _ }
  | Pexpr_guard { loc_; _ }
  | Pexpr_guard_let { loc_; _ }
  | Pexpr_pipe { loc_; _ }
  | Pexpr_map { loc_; _ }
  | Pexpr_group { loc_; _ }
  | Pexpr_multiline_string { loc_; _ } ->
      loc_
  | Pexpr_static_assert _ -> Rloc.no_location

let loc_of_pattern p =
  match p with
  | Ppat_alias { loc_; _ }
  | Ppat_any { loc_; _ }
  | Ppat_array { loc_; _ }
  | Ppat_constant { loc_; _ }
  | Ppat_constraint { loc_; _ }
  | Ppat_constr { loc_; _ }
  | Ppat_or { loc_; _ }
  | Ppat_tuple { loc_; _ }
  | Ppat_var { loc_; _ }
  | Ppat_range { loc_; _ }
  | Ppat_record { loc_; _ }
  | Ppat_map { loc_; _ } ->
      loc_

let loc_of_type_expression te =
  match te with
  | Ptype_any { loc_ }
  | Ptype_arrow { loc_; _ }
  | Ptype_tuple { loc_; _ }
  | Ptype_name { loc_; _ }
  | Ptype_option { loc_; _ } ->
      loc_
