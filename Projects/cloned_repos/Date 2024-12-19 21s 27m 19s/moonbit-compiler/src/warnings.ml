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
type loc = Loc.t
type unused_kind = Unused | No_construct | No_read

include struct
  let _ = fun (_ : unused_kind) -> ()

  let sexp_of_unused_kind =
    (function
     | Unused -> S.Atom "Unused"
     | No_construct -> S.Atom "No_construct"
     | No_read -> S.Atom "No_read"
      : unused_kind -> S.t)

  let _ = sexp_of_unused_kind
  let equal_unused_kind = (Stdlib.( = ) : unused_kind -> unused_kind -> bool)
  let _ = equal_unused_kind
  let compare_unused_kind = (Stdlib.compare : unused_kind -> unused_kind -> int)
  let _ = compare_unused_kind
end

type kind =
  | Unused_func of string
  | Unused_var of { var_name : string; is_toplevel : bool }
  | Unused_type_declaration of string
  | Unused_abstract_type of string
  | Unused_tvar of string
  | Unused_constructor of { constr : string; kind : unused_kind }
  | Unused_field of string
  | Unused_constr_arg of { constr : string; index : int }
  | Unused_constr_field of {
      constr : string;
      label : string;
      is_mutated : bool;
    }
  | Redundant_modifier of { modifier : string; field : string }
  | Struct_never_constructed of string
  | Unused_pat
  | Partial_match of string list
  | Unreachable
  | Unresolved_tvar of string
  | Lowercase_type_name of string
  | Unused_mutability of string
  | Parser_inconsistency of {
      file_name : string;
      segment : string;
      is_menhir_succeed : bool;
      is_handrolled_succeed : bool;
    }
  | Useless_loop
  | Toplevel_not_left_aligned
  | Unexpected_pragmas of string
  | Omitted_constr_argument of { constr : string; labels : string list }
  | Ambiguous_block
  | Useless_try
  | Useless_error_type
  | Useless_catch_all
  | Deprecated_syntax of {
      old_usage : string;
      purpose : string;
      new_usage : string option;
    }
  | Todo
  | Unused_package of { name : string; is_alias : bool }
  | Empty_package_alias
  | Optional_arg_never_supplied of string
  | Optional_arg_always_supplied of string
  | Unused_import_value of string
  | Deprecated_prefix_label_syntax of string
  | Reserved_keyword of string

include struct
  let _ = fun (_ : kind) -> ()

  let sexp_of_kind =
    (function
     | Unused_func arg0__005_ ->
         let res0__006_ = Moon_sexp_conv.sexp_of_string arg0__005_ in
         S.List [ S.Atom "Unused_func"; res0__006_ ]
     | Unused_var { var_name = var_name__008_; is_toplevel = is_toplevel__010_ }
       ->
         let bnds__007_ = ([] : _ Stdlib.List.t) in
         let bnds__007_ =
           let arg__011_ = Moon_sexp_conv.sexp_of_bool is_toplevel__010_ in
           (S.List [ S.Atom "is_toplevel"; arg__011_ ] :: bnds__007_
             : _ Stdlib.List.t)
         in
         let bnds__007_ =
           let arg__009_ = Moon_sexp_conv.sexp_of_string var_name__008_ in
           (S.List [ S.Atom "var_name"; arg__009_ ] :: bnds__007_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Unused_var" :: bnds__007_)
     | Unused_type_declaration arg0__012_ ->
         let res0__013_ = Moon_sexp_conv.sexp_of_string arg0__012_ in
         S.List [ S.Atom "Unused_type_declaration"; res0__013_ ]
     | Unused_abstract_type arg0__014_ ->
         let res0__015_ = Moon_sexp_conv.sexp_of_string arg0__014_ in
         S.List [ S.Atom "Unused_abstract_type"; res0__015_ ]
     | Unused_tvar arg0__016_ ->
         let res0__017_ = Moon_sexp_conv.sexp_of_string arg0__016_ in
         S.List [ S.Atom "Unused_tvar"; res0__017_ ]
     | Unused_constructor { constr = constr__019_; kind = kind__021_ } ->
         let bnds__018_ = ([] : _ Stdlib.List.t) in
         let bnds__018_ =
           let arg__022_ = sexp_of_unused_kind kind__021_ in
           (S.List [ S.Atom "kind"; arg__022_ ] :: bnds__018_ : _ Stdlib.List.t)
         in
         let bnds__018_ =
           let arg__020_ = Moon_sexp_conv.sexp_of_string constr__019_ in
           (S.List [ S.Atom "constr"; arg__020_ ] :: bnds__018_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Unused_constructor" :: bnds__018_)
     | Unused_field arg0__023_ ->
         let res0__024_ = Moon_sexp_conv.sexp_of_string arg0__023_ in
         S.List [ S.Atom "Unused_field"; res0__024_ ]
     | Unused_constr_arg { constr = constr__026_; index = index__028_ } ->
         let bnds__025_ = ([] : _ Stdlib.List.t) in
         let bnds__025_ =
           let arg__029_ = Moon_sexp_conv.sexp_of_int index__028_ in
           (S.List [ S.Atom "index"; arg__029_ ] :: bnds__025_
             : _ Stdlib.List.t)
         in
         let bnds__025_ =
           let arg__027_ = Moon_sexp_conv.sexp_of_string constr__026_ in
           (S.List [ S.Atom "constr"; arg__027_ ] :: bnds__025_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Unused_constr_arg" :: bnds__025_)
     | Unused_constr_field
         {
           constr = constr__031_;
           label = label__033_;
           is_mutated = is_mutated__035_;
         } ->
         let bnds__030_ = ([] : _ Stdlib.List.t) in
         let bnds__030_ =
           let arg__036_ = Moon_sexp_conv.sexp_of_bool is_mutated__035_ in
           (S.List [ S.Atom "is_mutated"; arg__036_ ] :: bnds__030_
             : _ Stdlib.List.t)
         in
         let bnds__030_ =
           let arg__034_ = Moon_sexp_conv.sexp_of_string label__033_ in
           (S.List [ S.Atom "label"; arg__034_ ] :: bnds__030_
             : _ Stdlib.List.t)
         in
         let bnds__030_ =
           let arg__032_ = Moon_sexp_conv.sexp_of_string constr__031_ in
           (S.List [ S.Atom "constr"; arg__032_ ] :: bnds__030_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Unused_constr_field" :: bnds__030_)
     | Redundant_modifier { modifier = modifier__038_; field = field__040_ } ->
         let bnds__037_ = ([] : _ Stdlib.List.t) in
         let bnds__037_ =
           let arg__041_ = Moon_sexp_conv.sexp_of_string field__040_ in
           (S.List [ S.Atom "field"; arg__041_ ] :: bnds__037_
             : _ Stdlib.List.t)
         in
         let bnds__037_ =
           let arg__039_ = Moon_sexp_conv.sexp_of_string modifier__038_ in
           (S.List [ S.Atom "modifier"; arg__039_ ] :: bnds__037_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Redundant_modifier" :: bnds__037_)
     | Struct_never_constructed arg0__042_ ->
         let res0__043_ = Moon_sexp_conv.sexp_of_string arg0__042_ in
         S.List [ S.Atom "Struct_never_constructed"; res0__043_ ]
     | Unused_pat -> S.Atom "Unused_pat"
     | Partial_match arg0__044_ ->
         let res0__045_ =
           Moon_sexp_conv.sexp_of_list Moon_sexp_conv.sexp_of_string arg0__044_
         in
         S.List [ S.Atom "Partial_match"; res0__045_ ]
     | Unreachable -> S.Atom "Unreachable"
     | Unresolved_tvar arg0__046_ ->
         let res0__047_ = Moon_sexp_conv.sexp_of_string arg0__046_ in
         S.List [ S.Atom "Unresolved_tvar"; res0__047_ ]
     | Lowercase_type_name arg0__048_ ->
         let res0__049_ = Moon_sexp_conv.sexp_of_string arg0__048_ in
         S.List [ S.Atom "Lowercase_type_name"; res0__049_ ]
     | Unused_mutability arg0__050_ ->
         let res0__051_ = Moon_sexp_conv.sexp_of_string arg0__050_ in
         S.List [ S.Atom "Unused_mutability"; res0__051_ ]
     | Parser_inconsistency
         {
           file_name = file_name__053_;
           segment = segment__055_;
           is_menhir_succeed = is_menhir_succeed__057_;
           is_handrolled_succeed = is_handrolled_succeed__059_;
         } ->
         let bnds__052_ = ([] : _ Stdlib.List.t) in
         let bnds__052_ =
           let arg__060_ =
             Moon_sexp_conv.sexp_of_bool is_handrolled_succeed__059_
           in
           (S.List [ S.Atom "is_handrolled_succeed"; arg__060_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__058_ =
             Moon_sexp_conv.sexp_of_bool is_menhir_succeed__057_
           in
           (S.List [ S.Atom "is_menhir_succeed"; arg__058_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__056_ = Moon_sexp_conv.sexp_of_string segment__055_ in
           (S.List [ S.Atom "segment"; arg__056_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__054_ = Moon_sexp_conv.sexp_of_string file_name__053_ in
           (S.List [ S.Atom "file_name"; arg__054_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Parser_inconsistency" :: bnds__052_)
     | Useless_loop -> S.Atom "Useless_loop"
     | Toplevel_not_left_aligned -> S.Atom "Toplevel_not_left_aligned"
     | Unexpected_pragmas arg0__061_ ->
         let res0__062_ = Moon_sexp_conv.sexp_of_string arg0__061_ in
         S.List [ S.Atom "Unexpected_pragmas"; res0__062_ ]
     | Omitted_constr_argument { constr = constr__064_; labels = labels__066_ }
       ->
         let bnds__063_ = ([] : _ Stdlib.List.t) in
         let bnds__063_ =
           let arg__067_ =
             Moon_sexp_conv.sexp_of_list Moon_sexp_conv.sexp_of_string
               labels__066_
           in
           (S.List [ S.Atom "labels"; arg__067_ ] :: bnds__063_
             : _ Stdlib.List.t)
         in
         let bnds__063_ =
           let arg__065_ = Moon_sexp_conv.sexp_of_string constr__064_ in
           (S.List [ S.Atom "constr"; arg__065_ ] :: bnds__063_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Omitted_constr_argument" :: bnds__063_)
     | Ambiguous_block -> S.Atom "Ambiguous_block"
     | Useless_try -> S.Atom "Useless_try"
     | Useless_error_type -> S.Atom "Useless_error_type"
     | Useless_catch_all -> S.Atom "Useless_catch_all"
     | Deprecated_syntax
         {
           old_usage = old_usage__069_;
           purpose = purpose__071_;
           new_usage = new_usage__073_;
         } ->
         let bnds__068_ = ([] : _ Stdlib.List.t) in
         let bnds__068_ =
           let arg__074_ =
             Moon_sexp_conv.sexp_of_option Moon_sexp_conv.sexp_of_string
               new_usage__073_
           in
           (S.List [ S.Atom "new_usage"; arg__074_ ] :: bnds__068_
             : _ Stdlib.List.t)
         in
         let bnds__068_ =
           let arg__072_ = Moon_sexp_conv.sexp_of_string purpose__071_ in
           (S.List [ S.Atom "purpose"; arg__072_ ] :: bnds__068_
             : _ Stdlib.List.t)
         in
         let bnds__068_ =
           let arg__070_ = Moon_sexp_conv.sexp_of_string old_usage__069_ in
           (S.List [ S.Atom "old_usage"; arg__070_ ] :: bnds__068_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Deprecated_syntax" :: bnds__068_)
     | Todo -> S.Atom "Todo"
     | Unused_package { name = name__076_; is_alias = is_alias__078_ } ->
         let bnds__075_ = ([] : _ Stdlib.List.t) in
         let bnds__075_ =
           let arg__079_ = Moon_sexp_conv.sexp_of_bool is_alias__078_ in
           (S.List [ S.Atom "is_alias"; arg__079_ ] :: bnds__075_
             : _ Stdlib.List.t)
         in
         let bnds__075_ =
           let arg__077_ = Moon_sexp_conv.sexp_of_string name__076_ in
           (S.List [ S.Atom "name"; arg__077_ ] :: bnds__075_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Unused_package" :: bnds__075_)
     | Empty_package_alias -> S.Atom "Empty_package_alias"
     | Optional_arg_never_supplied arg0__080_ ->
         let res0__081_ = Moon_sexp_conv.sexp_of_string arg0__080_ in
         S.List [ S.Atom "Optional_arg_never_supplied"; res0__081_ ]
     | Optional_arg_always_supplied arg0__082_ ->
         let res0__083_ = Moon_sexp_conv.sexp_of_string arg0__082_ in
         S.List [ S.Atom "Optional_arg_always_supplied"; res0__083_ ]
     | Unused_import_value arg0__084_ ->
         let res0__085_ = Moon_sexp_conv.sexp_of_string arg0__084_ in
         S.List [ S.Atom "Unused_import_value"; res0__085_ ]
     | Deprecated_prefix_label_syntax arg0__086_ ->
         let res0__087_ = Moon_sexp_conv.sexp_of_string arg0__086_ in
         S.List [ S.Atom "Deprecated_prefix_label_syntax"; res0__087_ ]
     | Reserved_keyword arg0__088_ ->
         let res0__089_ = Moon_sexp_conv.sexp_of_string arg0__088_ in
         S.List [ S.Atom "Reserved_keyword"; res0__089_ ]
      : kind -> S.t)

  let _ = sexp_of_kind

  let compare_kind =
    (fun a__090_ b__091_ ->
       if Stdlib.( == ) a__090_ b__091_ then 0
       else
         match (a__090_, b__091_) with
         | Unused_func _a__092_, Unused_func _b__093_ ->
             Stdlib.compare (_a__092_ : string) _b__093_
         | Unused_func _, _ -> -1
         | _, Unused_func _ -> 1
         | Unused_var _a__094_, Unused_var _b__095_ -> (
             match
               Stdlib.compare (_a__094_.var_name : string) _b__095_.var_name
             with
             | 0 ->
                 Stdlib.compare
                   (_a__094_.is_toplevel : bool)
                   _b__095_.is_toplevel
             | n -> n)
         | Unused_var _, _ -> -1
         | _, Unused_var _ -> 1
         | Unused_type_declaration _a__096_, Unused_type_declaration _b__097_ ->
             Stdlib.compare (_a__096_ : string) _b__097_
         | Unused_type_declaration _, _ -> -1
         | _, Unused_type_declaration _ -> 1
         | Unused_abstract_type _a__098_, Unused_abstract_type _b__099_ ->
             Stdlib.compare (_a__098_ : string) _b__099_
         | Unused_abstract_type _, _ -> -1
         | _, Unused_abstract_type _ -> 1
         | Unused_tvar _a__100_, Unused_tvar _b__101_ ->
             Stdlib.compare (_a__100_ : string) _b__101_
         | Unused_tvar _, _ -> -1
         | _, Unused_tvar _ -> 1
         | Unused_constructor _a__102_, Unused_constructor _b__103_ -> (
             match
               Stdlib.compare (_a__102_.constr : string) _b__103_.constr
             with
             | 0 -> compare_unused_kind _a__102_.kind _b__103_.kind
             | n -> n)
         | Unused_constructor _, _ -> -1
         | _, Unused_constructor _ -> 1
         | Unused_field _a__104_, Unused_field _b__105_ ->
             Stdlib.compare (_a__104_ : string) _b__105_
         | Unused_field _, _ -> -1
         | _, Unused_field _ -> 1
         | Unused_constr_arg _a__106_, Unused_constr_arg _b__107_ -> (
             match
               Stdlib.compare (_a__106_.constr : string) _b__107_.constr
             with
             | 0 -> Stdlib.compare (_a__106_.index : int) _b__107_.index
             | n -> n)
         | Unused_constr_arg _, _ -> -1
         | _, Unused_constr_arg _ -> 1
         | Unused_constr_field _a__108_, Unused_constr_field _b__109_ -> (
             match
               Stdlib.compare (_a__108_.constr : string) _b__109_.constr
             with
             | 0 -> (
                 match
                   Stdlib.compare (_a__108_.label : string) _b__109_.label
                 with
                 | 0 ->
                     Stdlib.compare
                       (_a__108_.is_mutated : bool)
                       _b__109_.is_mutated
                 | n -> n)
             | n -> n)
         | Unused_constr_field _, _ -> -1
         | _, Unused_constr_field _ -> 1
         | Redundant_modifier _a__110_, Redundant_modifier _b__111_ -> (
             match
               Stdlib.compare (_a__110_.modifier : string) _b__111_.modifier
             with
             | 0 -> Stdlib.compare (_a__110_.field : string) _b__111_.field
             | n -> n)
         | Redundant_modifier _, _ -> -1
         | _, Redundant_modifier _ -> 1
         | Struct_never_constructed _a__112_, Struct_never_constructed _b__113_
           ->
             Stdlib.compare (_a__112_ : string) _b__113_
         | Struct_never_constructed _, _ -> -1
         | _, Struct_never_constructed _ -> 1
         | Unused_pat, Unused_pat -> 0
         | Unused_pat, _ -> -1
         | _, Unused_pat -> 1
         | Partial_match _a__114_, Partial_match _b__115_ ->
             Ppx_base.compare_list
               (fun a__116_ b__117_ ->
                 Stdlib.compare (a__116_ : string) b__117_)
               _a__114_ _b__115_
         | Partial_match _, _ -> -1
         | _, Partial_match _ -> 1
         | Unreachable, Unreachable -> 0
         | Unreachable, _ -> -1
         | _, Unreachable -> 1
         | Unresolved_tvar _a__118_, Unresolved_tvar _b__119_ ->
             Stdlib.compare (_a__118_ : string) _b__119_
         | Unresolved_tvar _, _ -> -1
         | _, Unresolved_tvar _ -> 1
         | Lowercase_type_name _a__120_, Lowercase_type_name _b__121_ ->
             Stdlib.compare (_a__120_ : string) _b__121_
         | Lowercase_type_name _, _ -> -1
         | _, Lowercase_type_name _ -> 1
         | Unused_mutability _a__122_, Unused_mutability _b__123_ ->
             Stdlib.compare (_a__122_ : string) _b__123_
         | Unused_mutability _, _ -> -1
         | _, Unused_mutability _ -> 1
         | Parser_inconsistency _a__124_, Parser_inconsistency _b__125_ -> (
             match
               Stdlib.compare (_a__124_.file_name : string) _b__125_.file_name
             with
             | 0 -> (
                 match
                   Stdlib.compare (_a__124_.segment : string) _b__125_.segment
                 with
                 | 0 -> (
                     match
                       Stdlib.compare
                         (_a__124_.is_menhir_succeed : bool)
                         _b__125_.is_menhir_succeed
                     with
                     | 0 ->
                         Stdlib.compare
                           (_a__124_.is_handrolled_succeed : bool)
                           _b__125_.is_handrolled_succeed
                     | n -> n)
                 | n -> n)
             | n -> n)
         | Parser_inconsistency _, _ -> -1
         | _, Parser_inconsistency _ -> 1
         | Useless_loop, Useless_loop -> 0
         | Useless_loop, _ -> -1
         | _, Useless_loop -> 1
         | Toplevel_not_left_aligned, Toplevel_not_left_aligned -> 0
         | Toplevel_not_left_aligned, _ -> -1
         | _, Toplevel_not_left_aligned -> 1
         | Unexpected_pragmas _a__126_, Unexpected_pragmas _b__127_ ->
             Stdlib.compare (_a__126_ : string) _b__127_
         | Unexpected_pragmas _, _ -> -1
         | _, Unexpected_pragmas _ -> 1
         | Omitted_constr_argument _a__128_, Omitted_constr_argument _b__129_
           -> (
             match
               Stdlib.compare (_a__128_.constr : string) _b__129_.constr
             with
             | 0 ->
                 Ppx_base.compare_list
                   (fun a__130_ b__131_ ->
                     Stdlib.compare (a__130_ : string) b__131_)
                   _a__128_.labels _b__129_.labels
             | n -> n)
         | Omitted_constr_argument _, _ -> -1
         | _, Omitted_constr_argument _ -> 1
         | Ambiguous_block, Ambiguous_block -> 0
         | Ambiguous_block, _ -> -1
         | _, Ambiguous_block -> 1
         | Useless_try, Useless_try -> 0
         | Useless_try, _ -> -1
         | _, Useless_try -> 1
         | Useless_error_type, Useless_error_type -> 0
         | Useless_error_type, _ -> -1
         | _, Useless_error_type -> 1
         | Useless_catch_all, Useless_catch_all -> 0
         | Useless_catch_all, _ -> -1
         | _, Useless_catch_all -> 1
         | Deprecated_syntax _a__132_, Deprecated_syntax _b__133_ -> (
             match
               Stdlib.compare (_a__132_.old_usage : string) _b__133_.old_usage
             with
             | 0 -> (
                 match
                   Stdlib.compare (_a__132_.purpose : string) _b__133_.purpose
                 with
                 | 0 -> (
                     match (_a__132_.new_usage, _b__133_.new_usage) with
                     | None, None -> 0
                     | None, Some _ -> -1
                     | Some _, None -> 1
                     | Some __option_x, Some __option_y ->
                         (fun a__134_ b__135_ ->
                           Stdlib.compare (a__134_ : string) b__135_)
                           __option_x __option_y)
                 | n -> n)
             | n -> n)
         | Deprecated_syntax _, _ -> -1
         | _, Deprecated_syntax _ -> 1
         | Todo, Todo -> 0
         | Todo, _ -> -1
         | _, Todo -> 1
         | Unused_package _a__136_, Unused_package _b__137_ -> (
             match Stdlib.compare (_a__136_.name : string) _b__137_.name with
             | 0 -> Stdlib.compare (_a__136_.is_alias : bool) _b__137_.is_alias
             | n -> n)
         | Unused_package _, _ -> -1
         | _, Unused_package _ -> 1
         | Empty_package_alias, Empty_package_alias -> 0
         | Empty_package_alias, _ -> -1
         | _, Empty_package_alias -> 1
         | ( Optional_arg_never_supplied _a__138_,
             Optional_arg_never_supplied _b__139_ ) ->
             Stdlib.compare (_a__138_ : string) _b__139_
         | Optional_arg_never_supplied _, _ -> -1
         | _, Optional_arg_never_supplied _ -> 1
         | ( Optional_arg_always_supplied _a__140_,
             Optional_arg_always_supplied _b__141_ ) ->
             Stdlib.compare (_a__140_ : string) _b__141_
         | Optional_arg_always_supplied _, _ -> -1
         | _, Optional_arg_always_supplied _ -> 1
         | Unused_import_value _a__142_, Unused_import_value _b__143_ ->
             Stdlib.compare (_a__142_ : string) _b__143_
         | Unused_import_value _, _ -> -1
         | _, Unused_import_value _ -> 1
         | ( Deprecated_prefix_label_syntax _a__144_,
             Deprecated_prefix_label_syntax _b__145_ ) ->
             Stdlib.compare (_a__144_ : string) _b__145_
         | Deprecated_prefix_label_syntax _, _ -> -1
         | _, Deprecated_prefix_label_syntax _ -> 1
         | Reserved_keyword _a__146_, Reserved_keyword _b__147_ ->
             Stdlib.compare (_a__146_ : string) _b__147_
      : kind -> kind -> int)

  let _ = compare_kind

  let equal_kind =
    (fun a__148_ b__149_ ->
       if Stdlib.( == ) a__148_ b__149_ then true
       else
         match (a__148_, b__149_) with
         | Unused_func _a__150_, Unused_func _b__151_ ->
             Stdlib.( = ) (_a__150_ : string) _b__151_
         | Unused_func _, _ -> false
         | _, Unused_func _ -> false
         | Unused_var _a__152_, Unused_var _b__153_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__152_.var_name : string) _b__153_.var_name)
               (Stdlib.( = ) (_a__152_.is_toplevel : bool) _b__153_.is_toplevel)
         | Unused_var _, _ -> false
         | _, Unused_var _ -> false
         | Unused_type_declaration _a__154_, Unused_type_declaration _b__155_ ->
             Stdlib.( = ) (_a__154_ : string) _b__155_
         | Unused_type_declaration _, _ -> false
         | _, Unused_type_declaration _ -> false
         | Unused_abstract_type _a__156_, Unused_abstract_type _b__157_ ->
             Stdlib.( = ) (_a__156_ : string) _b__157_
         | Unused_abstract_type _, _ -> false
         | _, Unused_abstract_type _ -> false
         | Unused_tvar _a__158_, Unused_tvar _b__159_ ->
             Stdlib.( = ) (_a__158_ : string) _b__159_
         | Unused_tvar _, _ -> false
         | _, Unused_tvar _ -> false
         | Unused_constructor _a__160_, Unused_constructor _b__161_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__160_.constr : string) _b__161_.constr)
               (equal_unused_kind _a__160_.kind _b__161_.kind)
         | Unused_constructor _, _ -> false
         | _, Unused_constructor _ -> false
         | Unused_field _a__162_, Unused_field _b__163_ ->
             Stdlib.( = ) (_a__162_ : string) _b__163_
         | Unused_field _, _ -> false
         | _, Unused_field _ -> false
         | Unused_constr_arg _a__164_, Unused_constr_arg _b__165_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__164_.constr : string) _b__165_.constr)
               (Stdlib.( = ) (_a__164_.index : int) _b__165_.index)
         | Unused_constr_arg _, _ -> false
         | _, Unused_constr_arg _ -> false
         | Unused_constr_field _a__166_, Unused_constr_field _b__167_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__166_.constr : string) _b__167_.constr)
               (Stdlib.( && )
                  (Stdlib.( = ) (_a__166_.label : string) _b__167_.label)
                  (Stdlib.( = )
                     (_a__166_.is_mutated : bool)
                     _b__167_.is_mutated))
         | Unused_constr_field _, _ -> false
         | _, Unused_constr_field _ -> false
         | Redundant_modifier _a__168_, Redundant_modifier _b__169_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__168_.modifier : string) _b__169_.modifier)
               (Stdlib.( = ) (_a__168_.field : string) _b__169_.field)
         | Redundant_modifier _, _ -> false
         | _, Redundant_modifier _ -> false
         | Struct_never_constructed _a__170_, Struct_never_constructed _b__171_
           ->
             Stdlib.( = ) (_a__170_ : string) _b__171_
         | Struct_never_constructed _, _ -> false
         | _, Struct_never_constructed _ -> false
         | Unused_pat, Unused_pat -> true
         | Unused_pat, _ -> false
         | _, Unused_pat -> false
         | Partial_match _a__172_, Partial_match _b__173_ ->
             Ppx_base.equal_list
               (fun a__174_ b__175_ -> Stdlib.( = ) (a__174_ : string) b__175_)
               _a__172_ _b__173_
         | Partial_match _, _ -> false
         | _, Partial_match _ -> false
         | Unreachable, Unreachable -> true
         | Unreachable, _ -> false
         | _, Unreachable -> false
         | Unresolved_tvar _a__176_, Unresolved_tvar _b__177_ ->
             Stdlib.( = ) (_a__176_ : string) _b__177_
         | Unresolved_tvar _, _ -> false
         | _, Unresolved_tvar _ -> false
         | Lowercase_type_name _a__178_, Lowercase_type_name _b__179_ ->
             Stdlib.( = ) (_a__178_ : string) _b__179_
         | Lowercase_type_name _, _ -> false
         | _, Lowercase_type_name _ -> false
         | Unused_mutability _a__180_, Unused_mutability _b__181_ ->
             Stdlib.( = ) (_a__180_ : string) _b__181_
         | Unused_mutability _, _ -> false
         | _, Unused_mutability _ -> false
         | Parser_inconsistency _a__182_, Parser_inconsistency _b__183_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__182_.file_name : string) _b__183_.file_name)
               (Stdlib.( && )
                  (Stdlib.( = ) (_a__182_.segment : string) _b__183_.segment)
                  (Stdlib.( && )
                     (Stdlib.( = )
                        (_a__182_.is_menhir_succeed : bool)
                        _b__183_.is_menhir_succeed)
                     (Stdlib.( = )
                        (_a__182_.is_handrolled_succeed : bool)
                        _b__183_.is_handrolled_succeed)))
         | Parser_inconsistency _, _ -> false
         | _, Parser_inconsistency _ -> false
         | Useless_loop, Useless_loop -> true
         | Useless_loop, _ -> false
         | _, Useless_loop -> false
         | Toplevel_not_left_aligned, Toplevel_not_left_aligned -> true
         | Toplevel_not_left_aligned, _ -> false
         | _, Toplevel_not_left_aligned -> false
         | Unexpected_pragmas _a__184_, Unexpected_pragmas _b__185_ ->
             Stdlib.( = ) (_a__184_ : string) _b__185_
         | Unexpected_pragmas _, _ -> false
         | _, Unexpected_pragmas _ -> false
         | Omitted_constr_argument _a__186_, Omitted_constr_argument _b__187_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__186_.constr : string) _b__187_.constr)
               (Ppx_base.equal_list
                  (fun a__188_ b__189_ ->
                    Stdlib.( = ) (a__188_ : string) b__189_)
                  _a__186_.labels _b__187_.labels)
         | Omitted_constr_argument _, _ -> false
         | _, Omitted_constr_argument _ -> false
         | Ambiguous_block, Ambiguous_block -> true
         | Ambiguous_block, _ -> false
         | _, Ambiguous_block -> false
         | Useless_try, Useless_try -> true
         | Useless_try, _ -> false
         | _, Useless_try -> false
         | Useless_error_type, Useless_error_type -> true
         | Useless_error_type, _ -> false
         | _, Useless_error_type -> false
         | Useless_catch_all, Useless_catch_all -> true
         | Useless_catch_all, _ -> false
         | _, Useless_catch_all -> false
         | Deprecated_syntax _a__190_, Deprecated_syntax _b__191_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__190_.old_usage : string) _b__191_.old_usage)
               (Stdlib.( && )
                  (Stdlib.( = ) (_a__190_.purpose : string) _b__191_.purpose)
                  (match (_a__190_.new_usage, _b__191_.new_usage) with
                  | None, None -> true
                  | None, Some _ -> false
                  | Some _, None -> false
                  | Some __option_x, Some __option_y ->
                      (fun a__192_ b__193_ ->
                        Stdlib.( = ) (a__192_ : string) b__193_)
                        __option_x __option_y))
         | Deprecated_syntax _, _ -> false
         | _, Deprecated_syntax _ -> false
         | Todo, Todo -> true
         | Todo, _ -> false
         | _, Todo -> false
         | Unused_package _a__194_, Unused_package _b__195_ ->
             Stdlib.( && )
               (Stdlib.( = ) (_a__194_.name : string) _b__195_.name)
               (Stdlib.( = ) (_a__194_.is_alias : bool) _b__195_.is_alias)
         | Unused_package _, _ -> false
         | _, Unused_package _ -> false
         | Empty_package_alias, Empty_package_alias -> true
         | Empty_package_alias, _ -> false
         | _, Empty_package_alias -> false
         | ( Optional_arg_never_supplied _a__196_,
             Optional_arg_never_supplied _b__197_ ) ->
             Stdlib.( = ) (_a__196_ : string) _b__197_
         | Optional_arg_never_supplied _, _ -> false
         | _, Optional_arg_never_supplied _ -> false
         | ( Optional_arg_always_supplied _a__198_,
             Optional_arg_always_supplied _b__199_ ) ->
             Stdlib.( = ) (_a__198_ : string) _b__199_
         | Optional_arg_always_supplied _, _ -> false
         | _, Optional_arg_always_supplied _ -> false
         | Unused_import_value _a__200_, Unused_import_value _b__201_ ->
             Stdlib.( = ) (_a__200_ : string) _b__201_
         | Unused_import_value _, _ -> false
         | _, Unused_import_value _ -> false
         | ( Deprecated_prefix_label_syntax _a__202_,
             Deprecated_prefix_label_syntax _b__203_ ) ->
             Stdlib.( = ) (_a__202_ : string) _b__203_
         | Deprecated_prefix_label_syntax _, _ -> false
         | _, Deprecated_prefix_label_syntax _ -> false
         | Reserved_keyword _a__204_, Reserved_keyword _b__205_ ->
             Stdlib.( = ) (_a__204_ : string) _b__205_
      : kind -> kind -> bool)

  let _ = equal_kind
end

type t = { loc : Loc.t; kind : kind }

include struct
  let _ = fun (_ : t) -> ()

  let sexp_of_t =
    (fun { loc = loc__207_; kind = kind__209_ } ->
       let bnds__206_ = ([] : _ Stdlib.List.t) in
       let bnds__206_ =
         let arg__210_ = sexp_of_kind kind__209_ in
         (S.List [ S.Atom "kind"; arg__210_ ] :: bnds__206_ : _ Stdlib.List.t)
       in
       let bnds__206_ =
         let arg__208_ = Loc.sexp_of_t loc__207_ in
         (S.List [ S.Atom "loc"; arg__208_ ] :: bnds__206_ : _ Stdlib.List.t)
       in
       S.List bnds__206_
      : t -> S.t)

  let _ = sexp_of_t

  let compare =
    (fun a__211_ b__212_ ->
       if Stdlib.( == ) a__211_ b__212_ then 0
       else
         match Loc.compare a__211_.loc b__212_.loc with
         | 0 -> compare_kind a__211_.kind b__212_.kind
         | n -> n
      : t -> t -> int)

  let _ = compare

  let equal =
    (fun a__213_ b__214_ ->
       if Stdlib.( == ) a__213_ b__214_ then true
       else
         Stdlib.( && )
           (Loc.equal a__213_.loc b__214_.loc)
           (equal_kind a__213_.kind b__214_.kind)
      : t -> t -> bool)

  let _ = equal
end

let number = function
  | Unused_func _ -> 1
  | Unused_var _ -> 2
  | Unused_type_declaration _ -> 3
  | Unused_abstract_type _ -> 4
  | Unused_tvar _ -> 5
  | Unused_constructor _ -> 6
  | Unused_field _ | Unused_constr_arg _ | Unused_constr_field _ -> 7
  | Redundant_modifier _ -> 8
  | Struct_never_constructed _ -> 9
  | Unused_pat -> 10
  | Partial_match _ -> 11
  | Unreachable -> 12
  | Unresolved_tvar _ -> 13
  | Lowercase_type_name _ -> 14
  | Unused_mutability _ -> 15
  | Parser_inconsistency _ -> 16
  | Useless_loop -> 18
  | Toplevel_not_left_aligned -> 19
  | Unexpected_pragmas _ -> 20
  | Omitted_constr_argument _ -> 21
  | Ambiguous_block -> 22
  | Useless_try -> 23
  | Useless_error_type -> 24
  | Useless_catch_all -> 26
  | Deprecated_syntax _ -> 27
  | Todo -> 28
  | Unused_package _ -> 29
  | Empty_package_alias -> 30
  | Optional_arg_never_supplied _ -> 31
  | Optional_arg_always_supplied _ -> 32
  | Unused_import_value _ -> 33
  | Deprecated_prefix_label_syntax _ -> 34
  | Reserved_keyword _ -> 35

let last_warning_id = 35

let message ?(as_error = false) (x : kind) : string =
  let leading = if as_error then "Error (warning): " else "Warning: " in
  let msg =
    match x with
    | Unused_func s -> ("Unused function '" ^ s ^ "'" : Stdlib.String.t)
    | Unused_var { var_name; is_toplevel } ->
        if is_toplevel then
          ("Unused toplevel variable '" ^ var_name
           ^ "'. Note if the body contains side effect, it will not happen. \
              Use `fn init { .. }` to wrap the effect."
            : Stdlib.String.t)
        else ("Unused variable '" ^ var_name ^ "'" : Stdlib.String.t)
    | Unused_type_declaration s -> ("Unused type '" ^ s ^ "'" : Stdlib.String.t)
    | Unused_abstract_type s ->
        ("Unused abstract type '" ^ s ^ "'" : Stdlib.String.t)
    | Unused_tvar s ->
        ("Unused generic type variable '" ^ s ^ "'" : Stdlib.String.t)
    | Unused_constructor { constr; kind } -> (
        match kind with
        | Unused -> ("Variant '" ^ constr ^ "' is unused" : Stdlib.String.t)
        | No_construct ->
            ("Variant '" ^ constr ^ "' is never constructed" : Stdlib.String.t)
        | No_read ->
            ("Variant '" ^ constr ^ "' is never read" : Stdlib.String.t))
    | Unused_field s -> ("Field '" ^ s ^ "' is never read" : Stdlib.String.t)
    | Unused_constr_arg { constr; index } ->
        let index =
          match index with
          | 0 -> "1st"
          | 1 -> "2nd"
          | 2 -> "3rd"
          | n -> (Int.to_string (n + 1) ^ "th" : Stdlib.String.t)
        in
        Stdlib.String.concat ""
          [
            "The ";
            index;
            " positional argument of constructor '";
            constr;
            "' is unused.";
          ]
    | Unused_constr_field { constr; label; is_mutated = false } ->
        Stdlib.String.concat ""
          [ "Field '"; label; "' of constructor '"; constr; "' is unused." ]
    | Unused_constr_field { constr; label; is_mutated = true } ->
        Stdlib.String.concat ""
          [ "Field '"; label; "' of constructor '"; constr; "' is never read." ]
    | Redundant_modifier { modifier; field } ->
        Stdlib.String.concat ""
          [
            "The ";
            modifier;
            " modifier is redundant here since field ";
            field;
            " is ";
            modifier;
            " by default";
          ]
    | Struct_never_constructed name ->
        ("The struct " ^ name ^ " is never constructed" : Stdlib.String.t)
    | Unused_pat -> "Unused pattern"
    | Partial_match _->
        "Partial match"
    | Unreachable -> "Unreachable code"
    | Unresolved_tvar s ->
        ("The type of this expression is " ^ s
         ^ ", which contains unresolved type variables. The type variable is \
            default to Unit."
          : Stdlib.String.t)
    | Lowercase_type_name s ->
        ("Type name '" ^ s ^ "' should be capitalized." : Stdlib.String.t)
    | Unused_mutability s ->
        ("The mutability of " ^ s ^ " is never used." : Stdlib.String.t)
    | Parser_inconsistency
        { file_name; segment; is_menhir_succeed; is_handrolled_succeed } ->
        let p x = if x then "succeed" else "failed" in
        Stdlib.String.concat ""
          [
            "parser consistency check failed at '";
            file_name;
            "' ";
            segment;
            " (menhir parser ";
            p is_menhir_succeed;
            ", handrolled parser ";
            p is_handrolled_succeed;
            ")";
          ]
    | Useless_loop ->
        ("There is no [continue] in this loop expression, so [loop] is useless \
          here."
          : Stdlib.String.t)
    | Toplevel_not_left_aligned ->
        ("Toplevel declaration is not left aligned." : Stdlib.String.t)
    | Unexpected_pragmas s -> ("Invalid pragma, " ^ s : Stdlib.String.t)
    | Omitted_constr_argument { constr; labels } ->
        let labels_str = String.concat "," labels in
        Stdlib.String.concat ""
          [
            "The argument(s) ";
            labels_str;
            " of constructor ";
            constr;
            " are omitted (To ignore them, add \"..\" to the end of argument \
             list).";
          ]
    | Ambiguous_block ->
        ("Ambiguous block expression. Use `id` directly, or use `{ id, }` to \
          clarify a struct literal."
          : Stdlib.String.t)
    | Useless_try -> "The body of this try expression never raises any error."
    | Useless_error_type -> "The error type of this function is never used."
    | Useless_catch_all ->
        "The patterns are complete so the usage of `catch!` is useless. Use \
         `catch` instead."
    | Deprecated_syntax { old_usage; purpose; new_usage } ->
        let new_usage =
          match new_usage with
          | None -> ""
          | Some new_usage ->
              (" Use " ^ new_usage ^ " instead." : Stdlib.String.t)
        in
        Stdlib.String.concat ""
          [
            "The syntax ";
            old_usage;
            " for ";
            purpose;
            " is deprecated.";
            new_usage;
          ]
    | Todo -> "unfinished code"
    | Unused_package { name; is_alias } ->
        if is_alias then
          ("Unused package alias '" ^ name ^ "'" : Stdlib.String.t)
        else ("Unused package '" ^ name ^ "'" : Stdlib.String.t)
    | Empty_package_alias ->
        "The package alias is empty. The default package alias will be used \
         instead."
    | Optional_arg_never_supplied label ->
        ("The optional argument '" ^ label ^ "' is never supplied."
          : Stdlib.String.t)
    | Optional_arg_always_supplied label ->
        ("Default value of optional argument '" ^ label ^ "' is unused."
          : Stdlib.String.t)
    | Unused_import_value name ->
        ("The import value " ^ name ^ " is never used directly. "
          : Stdlib.String.t)
    | Deprecated_prefix_label_syntax label ->
        Stdlib.String.concat ""
          [
            "The syntax `~";
            label;
            "` is deprecated. Use postfix style like `";
            label;
            "~` or `";
            label;
            "?` instead.";
          ]
    | Reserved_keyword s ->
        ("The word `" ^ s
         ^ "` is reserved for possible future use. Please consider using \
            another name."
          : Stdlib.String.t)
  in
  leading ^ msg

type state = { active : bool array; error : bool array }

let current =
  ref
    {
      active = Array.make (last_warning_id + 1) true;
      error = Array.make (last_warning_id + 1) false;
    }

let disabled = ref false
let disable_warn_as_error = ref false
let is_active x = (not !disabled) && !current.active.(number x)
let is_error x = (not !disable_warn_as_error) && !current.error.(number x)
let without_warn_as_error f = Basic_ref.protect disable_warn_as_error true f

let parse_opt active error flags s =
  let set i = flags.(i) <- true in
  let reset i =
    flags.(i) <- false;
    error.(i) <- false
  in
  let both i =
    active.(i) <- true;
    error.(i) <- true
  in
  let error msg = raise (Arg.Bad ("Ill-formed list of warnings: " ^ msg)) in
  let unknown_token c = error ("unexpected token '" ^ String.make 1 c ^ "'") in
  let readint i =
    let rec go acc i =
      if i >= String.length s then (i, acc)
      else
        match s.[i] with
        | '0' .. '9' -> go ((10 * acc) + Char.code s.[i] - Char.code '0') (i + 1)
        | _ -> (i, acc)
    in
    go 0 i
  in
  let readrange i =
    let i, n1 = readint i in
    if i + 2 < String.length s && s.[i] = '.' && s.[i + 1] = '.' then
      let i, n2 = readint (i + 2) in
      if n2 < n1 then
        error (string_of_int n2 ^ " is smaller than " ^ string_of_int n1)
      else (i, n1, n2)
    else (i, n1, n1)
  in
  let alpha f i =
    if i >= String.length s then error "unexpected end"
    else
      match s.[i] with
      | 'A' | 'a' ->
          for j = 1 to last_warning_id do
            f j
          done;
          i + 1
      | '0' .. '9' ->
          let i, n1, n2 = readrange i in
          for j = n1 to Int.min n2 last_warning_id do
            f j
          done;
          i
      | _ -> unknown_token s.[i]
  in
  let rec loop i =
    if i < String.length s then
      match s.[i] with
      | 'A' | 'a' -> loop (alpha set i)
      | '+' -> loop (alpha set (i + 1))
      | '-' -> loop (alpha reset (i + 1))
      | '@' -> loop (alpha both (i + 1))
      | _ -> unknown_token s.[i]
  in
  loop 0

let parse_options errflag s =
  let error = Array.copy !current.error in
  let active = Array.copy !current.active in
  parse_opt active error (if errflag then error else active) s;
  current := { active; error }

let default_warnings = "+a-31-32"
let default_warnings_as_errors = "-a+15+19+23+24"

let reset () =
  parse_options false default_warnings;
  parse_options true default_warnings_as_errors

let () = reset ()

let descriptions =
  [
    (1, "Unused function.");
    (2, "Unused variable.");
    (3, "Unused type declaration.");
    (4, "Redundant case in a pattern matching (unused match case).");
    (5, "Unused function argument.");
    (6, "Unused constructor.");
    (7, "Unused module declaration.");
    (8, "Unused struct field.");
    (10, "Unused generic type variable.");
    (11, "Partial pattern matching.");
    (12, "Unreachable code.");
    (13, "Unresolved type variable.");
    (14, "Lowercase type name.");
    (15, "Unused mutability.");
    (16, "Parser inconsistency.");
    (18, "Useless loop expression.");
    (19, "Top-level declaration is not left aligned.");
    (20, "Invalid pragma");
    (21, "Some arguments of constructor are omitted in pattern.");
    (22, "Ambiguous block.");
    (23, "Useless try expression.");
    (24, "Useless error type.");
    (26, "Useless catch all.");
    (27, "Deprecated syntax.");
    (28, "Todo");
    (29, "Unused package.");
    (30, "Empty package alias.");
    (31, "Optional argument never supplied.");
    (32, "Default value of optional argument never used.");
    (33, "Unused import value");
  ]

let help_warnings () =
  print_endline "Available warnings: ";
  List.iter (fun (i, s) -> Printf.printf "%3i %s\n" i s) descriptions;
  print_endline "  A all warnings";
  exit 0
