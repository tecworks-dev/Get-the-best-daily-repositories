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


module Longident = Basic_longident
module Type_path = Basic_type_path
module Ident = Basic_ident
module Lst = Basic_lst

type report = Local_diagnostics.report
type loc = Rloc.t

let type_path_name p =
  Type_path.short_name ~cur_pkg_name:(Some !Basic_config.current_package) p

let swallow_error : report = Local_diagnostics.swallow_error

let internal message : Diagnostics.report =
  {
    message = "Compiler Internal Error: " ^ message;
    loc = Loc.no_location;
    error_code = Error_code.internal;
  }

let lexing_error ~loc_start ~loc_end message : Diagnostics.report =
  {
    loc = Loc.of_menhir (loc_start, loc_end);
    message = "Lexing error: " ^ message;
    error_code = Error_code.lexing_error;
  }

let parse_error ~loc_start ~loc_end message : Diagnostics.report =
  {
    loc = Loc.of_menhir (loc_start, loc_end);
    message;
    error_code = Error_code.parse_error;
  }

let json_parse_error ~loc_start ~loc_end message : Diagnostics.report =
  {
    loc = Loc.of_menhir (loc_start, loc_end);
    message;
    error_code = Error_code.json_parse_error;
  }

let invalid_init_or_main ~(kind : [ `Init | `Main ]) ~loc : Diagnostics.report =
  let message = match kind with `Init -> "Init" | `Main -> "Main" in
  {
    loc;
    message =
      (message ^ " function must have no arguments and no return value."
        : Stdlib.String.t);
    error_code = Error_code.invalid_init_or_main;
  }

let missing_parameter_list ~name ~loc : report =
  let message =
    ("Missing parameters list. Add `()` if function `" ^ name
     ^ "` has 0 parameter."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.missing_parameter_list }

let unexpected_token ~found ~expected ~loc : Diagnostics.report =
  let message =
    Stdlib.String.concat ""
      [
        "Parse error, unexpected token ";
        found;
        ", you may expect ";
        expected;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.parse_error }

let unexpected_line_break ~expected ~loc : Diagnostics.report =
  let message =
    ("Unexpected line break here, missing " ^ expected
     ^ " at the end of this line."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.parse_error }

let unexpected_token_maybe_forget_indent ~expected ~next ~loc :
    Diagnostics.report =
  let message =
    Stdlib.String.concat ""
      [
        "Parse error, expect ";
        expected;
        ". Did you forget to indent the local ";
        next;
        "?";
      ]
  in
  { message; loc; error_code = Error_code.parse_error }

let invalid_visibility ~entity ~vis ~loc : report =
  let message =
    Stdlib.String.concat "" [ "No '"; vis; "' visibility for "; entity; "." ]
  in
  { message; loc; error_code = Error_code.invalid_visibility }

let enum_no_individual_visibility loc : report =
  {
    message = "No individual visibility for enum constructor.";
    loc;
    error_code = Error_code.enum_no_individual_visibility;
  }

let dotdot_in_middle_of_pattern ~(kind : [ `Record | `Constr ]) ~loc : report =
  let component, kind =
    match kind with
    | `Record -> ("field", "record")
    | `Constr -> ("argument", "constructor arguments")
  in
  let message =
    Stdlib.String.concat ""
      [
        "Unexpected `..` here, add `, ..` behind the last ";
        component;
        " to ignore the rest of ";
        kind;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.dotdot_in_middle_of_pattern }

let array_pat_multiple_dotdot loc : report =
  {
    message = "At most one `..` is allowed in array pattern.";
    loc;
    error_code = Error_code.array_pat_multiple_dotdot;
  }

let record_pattern_only_dotdot loc : report =
  {
    message =
      "Record pattern cannot contain only `..`, use wildcard pattern `_` \
       instead.";
    loc;
    error_code = Error_code.record_pattern_only_dotdot;
  }

let positional_argument_no_default loc : report =
  {
    message = "Only labelled arguments can have default value.";
    loc;
    error_code = Error_code.positional_argument_no_default;
  }

let invalid_left_value loc : report =
  {
    message = "Invalid left value for assignment.";
    loc;
    error_code = Error_code.invalid_left_value;
  }

let bad_range_pattern_operand loc : report =
  {
    message =
      "Bounds of range pattern must be constant, named constant or wildcard.";
    loc;
    error_code = Error_code.bad_range_pattern_operand;
  }

let inclusive_range_pattern_no_upper_bound loc : report =
  {
    message = "Inclusive range pattern `a..=b` cannot have `_` as upper bound";
    loc;
    error_code = Error_code.inclusive_range_pattern_no_upper_bound;
  }

let cannot_mix_record_and_map_pat loc : report =
  {
    message = "Record pattern and map pattern cannot be mixed.";
    loc;
    error_code = Error_code.cannot_mix_record_and_map_pat;
  }

let map_pattern_always_open loc : report =
  {
    message = "Map patterns are always open, the `..` is useless.";
    loc;
    error_code = Error_code.map_pattern_always_open;
  }

let inline_wasm_syntax_error ~message ~loc_inside_wasm ~loc : Diagnostics.report
    =
  {
    message =
      Stdlib.String.concat ""
        [
          "Inline wasm syntax error: ";
          message;
          " at ";
          Loc.loc_range_string_no_filename loc_inside_wasm;
        ];
    loc;
    error_code = Error_code.inline_wasm_syntax_error;
  }

let duplicate_tvar ~name ~loc : report =
  {
    loc;
    message =
      ("Generic type variable name '" ^ name ^ "' is already used."
        : Stdlib.String.t);
    error_code = Error_code.duplicate_tvar;
  }

let field_visibility ~field_vis ~type_vis ~loc : report =
  {
    loc;
    message =
      Stdlib.String.concat ""
        [
          "A ";
          field_vis;
          " field cannot be declared within a ";
          type_vis;
          " struct.";
        ];
    error_code = Error_code.field_visibility;
  }

let unsupported_modifier ~modifier ~loc : report =
  let message =
    ("The " ^ modifier ^ " modifier is not supported here" : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unsupported_modifier }

let reserved_type_name ~(decl_kind : [ `Tvar | `Type | `Trait ]) ~name ~loc :
    report =
  let name_decl_kind =
    match decl_kind with
    | `Tvar -> "type variable"
    | `Type -> "type"
    | `Trait -> "trait"
  in
  let message =
    Stdlib.String.concat ""
      [
        "\"";
        name;
        "\" is a reserved type name. Cannot declare it as ";
        name_decl_kind;
      ]
  in
  { message; loc; error_code = Error_code.reserved_type_name }

let trait_method_cannot_poly loc : report =
  let message = "polymorphic trait method is not supported" in
  { message; loc; error_code = Error_code.trait_method_cannot_poly }

let trait_duplicate_method ~trait ~name ~first ~second : report =
  let message =
    Stdlib.String.concat ""
      [
        "method ";
        name;
        " of trait ";
        trait;
        " is declared multiple times (first at ";
        Loc.to_string first;
        ")";
      ]
  in
  { message; loc = second; error_code = Error_code.trait_duplicate_method }

let duplicate_local_fns ~name ~loc ~prev_loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "local function '";
        name;
        "' is already defined at ";
        Int.to_string (Loc.line_number prev_loc);
        ":";
        Int.to_string (Loc.column_number prev_loc);
      ]
  in
  { message; loc; error_code = Error_code.duplicate_local_fns }

let illform_constr_arg loc : report =
  let message = ("constructor can't take unit as argument" : Stdlib.String.t) in
  { message; loc; error_code = Error_code.illform_constr_arg }

let ffi_cannot_poly loc : Diagnostics.report =
  let message = "FFI function cannot have type parameters." in
  { message; loc; error_code = Error_code.ffi_cannot_poly }

let matchfn_arity_mismatch ~loc ~expected ~actual : report =
  let message =
    Stdlib.String.concat ""
      [
        "Match function expects ";
        Int.to_string expected;
        " arguments, but ";
        Int.to_string actual;
        " arguments are provided.";
      ]
  in
  { message; loc; error_code = Error_code.matchfn_arity_mismatch }

let no_vis_on_default_impl loc : report =
  let message =
    ("`pub` is not allowed on default implementation for traits."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.no_vis_on_default_impl }

let no_quantifiers_on_default_impl loc : report =
  let message =
    ("Type parameters are not allowed on default implementation for traits."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.no_quantifiers_on_default_impl }

let constr_no_mut_positional_field loc : report =
  let message =
    ("Mutable constructor fields are only allowed on labelled arguments."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.constr_no_mut_positional_field }

let func_param_num_mismatch ~expected ~actual ~ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "This function has type ";
        ty;
        ", which expects ";
        Int.to_string expected;
        " argument(s), but is given ";
        Int.to_string actual;
        " argument(s).";
      ]
  in
  { message; loc; error_code = Error_code.func_param_num_mismatch }

let generic_type_mismatch ~header ~expected ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        header;
        "\n        has type : ";
        actual;
        "\n        wanted   : ";
        expected;
      ]
  in
  { message; loc; error_code = Error_code.type_mismatch }

let type_mismatch ~expected ~actual ~loc =
  generic_type_mismatch ~header:"Type Mismatch" ~expected ~actual ~loc

let expr_unify ~expected ~actual ~loc =
  generic_type_mismatch ~header:"Expr Type Mismatch" ~expected ~actual ~loc

let pat_unify ~expected ~actual ~loc =
  generic_type_mismatch ~header:"Pattern Type Mismatch" ~expected ~actual ~loc

let param_unify ~name ~expected ~actual ~loc =
  generic_type_mismatch
    ~header:
      ("Parameter Type Mismatch(parameter " ^ name ^ ")" : Stdlib.String.t)
    ~expected ~actual ~loc

let constr_unify ~name ~expected ~actual ~loc =
  generic_type_mismatch
    ~header:("Constr Type Mismatch(constructor " ^ name ^ ")" : Stdlib.String.t)
    ~expected ~actual ~loc

let cascade_type_mismatch ~actual ~loc : report =
  let message =
    ("This method returns " ^ actual
     ^ ", but only methods that return Unit can be used with `..`."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.type_mismatch }

let cannot_resolve_method ~ty ~name ~loc ~hint : report =
  let message =
    match hint with
    | `Record ->
        Stdlib.String.concat ""
          [
            "Type ";
            ty;
            " has no method ";
            name;
            ".\n  (hint: to apply record field as function, write `(x.";
            name;
            ")(...)` instead)";
          ]
    | `Trait ->
        Stdlib.String.concat "" [ "Trait "; ty; " has no method "; name; "." ]
    | `No_hint ->
        Stdlib.String.concat "" [ "Type "; ty; " has no method "; name; "." ]
  in
  { message; loc; error_code = Error_code.cannot_resolve_method }

let cannot_resolve_infix_op ~method_name ~op_name ~ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Please implement the method ";
        method_name;
        " for the type ";
        ty;
        " to use the infix operator \"";
        op_name;
        "\".";
      ]
  in
  { message; loc; error_code = Error_code.cannot_resolve_infix_op }

let ambiguous_trait_method ~label ~ty ~first ~second ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        label;
        " of type ";
        ty;
        " is ambiguious, it may come from trait ";
        first;
        " or ";
        second;
      ]
  in
  { message; loc; error_code = Error_code.ambiguous_trait_method }

let cannot_resolve_trait ~loc ~message : report =
  { message; loc; error_code = Error_code.cannot_resolve_trait }

let interp_to_string_incorrect_type ~self_ty ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Method `to_string` of type ";
        self_ty;
        " has type ";
        actual;
        ", but is expected to have type (";
        self_ty;
        ") -> String.";
      ]
  in
  { message; loc; error_code = Error_code.type_mismatch }

let duplicated_label_in_decl ~label ~first_loc ~second_loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "The label ";
        label;
        "~ is declared twice in this function, first in ";
        Loc.to_string first_loc;
      ]
  in
  {
    message;
    loc = second_loc;
    error_code = Error_code.duplicated_label_in_decl;
  }

let pkg_not_loaded ~pkg ~loc : report =
  let message =
    ("Package \"" ^ pkg ^ "\" not found in the loaded packages."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.pkg_not_loaded }

let unbound_value ~name ~loc : report =
  let message =
    match (name : Longident.t) with
    | Lident id ->
        ("The value identifier " ^ id ^ " is unbound." : Stdlib.String.t)
    | Ldot { pkg; id } ->
        Printf.sprintf "Value %s not found in package %S." id pkg
  in
  { message; loc; error_code = Error_code.unbound_value }

let unbound_type ~name ~loc : report =
  let message =
    match (name : Longident.t) with
    | Lident id ->
        ("The type constructor " ^ id ^ " is not found." : Stdlib.String.t)
    | Ldot { pkg; id } ->
        Printf.sprintf "Type %s not found in package %S." id pkg
  in
  { message; loc; error_code = Error_code.unbound_type }

let unbound_trait ~name ~loc : report =
  let message =
    match (name : Longident.t) with
    | Lident id -> ("The trait " ^ id ^ " is not found." : Stdlib.String.t)
    | Ldot { pkg; id } ->
        Printf.sprintf "Trait %s not found in package %S." id pkg
  in
  { message; loc; error_code = Error_code.unbound_trait }

let unbound_type_or_trait ~name ~loc : report =
  let message =
    ("The type/trait " ^ Longident.to_string name ^ " is not found."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unbound_type_or_trait }

let ambiguous_method ~name ~type_locs ~loc : report =
  let info =
    type_locs
    |> List.map (fun (s, loc) : Stdlib.String.t ->
           Loc.to_string loc ^ " " ^ type_path_name s)
    |> String.concat "\n"
  in
  let message =
    Stdlib.String.concat ""
      [
        "Method ";
        name;
        " has been defined for the following types:\n";
        info;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.ambiguous_method }

let unbound_field ~name ~loc : report =
  let message = ("The field " ^ name ^ " is not found." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.unbound_field }

let not_a_record ~may_be_method ~ty ~kind ~loc : report =
  let base_message =
    Stdlib.String.concat ""
      [
        "This expression has type ";
        ty;
        ", which is a ";
        kind;
        " type and not a record.";
      ]
  in
  let message =
    match may_be_method with
    | Some field ->
        Stdlib.String.concat ""
          [
            base_message;
            "\n  (hint: to pass method as function, write `fn (...) { x.";
            field;
            "(...) }` instead)";
          ]
    | None -> base_message
  in
  { message; loc; error_code = Error_code.not_a_record }

let not_a_variant ~ty ~kind ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "This expression has type ";
        ty;
        ", which is a ";
        kind;
        " type and not a variant.";
      ]
  in
  { message; loc; error_code = Error_code.not_a_variant }

let field_not_found ~ty ~label ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "The record type "; ty; " does not have the field "; label; "." ]
  in
  { message; loc; error_code = Error_code.field_not_found }

let constr_not_found ~ty ~constr ~loc : report =
  let message =
    match ty with
    | None -> ("The value " ^ constr ^ " is undefined." : Stdlib.String.t)
    | Some ty ->
        Stdlib.String.concat ""
          [
            "The variant type ";
            ty;
            " does not have the constructor ";
            constr;
            ".";
          ]
  in
  { message; loc; error_code = Error_code.constr_not_found }

let type_not_found ~tycon ~loc : report =
  let message = ("The type " ^ tycon ^ " is undefined." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.type_not_found }

let cannot_resolve_record ~labels ~loc : report =
  let labels = String.concat ", " labels in
  let message =
    ("There is no record definition with the fields: " ^ labels ^ "."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.cannot_resolve_record }

let ambiguous_record ~names ~loc : report =
  let names = String.concat ", " names in
  let message =
    ("Mutiple possible record types detected: " ^ names
     ^ ", please add more annotation."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.ambiguous_record }

let readonly_type ~name ~loc : report =
  let message =
    ("Cannot create values of the read-only type: " ^ name ^ "."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.readonly_type }

let cannot_create_struct_with_priv_field ~name ~loc : report =
  let message =
    ("Cannot create values of struct type " ^ name
     ^ " because it contains private field(s)."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.readonly_type }

let pkg_not_imported ~name ~action ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "Cannot "; action; ": package "; name; " is not imported." ]
  in
  { message; loc; error_code = Error_code.pkg_not_imported }

let type_not_object_safe ~name ~reasons ~loc : report =
  let header =
    ("Trait object for " ^ name ^ " is not allowed:" : Stdlib.String.t)
  in
  let message =
    match reasons with
    | [] -> assert false
    | reason :: [] -> header ^ " " ^ reason
    | reasons -> header ^ "\n    " ^ String.concat "\n    " reasons
  in
  { message; loc; error_code = Error_code.type_not_object_safe }

let method_not_found_in_trait ~trait ~method_name ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "There is no method "; method_name; " in trait "; type_path_name trait ]
  in
  { message; loc; error_code = Error_code.method_not_found_in_trait }

let cannot_use_method_of_abstract_trait ~trait ~method_name ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Cannot use method ";
        method_name;
        " of abstract trait ";
        type_path_name trait;
      ]
  in
  { message; loc; error_code = Error_code.method_not_found_in_trait }

let type_constr_arity_mismatch ~kind ~id ~expected ~actual ~loc : report =
  let message =
    Printf.sprintf
      "The %s %s expects %d argument(s), but is here given %d argument(s)." kind
      (Longident.to_string id) expected actual
  in
  { message; loc; error_code = Error_code.type_constr_arity_mismatch }

let unexpected_partial_type loc : report =
  let message = "Partial type is not allowed in toplevel declarations." in
  { message; loc; error_code = Error_code.unexpected_partial_type }

let invalid_stub_type loc : report =
  {
    message = "Invalid stub type.";
    loc;
    error_code = Error_code.invalid_stub_type;
  }

let duplicate_record_field ~label ~context ~loc : report =
  let message =
    match context with
    | `Pattern ->
        ("The record field " ^ label
         ^ " is matched several times in this pattern."
          : Stdlib.String.t)
    | `Creation ->
        ("The record field " ^ label ^ " is defined several times."
          : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.duplicate_record_field }

let missing_fields_in_record ~labels ~ty ~context ~loc : report =
  let message =
    let labels_str = String.concat ", " labels in
    match context with
    | `Pattern ->
        ("Record fields " ^ labels_str
         ^ " are unmatched, use `..` to ignore them."
          : Stdlib.String.t)
    | `Creation ->
        Stdlib.String.concat ""
          [ "Record fields "; labels_str; " are undefined for type "; ty ]
  in
  { message; loc; error_code = Error_code.missing_fields_in_record }

let superfluous_field ~label ~ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "The fields "; label; " is not defined in the record type "; ty; "." ]
  in
  { message; loc; error_code = Error_code.superfluous_field }

let cannot_depend_private ~entity ~loc : report =
  let message =
    ("A public definition cannot depend on private " ^ entity : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.cannot_depend_private }

let alias_with_priv_target_in_pub_sig ~alias ~priv_type ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "The alias ";
        alias;
        " cannot be used in public signature, becasu it mentions private type ";
        priv_type;
      ]
  in
  { message; loc; error_code = Error_code.cannot_depend_private }

let pkg_not_found ~pkg : Diagnostics.report =
  let message =
    ("Package " ^ pkg ^ " not found when loading packages." : Stdlib.String.t)
  in
  { message; loc = Loc.no_location; error_code = Error_code.pkg_not_found }

let pkg_wrong_format ~pkg : Diagnostics.report =
  let message =
    ("The package file of " ^ pkg ^ " is in wrong format" : Stdlib.String.t)
  in
  { message; loc = Loc.no_location; error_code = Error_code.pkg_wrong_format }

let pkg_magic_mismatch ~pkg : Diagnostics.report =
  let message =
    ("Magic number mismatch for the package file of " ^ pkg : Stdlib.String.t)
  in
  { message; loc = Loc.no_location; error_code = Error_code.pkg_magic_mismatch }

let cycle_definitions ~cycle ~locs : Diagnostics.report list =
  let rec make_string (idents : Ident.t list) head =
    match idents with
    | [] -> ""
    | i :: [] -> Ident.base_name i ^ " -> " ^ Ident.base_name head
    | i :: rest -> Ident.base_name i ^ " -> " ^ make_string rest head
  in
  let make_report loc =
    let message =
      Printf.sprintf "Definition cycle detected : %s"
        (make_string cycle (List.hd cycle))
    in
    ({ message; loc; error_code = Error_code.cycle_definitions }
      : Diagnostics.report)
  in
  Lst.map locs make_report

let redeclare ~kind ~name ~first_loc ~second_loc
    ~(extra_message : string option) : Diagnostics.report =
  let message =
    Stdlib.String.concat ""
      [
        "The ";
        kind;
        " ";
        name;
        " is declared twice: it was previously defined at ";
        Loc.to_string first_loc;
        ".";
      ]
  in
  let message =
    match extra_message with
    | None -> message
    | Some extra_message -> message ^ "\n  " ^ extra_message
  in
  { message; loc = second_loc; error_code = Error_code.redeclare }

let value_redeclare ~name ~first_loc ~second_loc ~extra_message =
  redeclare ~kind:"toplevel identifier" ~name ~first_loc ~second_loc
    ~extra_message

let type_redeclare ~name ~first_loc ~second_loc ~extra_message =
  redeclare ~kind:"type" ~name ~first_loc ~second_loc ~extra_message

let trait_redeclare ~name ~first_loc ~second_loc =
  redeclare ~kind:"trait" ~name ~first_loc ~second_loc ~extra_message:None

let direct_use_redeclare ~name ~first_loc ~second_loc =
  redeclare ~kind:"import item" ~name ~first_loc ~second_loc ~extra_message:None

let type_trait_duplicate ~name ~first_kind ~first_loc ~second_kind ~second_loc
    ~extra_message : Diagnostics.report =
  let message =
    Stdlib.String.concat ""
      [
        "The ";
        second_kind;
        " ";
        name;
        " duplicates with ";
        first_kind;
        " ";
        name;
        " previously defined at ";
        Loc.to_string first_loc;
        ".";
      ]
  in
  let message =
    match extra_message with
    | None -> message
    | Some extra_message -> message ^ "\n  " ^ extra_message
  in
  { message; loc = second_loc; error_code = Error_code.type_trait_duplicate }

let invalid_self_type loc : report =
  let message = "Invalid type for \"self\": must be a type constructor" in
  { message; loc; error_code = Error_code.invalid_self_type }

let cannot_determine_self_type loc : report =
  let message =
    ("Cannot determine self type of extension method. [Self] does not occur in \
      the signature of the method"
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.cannot_determine_self_type }

let field_duplicate ~name ~loc : report =
  let message = ("field " ^ name ^ " is already declared." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.field_duplicate }

let method_duplicate ~method_name ~type_name ~first_loc ~second_loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "The method ";
        method_name;
        " for type ";
        type_path_name type_name;
        " has been defined at ";
        Loc.to_string first_loc;
        ".";
      ]
  in
  { message; loc = second_loc; error_code = Error_code.method_duplicate }

let constructor_duplicate ~name ~loc : report =
  let message =
    ("The constructor " ^ name ^ " is duplicate." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.constructor_duplicate }

let method_func_duplicate ~name ~first_loc ~second_loc ~extra_message : report =
  let message =
    Stdlib.String.concat ""
      [
        "The method ";
        name;
        " conflicts with the toplevel function defined at ";
        Loc.to_string first_loc;
        ".";
      ]
  in
  let message =
    match extra_message with
    | None -> message
    | Some extra_message -> message ^ "\n  " ^ extra_message
  in
  { message; loc = second_loc; error_code = Error_code.method_func_duplicate }

let method_on_foreign_type ~method_name ~type_name ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Cannot define method ";
        method_name;
        " for foreign type ";
        type_path_name type_name;
      ]
  in
  { message; loc; error_code = Error_code.method_on_foreign_type }

let ext_method_type_mismatch ~trait ~method_name ~expected ~actual ~loc : report
    =
  let trait =
    Type_path.short_name ~cur_pkg_name:(Some !Basic_config.current_package)
      trait
  in
  let message =
    Stdlib.String.concat ""
      [
        "Method ";
        method_name;
        " of trait ";
        trait;
        " is expected to have type ";
        expected;
        ", it cannot be implemened with type ";
        actual;
      ]
  in
  { message; loc; error_code = Error_code.ext_method_type_mismatch }

let ext_method_foreign_trait_foreign_type ~trait ~type_name ~method_name ~loc :
    report =
  let trait = type_path_name trait in
  let type_name = type_path_name type_name in
  let message =
    Stdlib.String.concat ""
      [
        "Cannot define method ";
        method_name;
        " of foreign trait ";
        trait;
        " for foreign type ";
        type_name;
      ]
  in
  {
    message;
    loc;
    error_code = Error_code.ext_method_foreign_trait_foreign_type;
  }

let priv_ext_shadows_pub_method ~method_name ~trait ~type_name ~prev_loc ~loc :
    Diagnostics.report =
  let message =
    Stdlib.String.concat ""
      [
        "This `impl` shadows method ";
        method_name;
        " of ";
        type_path_name type_name;
        " previously defined at ";
        Loc.to_string prev_loc;
        ". This will result in different implementations for ";
        type_path_name trait;
        " inside and outside current package.";
      ]
  in
  { message; loc; error_code = Error_code.priv_ext_shadows_pub_method }

let trait_not_implemented ~trait ~type_name ~failure_reasons ~loc :
    Diagnostics.report =
  let reasons = String.concat "\n  " failure_reasons in
  let message =
    Stdlib.String.concat ""
      [
        "Type ";
        type_path_name type_name;
        " does not implement trait ";
        type_path_name trait;
        ", although an `impl` is defined. hint:\n  ";
        reasons;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.trait_not_implemented }

let bad_operator_arity ~method_name ~expected ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "overloaded operator \"";
        method_name;
        "\" should accept ";
        Int.to_string expected;
        " arguments, but it accepts ";
        Int.to_string actual;
        " arguments";
      ]
  in
  { message; loc; error_code = Error_code.bad_operator_arity }

let bad_operator_type ~method_name ~first ~second ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "overloaded operator \"";
        method_name;
        "\" has inconsistent parameter type: first parameter has type ";
        first;
        " while second parameter has type ";
        second;
      ]
  in
  { message; loc; error_code = Error_code.bad_operator_type }

let missing_main ~loc : Diagnostics.report =
  {
    message = "Missing main function in the main package.";
    loc;
    error_code = Error_code.missing_main;
  }

let multiple_main ~first_loc ~second_loc : Diagnostics.report =
  let message =
    ("Main function is already defined at " ^ Loc.to_string first_loc ^ "."
      : Stdlib.String.t)
  in
  { message; loc = second_loc; error_code = Error_code.multiple_main }

let unexpected_main loc : Diagnostics.report =
  {
    loc;
    message = "Unexpected main function in the non-main package.";
    error_code = Error_code.unexpected_main;
  }

let unknown_intrinsic ~name ~loc : Diagnostics.report =
  let message = ("Unknown intrinsic " ^ name ^ "." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.unknown_intrinsic }

let multiple_intrinsic loc : Diagnostics.report =
  let message = "Multiple intrinsic is not unsupported." in
  { message; loc; error_code = Error_code.multiple_intrinsic }

let default_method_duplicate ~trait ~method_name ~first_loc ~second_loc : report
    =
  let message =
    Stdlib.String.concat ""
      [
        "Method ";
        method_name;
        " of trait ";
        type_path_name trait;
        " already has a default implementation at ";
        Loc.to_string first_loc;
      ]
  in
  {
    message;
    loc = second_loc;
    error_code = Error_code.default_method_duplicate;
  }

let default_method_on_foreign ~trait ~loc : report =
  let message =
    ("Cannot provide default implementation for foreign trait "
     ^ type_path_name trait
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.default_method_on_foreign }

let let_missing_annot ~name ~loc ~reason : report =
  let message =
    match reason with
    | `Cannot_infer ->
        ("Cannot infer the type of variable " ^ name
         ^ ", please add more type annotation."
          : Stdlib.String.t)
    | `Pub_not_literal ->
        ("Public definition " ^ name ^ " must be annotated with its type."
          : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.let_missing_annot }

let missing_param_annot ~name ~loc : report =
  let message =
    ("Missing type annotation for the parameter " ^ name ^ "."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.missing_param_annot }

let missing_return_annot loc : report =
  let message =
    ("Missing type annotation for the return value." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.missing_return_annot }

let derive_unsupported_trait ~tycon ~trait ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Don't know how to derive trait ";
        Longident.to_string trait;
        " for type ";
        type_path_name tycon;
      ]
  in
  { message; loc; error_code = Error_code.derive_unsupported_trait }

let cannot_derive ~tycon ~trait ~reason ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Cannot derive trait ";
        Longident.to_string trait;
        " for type ";
        tycon;
        ": ";
        reason;
      ]
  in
  { message; loc; error_code = Error_code.cannot_derive }

let derive_method_exists ~trait ~type_name ~method_name ~prev_loc ~loc : report
    =
  let message =
    Stdlib.String.concat ""
      [
        "Cannot derive trait ";
        type_path_name trait;
        " for ";
        type_path_name type_name;
        ": method ";
        method_name;
        " is already defined at ";
        Loc.to_string prev_loc;
      ]
  in
  { message; loc; error_code = Error_code.derive_method_exists }

let arity_mismatch ~label ~expected ~actual ~has_label ~loc : report =
  let arg_desc = if has_label then "positional arguments" else "arguments" in
  let message =
    Stdlib.String.concat ""
      [
        label;
        " requires ";
        Int.to_string expected;
        " ";
        arg_desc;
        ", but is given ";
        Int.to_string actual;
        " ";
        arg_desc;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.arity_mismatch }

let constr_arity_mismatch ~name ~expected ~actual ~has_label ~loc =
  arity_mismatch
    ~label:("The constructor " ^ name : Stdlib.String.t)
    ~expected ~actual ~has_label ~loc

let fn_arity_mismatch ~func_ty ~expected ~actual ~has_label ~loc =
  arity_mismatch
    ~label:("This function has type " ^ func_ty ^ ", which" : Stdlib.String.t)
    ~expected ~actual ~has_label ~loc

let non_linear_pattern ~name ~loc : report =
  let message =
    ("The identifier " ^ name ^ " is bound more than once in the same pattern."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.non_linear_pattern }

let inconsistent_or_pattern ~name ~loc : report =
  let message =
    ("Variable " ^ name ^ " is not bound in all patterns." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.inconsistent_or_pattern }

let no_op_as_view ~ty ~loc : report =
  let message =
    ("The type " ^ ty ^ " does not implement `op_as_view` method."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.no_op_as_view }

let duplicated_fn_label ~label ~second_loc : report =
  let message =
    ("The label " ^ label ^ "~ is supplied twice." : Stdlib.String.t)
  in
  { message; loc = second_loc; error_code = Error_code.duplicated_fn_label }

let superfluous_arg_label ~label ~(kind : string) ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "This "; kind; " has no parameter with label "; label; "~." ]
  in
  { message; loc; error_code = Error_code.superfluous_fn_label }

let missing_fn_label ~labels ~loc : report =
  let labels_str = String.concat ", " (Lst.map labels (fun l -> l ^ "~")) in
  let message =
    ("The labels " ^ labels_str
     ^ " are required by this function, but not supplied."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.missing_fn_label }

let not_mutable ~id ~loc : report =
  let message =
    ("The variable " ^ Longident.to_string id ^ " is not mutable."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.not_mutable }

let immutable_field ~label ~loc : report =
  let message =
    ("The record field " ^ label ^ " is immutable." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.immutable_field }

let no_tuple_index ~required ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        Int.to_string actual;
        "-tuple has no field `";
        Int.to_string required;
        "`";
      ]
  in
  { message; loc; error_code = Error_code.no_tuple_index }

let tuple_not_mutable loc : report =
  let message = "tuples are not mutable" in
  { message; loc; error_code = Error_code.tuple_not_mutable }

let no_such_field ~ty ~field ~may_be_method ~loc : report =
  let message =
    if may_be_method then
      Stdlib.String.concat ""
        [
          "The type ";
          ty;
          " has no field ";
          field;
          ".\n  (hint: to pass method as function, write `fn (...) { x.";
          field;
          "(...) }` instead)";
        ]
    else
      Stdlib.String.concat "" [ "The type "; ty; " has no field "; field; "." ]
  in
  { message; loc; error_code = Error_code.no_such_field }

let record_type_missing loc : report =
  let message = "Missing annotation for this empty record." in
  { message; loc; error_code = Error_code.record_type_missing }

let not_a_record_type ~name ~loc : report =
  let message =
    ("The type " ^ name ^ " is not a record type" : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.not_a_record_type }

let mutate_readonly_field ~label ~loc : report =
  let message =
    ("Cannot modify a read-only field: " ^ label : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.mutate_readonly_field }

let overflow ~value ~loc : report =
  let message =
    ("Integer literal " ^ value ^ " is out of range." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.overflow }

let invalid_newtype_index ~ty ~loc : report =
  let message =
    ("Field of newtype " ^ ty ^ " can only be accessed via index 0."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_newtype_index }

let not_a_trait ~name ~loc : report =
  let message = ("The type " ^ name ^ " is not a trait" : Stdlib.String.t) in
  { message; loc; error_code = Error_code.not_a_trait }

let unsupported_pipe_expr loc : report =
  let message = "Unsupported expression after the pipe operator." in
  { message; loc; error_code = Error_code.unsupported_pipe_expr }

let outside_loop ~msg ~loc : report =
  {
    loc;
    message = ("'" ^ msg ^ "' outside of a loop" : Stdlib.String.t);
    error_code = Error_code.outside_loop;
  }

let loop_pat_arity_mismatch ~expected ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "This loop has ";
        Int.to_string expected;
        " arguments, but ";
        Int.to_string actual;
        " patterns are supplied";
      ]
  in
  { message; loc; error_code = Error_code.loop_pat_arity_mismatch }

let continue_arity_mismatch ~expected ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Current loop expects ";
        Int.to_string expected;
        " arguments, but `continue` is supplied with ";
        Int.to_string actual;
        " arguments";
      ]
  in
  { message; loc; error_code = Error_code.continue_arity_mismatch }

let break_type_mismatch ~expected ~actual ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Current loop has result type ";
        expected;
        ", but `break` is supplied with ";
        actual;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.break_type_mismatch }

let unknown_binder_in_for_steps ~name ~loc : report =
  let message =
    ("Unknown binder " ^ name
     ^ " in the for-loop steps. Binders in the steps must be declared in the \
        initialization block of the for-loop."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unknown_binder_in_for_steps }

let duplicate_for_binder ~name ~loc : report =
  let message =
    (name ^ " is declared multiple times in this for-loop" : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.duplicate_for_binder }

let need_else_branch ~loop_kind ~ty ~loc : report =
  let loop_kind_string =
    match loop_kind with `For -> "for" | `While -> "while"
  in
  let message =
    Stdlib.String.concat ""
      [
        "The ";
        loop_kind_string;
        " loop is expected to yield a value of type ";
        ty;
        ", please add an `else` branch.";
      ]
  in
  { message; loc; error_code = Error_code.need_else_branch }

let invalid_return loc : report =
  {
    loc;
    message = ("Return must be inside a function." : Stdlib.String.t);
    error_code = Error_code.invalid_return;
  }

let invalid_break ~loop_kind ~loc : report =
  let loop_kind_string =
    match loop_kind with `For -> "for" | `While -> "while"
  in
  let message =
    ("The " ^ loop_kind_string
     ^ " loop is not expected to yield a value, please remove the argument of \
        the `break` or add an `else` branch."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_break }

let ambiguous_break loc : report =
  let message = ("The usage break statement is invalid." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.ambiguous_break }

let ambiguous_continue loc : report =
  let message =
    ("The usage continue statement is invalid." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.ambiguous_continue }

let constr_no_such_field ~ty ~constr ~field ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Constructor ";
        constr;
        " of type ";
        type_path_name ty;
        " has no field ";
        field;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.constr_no_such_field }

let no_local_labelled_function loc : report =
  let message = "Only toplevel functions can have labelled arguments." in
  { message; loc; error_code = Error_code.no_local_labelled_function }

let unsupported_autofill ~ty ~loc : report =
  let message =
    ("Cannot auto-fill parameter of type " ^ ty : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unsupported_autofill }

let found_hole loc : report =
  let message = "Found hole _ " in
  { message; loc; error_code = Error_code.found_hole }

let no_first_class_labelled_function loc : report =
  let message =
    "Function with labelled arguments can only be applied directly."
  in
  { message; loc; error_code = Error_code.no_first_class_labelled_function }

let cannot_use_map_pattern_no_method ~ty ~loc : report =
  let message =
    ("Please implement method `op_get` for type " ^ ty
     ^ " to match it with map pattern."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.cannot_use_map_pattern }

let cannot_use_map_pattern_method_type_mismatch ~ty ~actual_ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Cannot match type ";
        ty;
        " with map pattern: its method `op_get` has incorrect type:\n\
        \    wanted: (Self, K) -> Option[V]\n\
        \    has   : ";
        actual_ty;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.cannot_use_map_pattern }

let nontoplevel_func_cannot_have_labelled_arg ~loc : report =
  let message =
    ("This function is not a toplevel function, so it cannot have labelled \
      arguments"
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unknown_func_labelled_arg }

let error_type_mismatch ~expected_ty ~actual_ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "The error type is mismatched:\n    wanted: ";
        expected_ty;
        "\n    has   : ";
        actual_ty;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.cannot_use_map_pattern }

let unhandled_error ~err_ty ~loc : report =
  let message =
    ("The application might raise errors of type " ^ err_ty
     ^ ", but it's not handled. Try adding a infix operator `!` or `?` to the \
        application, so that it looks like `...!(...)` or `...?(...)`."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.unhandled_error }

let invalid_apply_attr ~(kind : [ `UnknownType | `NoErrorType | `Constructor ])
    ~loc : report =
  let message =
    match kind with
    | `Constructor ->
        ("The attribute `!` or `?` cannot be used on constructors."
          : Stdlib.String.t)
    | `UnknownType ->
        ("The type of function is unknown so the attribute `!` or `?` cannot \
          be used."
          : Stdlib.String.t)
    | `NoErrorType ->
        ("The attribute `!` or `?` cannot be used on application that does not \
          raise errors"
          : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_apply_attr }

let invalid_raise ~(kind : [ `Catchall | `Rethrow | `Raise ]) ~loc : report =
  let reason =
    match kind with
    | `Catchall -> "catch! will rethrow unhandled error, and"
    | `Rethrow ->
        "! operator will rethrow the error raised in the function application, \
         and"
    | `Raise -> "raise"
  in
  {
    loc;
    message =
      (reason
       ^ " can only be used inside a function with error types in its \
          signature. Please fix the return type of this function. \n\
          For local functions, you could use `fn func_name!()` to declare the \
          function with an error type, and let the compiler infer the error \
          type.\n\
          For anonymous functions, use `fn!` instead of `fn` to declare the \
          function with an error type."
        : Stdlib.String.t);
    error_code = Error_code.invalid_raise;
  }

let ambiguous_constructor ~name ~first_ty ~second_ty ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "The constructor ";
        name;
        " is ambiguous: it may come from type ";
        first_ty;
        " or ";
        second_ty;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.ambiguous_constructor }

let double_exclamation_with_cascade loc : report =
  let message = ("`?` operator cannot be used with `..`." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.double_exclamation_with_cascade }

let not_error_subtype ty loc : report =
  let message = ("Type " ^ ty ^ " is not an error type." : Stdlib.String.t) in
  { message; loc; error_code = Error_code.not_error_subtype }

let invalid_type_alias_target loc : report =
  let message =
    ("Target of type alias must not be a type parameter." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_type_alias_target }

let cycle_in_type_alias ~cycle ~loc : report =
  let cycle = String.concat " -> " cycle in
  let message =
    ("Found cycle " ^ cycle ^ " in type alias." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.cycle_in_type_alias }

let type_alias_cannot_derive loc : report =
  let message = "`derive` is not allowed for type alias" in
  { message; loc; error_code = Error_code.type_alias_cannot_derive }

let type_alias_not_a_constructor ~alias_name ~loc : report =
  let message =
    ("The type alias " ^ alias_name
     ^ " is a function type, not a type constructor."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.type_alias_not_a_constructor }

let invalid_test_parameter loc : Diagnostics.report =
  let message =
    "Invalid test parameter. Only one parameter with type `@test.T` is allowed."
  in
  { message; loc; error_code = Error_code.invalid_test_parameter }

let foreach_loop_variable_count_mismatch ~actual ~expected ~loc : report =
  let expected =
    match expected with
    | None -> "at most 2"
    | Some expected -> Int.to_string expected
  in
  let message =
    Stdlib.String.concat ""
      [
        "This `for .. in` loop has ";
        Int.to_string actual;
        " loop variables, but ";
        expected;
        " is expected.";
      ]
  in
  { message; loc; error_code = Error_code.foreach_loop_variable_count_mismatch }

let anonymous_missing_error_annotation loc : report =
  let message =
    ("The return type of this anonymous function is expected include an error \
      type. Please add the error type to the return type annotation or use \
      `fn!` instead."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.anonymous_missing_error_annotation }

let static_assert_failure ~type_ ~trait ~required_by ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "Type ";
        type_;
        " does not implement trait ";
        trait;
        ".\n  note: this constraint is required by ";
        required_by;
      ]
  in
  { message; loc; error_code = Error_code.cannot_resolve_trait }

let inconsistent_impl ~trait ~type_name ~reason ~loc1 ~loc2 : Diagnostics.report
    =
  let reason =
    match reason with
    | `Self_type_mismatch (ty1, ty2) ->
        Stdlib.String.concat ""
          [
            "implementations have different self type:\n\
            \  self type of first impl : ";
            ty1;
            "\n  self type of second impl: ";
            ty2;
          ]
    | `Type_parameter_bound ->
        ("type parameters of implementations have different constraints"
          : Stdlib.String.t)
  in
  let message =
    Stdlib.String.concat ""
      [
        "Inconsistent `impl` of trait ";
        trait;
        " for ";
        type_name;
        " at ";
        Loc.to_string loc1;
        " and ";
        Loc.to_string loc2;
        ": ";
        reason;
      ]
  in
  { message; loc = loc2; error_code = Error_code.inconsistent_impl }

let no_default_for_question_optional ~label ~loc : report =
  let message =
    ("The parameter " ^ label ^ "? already has default value `None`."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.no_default_for_question_optional }

let invalid_extra_delimiter ~delimiter ~loc : Diagnostics.report =
  let message =
    ("Expecting a newline or `;` here, but encountered " ^ delimiter ^ "."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_extra_delimiter }

let not_a_newtype ~actual_ty ~loc : report =
  let message =
    ("This expression has type " ^ actual_ty ^ ", which is not a newtype."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.not_a_newtype }

let range_operator_only_in_for loc : report =
  let message =
    ("Range operators are currently only supported in `for .. in` loops."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.range_operator_only_in_for }

let range_operator_unsupported_type ~actual_ty ~loc : report =
  let message =
    ("Range operators only support builtin integer types, they cannot be used \
      on type " ^ actual_ty ^ "."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.range_operator_unsupported_type }

let non_unit_cannot_be_ignored ~ty ~loc : report =
  let message =
    ("This expression has type " ^ ty
     ^ ", its value cannot be implicitly ignored (hint: use `ignore(...)` or \
        `let _ = ...` to explicitly ignore it)."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.non_unit_cannot_be_ignored }

let c_stub_invalid_function_name loc : Diagnostics.report =
  let message = "Invalid C function name in extern \"C\" declaration" in
  { message; loc; error_code = Error_code.c_stub_invalid_function_name }

let invalid_question_arg_application ~label ~loc : report =
  let message =
    Stdlib.String.concat ""
      [
        "This form of application is invalid for argument ";
        label;
        "~, because it is not declared with ";
        label;
        "? : _.";
      ]
  in
  { message; loc; error_code = Error_code.invalid_question_arg_application }

let direct_use_not_found ~pkg ~id ~loc : Diagnostics.report =
  let message = Printf.sprintf "Value %s not found in package %S." id pkg in
  { message; loc; error_code = Error_code.unbound_value }

let direct_use_ambiguous_method ~name ~type_locs ~pkg ~loc : Diagnostics.report
    =
  let info =
    type_locs
    |> List.map (fun (s, loc) : Stdlib.String.t ->
           Loc.to_string loc ^ " " ^ type_path_name s)
    |> String.concat "\n"
  in
  let message =
    Stdlib.String.concat ""
      [
        "The method ";
        name;
        " has been defined for the following types in packae ";
        pkg;
        ":\n";
        info;
        ".";
      ]
  in
  { message; loc; error_code = Error_code.ambiguous_method }

let constant_not_constant loc : report =
  let message = "This 'const' declaration is not constant." in
  { message; loc; error_code = Error_code.constant_not_constant }

let invalid_constant_type ~ty ~loc : report =
  let message =
    (ty
     ^ " is not a valid constant type, only immutable primitive types are \
        allowed."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.invalid_constant_type }

let constant_constr_duplicate ~name ~const_loc ~constr_loc : Diagnostics.report
    =
  let message =
    Stdlib.String.concat ""
      [
        "The constant ";
        name;
        " duplicates with constructor declared at ";
        Loc.to_string constr_loc;
        ".";
      ]
  in
  { message; loc = const_loc; error_code = Error_code.redeclare }

let constant_pat_with_args ~name ~loc : report =
  let message =
    ("'" ^ name
     ^ "' is a constant, not a constructor, it cannot be applied to arguments."
      : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.constant_pat_with_args }

let cannot_implement_sealed_trait ~trait ~trait_vis ~loc : report =
  let message =
    Stdlib.String.concat ""
      [ "Cannot implement trait '"; trait; "' because it is "; trait_vis; "." ]
  in
  { message; loc; error_code = Error_code.cannot_implement_abstract_trait }

let range_pattern_unsupported_type ~ty ~loc : report =
  let message =
    ("Type " ^ ty ^ " is not supported by range pattern." : Stdlib.String.t)
  in
  { message; loc; error_code = Error_code.range_pattern_unsupported_type }

let range_pattern_invalid_range ~inclusive ~loc : report =
  let message =
    if inclusive then "Range pattern `a..=b` must satisfy `a <= b`."
    else "Range pattern `a..<b` must satisfy `a < b`."
  in
  { message; loc; error_code = Error_code.range_pattern_invalid_range }
