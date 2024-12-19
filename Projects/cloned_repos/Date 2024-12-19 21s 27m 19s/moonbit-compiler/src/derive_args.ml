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


module StringMap = Basic_map_string
module Syntax = Parsing_syntax

type meta_labelled_arg_spec = {
  name : string;
  alias : string list;
  allow_no_value : bool;
  allow_value : bool;
  on_repeat : [ `Error | `Append | `Replace ];
  action : (Syntax.argument list -> unit) option;
}

type meta_arg_parser = {
  reverse_args : meta_labelled_arg_spec StringMap.t;
  allow_positional : bool;
  allow_unknown_labelled : bool;
}

let decl_meta_arg name ?(alias : string list = []) ?(allow_no_value = false)
    ?(allow_value = false) ?(action = None) ?(on_repeat = `Error) () =
  { name; alias; allow_no_value; allow_value; action; on_repeat }

let mk_parser (args : meta_labelled_arg_spec list) ?(allow_positional = false)
    ?(allow_unknown_labelled = false) () =
  let reverse_args =
    List.fold_left
      (fun acc (arg : meta_labelled_arg_spec) ->
        List.fold_left
          (fun acc alias -> StringMap.add acc alias arg)
          acc (arg.name :: arg.alias))
      StringMap.empty args
  in
  { reverse_args; allow_positional; allow_unknown_labelled }

type parsed_meta_args = {
  positional : Syntax.argument list;
  labelled : Syntax.argument list StringMap.t;
}

let mk_diag_emitter ~(host_type : string) ~(trait_name : Basic_longident.t)
    (diag : Local_diagnostics.t) (msg : string) (loc : Rloc.t) =
  let report =
    Errors.cannot_derive ~tycon:host_type ~trait:trait_name ~reason:msg ~loc
  in
  Local_diagnostics.add_error diag report

let parse (parser : meta_arg_parser) ~(host_type : string)
    (directive : Syntax.deriving_directive) (diag : Local_diagnostics.t) =
  let met_error = ref false in
  let emit_diag msg loc =
    met_error := true;
    mk_diag_emitter ~host_type ~trait_name:directive.type_name_.name diag msg
      loc
  in
  let args = directive.args in
  let positional = ref [] in
  let labelled = ref StringMap.empty in
  let do_add_labelled is_replace id args =
    let old = StringMap.find_opt !labelled id in
    let new_args =
      match old with
      | None -> args
      | Some old_args -> if is_replace then args else old_args @ args
    in
    labelled := StringMap.add !labelled id new_args
  in
  let try_add_labelled id (args : Syntax.argument list) loc =
    let meta_arg_desc = StringMap.find_opt parser.reverse_args id in
    match meta_arg_desc with
    | None when not parser.allow_unknown_labelled ->
        emit_diag ("Unknown labelled argument: " ^ id : Stdlib.String.t) loc
    | None -> do_add_labelled false id args
    | Some meta_arg_desc -> (
        let { allow_no_value; allow_value; on_repeat; action; _ } =
          meta_arg_desc
        in
        match args with
        | [] when not allow_no_value ->
            emit_diag ("Expected value for " ^ id : Stdlib.String.t) loc
        | _ :: _ when not allow_value ->
            emit_diag ("Unexpected value for " ^ id : Stdlib.String.t) loc
        | _ -> (
            let id = meta_arg_desc.name in
            (match on_repeat with
            | `Error when StringMap.mem !labelled id ->
                emit_diag ("Duplicate argument: " ^ id : Stdlib.String.t) loc
            | `Error -> do_add_labelled false id args
            | `Append -> do_add_labelled false id args
            | `Replace -> do_add_labelled true id args);
            match action with None -> () | Some action -> action args))
  in
  let mk_positional arg : Syntax.argument =
    { arg_kind = Syntax.Positional; arg_value = arg }
  in
  let parse_single (arg : Syntax.argument) =
    match arg.arg_kind with
    | Syntax.Positional -> (
        match arg.arg_value with
        | Syntax.Pexpr_ident
            { id = { var_name = Basic_longident.Lident id; _ }; loc_ } ->
            try_add_labelled id [] loc_
        | Syntax.Pexpr_ident { loc_; _ } ->
            emit_diag "Non-local identifier not supported" loc_
        | Syntax.Pexpr_apply
            {
              func =
                Syntax.Pexpr_ident
                  { id = { var_name = Basic_longident.Lident id; _ }; _ };
              args;
              loc_;
              _;
            } ->
            try_add_labelled id args loc_
        | Syntax.Pexpr_apply { loc_; _ } ->
            emit_diag "Non-local identifier not supported" loc_
        | expr ->
            if parser.allow_positional then positional := arg :: !positional
            else
              emit_diag "Positional arguments are not allowed"
                (Syntax.loc_of_expression expr))
    | Syntax.Labelled lbl | Syntax.Labelled_option { label = lbl; _ } ->
        let name = lbl.label_name in
        let arg = mk_positional arg.arg_value in
        let loc = lbl.loc_ in
        try_add_labelled name [ arg ] loc
    | Syntax.Labelled_pun _ | Syntax.Labelled_option_pun _ ->
        emit_diag "Labelled pun arguments are not allowed"
          (Syntax.loc_of_expression arg.arg_value)
  in
  List.iter parse_single args;
  if !met_error then Error ()
  else Ok { positional = !positional; labelled = !labelled }

let deny_all_args ~host_type ~trait_name diag
    (directive : Syntax.deriving_directive) =
  if directive.args <> [] then (
    mk_diag_emitter ~host_type ~trait_name diag
      ("{trait_name} does not accept any arguments." : Stdlib.String.t)
      directive.loc_;
    true)
  else false
