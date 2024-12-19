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


module Lst = Basic_lst
module Loc = Rloc

let type_path_json = Basic_type_path.Builtin.type_path_json

type json_constructor = string

let true_ : json_constructor = "True"
let false_ : json_constructor = "False"
let number : json_constructor = "Number"
let string : json_constructor = "String"
let array : json_constructor = "Array"
let object_ : json_constructor = "Object"

let get_json_constr_tag ~global_env ~diagnostics constr_name ~action ~loc =
  match Global_env.find_type_by_path global_env type_path_json with
  | Some { ty_desc = Variant_type constrs; _ } -> (
      match
        Lst.find_first constrs (fun constr -> constr.constr_name = constr_name)
      with
      | None -> Typeutil.unknown_tag
      | Some constr -> constr.cs_tag)
  | Some _ -> Typeutil.unknown_tag
  | None ->
      Typeutil.add_local_typing_error diagnostics
        (Errors.pkg_not_imported ~name:"moonbitlang/core/json"
           ~action:
             (action ^ " of type " ^ Type_path_util.name type_path_json
               : Stdlib.String.t)
           ~loc);
      Typeutil.unknown_tag

let make_json_pat ~global_env ~diagnostics constr_name args ~loc : Typedtree.pat
    =
  let args =
    Lst.mapi args (fun pos pat : Typedtree.constr_pat_arg ->
        Constr_pat_arg { pat; kind = Positional; pos })
  in
  Tpat_constr
    {
      constr =
        {
          constr_name = { name = constr_name; loc_ = Loc.no_location };
          extra_info = No_extra_info;
          loc_ = Loc.no_location;
        };
      args;
      tag =
        get_json_constr_tag ~global_env ~diagnostics constr_name ~loc
          ~action:"destruct value";
      ty = Stype.json;
      used_error_subtyping = false;
      loc_ = Loc.no_location;
    }

let make_json_const_expr ~global_env ~diagnostics constr_name ~loc :
    Typedtree.expr =
  Texpr_constr
    {
      constr =
        {
          constr_name = { name = constr_name; loc_ = Loc.no_location };
          extra_info = No_extra_info;
          loc_ = Loc.no_location;
        };
      tag =
        get_json_constr_tag ~global_env ~diagnostics constr_name ~loc
          ~action:"create value";
      ty = Stype.json;
      arity_ = Fn_arity.simple 0;
      loc_ = loc;
    }

let make_json_expr ~global_env ~diagnostics constr_name arg ~loc :
    Typedtree.expr =
  Texpr_apply
    {
      func =
        Texpr_constr
          {
            constr =
              {
                constr_name = { name = constr_name; loc_ = Loc.no_location };
                extra_info = No_extra_info;
                loc_ = Loc.no_location;
              };
            tag =
              get_json_constr_tag ~global_env ~diagnostics constr_name ~loc
                ~action:"create value";
            ty =
              Builtin.type_arrow
                [ Typedtree_util.type_of_typed_expr arg ]
                Stype.json ~err_ty:None;
            arity_ = Fn_arity.simple 1;
            loc_ = Loc.no_location;
          };
      args = [ { arg_value = arg; arg_kind = Positional } ];
      ty = Stype.json;
      kind_ = Normal;
      loc_ = loc;
    }
