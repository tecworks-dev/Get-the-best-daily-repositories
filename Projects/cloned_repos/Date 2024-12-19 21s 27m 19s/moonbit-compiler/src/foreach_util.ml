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


module Ident = Basic_core_ident

let iter_result_end : Core.expr = Core.const (C_int { v = 0l; repr = None })

let iter_result_continue : Core.expr =
  Core.const (C_int { v = 1l; repr = None })

let foreach_result_params : Tvar_env.t =
  Tvar_env.of_list_mapi [ "B"; "R"; "E" ] (fun index name ->
      Tvar_env.tparam_info ~name
        ~typ:(Tparam { index; name_ = name })
        ~constraints:[] ~loc:Rloc.no_location)

let [ foreach_param_break; foreach_param_ret; foreach_param_err ] =
  Tvar_env.get_types foreach_result_params
[@@warning "-partial-match"]

let foreach_result_tpath = Basic_type_path.Builtin.type_path_foreach_result

let type_foreach_result break return error : Stype.t =
  T_constr
    {
      type_constructor = foreach_result_tpath;
      tys = [ break; return; error ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let foreach_result_type_poly : Stype.t =
  T_constr
    {
      type_constructor = foreach_result_tpath;
      tys = Tvar_env.get_types foreach_result_params;
      generic_ = true;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let continue : Typedecl_info.constructor =
  {
    constr_name = "Continue";
    cs_res = foreach_result_type_poly;
    cs_args = [];
    cs_tag =
      Constr_tag_regular
        { total = 4; index = 0; name_ = "Continue"; is_constant_ = true };
    cs_vis = Read_write;
    cs_ty_params_ = foreach_result_params;
    cs_arity_ = Fn_arity.simple 0;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let break : Typedecl_info.constructor =
  {
    constr_name = "Break";
    cs_res = foreach_result_type_poly;
    cs_args = [ foreach_param_break ];
    cs_tag =
      Constr_tag_regular
        { total = 4; index = 1; name_ = "Break"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = foreach_result_params;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let return : Typedecl_info.constructor =
  {
    constr_name = "Return";
    cs_res = foreach_result_type_poly;
    cs_args = [ foreach_param_ret ];
    cs_tag =
      Constr_tag_regular
        { total = 4; index = 2; name_ = "Return"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = foreach_result_params;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let error : Typedecl_info.constructor =
  {
    constr_name = "Error";
    cs_res = foreach_result_type_poly;
    cs_args = [ foreach_param_err ];
    cs_tag =
      Constr_tag_regular
        { total = 4; index = 3; name_ = "Error"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = foreach_result_params;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let foreach_result : Typedecl_info.t =
  {
    ty_constr = foreach_result_tpath;
    ty_arity = 3;
    ty_desc = Variant_type [ continue; break; return; error ];
    ty_vis = Vis_fully_pub;
    ty_params_ = foreach_result_params;
    ty_loc_ = Loc.no_location;
    ty_doc_ = Docstring.empty;
    ty_is_only_tag_enum_ = false;
    ty_is_suberror_ = false;
  }

let get_first_enum_field (obj : Ident.t) (constr : Typedecl_info.constructor)
    ~ty ~constr_ty =
  Core.prim ~ty
    (Penum_field { index = 0; tag = constr.cs_tag })
    [ Core.var obj ~ty:(Type.make_constr_type constr_ty ~tag:constr.cs_tag) ]
