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


module B = Stype.Type_path.Builtin
module Type_path = Basic_type_path

let f desc t : Stype.t =
  T_constr
    {
      type_constructor = desc;
      tys = [ t ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let type_array = f B.type_path_array
let type_fixedarray = f B.type_path_fixedarray
let type_maybe_uninit = f B.type_path_maybe_uninit

let type_arrow t1 t2 ~err_ty : Stype.t =
  Tarrow { params_ty = t1; ret_ty = t2; generic_ = false; err_ty }

let type_product ts : Stype.t =
  T_constr
    {
      type_constructor = Type_path.tuple (List.length ts);
      tys = ts;
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let type_ref ty : Stype.t =
  T_constr
    {
      type_constructor = B.type_path_ref;
      tys = [ ty ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let type_option ty : Stype.t =
  T_constr
    {
      type_constructor = B.type_path_option;
      tys = [ ty ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let type_map key value : Stype.t =
  T_constr
    {
      type_constructor = B.type_path_map;
      tys = [ key; value ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let type_iter ty : Stype.t = f B.type_path_iter ty

let type_iter2 key value : Stype.t =
  T_constr
    {
      type_constructor = B.type_path_iter2;
      tys = [ key; value ];
      generic_ = false;
      only_tag_enum_ = false;
      is_suberror_ = false;
    }

let tvar_env_1 = Tvar_env.tvar_env_1
let generic_var = Stype.param0

let constr_none : Typedecl_info.constructor =
  let arg = generic_var in
  {
    constr_name = "None";
    cs_res =
      T_constr
        {
          type_constructor = B.type_path_option;
          tys = [ arg ];
          generic_ = true;
          only_tag_enum_ = false;
          is_suberror_ = false;
        };
    cs_args = [];
    cs_tag =
      Constr_tag_regular
        { total = 2; index = 0; name_ = "None"; is_constant_ = true };
    cs_vis = Read_write;
    cs_ty_params_ = tvar_env_1;
    cs_arity_ = Fn_arity.simple 0;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let constr_some : Typedecl_info.constructor =
  let arg = generic_var in
  {
    constr_name = "Some";
    cs_res =
      T_constr
        {
          type_constructor = B.type_path_option;
          tys = [ arg ];
          generic_ = true;
          only_tag_enum_ = false;
          is_suberror_ = false;
        };
    cs_args = [ arg ];
    cs_tag =
      Constr_tag_regular
        { total = 2; index = 1; name_ = "Some"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = tvar_env_1;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let field_val : Typedecl_info.field =
  let arg = generic_var in
  {
    field_name = "val";
    pos = 0;
    ty_field = arg;
    ty_record =
      T_constr
        {
          type_constructor = B.type_path_ref;
          tys = [ arg ];
          generic_ = true;
          only_tag_enum_ = false;
          is_suberror_ = false;
        };
    mut = true;
    vis = Read_write;
    all_labels = [ "val" ];
    ty_params_ = tvar_env_1;
    label_loc_ = Loc.no_location;
    loc_ = Loc.no_location;
  }

let tvar_env_2 = Tvar_env.tvar_env_2
let generic_var_1 = (Tvar_env.find_by_index_exn tvar_env_2 0).typ
let generic_var_2 = (Tvar_env.find_by_index_exn tvar_env_2 1).typ

let constr_ok : Typedecl_info.constructor =
  let ty_ok = generic_var_1 in
  let ty_err = generic_var_2 in
  {
    constr_name = "Ok";
    cs_args = [ ty_ok ];
    cs_res =
      T_constr
        {
          type_constructor = B.type_path_result;
          tys = [ ty_ok; ty_err ];
          generic_ = true;
          only_tag_enum_ = false;
          is_suberror_ = false;
        };
    cs_tag =
      Constr_tag_regular
        { total = 2; index = 1; name_ = "Ok"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = tvar_env_2;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

let constr_err : Typedecl_info.constructor =
  let ty_ok = generic_var_1 in
  let ty_err = generic_var_2 in
  {
    constr_name = "Err";
    cs_args = [ ty_err ];
    cs_res =
      T_constr
        {
          type_constructor = B.type_path_result;
          tys = [ ty_ok; ty_err ];
          generic_ = true;
          only_tag_enum_ = false;
          is_suberror_ = false;
        };
    cs_tag =
      Constr_tag_regular
        { total = 2; index = 0; name_ = "Err"; is_constant_ = false };
    cs_vis = Read_write;
    cs_ty_params_ = tvar_env_2;
    cs_arity_ = Fn_arity.simple 1;
    cs_constr_loc_ = Loc.no_location;
    cs_loc_ = Loc.no_location;
  }

type local_typedecl_info = {
  ty_constr : Type_path.t;
  ty_arity : int;
  ty_desc : Typedecl_info.type_components;
}

let f_poly1 (t : local_typedecl_info) : Typedecl_info.t =
  {
    ty_constr = t.ty_constr;
    ty_arity = t.ty_arity;
    ty_desc = t.ty_desc;
    ty_vis = Vis_fully_pub;
    ty_params_ = tvar_env_1;
    ty_loc_ = Loc.no_location;
    ty_doc_ = Docstring.empty;
    ty_is_only_tag_enum_ = false;
    ty_is_suberror_ = false;
  }

let f_poly2 (t : local_typedecl_info) : Typedecl_info.t =
  {
    ty_constr = t.ty_constr;
    ty_arity = t.ty_arity;
    ty_desc = t.ty_desc;
    ty_vis = Vis_fully_pub;
    ty_params_ = tvar_env_2;
    ty_loc_ = Loc.no_location;
    ty_doc_ = Docstring.empty;
    ty_is_only_tag_enum_ = false;
    ty_is_suberror_ = false;
  }

let ty_constr_option : Typedecl_info.t =
  f_poly1
    {
      ty_constr = B.type_path_option;
      ty_arity = 1;
      ty_desc = Variant_type [ constr_none; constr_some ];
    }

let ty_constr_fixedarray : Typedecl_info.t =
  f_poly1
    {
      ty_constr = B.type_path_fixedarray;
      ty_arity = 1;
      ty_desc = Abstract_type;
    }

let ty_constr_ref : Typedecl_info.t =
  f_poly1
    {
      ty_constr = B.type_path_ref;
      ty_arity = 1;
      ty_desc =
        Record_type { fields = [ field_val ]; has_private_field_ = false };
    }

let ty_constr_result : Typedecl_info.t =
  f_poly2
    {
      ty_constr = B.type_path_result;
      ty_arity = 2;
      ty_desc = Variant_type [ constr_err; constr_ok ];
    }

let ty_constr_error : Typedecl_info.t =
  {
    ty_constr = B.type_path_error;
    ty_arity = 0;
    ty_desc = Abstract_type;
    ty_vis = Vis_fully_pub;
    ty_params_ = tvar_env_1;
    ty_loc_ = Loc.no_location;
    ty_doc_ = Docstring.empty;
    ty_is_only_tag_enum_ = false;
    ty_is_suberror_ = false;
  }

let builtin_types = Typing_info.make_types ()

let _ =
  [
    ty_constr_option;
    ty_constr_fixedarray;
    ty_constr_ref;
    ty_constr_result;
    ty_constr_error;
  ]
  |> List.iter (fun (t : Typedecl_info.t) ->
         Typing_info.add_type builtin_types (Type_path_util.name t.ty_constr) t)

let builtin_values : Typing_info.values = Typing_info.make_values ()

let _ =
  List.iter
    (Typing_info.add_constructor builtin_values)
    [ constr_none; constr_some; constr_err; constr_ok ];
  Typing_info.add_field builtin_values ~field:field_val
