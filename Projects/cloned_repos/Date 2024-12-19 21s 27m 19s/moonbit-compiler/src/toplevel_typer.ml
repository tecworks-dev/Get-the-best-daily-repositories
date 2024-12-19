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


module Qual_ident = Basic_qual_ident
module Config = Basic_config
module Constr_info = Basic_constr_info
module Lst = Basic_lst
module Vec = Basic_vec
module Type_path = Basic_type_path
module Hash_string = Basic_hash_string
module Strutil = Basic_strutil
module Syntax = Parsing_syntax
module Operators = Parsing_operators

type docstring = Docstring.t

let trait_not_implemented ~trait ~type_name ~failure_reasons ~loc =
  let failure_reasons =
    Lst.map failure_reasons (fun reason ->
        match reason with
        | `Method_missing name ->
            ("method " ^ name ^ " is missing" : Stdlib.String.t)
        | `Private_method name ->
            ("implementation of method " ^ name ^ " is private"
              : Stdlib.String.t)
        | `Type_mismatch (name, expected, actual) ->
            Stdlib.String.concat ""
              [
                "method ";
                name;
                " is expected to have type ";
                expected;
                ", but has type ";
                actual;
              ]
        | `Method_constraint_not_satisfied name ->
            ("constraints of method " ^ name ^ " are not satisfied"
              : Stdlib.String.t))
  in
  Errors.trait_not_implemented ~trait ~type_name ~failure_reasons ~loc

let add_error = Local_diagnostics.add_error
let add_global_error = Diagnostics.add_error

let take_info_partial partial ~diagnostics =
  Local_diagnostics.take_partial_info partial ~diagnostics
[@@inline]

let rec check_duplicate (xs : 'a list) (f : 'a -> 'a -> bool) =
  match xs with
  | [] -> None
  | x :: rest ->
      if Lst.exists rest (f x) then Some x else check_duplicate rest f

let duplicate_import_help_message kind : Stdlib.String.t =
  "This " ^ kind
  ^ " is imported implicitly because \"test-import-all\" is set to true in the \
     config file"

let get_duplicate_loc_info (v : Value_info.toplevel) =
  match v.direct_use_loc_ with
  | Not_direct_use -> (v.loc_, None)
  | Explicit_import loc_ -> (loc_, None)
  | Implicit_import_all loc_ ->
      (loc_, Some (duplicate_import_help_message "value"))

let typing_type ~(allow_private : bool)
    ?(placeholder_env : Placeholder_env.t option) (te : Syntax.typ)
    ~(tvar_env : Tvar_env.t) ~(types : Global_env.All_types.t) ~diagnostics :
    Typedtree.typ =
  Typeutil.typing_type ~allow_private ~is_toplevel:true ?placeholder_env
    ~tvar_env ~types te
  |> take_info_partial ~diagnostics

let typing_error_type ~(allow_private : bool)
    ?(placeholder_env : Placeholder_env.t option) (err_ty : Syntax.error_typ)
    ~(tvar_env : Tvar_env.t) ~(types : Global_env.All_types.t) ~has_error
    ~header_loc ~diagnostics : Stype.t option * Typedtree.error_typ =
  match err_ty with
  | Error_typ { ty } ->
      let typ =
        typing_type ~allow_private ?placeholder_env ~tvar_env ~types
          ~diagnostics ty
      in
      let stype = Typedtree_util.stype_of_typ typ in
      if not (Type.is_error_type ~tvar_env stype) then
        add_error diagnostics
          (Errors.not_error_subtype
             (Printer.type_to_string stype)
             (Typedtree.loc_of_typ typ));
      (Some stype, Error_typ { ty = typ })
  | Default_error_typ { loc_ } ->
      (Some Typeutil.default_err_type, Default_error_typ { loc_ })
  | No_error_typ when has_error ->
      (Some Typeutil.default_err_type, Default_error_typ { loc_ = header_loc })
  | No_error_typ -> (None, No_error_typ)

let typing_trait_name ~types ~allow_private trait ~diagnostics =
  Typeutil.typing_trait_name ~types ~allow_private trait
  |> take_info_partial ~diagnostics

let collect_types_and_traits (impls : Syntax.impls) ~foreign_types
    ~(diagnostics : Diagnostics.t) : Placeholder_env.t =
  let type_defs : Syntax.type_decl Hash_string.t = Hash_string.create 17 in
  let trait_defs : Syntax.trait_decl Hash_string.t = Hash_string.create 17 in
  let find_declared_type name =
    match Hash_string.find_opt type_defs name with
    | Some { loc_; _ } -> Some (loc_, None)
    | None -> (
        match
          Global_env.All_types.find_type_alias foreign_types
            (Basic_longident.Lident name)
        with
        | Some alias ->
            Some (alias.loc_, Some (duplicate_import_help_message "type"))
        | None -> None)
  in
  let go (impl : Syntax.impl) =
    match impl with
    | Ptop_expr _ | Ptop_test _ | Ptop_letdef _ | Ptop_funcdef _ | Ptop_impl _
    | Ptop_impl_relation _ ->
        ()
    | Ptop_typedef decl -> (
        let base = decl.loc_ in
        let tycon_loc_ = Rloc.to_loc ~base decl.tycon_loc_ in
        let name = decl.tycon in
        match find_declared_type name with
        | Some (first_loc, extra_message) ->
            add_global_error diagnostics
              (Errors.type_redeclare ~name ~first_loc ~second_loc:tycon_loc_
                 ~extra_message);
            if extra_message <> None then Hash_string.add type_defs name decl
        | None -> (
            Hash_string.add type_defs name decl;
            match Hash_string.find_opt trait_defs name with
            | Some i ->
                add_global_error diagnostics
                  (Errors.type_trait_duplicate ~name ~first_kind:"trait"
                     ~first_loc:i.trait_loc_ ~second_kind:"type"
                     ~second_loc:tycon_loc_ ~extra_message:None)
            | None -> ()))
    | Ptop_trait trait_decl -> (
        let name = trait_decl.trait_name.binder_name in
        let trait_loc_ = trait_decl.trait_loc_ in
        match Hash_string.find_opt trait_defs name with
        | Some i ->
            add_global_error diagnostics
              (Errors.trait_redeclare ~name ~first_loc:i.trait_loc_
                 ~second_loc:trait_loc_)
        | None -> (
            Hash_string.add trait_defs name trait_decl;
            match find_declared_type name with
            | Some (loc, extra_message) ->
                add_global_error diagnostics
                  (Errors.type_trait_duplicate ~name ~first_kind:"type"
                     ~first_loc:loc ~second_kind:"trait" ~second_loc:trait_loc_
                     ~extra_message)
            | None -> ()))
  in
  Lst.iter impls go;
  Placeholder_env.make ~foreign_types ~type_defs ~trait_defs

let check_operator_decl ~(diagnostics : Local_diagnostics.t) ~(loc : Rloc.t)
    method_name ty =
  match Operators.find_by_method_opt method_name with
  | Some op -> (
      let expected_arity =
        match op.kind with
        | Prefix -> 1
        | Infix _ -> 2
        | Mixfix { arity; _ } -> arity
      in
      let actual_arity = Stype.arity_of_typ ty in
      if expected_arity <> actual_arity then
        add_error diagnostics
          (Errors.bad_operator_arity ~method_name ~expected:expected_arity
             ~actual:actual_arity ~loc)
      else
        match op.kind with
        | Infix { should_have_equal_param_type = true } -> (
            match ty with
            | Tarrow { params_ty = [ param_typ1; param_typ2 ]; _ } ->
                if not (Type.same_type param_typ1 param_typ2) then
                  add_error diagnostics
                    (Errors.bad_operator_type ~method_name
                       ~first:(Printer.type_to_string param_typ1)
                       ~second:(Printer.type_to_string param_typ2)
                       ~loc)
            | _ -> assert false)
        | _ -> ())
  | None -> ()

let typing_trait_method_decl ~is_pub (method_decl : Syntax.trait_method_decl)
    ~(placeholder_env : Placeholder_env.t) ~(types : Global_env.All_types.t)
    ~(base : Loc.t) ~diagnostics :
    Trait_decl.method_decl * Typedtree.method_decl =
  let (Trait_method
        { name; has_error; quantifiers = _; params; return_type; loc_ }) =
    method_decl
  in
  let meth_loc = Rloc.to_loc ~base loc_ in
  let tvar_env = Tvar_env.tvar_env_self in
  let params_ty, tast_params =
    Lst.map_split params (fun p ->
        let typ =
          typing_type ~allow_private:(not is_pub) ~placeholder_env ~types
            ~tvar_env p.tmparam_typ ~diagnostics
        in
        (Typedtree_util.stype_of_typ typ, (p.tmparam_label, typ)))
  in
  let ret_ty, ret_typ, err_ty, err_typ =
    match return_type with
    | Some (return_type, err_type) ->
        let typ =
          typing_type ~allow_private:(not is_pub) ~placeholder_env ~types
            ~tvar_env return_type ~diagnostics
        in
        let error_stype, error_typ =
          typing_error_type ~allow_private:(not is_pub) ~placeholder_env ~types
            ~tvar_env ~diagnostics ~has_error ~header_loc:name.loc_ err_type
        in
        let error_typ =
          match error_typ with
          | Error_typ { ty } -> Some ty
          | Default_error_typ _ -> Some Typeutil.default_err_typedtree_type
          | No_error_typ -> None
        in
        (Typedtree_util.stype_of_typ typ, Some typ, error_stype, error_typ)
    | None ->
        add_error diagnostics (Errors.missing_return_annot name.loc_);
        (Stype.blackhole, None, None, None)
  in
  let method_arity =
    Fn_arity.from_trait_method_params params ~base
    |> take_info_partial ~diagnostics
  in
  let method_name = name.binder_name in
  let method_typ : Stype.t =
    Tarrow { params_ty; ret_ty; err_ty; generic_ = true }
  in
  check_operator_decl ~diagnostics ~loc:name.loc_ method_name method_typ;
  let method_decl : Trait_decl.method_decl =
    {
      method_name = name.binder_name;
      method_typ;
      method_arity;
      method_loc_ = meth_loc;
    }
  in
  let tast_decl : Typedtree.method_decl =
    {
      method_name = name;
      method_params = tast_params;
      method_ret = ret_typ;
      method_err = err_typ;
      method_loc_ = loc_;
    }
  in
  (method_decl, tast_decl)

type toplevel_env = Typing_info.values

let transl_type_vis ~diagnostics (vis : Syntax.visibility) :
    Typedecl_info.visibility =
  match vis with
  | Vis_priv _ -> Vis_priv
  | Vis_default -> Vis_default
  | Vis_pub { attr = None; loc_ } ->
      Local_diagnostics.add_warning diagnostics
        {
          loc = loc_;
          kind =
            Deprecated_syntax
              {
                old_usage = "`pub`";
                purpose = "declaring a fully public struct/enum/newtype";
                new_usage =
                  Some
                    "`pub` will default to the semantic of `pub(readonly)` in \
                     the future, to declare a fully public type, use \
                     `pub(all)`";
              };
        };
      Vis_fully_pub
  | Vis_pub { attr = Some "all" } -> Vis_fully_pub
  | Vis_pub { attr = Some "" } -> Vis_fully_pub
  | Vis_pub { attr = Some "readonly" } -> Vis_readonly
  | Vis_pub { attr = Some attr; loc_ } ->
      add_error diagnostics
        (Errors.unsupported_modifier
           ~modifier:("pub(" ^ attr ^ ")" : Stdlib.String.t)
           ~loc:loc_);
      Vis_fully_pub

let transl_trait_vis ~diagnostics (vis : Syntax.visibility) :
    Typedecl_info.visibility =
  match vis with
  | Vis_priv _ -> Vis_priv
  | Vis_default -> Vis_default
  | Vis_pub { attr = None; loc_ } ->
      Local_diagnostics.add_warning diagnostics
        {
          loc = loc_;
          kind =
            Deprecated_syntax
              {
                old_usage = "`pub trait`";
                purpose =
                  "declaring a fully public trait that can be implemented \
                   outside current package";
                new_usage =
                  Some
                    "`pub trait` will default to `pub(readonly)` trait in the \
                     future, to declare a fully public trait, use `pub(open)`";
              };
        };
      Vis_fully_pub
  | Vis_pub { attr = Some "" } -> Vis_fully_pub
  | Vis_pub { attr = Some "readonly" } -> Vis_readonly
  | Vis_pub { attr = Some "open" } -> Vis_fully_pub
  | Vis_pub { attr = Some attr; loc_ } ->
      add_error diagnostics
        (Errors.unsupported_modifier
           ~modifier:("pub(" ^ attr ^ ")" : Stdlib.String.t)
           ~loc:loc_);
      Vis_fully_pub

let transl_component_vis = function
  | Typedecl_info.Vis_default | Vis_priv -> Typedecl_info.Invisible
  | Vis_readonly -> Readable
  | Vis_fully_pub -> Read_write

let submit_variant (ty_res : Stype.t) (ty_vis : Typedecl_info.visibility)
    (cs : Syntax.constr_decl list) (tvar_env : Tvar_env.t)
    ~(placeholder_env : Placeholder_env.t) ~(types : Global_env.All_types.t)
    (toplevel : toplevel_env) ~(is_exn : bool) ~(type_name : string)
    ~(base : Loc.t) ~diagnostics :
    Typedecl_info.type_components * Typedtree.type_desc =
  let n = List.length cs in
  let allow_private =
    match ty_vis with
    | Vis_default | Vis_priv -> true
    | Vis_readonly | Vis_fully_pub -> false
  in
  let cds, tast_cds =
    Lst.mapi cs (fun i constr ->
        let ({ constr_name; constr_args = args; constr_loc_ = cs_loc_ }
              : Syntax.constr_decl) =
          constr
        in
        let name = constr_name.name in
        let args, cs_arity_ =
          match args with
          | None -> ([], Fn_arity.simple 0)
          | Some args ->
              ( args,
                take_info_partial ~diagnostics
                  (Fn_arity.from_constr_params ~base args) )
        in
        let cs_args, tast_args =
          Lst.map_split args (fun arg ->
              let typ =
                typing_type ~allow_private ~tvar_env ~placeholder_env ~types
                  ~diagnostics arg.cparam_typ
              in
              let tast_arg : Typedtree.constr_decl_arg =
                {
                  carg_typ = typ;
                  carg_mut = arg.cparam_mut;
                  carg_label = arg.cparam_label;
                }
              in
              (Typedtree_util.stype_of_typ typ, tast_arg))
        in
        let cs_vis = transl_component_vis ty_vis in
        let cs_tag : Constr_info.constr_tag =
          if is_exn then
            Extensible_tag
              {
                pkg = !Config.current_package;
                type_name;
                name;
                total = n;
                index = i;
              }
          else
            Constr_tag_regular
              {
                total = n;
                index = i;
                name_ = name;
                is_constant_ = cs_args = [];
              }
        in
        let constr : Typedecl_info.constructor =
          let cs_constr_loc_ = Rloc.to_loc ~base constr_name.loc_ in
          let cs_loc_ = Rloc.to_loc ~base cs_loc_ in
          {
            constr_name = name;
            cs_res = ty_res;
            cs_args;
            cs_tag;
            cs_vis;
            cs_ty_params_ = tvar_env;
            cs_arity_;
            cs_constr_loc_;
            cs_loc_;
          }
        in
        let constr_decl : Typedtree.constr_decl =
          {
            constr_name = { label_name = name; loc_ = constr_name.loc_ };
            constr_tag = cs_tag;
            constr_args = tast_args;
            constr_arity_ = cs_arity_;
            constr_loc_ = cs_loc_;
          }
        in
        Typing_info.add_constructor toplevel constr;
        (constr, constr_decl))
    |> List.split
  in
  if is_exn then (ErrorEnum_type cds, Td_error (Enum_payload tast_cds))
  else (Variant_type cds, Td_variant tast_cds)

let submit_field (ty_record : Stype.t) (ty_vis : Typedecl_info.visibility)
    (fs : Syntax.field_decl list) (tvar_env : Tvar_env.t)
    ~(placeholder_env : Placeholder_env.t) ~(types : Global_env.All_types.t)
    (toplevel : toplevel_env) ~diagnostics ~base :
    Typedecl_info.type_components * Typedtree.type_desc =
  let all_labels = Lst.map fs (fun f -> f.field_name.label) in
  let has_private_field = ref false in
  let fields, tast_fields =
    Lst.mapi fs (fun i f ->
        let ({
               field_name = name;
               field_ty = ty_expr;
               field_mut = mut;
               field_loc_ = loc_;
               field_vis;
             }
              : Syntax.field_decl) =
          f
        in
        let field_vis : Typedecl_info.visibility =
          match field_vis with
          | Vis_priv _ ->
              has_private_field := true;
              Vis_priv
          | Vis_pub _ | Vis_default -> ty_vis
        in
        let vis = transl_component_vis field_vis in
        let allow_private =
          match vis with Read_write | Readable -> false | Invisible -> true
        in
        let field_typ =
          typing_type ty_expr ~allow_private ~placeholder_env ~tvar_env ~types
            ~diagnostics
        in
        let field : Typedecl_info.field =
          {
            field_name = name.label;
            all_labels;
            pos = i;
            ty_field = Typedtree_util.stype_of_typ field_typ;
            mut;
            vis;
            ty_record;
            ty_params_ = tvar_env;
            label_loc_ = Rloc.to_loc ~base name.loc_;
            loc_ = Rloc.to_loc ~base loc_;
          }
        in
        let field_decl : Typedtree.field_decl =
          {
            field_label = { label_name = name.label; loc_ = name.loc_ };
            field_typ;
            field_mut = mut;
            field_vis;
            field_loc_ = loc_;
          }
        in
        Typing_info.add_field toplevel ~field;
        (field, field_decl))
    |> List.split
  in
  let has_private_field_ = !has_private_field in
  (Record_type { fields; has_private_field_ }, Td_record tast_fields)

let typing_typedef (decl : Syntax.type_decl)
    ~(placeholder_env : Placeholder_env.t) (types : Global_env.All_types.t)
    (toplevel : toplevel_env) ~(diagnostics : Local_diagnostics.t) :
    Typedtree.type_decl =
  let name = decl.tycon in
  let tvar_env = Typeutil.typing_type_decl_binders decl.params in
  let base = decl.loc_ in
  let ty_vis = transl_type_vis ~diagnostics decl.type_vis in
  let tycon_loc_ = Rloc.to_loc ~base decl.tycon_loc_ in
  let type_constructor =
    Type_path.toplevel_type ~pkg:!Basic_config.current_package decl.tycon
  in
  let ty_args = Tvar_env.get_types tvar_env in
  let only_tag_enum_ =
    match decl.components with
    | Ptd_variant cs ->
        Lst.for_all cs (fun { constr_args; _ } -> constr_args = None)
    | _ -> false
  in
  let is_suberror_ =
    match decl.components with Ptd_error _ -> true | _ -> false
  in
  let ty_res : Stype.t =
    T_constr
      {
        type_constructor;
        tys = ty_args;
        generic_ = ty_args <> [];
        only_tag_enum_;
        is_suberror_;
      }
  in
  let ty_comp, ty_desc =
    match decl.components with
    | Ptd_variant cs ->
        (match
           check_duplicate cs (fun c1 c2 ->
               c1.constr_name.name = c2.constr_name.name)
         with
        | Some { constr_name = { name; loc_ }; _ } ->
            add_error diagnostics (Errors.constructor_duplicate ~name ~loc:loc_)
        | None -> ());
        submit_variant ty_res ty_vis cs tvar_env ~placeholder_env ~types
          toplevel ~is_exn:false ~type_name:name ~diagnostics ~base
    | Ptd_error ty -> (
        let allow_private =
          match ty_vis with
          | Vis_priv | Vis_default -> true
          | Vis_readonly | Vis_fully_pub -> false
        in
        match ty with
        | No_payload ->
            let constr : Typedecl_info.constructor =
              {
                constr_name = decl.tycon;
                cs_res = ty_res;
                cs_args = [];
                cs_tag =
                  Extensible_tag
                    {
                      pkg = !Config.current_package;
                      type_name = name;
                      name;
                      total = 1;
                      index = 0;
                    };
                cs_vis = transl_component_vis ty_vis;
                cs_ty_params_ = tvar_env;
                cs_arity_ = Fn_arity.simple 0;
                cs_constr_loc_ = tycon_loc_;
                cs_loc_ = tycon_loc_;
              }
            in
            Typing_info.add_constructor toplevel constr;
            (Error_type constr, Td_error No_payload)
        | Single_payload ty ->
            let typ =
              typing_type ty ~allow_private ~placeholder_env ~tvar_env ~types
                ~diagnostics
            in
            let constr : Typedecl_info.constructor =
              {
                constr_name = decl.tycon;
                cs_res = ty_res;
                cs_args = [ Typedtree_util.stype_of_typ typ ];
                cs_tag =
                  Extensible_tag
                    {
                      pkg = !Config.current_package;
                      type_name = name;
                      name;
                      total = 1;
                      index = 0;
                    };
                cs_vis = transl_component_vis ty_vis;
                cs_ty_params_ = tvar_env;
                cs_arity_ = Fn_arity.simple 1;
                cs_constr_loc_ = tycon_loc_;
                cs_loc_ = tycon_loc_;
              }
            in
            Typing_info.add_constructor toplevel constr;
            (Error_type constr, Td_error (Single_payload typ))
        | Enum_payload cs ->
            (match
               check_duplicate cs (fun c1 c2 ->
                   c1.constr_name.name = c2.constr_name.name)
             with
            | Some { constr_name = { name; loc_ }; _ } ->
                add_error diagnostics
                  (Errors.constructor_duplicate ~name ~loc:loc_)
            | None -> ());
            submit_variant ty_res ty_vis cs tvar_env ~placeholder_env ~types
              toplevel ~is_exn:true ~type_name:name ~diagnostics ~base)
    | Ptd_newtype ty ->
        let allow_private =
          match ty_vis with
          | Vis_priv | Vis_default -> true
          | Vis_readonly | Vis_fully_pub -> false
        in
        let typ =
          typing_type ty ~allow_private ~placeholder_env ~tvar_env ~types
            ~diagnostics
        in
        let sty = Typedtree_util.stype_of_typ typ in
        let name = decl.tycon in
        let constr : Typedecl_info.constructor =
          {
            constr_name = name;
            cs_res = ty_res;
            cs_args = [ sty ];
            cs_tag =
              Constr_tag_regular
                { total = 1; index = 0; name_ = name; is_constant_ = false };
            cs_vis = transl_component_vis ty_vis;
            cs_ty_params_ = tvar_env;
            cs_arity_ = Fn_arity.simple 1;
            cs_constr_loc_ = tycon_loc_;
            cs_loc_ = tycon_loc_;
          }
        in
        Typing_info.add_constructor toplevel constr;
        let recursive = Placeholder_env.newtype_in_cycle placeholder_env name in
        ( New_type { newtype_constr = constr; underlying_typ = sty; recursive },
          Td_newtype typ )
    | Ptd_record fs ->
        (match
           check_duplicate fs (fun f1 f2 ->
               f1.field_name.label = f2.field_name.label)
         with
        | Some { field_name = { label = name; loc_ }; _ } ->
            add_error diagnostics (Errors.field_duplicate ~name ~loc:loc_)
        | None -> ());
        submit_field ty_res ty_vis fs tvar_env ~placeholder_env ~types toplevel
          ~diagnostics ~base
    | Ptd_abstract -> (Extern_type, Td_abstract)
    | Ptd_alias _ -> assert false
  in
  let ty_constr =
    Type_path.toplevel_type ~pkg:!Basic_config.current_package name
  in
  let typedecl_info : Typedecl_info.t =
    {
      ty_constr;
      ty_arity = List.length decl.params;
      ty_desc = ty_comp;
      ty_vis;
      ty_params_ = tvar_env;
      ty_loc_ = tycon_loc_;
      ty_doc_ = decl.doc_;
      ty_is_only_tag_enum_ = only_tag_enum_;
      ty_is_suberror_ = is_suberror_;
    }
  in
  Global_env.All_types.add_type types name typedecl_info;
  {
    td_binder = { name = ty_constr; kind = Type; loc_ = decl.tycon_loc_ };
    td_params = tvar_env;
    td_desc = ty_desc;
    td_vis = ty_vis;
    td_loc_ = decl.loc_;
    td_doc_ = decl.doc_;
    td_deriving_ =
      Lst.map decl.deriving_
        (fun { type_name_ = { name; loc_ }; _ } : Syntax.constrid_loc ->
          { lid = name; loc_ });
  }

let typing_types_and_traits (impls : Syntax.impl list)
    (types : Global_env.All_types.t) (toplevel_env : toplevel_env)
    ~(diagnostics : Diagnostics.t) =
  let placeholder_env =
    collect_types_and_traits ~foreign_types:types impls ~diagnostics
  in
  let tast_of_alias = Hash_string.create 17 in
  let rec typing_alias ~visiting name (decl : Syntax.type_decl) =
    let base = decl.loc_ in
    let local_diagnostics = Local_diagnostics.make ~base in
    match decl.components with
    | Ptd_alias typ ->
        (if Hash_string.mem tast_of_alias name then ()
         else if Lst.mem_string visiting name then
           add_error local_diagnostics
             (Errors.cycle_in_type_alias
                ~cycle:(List.rev (name :: visiting))
                ~loc:decl.tycon_loc_)
         else
           let tvar_env = Typeutil.typing_type_decl_binders decl.params in
           let is_pub =
             match decl.type_vis with
             | Vis_pub _ -> true
             | Vis_default -> false
             | Vis_priv { loc_ = vis_loc } ->
                 add_error local_diagnostics
                   (Errors.invalid_visibility ~vis:"priv" ~entity:"type alias"
                      ~loc:vis_loc);
                 false
           in
           let result : Typedecl_info.alias =
             {
               name;
               arity = List.length decl.params;
               ty_params = tvar_env;
               alias = T_blackhole;
               is_pub;
               doc_ = decl.doc_;
               loc_ = decl.loc_;
             }
           in
           Global_env.All_types.add_type_alias types name result;
           (match
              Placeholder_env.find_type_alias_deps placeholder_env decl.tycon
            with
           | None -> ()
           | Some deps ->
               Lst.iter deps (fun dep ->
                   match Placeholder_env.find_type_opt placeholder_env dep with
                   | None -> ()
                   | Some dep_decl ->
                       typing_alias ~visiting:(name :: visiting) dep dep_decl));
           let typ =
             typing_type ~allow_private:(not is_pub) ~placeholder_env ~tvar_env
               ~types ~diagnostics:local_diagnostics typ
           in
           let stype = Typedtree_util.stype_of_typ typ in
           (match stype with
           | T_constr _ | T_trait _ | T_builtin _ | Tarrow _ -> ()
           | T_blackhole -> ()
           | Tvar _ -> assert false
           | Tparam _ ->
               add_error local_diagnostics
                 (Errors.invalid_type_alias_target (Typedtree.loc_of_typ typ)));
           (match decl.deriving_ with
           | [] -> ()
           | trait1 :: rest ->
               let loc =
                 Lst.fold_left rest trait1.loc_ (fun l r -> Rloc.merge l r.loc_)
               in
               add_error local_diagnostics (Errors.type_alias_cannot_derive loc));
           Global_env.All_types.add_type_alias types name
             { result with alias = stype };
           let tast_node : Typedtree.type_decl =
             {
               td_binder =
                 {
                   name =
                     Type_path.toplevel_type
                       ~pkg:!Basic_config.current_package
                       name;
                   kind = Type;
                   loc_ = decl.tycon_loc_;
                 };
               td_params = tvar_env;
               td_desc = Td_alias typ;
               td_vis = (if is_pub then Vis_fully_pub else Vis_priv);
               td_loc_ = decl.loc_;
               td_doc_ = decl.doc_;
               td_deriving_ = [];
             }
           in
           Hash_string.add tast_of_alias name tast_node);
        Local_diagnostics.add_to_global local_diagnostics diagnostics
    | _ -> ()
  in
  Placeholder_env.iter_types placeholder_env (typing_alias ~visiting:[]);
  let type_decls =
    Placeholder_env.types_to_list_map placeholder_env (fun name decl ->
        let local_diagnostics = Local_diagnostics.make ~base:decl.loc_ in
        let res =
          match decl.components with
          | Ptd_alias _ -> Hash_string.find_exn tast_of_alias name
          | _ ->
              typing_typedef decl ~placeholder_env types toplevel_env
                ~diagnostics:local_diagnostics
        in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        res)
  in
  let method_decls_of_trait = Hash_string.create 17 in
  Placeholder_env.iter_traits placeholder_env (fun name trait_info ->
      let base = trait_info.decl.trait_loc_ in
      let local_diagnostics = Local_diagnostics.make ~base in
      let method_decls =
        let is_pub =
          match trait_info.decl.trait_vis with
          | Vis_pub _ -> true
          | Vis_priv _ | Vis_default -> false
        in
        Lst.map_split trait_info.decl.trait_methods
          (typing_trait_method_decl ~is_pub ~placeholder_env ~types ~base
             ~diagnostics:local_diagnostics)
      in
      Local_diagnostics.add_to_global local_diagnostics diagnostics;
      Hash_string.add method_decls_of_trait name method_decls);
  let trait_decls =
    Placeholder_env.traits_to_list_map placeholder_env
      (fun name trait_info : Typedtree.trait_decl ->
        let local_diagnostics =
          Local_diagnostics.make ~base:trait_info.decl.trait_loc_
        in
        let trait_vis =
          transl_trait_vis ~diagnostics:local_diagnostics
            trait_info.decl.trait_vis
        in
        let supers =
          Lst.fold_right trait_info.decl.trait_supers []
            (fun { tvc_trait; loc_ } acc ->
              let builtin_or_foreign () =
                match
                  typing_trait_name ~types ~diagnostics:local_diagnostics
                    ~allow_private:(Typedecl_info.vis_is_pub trait_vis)
                    { name = tvc_trait; loc_ }
                with
                | Some decl, _ -> decl.name :: acc
                | None, _ -> acc
                  [@@inline]
              in
              match tvc_trait with
              | Lident name -> (
                  match Placeholder_env.find_trait_opt placeholder_env name with
                  | Some { decl = super_decl; _ } ->
                      (match
                         (trait_info.decl.trait_vis, super_decl.trait_vis)
                       with
                      | Vis_pub _, Vis_priv _ ->
                          add_error local_diagnostics
                            (Errors.cannot_depend_private ~entity:"trait"
                               ~loc:loc_)
                      | _ -> ());
                      Type_path.toplevel_type
                        ~pkg:!Basic_config.current_package
                        name
                      :: acc
                  | None -> builtin_or_foreign ())
              | Ldot _ -> builtin_or_foreign ())
        in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        let methods, tast_methods =
          Hash_string.find_exn method_decls_of_trait name
        in
        let path =
          Type_path.toplevel_type ~pkg:!Basic_config.current_package name
        in
        let closure_methods =
          Lst.concat_map trait_info.closure (fun trait ->
              match trait with
              | Toplevel { pkg; id } when pkg = !Basic_config.current_package ->
                  fst (Hash_string.find_exn method_decls_of_trait id)
                  |> List.map (fun meth_decl -> (trait, meth_decl))
              | _ -> (
                  match Global_env.All_types.find_trait_by_path types trait with
                  | None -> []
                  | Some trait_decl ->
                      Lst.map trait_decl.methods (fun meth_decl ->
                          (trait, meth_decl))))
        in
        let trait_decl : Trait_decl.t =
          {
            name = path;
            supers;
            closure = trait_info.closure;
            closure_methods;
            methods;
            vis_ = trait_vis;
            loc_ = trait_info.decl.trait_loc_;
            doc_ = trait_info.decl.trait_doc_;
            object_safety_ = trait_info.object_safety_status;
          }
        in
        Global_env.All_types.add_trait types name trait_decl;
        {
          trait_name =
            {
              name = path;
              kind = Trait;
              loc_ = trait_info.decl.trait_name.loc_;
            };
          trait_methods = tast_methods;
          trait_vis;
          trait_doc_ = trait_info.decl.trait_doc_;
          trait_loc_ = trait_info.decl.trait_loc_;
        })
  in
  (type_decls, trait_decls)

type ret_annotation =
  | Annotated of (Typedtree.typ * Typedtree.error_typ)
  | Has_error_type of Rloc.t
  | No_error_type_annotated

type typed_fn_annotation = {
  params_ty : (Stype.t * Typedtree.typ option) list;
  ret_ty : Stype.t;
  err_ty : Stype.t option;
  ret_annotation : ret_annotation;
}

type value_worklist_item =
  | Wl_top_expr of {
      expr : Syntax.expr;
      is_main : bool;
      id : Qual_ident.t;
      loc_ : Loc.t;
    }
  | Wl_top_letdef of {
      binder : Syntax.binder;
      expr : top_let_body;
      is_pub : bool;
      loc_ : Loc.t;
      doc_ : docstring;
      konstraint : Typedtree.typ option;
      id : Qual_ident.t;
      typ : Stype.t;
    }
  | Wl_top_funcdef of {
      fun_binder : Syntax.binder;
      decl_params : Syntax.parameter list;
      params_loc : Rloc.t;
      is_pub : bool;
      doc : docstring;
      decl_body : Syntax.decl_body;
      loc_ : Loc.t;
      id : Qual_ident.t;
      kind : Typedtree.fun_decl_kind;
      arity : Fn_arity.t;
      tvar_env : Tvar_env.t;
      typed_fn_annotation : typed_fn_annotation;
    }
  | Wl_derive of {
      ty_decl : Typedecl_info.t;
      syn_decl : Syntax.type_decl;
      directive : Syntax.deriving_directive;
      trait_path : Type_path.t;
      loc_ : Loc.t;
    }

and top_let_body =
  | Wl_toplet_const of Typedtree.expr
  | Wl_toplet_normal of Syntax.expr

let add_method (type_name : Type_path.t) (fun_type : Stype.t)
    (meth : Syntax.binder) is_pub ~is_trait ~(doc : Docstring.t)
    ~(types : Global_env.All_types.t) ~tvar_env ~method_env
    ~(toplevel : toplevel_env) ~arity ~param_names ~prim
    ~(diagnostics : Local_diagnostics.t) ~(meth_loc : Loc.t) : Qual_ident.t =
  let method_name = meth.binder_name in
  let check_duplicate_method type_name =
    (if is_trait then
       match Global_env.All_types.find_trait_by_path types type_name with
       | None -> ()
       | Some trait_decl -> (
           match
             Lst.find_first trait_decl.methods (fun meth_decl ->
                 meth_decl.method_name = method_name)
           with
           | Some meth_decl ->
               add_error diagnostics
                 (Errors.method_duplicate ~method_name ~type_name
                    ~first_loc:meth_decl.method_loc_ ~second_loc:meth.loc_)
           | None -> ()));
    let report_duplicate (method_info : Method_env.method_info) =
      add_error diagnostics
        (Errors.method_duplicate ~method_name ~type_name
           ~first_loc:method_info.loc ~second_loc:meth.loc_)
        [@@inline]
    in
    Method_env.find_regular_method method_env ~type_name ~method_name
    |> Option.iter report_duplicate;
    (match type_name with
    | Toplevel { pkg; id = _ } when pkg <> !Config.current_package ->
        Pkg.find_regular_method
          (Global_env.All_types.get_pkg_tbl types)
          ~pkg ~type_name ~method_name
        |> Option.iter report_duplicate
    | _ -> ());
    if Type_path.can_be_extended_from_builtin type_name then
      Pkg.find_regular_method
        (Global_env.All_types.get_pkg_tbl types)
        ~pkg:Config.builtin_package ~type_name ~method_name
      |> Option.iter report_duplicate
  in
  let check_duplicate_function () =
    match Typing_info.find_value toplevel method_name with
    | Some v ->
        let first_loc, extra_message = get_duplicate_loc_info v in
        add_error diagnostics
          (Errors.method_func_duplicate ~name:method_name ~first_loc
             ~second_loc:meth.loc_ ~extra_message)
    | _ -> ()
  in
  let check_self_type () =
    match type_name with
    | _ when not (Type_path_util.is_foreign type_name) -> true
    | _ when !Basic_config.current_package = Basic_config.builtin_package ->
        true
    | _ ->
        add_error diagnostics
          (Errors.method_on_foreign_type ~method_name:meth.binder_name
             ~type_name ~loc:meth.loc_);
        false
  in
  check_operator_decl ~diagnostics ~loc:meth.loc_ method_name fun_type;
  check_duplicate_function ();
  check_duplicate_method type_name;
  let id : Basic_qual_ident.t =
    Qual_ident.meth ~self_typ:type_name ~name:method_name
  in
  (if check_self_type () then
     let method_info : Method_env.method_info =
       {
         id;
         prim;
         typ = fun_type;
         pub = is_pub;
         loc = meth_loc;
         doc_ = doc;
         ty_params_ = tvar_env;
         arity_ = arity;
         param_names_ = param_names;
       }
     in
     Method_env.add_method method_env ~type_name ~method_name ~method_info);
  id

let check_method_self_type ~(diagnostics : Local_diagnostics.t) ~(loc : Rloc.t)
    (self_type : Stype.t) =
  match Stype.type_repr self_type with
  | T_constr { type_constructor = p; tys = _ } -> Some (false, p)
  | T_builtin b -> Some (false, Stype.tpath_of_builtin b)
  | Tvar { contents = Tnolink Tvar_error } | T_blackhole -> None
  | Tvar { contents = Tnolink Tvar_normal } ->
      add_error diagnostics (Errors.cannot_determine_self_type loc);
      None
  | T_trait trait -> Some (true, trait)
  | Tarrow _ | Tparam _ ->
      add_error diagnostics (Errors.invalid_self_type loc);
      None
  | Tvar _ -> assert false

let add_self_method (self_type : Stype.t) (fun_type : Stype.t)
    (meth : Syntax.binder) is_pub ~(doc : docstring)
    ~(types : Global_env.All_types.t) ~tvar_env ~method_env
    ~(toplevel : toplevel_env) ~arity ~param_names ~prim
    ~(diagnostics : Local_diagnostics.t) ~meth_loc : Qual_ident.t =
  match check_method_self_type ~diagnostics ~loc:meth.loc_ self_type with
  | Some (is_trait, type_name) ->
      add_method type_name fun_type meth is_pub ~is_trait ~doc ~types ~tvar_env
        ~method_env ~toplevel ~arity ~param_names ~prim ~diagnostics ~meth_loc
  | None -> Qual_ident.toplevel_value ~name:meth.binder_name

let add_ext_method (trait_decl : Trait_decl.t) ~(self_ty : Stype.t)
    (fun_type : Stype.t) (meth : Syntax.binder) is_pub ~(doc : Docstring.t)
    ~types ~tvar_env ~ext_method_env ~trait_impls ~arity ~param_names ~prim
    ~(diagnostics : Local_diagnostics.t) ~(global_diagnostics : Diagnostics.t)
    ~header_loc ~meth_loc : Qual_ident.t =
  let trait = trait_decl.name in
  let method_name = meth.binder_name in
  match Trait_decl.find_method trait_decl method_name ~loc:meth.loc_ with
  | Error err ->
      add_error diagnostics err;
      Qual_ident.toplevel_value ~name:method_name
  | Ok meth_decl -> (
      let check_type () =
        let exception Arity_mismatch in
        let expected_ty =
          Poly_type.instantiate_method_decl meth_decl ~self:self_ty
        in
        try
          Ctype.unify_exn expected_ty fun_type;
          if not (Fn_arity.equal arity meth_decl.method_arity) then
            raise_notrace Arity_mismatch
        with _ ->
          add_error diagnostics
            (Errors.ext_method_type_mismatch ~trait ~method_name
               ~expected:
                 (Printer.toplevel_function_type_to_string
                    ~arity:meth_decl.method_arity expected_ty)
               ~actual:
                 (Printer.toplevel_function_type_to_string ~arity fun_type)
               ~loc:meth.loc_)
          [@@inline]
      in
      check_type ();
      match check_method_self_type ~diagnostics ~loc:meth.loc_ self_ty with
      | None -> Qual_ident.toplevel_value ~name:method_name
      | Some (is_trait, type_name) ->
          let check_duplication () =
            if is_trait && Type_path.equal type_name trait then
              add_error diagnostics
                (Errors.method_duplicate ~method_name ~type_name
                   ~first_loc:meth_decl.method_loc_ ~second_loc:meth.loc_);
            let prev_def =
              match
                Ext_method_env.find_method ext_method_env ~trait
                  ~self_type:type_name ~method_name
              with
              | Some _ as prev_def -> prev_def
              | None ->
                  Pkg.find_ext_method_opt
                    (Global_env.All_types.get_pkg_tbl types)
                    ~pkg:Config.builtin_package ~trait ~self_type:type_name
                    ~method_name
            in
            match prev_def with
            | None -> ()
            | Some prev_def ->
                let trait_name =
                  Type_path.short_name
                    ~cur_pkg_name:(Some !Basic_config.current_package) trait
                in
                add_error diagnostics
                  (Errors.method_duplicate
                     ~method_name:
                       (trait_name ^ "::" ^ method_name : Stdlib.String.t)
                     ~type_name ~first_loc:prev_def.loc ~second_loc:meth.loc_)
              [@@inline]
          in
          check_duplication ();
          let id : Basic_qual_ident.t =
            Qual_ident.ext_meth ~trait ~self_typ:type_name ~name:method_name
          in
          let method_info : Ext_method_env.method_info =
            {
              id;
              prim;
              typ = fun_type;
              pub = is_pub;
              loc = meth_loc;
              doc_ = doc;
              ty_params_ = tvar_env;
              arity_ = arity;
              param_names_ = param_names;
            }
          in
          if
            Type_path_util.is_foreign trait
            && Type_path_util.is_foreign type_name
          then
            add_error diagnostics
              (Errors.ext_method_foreign_trait_foreign_type ~trait ~type_name
                 ~method_name ~loc:meth.loc_)
          else (
            (match Trait_impl.find_impl trait_impls ~trait ~type_name with
            | None ->
                let impl : Trait_impl.impl =
                  {
                    trait;
                    self_ty;
                    ty_params = tvar_env;
                    is_pub;
                    loc_ = header_loc;
                  }
                in
                Trait_impl.add_impl trait_impls ~trait ~type_name impl
            | Some impl ->
                if not (Type.same_type impl.self_ty self_ty) then
                  add_global_error global_diagnostics
                    (Errors.inconsistent_impl
                       ~trait:(Type_path_util.name trait)
                       ~type_name:(Type_path_util.name type_name)
                       ~reason:
                         (`Self_type_mismatch
                           ( Printer.type_to_string impl.self_ty,
                             Printer.type_to_string self_ty ))
                       ~loc1:impl.loc_ ~loc2:header_loc)
                else if not (Tvar_env.equal impl.ty_params tvar_env) then
                  add_global_error global_diagnostics
                    (Errors.inconsistent_impl
                       ~trait:(Type_path_util.name trait)
                       ~type_name:(Type_path_util.name type_name)
                       ~reason:`Type_parameter_bound ~loc1:impl.loc_
                       ~loc2:header_loc)
                else if is_pub && not impl.is_pub then
                  Trait_impl.update trait_impls ~trait ~type_name (fun impl ->
                      { impl with is_pub = true; loc_ = header_loc }));
            Ext_method_env.add_method ext_method_env ~trait ~self_type:type_name
              ~method_name method_info);
          id)

let try_infer_expr ~types expr : (Stype.t * bool) option =
  let exception Cannot_infer in
  let is_literal = ref true in
  let rec go (expr : Syntax.expr) =
    match expr with
    | Pexpr_constant { c } -> Typeutil.type_of_constant c
    | Pexpr_interp _ | Pexpr_multiline_string _ -> Stype.string
    | Pexpr_array { exprs = [] } -> raise_notrace Cannot_infer
    | Pexpr_array { exprs = expr0 :: exprs } ->
        let ty0 = go expr0 in
        Lst.iter exprs (fun expr ->
            let ty = go expr in
            ignore (Ctype.try_unify ty0 ty));
        Builtin.type_array ty0
    | Pexpr_tuple { exprs } -> Builtin.type_product (List.map go exprs)
    | Pexpr_ident { id = { var_name = Ldot { pkg; id } } } -> (
        match Global_env.All_types.find_foreign_value types ~pkg ~name:id with
        | None -> raise_notrace Cannot_infer
        | Some (Local_imm _ | Local_mut _) -> assert false
        | Some (Toplevel_value { ty_params_; _ })
          when not (Tvar_env.is_empty ty_params_) ->
            raise_notrace Cannot_infer
        | Some (Toplevel_value { typ; _ }) ->
            is_literal := false;
            typ)
    | _ -> raise_notrace Cannot_infer
  in
  try
    let ty = go expr in
    Some (ty, !is_literal)
  with Cannot_infer -> None

let prim_of_decl (decl : Syntax.decl_body) (loc : Loc.t) ~(doc : Docstring.t)
    ~(diagnostics : Diagnostics.t) =
  match decl with
  | Decl_stubs (Embedded { language = None; code = Code_string s }) ->
      if Strutil.first_char_is s.string_val '%' then (
        let prim = Primitive.find_prim s.string_val in
        if prim = None then
          add_global_error diagnostics
            (Errors.unknown_intrinsic ~name:s.string_val ~loc);
        prim)
      else None
  | Decl_body _ -> (
      match
        Lst.fold_right (Docstring.pragmas doc) [] (fun pragma acc ->
            match pragma with
            | Pragma_alert _ -> acc
            | Pragma_gen_js _ -> acc
            | Pragma_intrinsic intrinsic -> intrinsic :: acc
            | Pragma_coverage_skip -> acc)
      with
      | [] -> None
      | intrinsic :: [] ->
          let prim = Primitive.find_prim intrinsic in
          if prim = None then
            add_global_error diagnostics
              (Errors.unknown_intrinsic ~name:intrinsic ~loc);
          prim
      | _ :: _ ->
          add_global_error diagnostics (Errors.multiple_intrinsic loc);
          None)
  | Decl_stubs _ -> None

let check_toplevel_decl ~(toplevel : toplevel_env) (impls : Syntax.impls)
    (types : Global_env.All_types.t) ~(method_env : Method_env.t)
    ~(ext_method_env : Ext_method_env.t) ~(trait_impls : Trait_impl.t)
    ~(worklist : value_worklist_item Vec.t) ~diagnostics
    ~(build_context : Typeutil.build_context) : unit =
  let add_symbol (toplevel : toplevel_env) (binder : Syntax.binder) ty is_pub
      doc (tvar_env : Tvar_env.t) ~kind ~arity ~param_names
      ~(binder_loc : Loc.t) : Qual_ident.t =
    let name = binder.binder_name in
    (match Typing_info.find_value toplevel name with
    | Some v ->
        let first_loc, extra_message = get_duplicate_loc_info v in
        add_global_error diagnostics
          (Errors.value_redeclare ~name ~first_loc ~second_loc:binder_loc
             ~extra_message)
    | None -> (
        match Method_env.find_methods_by_name method_env ~method_name:name with
        | [] -> ()
        | (_, m) :: _ ->
            add_global_error diagnostics
              (Errors.value_redeclare ~name ~first_loc:m.loc
                 ~second_loc:binder_loc ~extra_message:None)));
    let qid : Qual_ident.t = Qual_ident.toplevel_value ~name in
    Typing_info.add_value toplevel
      {
        id = qid;
        typ = ty;
        pub = is_pub;
        kind;
        loc_ = binder_loc;
        doc_ = doc;
        ty_params_ = tvar_env;
        arity_ = arity;
        param_names_ = param_names;
        direct_use_loc_ = Not_direct_use;
      };
    qid
  in
  let main_loc = ref None in
  let go (impl : Syntax.impl) : unit =
    match impl with
    | Ptop_expr { expr; is_main = true; local_types = _; loc_ } ->
        if build_context = Lib then
          add_global_error diagnostics (Errors.unexpected_main loc_);
        if !main_loc <> None then
          add_global_error diagnostics
            (Errors.multiple_main ~first_loc:(Option.get !main_loc)
               ~second_loc:loc_)
        else main_loc := Some loc_;
        let id =
          Basic_qual_ident.toplevel_value ~name:("*main" : Stdlib.String.t)
        in
        Vec.push worklist (Wl_top_expr { expr; is_main = true; id; loc_ })
    | Ptop_expr { expr; is_main = false; local_types = _; loc_ } ->
        let id =
          Basic_qual_ident.toplevel_value
            ~name:
              ("*init" ^ Int.to_string (Basic_uuid.next ()) : Stdlib.String.t)
        in
        Vec.push worklist (Wl_top_expr { expr; is_main = false; id; loc_ })
    | Ptop_test _ -> assert false
    | Ptop_typedef { components = Ptd_alias _; _ } -> ()
    | Ptop_typedef type_decl ->
        let desc : Typedecl_info.t =
          Global_env.All_types.find_toplevel_type_exn types type_decl.tycon
        in
        let local_diagnostics = Local_diagnostics.make ~base:type_decl.loc_ in
        Lst.iter type_decl.deriving_ (fun directive ->
            let trait = directive.type_name_ in
            let trait =
              { trait with name = Derive.resolve_derive_alias trait.name }
            in
            let trait_loc = Rloc.to_loc ~base:type_decl.loc_ trait.loc_ in
            Derive.generate_signatures ~types ~ext_method_env ~trait_impls
              ~diagnostics:local_diagnostics ~loc:trait_loc desc trait;
            match
              Global_env.All_types.find_trait types trait.name
                ~loc:Rloc.no_location
            with
            | Ok { name = trait_path; _ } ->
                Vec.push worklist
                  (Wl_derive
                     {
                       directive = { directive with type_name_ = trait };
                       ty_decl = desc;
                       syn_decl = type_decl;
                       trait_path;
                       loc_ = trait_loc;
                     })
            | Error _ -> ());
        Local_diagnostics.add_to_global local_diagnostics diagnostics
    | Ptop_trait _ -> ()
    | Ptop_letdef { binder; ty; expr; is_constant; is_pub; loc_; doc_ } ->
        let local_diagnostics = Local_diagnostics.make ~base:loc_ in
        let binder_loc = Rloc.to_loc ~base:loc_ binder.loc_ in
        let check_toplevel_let (binder : Syntax.binder) (expr : Syntax.expr)
            (ty_opt : Syntax.typ option) (toplevel : toplevel_env) : unit =
          let add_letdef konstraint typ =
            let non_constant () =
              (Value_info.Normal, Wl_toplet_normal expr)
                [@@inline]
            in
            let (kind : Value_info.value_kind), expr =
              if is_constant then (
                (match
                   Typing_info.find_constructor toplevel binder.binder_name
                 with
                | constr :: _ ->
                    Diagnostics.add_error diagnostics
                      (Errors.constant_constr_duplicate ~name:binder.binder_name
                         ~constr_loc:constr.cs_loc_ ~const_loc:binder_loc)
                | [] -> ());
                let is_valid_constant_type (b : Stype.builtin) =
                  match b with
                  | T_unit | T_bool | T_byte | T_char | T_int | T_int64 | T_uint
                  | T_uint64 | T_float | T_double | T_string ->
                      true
                  | T_bytes -> false
                in
                match Stype.type_repr typ with
                | T_builtin b as typ when is_valid_constant_type b -> (
                    match expr with
                    | Pexpr_constant { c; loc_ } ->
                        let actual_ty, c =
                          Typeutil.typing_constant ~expect_ty:(Some typ) c
                            ~loc:loc_
                          |> take_info_partial ~diagnostics:local_diagnostics
                        in
                        Ctype.unify_expr ~expect_ty:typ ~actual_ty loc_
                        |> Typeutil.store_error ~diagnostics:local_diagnostics;
                        ( Const c,
                          Wl_toplet_const
                            (Texpr_constant { c; ty = typ; name_ = None; loc_ })
                        )
                    | _ ->
                        add_error local_diagnostics
                          (Errors.constant_not_constant
                             (Syntax.loc_of_expression expr));
                        non_constant ())
                | T_blackhole | Tvar { contents = Tnolink Tvar_error } ->
                    non_constant ()
                | _ ->
                    let loc =
                      match konstraint with
                      | Some konstraint -> Typedtree.loc_of_typ konstraint
                      | None -> Syntax.loc_of_expression expr
                    in
                    add_error local_diagnostics
                      (Errors.invalid_constant_type
                         ~ty:(Printer.type_to_string typ)
                         ~loc);
                    non_constant ())
              else non_constant ()
            in
            let id =
              add_symbol toplevel binder typ is_pub doc_ Tvar_env.empty ~kind
                ~arity:None ~param_names:[] ~binder_loc
            in
            Vec.push worklist
              (Wl_top_letdef
                 { binder; expr; is_pub; loc_; doc_; konstraint; id; typ })
              [@@inline]
          in
          match (ty_opt, expr) with
          | Some ty_expr, _ | None, Pexpr_constraint { ty = ty_expr; _ } ->
              let konstraint =
                typing_type ~allow_private:(not is_pub) ty_expr
                  ~tvar_env:Tvar_env.empty ~types ~diagnostics:local_diagnostics
              in
              add_letdef (Some konstraint)
                (Typedtree_util.stype_of_typ konstraint)
          | None, expr -> (
              match try_infer_expr ~types expr with
              | Some (ty, is_literal) ->
                  if (not is_literal) && is_pub then
                    add_error local_diagnostics
                      (Errors.let_missing_annot ~name:binder.binder_name
                         ~loc:binder.loc_ ~reason:`Pub_not_literal);
                  add_letdef None ty
              | None ->
                  add_error local_diagnostics
                    (Errors.let_missing_annot ~name:binder.binder_name
                       ~loc:binder.loc_ ~reason:`Cannot_infer);
                  add_letdef None Stype.blackhole)
        in
        let res = check_toplevel_let binder expr ty toplevel in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        res
    | Ptop_funcdef { fun_decl; decl_body; loc_ } ->
        let local_diagnostics = Local_diagnostics.make ~base:loc_ in
        let check_toplevel_fun (fun_decl : Syntax.fun_decl)
            (toplevel : toplevel_env) =
          let ({
                 type_name;
                 name;
                 has_error;
                 quantifiers;
                 return_type;
                 decl_params;
                 params_loc_;
                 is_pub;
                 doc_;
               }
                : Syntax.fun_decl) =
            fun_decl
          in
          let tvar_env =
            Typeutil.typing_func_def_tparam_binders ~allow_private:(not is_pub)
              ~types quantifiers
            |> Typeutil.take_info_partial ~diagnostics:local_diagnostics
          in
          let decl_params = Option.value ~default:[] decl_params in
          let params_typ =
            Lst.map decl_params (fun (p : Syntax.parameter) ->
                match p.param_annot with
                | None ->
                    add_error local_diagnostics
                      (Errors.missing_param_annot
                         ~name:p.param_binder.binder_name
                         ~loc:p.param_binder.loc_);
                    (Stype.blackhole, None)
                | Some ty ->
                    let typ =
                      typing_type ~allow_private:(not is_pub) ty ~tvar_env
                        ~types ~diagnostics:local_diagnostics
                    in
                    (Typedtree_util.stype_of_typ typ, Some typ))
          in
          let ret_sty, err_sty, ret_annotation =
            match return_type with
            | None -> (
                match decl_body with
                | Decl_body _ ->
                    add_error local_diagnostics
                      (Errors.missing_return_annot fun_decl.name.loc_);
                    (Stype.blackhole, None, No_error_type_annotated)
                | Decl_stubs _ -> (Stype.unit, None, No_error_type_annotated))
            | Some (res_ty, err_ty) ->
                let typ =
                  typing_type ~allow_private:(not is_pub) res_ty ~tvar_env
                    ~types ~diagnostics:local_diagnostics
                in
                let err_sty, err_typ =
                  typing_error_type ~allow_private:(not is_pub) ~types ~tvar_env
                    err_ty ~diagnostics:local_diagnostics ~has_error
                    ~header_loc:name.loc_
                in
                ( Typedtree_util.stype_of_typ typ,
                  err_sty,
                  Annotated (typ, err_typ) )
          in
          let typ_generic = not (quantifiers = []) in
          let fun_type : Stype.t =
            Tarrow
              {
                params_ty = List.map fst params_typ;
                ret_ty = ret_sty;
                err_ty = err_sty;
                generic_ = typ_generic;
              }
          in
          let arity =
            Fn_arity.from_params decl_params ~base:loc_
            |> take_info_partial ~diagnostics:local_diagnostics
          in
          let param_names =
            Lst.map decl_params (fun p -> p.param_binder.binder_name)
          in
          let prim = prim_of_decl decl_body loc_ ~doc:doc_ ~diagnostics in
          let kind, id =
            match type_name with
            | Some type_name -> (
                match
                  Typeutil.typing_type_name type_name
                    ~allow_private:(not is_pub) ~tvar_env:Tvar_env.empty ~types
                  |> take_info_partial ~diagnostics:local_diagnostics
                with
                | Some (Tname_param _, _) -> assert false
                | Some
                    ( (Tname_predef ty_constr | Tname_defined { ty_constr; _ }),
                      type_name ) ->
                    let meth_loc = Rloc.to_loc ~base:loc_ name.loc_ in
                    ( Typedtree.Fun_kind_method (Some type_name),
                      add_method ty_constr fun_type name is_pub ~is_trait:false
                        ~doc:doc_ ~types ~tvar_env ~method_env ~toplevel
                        ~diagnostics:local_diagnostics ~arity ~param_names ~prim
                        ~meth_loc )
                | Some (Tname_trait trait_decl, type_name) ->
                    let meth_loc = Rloc.to_loc ~base:loc_ name.loc_ in
                    ( Typedtree.Fun_kind_method (Some type_name),
                      add_method trait_decl.name fun_type name is_pub
                        ~is_trait:true ~doc:doc_ ~types ~tvar_env ~method_env
                        ~toplevel ~diagnostics:local_diagnostics ~arity
                        ~param_names ~prim ~meth_loc )
                | None ->
                    ( Typedtree.Fun_kind_regular,
                      Qual_ident.toplevel_value ~name:name.binder_name ))
            | None -> (
                match Typeutil.classify_func decl_params with
                | Method _ ->
                    let meth_loc = Rloc.to_loc ~base:loc_ name.loc_ in
                    let self_ty, _ = List.hd params_typ in
                    ( Typedtree.Fun_kind_method None,
                      add_self_method self_ty fun_type name is_pub ~doc:doc_
                        ~types ~tvar_env ~method_env ~toplevel
                        ~diagnostics:local_diagnostics ~arity ~param_names ~prim
                        ~meth_loc )
                | Regular_func ->
                    let binder_loc = Rloc.to_loc ~base:loc_ name.loc_ in
                    ( Typedtree.Fun_kind_regular,
                      add_symbol toplevel name fun_type is_pub doc_ tvar_env
                        ~arity:(Some arity)
                        ~kind:
                          (match prim with None -> Normal | Some p -> Prim p)
                        ~param_names ~binder_loc ))
          in
          let typed_fn_annotation =
            {
              params_ty = params_typ;
              ret_ty = ret_sty;
              err_ty = err_sty;
              ret_annotation;
            }
          in
          Vec.push worklist
            (Wl_top_funcdef
               {
                 fun_binder = name;
                 decl_params;
                 params_loc = params_loc_;
                 is_pub;
                 doc = doc_;
                 decl_body;
                 loc_;
                 id;
                 kind;
                 arity;
                 tvar_env;
                 typed_fn_annotation;
               })
        in
        let res = check_toplevel_fun fun_decl toplevel in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        res
    | Ptop_impl
        {
          self_ty = None;
          trait;
          method_name = method_binder;
          has_error;
          quantifiers;
          params;
          ret_ty;
          body;
          is_pub = _;
          local_types;
          header_loc_;
          loc_;
          doc_;
        } ->
        let local_diagnostics = Local_diagnostics.make ~base:loc_ in
        let method_name = method_binder.binder_name in
        let is_partial = method_name = "*" in
        let is_pub, trait_decl, type_name =
          match
            typing_trait_name ~types ~allow_private:true trait
              ~diagnostics:local_diagnostics
          with
          | Some trait_decl, type_name ->
              let is_pub =
                match trait_decl.vis_ with
                | Vis_fully_pub -> true
                | Vis_priv | Vis_default | Vis_readonly -> false
              in
              (is_pub, Some trait_decl, type_name)
          | None, type_name -> (false, None, type_name)
        in
        let method_decl =
          match trait_decl with
          | None -> None
          | Some trait_decl -> (
              match
                Trait_decl.find_method trait_decl method_name
                  ~loc:method_binder.loc_
              with
              | Ok meth_decl -> Some meth_decl
              | Error err ->
                  if not is_partial then add_error local_diagnostics err;
                  None)
        in
        let tvar_env =
          Typeutil.typing_func_def_tparam_binders ~allow_private:(not is_pub)
            ~types
            ({
               tvar_name = "Self";
               tvar_constraints =
                 (if Option.is_some trait_decl then
                    [ { tvc_trait = trait.name; loc_ = Rloc.no_location } ]
                  else []);
               loc_ = Rloc.no_location;
             }
            :: quantifiers)
          |> Typeutil.take_info_partial ~diagnostics:local_diagnostics
        in
        let arity =
          Fn_arity.from_params params ~base:loc_
          |> take_info_partial ~diagnostics:local_diagnostics
        in
        let params_typ =
          Lst.map params (fun p ->
              match p.param_annot with
              | None -> (Stype.new_type_var Tvar_normal, None)
              | Some ty ->
                  let typ =
                    typing_type ~allow_private:(not is_pub) ty ~tvar_env ~types
                      ~diagnostics:local_diagnostics
                  in
                  (Typedtree_util.stype_of_typ typ, Some typ))
        in
        let ret_sty, err_sty, ret_annotation =
          match ret_ty with
          | None ->
              let has_error_type =
                match method_decl with
                | Some method_decl -> (
                    match method_decl.method_typ with
                    | Tarrow { err_ty = Some _; _ } -> true
                    | _ -> false)
                | None -> false
              in
              ( Stype.new_type_var Tvar_normal,
                (if has_error_type then Some (Stype.new_type_var Tvar_normal)
                 else None),
                if has_error_type then Has_error_type header_loc_
                else No_error_type_annotated )
          | Some (res_ty, err_ty) ->
              let typ =
                typing_type ~allow_private:(not is_pub) res_ty ~tvar_env ~types
                  ~diagnostics:local_diagnostics
              in
              let err_sty, err_typ =
                typing_error_type ~allow_private:(not is_pub) err_ty ~tvar_env
                  ~types ~diagnostics:local_diagnostics ~has_error
                  ~header_loc:method_binder.loc_
              in
              ( Typedtree_util.stype_of_typ typ,
                err_sty,
                Annotated (typ, err_typ) )
        in
        let ty_func : Stype.t =
          Tarrow
            {
              params_ty = List.map fst params_typ;
              ret_ty = ret_sty;
              err_ty = err_sty;
              generic_ = true;
            }
        in
        let typed_fn_annotation =
          {
            params_ty = params_typ;
            ret_ty = ret_sty;
            err_ty = err_sty;
            ret_annotation;
          }
        in
        let add_to_worklist id =
          Vec.push worklist
            (Wl_top_funcdef
               {
                 fun_binder = method_binder;
                 decl_params = params;
                 params_loc = Rloc.no_location;
                 is_pub;
                 doc = doc_;
                 decl_body = Decl_body { expr = body; local_types };
                 loc_;
                 id;
                 kind = Fun_kind_default_impl type_name;
                 arity;
                 tvar_env;
                 typed_fn_annotation;
               })
            [@@inline]
        in
        let res =
          match trait_decl with
          | None ->
              add_to_worklist (Qual_ident.toplevel_value ~name:method_name)
          | Some trait_decl -> (
              let trait = trait_decl.name in
              let id =
                Qual_ident.ext_meth ~trait
                  ~self_typ:Type_path.Builtin.default_impl_placeholder
                  ~name:method_name
              in
              add_to_worklist id;
              match
                Trait_decl.find_method trait_decl method_name
                  ~loc:method_binder.loc_
              with
              | Error err -> add_error local_diagnostics err
              | Ok meth_decl -> (
                  let exception Arity_mismatch in
                  (try
                     Ctype.unify_exn meth_decl.method_typ ty_func;
                     if not (Fn_arity.equal arity meth_decl.method_arity) then
                       raise_notrace Arity_mismatch
                   with _ ->
                     add_error local_diagnostics
                       (Errors.ext_method_type_mismatch ~trait ~method_name
                          ~expected:
                            (Printer.toplevel_function_type_to_string
                               ~arity:meth_decl.method_arity
                               meth_decl.method_typ)
                          ~actual:
                            (Printer.toplevel_function_type_to_string ~arity
                               ty_func)
                          ~loc:method_binder.loc_));
                  if Type_path_util.is_foreign trait then
                    add_error local_diagnostics
                      (Errors.default_method_on_foreign ~trait
                         ~loc:method_binder.loc_)
                  else
                    match
                      Ext_method_env.find_method ext_method_env ~trait
                        ~self_type:Type_path.Builtin.default_impl_placeholder
                        ~method_name
                    with
                    | Some mi ->
                        add_error local_diagnostics
                          (Errors.default_method_duplicate ~trait ~method_name
                             ~first_loc:mi.loc ~second_loc:method_binder.loc_)
                    | None ->
                        let method_info : Ext_method_env.method_info =
                          {
                            id;
                            prim = None;
                            typ = ty_func;
                            pub = is_pub;
                            loc = loc_;
                            doc_;
                            ty_params_ = tvar_env;
                            arity_ = arity;
                            param_names_ =
                              Lst.map params (fun p ->
                                  p.param_binder.binder_name);
                          }
                        in
                        Ext_method_env.add_method ext_method_env ~trait
                          ~self_type:Type_path.Builtin.default_impl_placeholder
                          ~method_name method_info))
        in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        res
    | Ptop_impl
        {
          self_ty = Some self_ty;
          trait;
          method_name = method_binder;
          has_error;
          quantifiers;
          params;
          ret_ty;
          body;
          is_pub;
          local_types;
          header_loc_;
          loc_;
          doc_;
        } ->
        let local_diagnostics = Local_diagnostics.make ~base:loc_ in
        let method_name = method_binder.binder_name in
        let tvar_env =
          Typeutil.typing_func_def_tparam_binders ~allow_private:(not is_pub)
            ~types quantifiers
          |> Typeutil.take_info_partial ~diagnostics:local_diagnostics
        in
        let self_typ =
          typing_type ~allow_private:(not is_pub) self_ty ~tvar_env ~types
            ~diagnostics:local_diagnostics
        in
        let arity =
          Fn_arity.from_params params ~base:loc_
          |> take_info_partial ~diagnostics:local_diagnostics
        in
        let params_typ =
          Lst.map params (fun p ->
              match p.param_annot with
              | None -> (Stype.new_type_var Tvar_normal, None)
              | Some ty ->
                  let typ =
                    typing_type ~allow_private:(not is_pub) ty ~tvar_env ~types
                      ~diagnostics:local_diagnostics
                  in
                  (Typedtree_util.stype_of_typ typ, Some typ))
        in
        let ret_sty, err_sty, ret_annotation =
          match ret_ty with
          | None ->
              let has_error_typ =
                match
                  Global_env.All_types.find_trait types trait.name
                    ~loc:trait.loc_
                with
                | Error _ -> false
                | Ok trait_decl -> (
                    match
                      Trait_decl.find_method trait_decl method_name
                        ~loc:method_binder.loc_
                    with
                    | Ok method_decl -> (
                        match method_decl.method_typ with
                        | Tarrow { err_ty = Some _; _ } -> true
                        | _ -> false)
                    | Error _ -> false)
              in
              ( Stype.new_type_var Tvar_normal,
                (if has_error_typ then Some (Stype.new_type_var Tvar_normal)
                 else None),
                if has_error_typ then Has_error_type header_loc_
                else No_error_type_annotated )
          | Some (res_ty, err_ty) ->
              let typ =
                typing_type ~allow_private:(not is_pub) res_ty ~tvar_env ~types
                  ~diagnostics:local_diagnostics
              in
              let err_sty, err_typ =
                typing_error_type ~allow_private:(not is_pub) ~types ~tvar_env
                  err_ty ~diagnostics:local_diagnostics ~has_error
                  ~header_loc:method_binder.loc_
              in
              ( Typedtree_util.stype_of_typ typ,
                err_sty,
                Annotated (typ, err_typ) )
        in
        let ty_func : Stype.t =
          Tarrow
            {
              params_ty = List.map fst params_typ;
              ret_ty = ret_sty;
              err_ty = err_sty;
              generic_ = true;
            }
        in
        let typed_fn_annotation =
          {
            params_ty = params_typ;
            ret_ty = ret_sty;
            err_ty = err_sty;
            ret_annotation;
          }
        in
        let add_to_worklist id type_name =
          Vec.push worklist
            (Wl_top_funcdef
               {
                 fun_binder = method_binder;
                 decl_params = params;
                 params_loc = Rloc.no_location;
                 is_pub;
                 doc = doc_;
                 decl_body = Decl_body { expr = body; local_types };
                 loc_;
                 id;
                 kind = Fun_kind_impl { self_ty = self_typ; trait = type_name };
                 arity;
                 tvar_env;
                 typed_fn_annotation;
               })
            [@@inline]
        in
        let res =
          match
            typing_trait_name ~types ~allow_private:(not is_pub) trait
              ~diagnostics:local_diagnostics
          with
          | None, type_name ->
              add_to_worklist
                (Qual_ident.toplevel_value ~name:method_name)
                type_name
          | ( Some ({ vis_ = Vis_default | Vis_readonly; _ } as trait_decl),
              type_name )
            when Type_path_util.is_foreign trait_decl.name ->
              let trait_vis =
                match trait_decl.vis_ with
                | Vis_default -> "abstract"
                | Vis_readonly -> "readonly"
                | Vis_priv | Vis_fully_pub -> assert false
              in
              add_error local_diagnostics
                (Errors.cannot_implement_sealed_trait
                   ~trait:(Type_path_util.name trait_decl.name)
                   ~trait_vis ~loc:trait.loc_);
              add_to_worklist
                (Qual_ident.toplevel_value ~name:method_name)
                type_name
          | Some trait_decl, type_name ->
              let meth_loc = Rloc.to_loc ~base:loc_ method_binder.loc_ in
              let header_loc =
                Loc.merge loc_
                  (Rloc.to_loc ~base:loc_ (Typedtree.loc_of_typ self_typ))
              in
              let id =
                add_ext_method trait_decl ty_func method_binder is_pub
                  ~self_ty:(Typedtree_util.stype_of_typ self_typ)
                  ~doc:doc_ ~types ~tvar_env ~ext_method_env ~trait_impls ~arity
                  ~param_names:
                    (Lst.map params (fun p -> p.param_binder.binder_name))
                  ~prim:None ~diagnostics:local_diagnostics
                  ~global_diagnostics:diagnostics ~meth_loc ~header_loc
              in
              add_to_worklist id type_name
        in
        Local_diagnostics.add_to_global local_diagnostics diagnostics;
        res
    | Ptop_impl_relation _ -> assert false
  in
  Lst.iter impls go;
  match build_context with
  | Exec { is_main_loc } when !main_loc = None ->
      add_global_error diagnostics (Errors.missing_main ~loc:is_main_loc)
  | _ -> ()

let check_traits_implemented ~global_env ~diagnostics =
  let check_method ~trait ~(trait_vis : Typedecl_info.visibility) ~self_typ
      ~(request : Trait_impl.impl) (method_decl : Trait_decl.method_decl) =
    let method_name = method_decl.method_name in
    let method_impl =
      Global_env.find_trait_method global_env ~trait ~type_name:self_typ
        ~method_name
    in
    (match method_impl with
    | Some ({ id = Qext_method _; _ } as method_info) ->
        Method_env.add_impl
          (Global_env.get_method_env global_env)
          ~type_name:self_typ ~method_name ~method_info
    | _ -> ());
    match method_impl with
    | None -> Some (`Method_missing method_name)
    | Some { id = Qext_method { self_typ; _ }; _ }
      when Type_path.equal self_typ Type_path.Builtin.default_impl_placeholder
      ->
        None
    | Some { pub = false; _ } when request.is_pub ->
        Some (`Private_method method_name)
    | Some { id = Qext_method _; pub = false; loc; _ } ->
        (match trait_vis with
        | Vis_fully_pub -> (
            match
              Global_env.find_regular_method global_env ~type_name:self_typ
                ~method_name
            with
            | Some { pub = true; loc = prev_loc; _ } ->
                add_global_error diagnostics
                  (Errors.priv_ext_shadows_pub_method ~method_name ~trait
                     ~type_name:self_typ ~prev_loc ~loc)
            | _ -> ())
        | _ -> ());
        None
    | Some method_info ->
        let expected =
          Poly_type.instantiate_method_decl method_decl ~self:request.self_ty
        in
        let cenv = Poly_type.make () in
        let actual, _ = Poly_type.instantiate_method ~cenv method_info in
        if Ctype.try_unify expected actual then (
          let aux_diagnostics = Local_diagnostics.make ~base:Loc.no_location in
          Type_constraint.solve_constraints cenv ~tvar_env:request.ty_params
            ~global_env ~diagnostics:aux_diagnostics;
          if Local_diagnostics.has_fatal_errors aux_diagnostics then
            Some (`Method_constraint_not_satisfied method_name)
          else None)
        else
          Some
            (`Type_mismatch
              ( method_name,
                Printer.type_to_string expected,
                Printer.type_to_string method_info.typ ))
      [@@inline]
  in
  Trait_impl.iter (Global_env.get_trait_impls global_env)
    (fun ~trait ~type_name request ->
      match Global_env.find_trait_by_path global_env trait with
      | None -> ()
      | Some trait_decl -> (
          let failure_reasons =
            Lst.fold_right trait_decl.methods [] (fun meth_decl acc ->
                match
                  check_method ~trait ~trait_vis:trait_decl.vis_
                    ~self_typ:type_name ~request meth_decl
                with
                | Some err -> err :: acc
                | None -> acc)
          in
          match failure_reasons with
          | [] -> ()
          | failure_reasons ->
              add_global_error diagnostics
                (trait_not_implemented ~trait ~type_name ~failure_reasons
                   ~loc:request.loc_)))

type output = {
  global_env : Global_env.t;
  values : value_worklist_item Vec.t;
  type_decls : Typedtree.type_decl list;
  trait_decls : Typedtree.trait_decl list;
}

let check_toplevel ?(pkgs : Pkg.pkg_tbl option) ~diagnostics
    ?(build_context : Typeutil.build_context = SingleFile) list_of_impls :
    output =
  let pkgs = match pkgs with Some pkgs -> pkgs | None -> Pkg.create_tbl () in
  let impls = List.concat list_of_impls in
  let builtin_types = Builtin.builtin_types in
  let toplevel_types = Typing_info.make_types () in
  let type_alias = Hash_string.create 17 in
  let types =
    Global_env.All_types.make ~toplevel:toplevel_types ~builtin:builtin_types
      ~type_alias ~pkgs
  in
  let toplevel_values = Typing_info.make_values () in
  Pkg.load_direct_uses pkgs toplevel_values type_alias ~diagnostics;
  let worklist = Vec.empty () in
  let type_decls, trait_decls =
    typing_types_and_traits impls types toplevel_values ~diagnostics
  in
  let method_env = Method_env.empty () in
  let ext_method_env = Ext_method_env.empty () in
  let trait_impls = Trait_impl.make () in
  check_toplevel_decl impls types ~method_env ~toplevel:toplevel_values
    ~ext_method_env ~trait_impls ~worklist ~diagnostics ~build_context;
  let builtin_values = Builtin.builtin_values in
  let global_env =
    Global_env.make ~types ~builtin:builtin_values ~toplevel:toplevel_values
      ~method_env ~ext_method_env ~trait_impls
  in
  check_traits_implemented ~global_env ~diagnostics;
  { global_env; values = worklist; type_decls; trait_decls }
