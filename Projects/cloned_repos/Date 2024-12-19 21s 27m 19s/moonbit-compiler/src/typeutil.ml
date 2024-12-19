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


type typ = Stype.t

module Ident = Basic_ident
module Lst = Basic_lst
module Hashset_string = Basic_hashset_string
module Longident = Basic_longident
module Type_path = Basic_type_path
module Syntax = Parsing_syntax
module UInt32 = Basic_uint32
module UInt64 = Basic_uint64
module Bigint = Basic_bigint

type build_context = Exec of { is_main_loc : Loc.t } | Lib | SingleFile

module Loc = Rloc

let type_of_constant (c : Syntax.constant) =
  match c with
  | Const_bool _ -> Stype.bool
  | Const_byte _ -> Stype.byte
  | Const_bytes _ -> Stype.bytes
  | Const_char _ -> Stype.char
  | Const_int _ -> Stype.int
  | Const_int64 _ -> Stype.int64
  | Const_double _ -> Stype.double
  | Const_string _ -> Stype.string
  | Const_uint _ -> Stype.uint
  | Const_uint64 _ -> Stype.uint64
  | Const_bigint _ -> Stype.bigint

let unknown_tag : Typedtree.constr_tag =
  Constr_tag_regular
    { total = -1; index = -1; name_ = ""; is_constant_ = false }

let unknown_pos : int = -1
let default_err_type = Stype.error

let default_err_typedtree_type =
  Typedtree.Tname
    {
      constr = { lid = Lident "string"; loc_ = Loc.no_location };
      params = [];
      ty = default_err_type;
      is_alias_ = false;
      loc_ = Loc.no_location;
    }

let typing_type ~(is_toplevel : bool) ~(allow_private : bool)
    ?(placeholder_env : Placeholder_env.t option) (te : Syntax.typ)
    ~(tvar_env : Tvar_env.t) ~(types : Global_env.All_types.t) :
    Typedtree.typ Local_diagnostics.partial_info =
  let errors = ref [] in
  let check_arity ~kind id expected ~loc
      (tys : [ `Wildcard | `Fixed of Stype.t list ]) =
    match tys with
    | `Wildcard ->
        if expected = 0 then
          errors :=
            Errors.type_constr_arity_mismatch ~kind ~id ~expected ~actual:1 ~loc
            :: !errors;
        if is_toplevel then List.init expected (fun _ -> Stype.blackhole)
        else Type.type_var_list expected Tvar_normal
    | `Fixed tys ->
        let actual = List.length tys in
        if expected <> actual then
          errors :=
            Errors.type_constr_arity_mismatch ~kind ~id ~expected ~actual ~loc
            :: !errors;
        if expected = actual then tys
        else if expected < actual then Lst.take expected tys
        else if is_toplevel then
          tys @ List.init (expected - actual) (fun _ -> Stype.blackhole)
        else tys @ Type.type_var_list (expected - actual) Tvar_error
      [@@inline]
  in
  let rec go (te : Syntax.typ) : typ * Typedtree.typ * bool =
    match te with
    | Ptype_any { loc_ } ->
        if is_toplevel then
          errors := Errors.unexpected_partial_type loc_ :: !errors;
        let ty =
          if is_toplevel then Stype.blackhole
          else Stype.new_type_var Tvar_normal
        in
        (ty, Tany { ty; loc_ }, false)
    | Ptype_arrow { ty_arg; ty_res; ty_err; loc_ } ->
        let params_ty, params, params_is_generic = go_list ty_arg in
        let ret_ty, return, ret_is_generic = go ty_res in
        let err_ty, err, err_is_generic =
          match ty_err with
          | Error_typ { ty = ty_err } ->
              let err_ty, err, err_is_generic = go ty_err in
              if not (Type.is_error_type ~tvar_env err_ty) then
                errors :=
                  Errors.not_error_subtype (Printer.type_to_string err_ty) loc_
                  :: !errors;
              (Some err_ty, Typedtree.Error_typ { ty = err }, err_is_generic)
          | Default_error_typ { loc_ } ->
              (Some default_err_type, Default_error_typ { loc_ }, false)
          | No_error_typ -> (None, No_error_typ, false)
        in
        let generic_ = params_is_generic || ret_is_generic || err_is_generic in
        let ty : Stype.t = Tarrow { params_ty; ret_ty; err_ty; generic_ } in
        (ty, Tarrow { ty; params; return; err_ty = err; loc_ }, generic_)
    | Ptype_name { constr_id; tys; loc_ } -> (
        let tys, params, generic_ = go_list tys in
        let make_typ ty ~is_alias ~generic_ =
          ( ty,
            Typedtree.Tname
              { ty; constr = constr_id; params; is_alias_ = is_alias; loc_ },
            generic_ )
            [@@inline]
        in
        let tys =
          match params with Tany _ :: [] -> `Wildcard | _ -> `Fixed tys
        in
        match constr_id.lid with
        | Lident id -> (
            match Tvar_env.find_by_name tvar_env id with
            | Some tvar_info ->
                let _ =
                  check_arity ~kind:"type parameter" constr_id.lid 0 tys
                    ~loc:loc_
                in
                make_typ ~generic_:true ~is_alias:false tvar_info.typ
            | None ->
                let ty, is_alias = go_constr ~generic_ constr_id.lid tys loc_ in
                make_typ ~generic_ ~is_alias ty)
        | _ ->
            let ty, is_alias = go_constr ~generic_ constr_id.lid tys loc_ in
            make_typ ~generic_ ~is_alias ty)
    | Ptype_tuple { tys; loc_ } ->
        let tys, params, generic_ = go_list tys in
        let ty : Stype.t =
          T_constr
            {
              type_constructor = Type_path.tuple (List.length tys);
              tys;
              generic_;
              only_tag_enum_ = false;
              is_suberror_ = false;
            }
        in
        (ty, T_tuple { ty; params; loc_ }, generic_)
    | Ptype_option { ty; loc_; question_loc } ->
        let tys, params, generic_ = go_list [ ty ] in
        let ty : Stype.t =
          T_constr
            {
              type_constructor = Type_path.Builtin.type_path_option;
              tys;
              generic_;
              only_tag_enum_ = false;
              is_suberror_ = false;
            }
        in
        ( ty,
          Tname
            {
              params;
              ty;
              loc_;
              constr = { lid = Lident "Option"; loc_ = question_loc };
              is_alias_ = false;
            },
          generic_ )
  and go_list tys =
    match tys with
    | [] -> ([], [], false)
    | ty :: tys ->
        let sty, typ, ty_is_generic = go ty in
        let stys, typs, tys_is_generic = go_list tys in
        (sty :: stys, typ :: typs, ty_is_generic || tys_is_generic)
  and go_constr ~generic_ (constr_id : Longident.t) tys (loc_ : Loc.t) :
      Stype.t * bool =
    let exception
      Found_type of {
        arity : int;
        type_constructor : Type_path.t;
        only_tag_enum_ : bool;
        is_private : bool;
        is_suberror_ : bool;
      }
    in
    let exception Found_predef of Stype.t in
    let exception
      Found_trait of {
        trait : Type_path.t;
        is_private : bool;
        object_safety_status : Trait_decl.object_safety_status;
      }
    in
    let exception Found_alias of Typedecl_info.alias in
    try
      (match Global_env.All_types.find_type_alias types constr_id with
      | None -> ()
      | Some alias -> raise_notrace (Found_alias alias));
      (match (placeholder_env, constr_id) with
      | Some placeholder_env, Lident name -> (
          match Placeholder_env.find_type_opt placeholder_env name with
          | Some decl ->
              let type_constructor =
                Type_path.toplevel_type
                  ~pkg:!Basic_config.current_package
                  decl.tycon
              in
              let only_tag_enum_ =
                match decl.components with
                | Ptd_variant cs ->
                    Lst.for_all cs (fun { constr_args; _ } ->
                        constr_args = None)
                | _ -> false
              in
              let is_suberror_ =
                match decl.components with Ptd_error _ -> true | _ -> false
              in
              let arity = List.length decl.params in
              raise_notrace
                (Found_type
                   {
                     arity;
                     type_constructor;
                     only_tag_enum_;
                     is_private =
                       (match decl.type_vis with
                       | Vis_priv _ -> true
                       | _ -> false);
                     is_suberror_;
                   })
          | None -> (
              match Placeholder_env.find_trait_opt placeholder_env name with
              | Some { decl = { trait_vis; _ }; object_safety_status; _ } ->
                  let trait =
                    Type_path.toplevel_type
                      ~pkg:!Basic_config.current_package
                      name
                  in
                  let is_private =
                    match trait_vis with Vis_priv _ -> true | _ -> false
                  in
                  raise_notrace
                    (Found_trait { trait; is_private; object_safety_status })
              | None -> ()))
      | _ -> ());
      (match
         Global_env.All_types.find_type_or_trait types constr_id ~loc:loc_
       with
      | `Type
          {
            ty_arity;
            ty_constr;
            ty_is_only_tag_enum_;
            ty_vis;
            ty_is_suberror_;
            _;
          } ->
          raise_notrace
            (Found_type
               {
                 arity = ty_arity;
                 type_constructor = ty_constr;
                 only_tag_enum_ = ty_is_only_tag_enum_;
                 is_private = ty_vis = Vis_priv;
                 is_suberror_ = ty_is_suberror_;
               })
      | `Trait trait_decl ->
          raise_notrace
            (Found_trait
               {
                 trait = trait_decl.name;
                 is_private = trait_decl.vis_ = Vis_priv;
                 object_safety_status = trait_decl.object_safety_;
               })
      | `Error _ -> ());
      (match constr_id with
      | Lident "Unit" -> raise_notrace (Found_predef Stype.unit)
      | Lident "Bool" -> raise_notrace (Found_predef Stype.bool)
      | Lident "Byte" -> raise_notrace (Found_predef Stype.byte)
      | Lident "Char" -> raise_notrace (Found_predef Stype.char)
      | Lident "Int" -> raise_notrace (Found_predef Stype.int)
      | Lident "Int64" -> raise_notrace (Found_predef Stype.int64)
      | Lident "UInt" -> raise_notrace (Found_predef Stype.uint)
      | Lident "UInt64" -> raise_notrace (Found_predef Stype.uint64)
      | Lident "Float" -> raise_notrace (Found_predef Stype.float)
      | Lident "Double" -> raise_notrace (Found_predef Stype.double)
      | Lident "String" -> raise_notrace (Found_predef Stype.string)
      | Lident "Bytes" -> raise_notrace (Found_predef Stype.bytes)
      | _ -> ());
      let error_message =
        match constr_id with
        | Ldot { pkg; id = _ }
          when Option.is_none
                 (Pkg.find_pkg_opt
                    (Global_env.All_types.get_pkg_tbl types)
                    ~pkg) ->
            Errors.pkg_not_loaded ~pkg ~loc:loc_
        | _ -> Errors.unbound_type ~name:constr_id ~loc:loc_
      in
      errors := error_message :: !errors;
      ( (if is_toplevel then Stype.blackhole else Stype.new_type_var Tvar_error),
        false )
    with
    | Found_type
        { arity; type_constructor; only_tag_enum_; is_private; is_suberror_ } ->
        if (not allow_private) && is_private then
          errors :=
            Errors.cannot_depend_private ~entity:"type" ~loc:loc_ :: !errors;
        let tys =
          check_arity ~kind:"type constructor" constr_id arity tys ~loc:loc_
        in
        ( T_constr
            { type_constructor; tys; generic_; only_tag_enum_; is_suberror_ },
          false )
    | Found_predef ty ->
        let _ =
          check_arity ~kind:"type constructor" constr_id 0 tys ~loc:loc_
        in
        (ty, false)
    | Found_trait { trait; is_private; object_safety_status } ->
        if (not allow_private) && is_private then
          errors :=
            Errors.cannot_depend_private ~entity:"trait" ~loc:loc_ :: !errors;
        let _ = check_arity ~kind:"trait" constr_id 0 tys ~loc:loc_ in
        (match
           Trait_decl.check_object_safety
             ~name:(Type_path_util.name trait)
             ~loc:loc_ object_safety_status
         with
        | None -> ()
        | Some err -> errors := err :: !errors);
        (T_trait trait, false)
    | Found_alias ({ arity; _ } as alias) ->
        let tys =
          check_arity ~kind:"type alias" constr_id arity tys ~loc:loc_
        in
        let exception Alias_mentions_priv_type of Type_path.t in
        let rec check_priv (ty : Stype.t) =
          let check_path (p : Type_path.t) =
            match p with
            | Toplevel { pkg; id } when pkg = !Basic_config.current_package -> (
                match placeholder_env with
                | Some placeholder_env -> (
                    match Placeholder_env.find_type_opt placeholder_env id with
                    | None -> ()
                    | Some decl -> (
                        match decl.type_vis with
                        | Vis_priv _ ->
                            raise_notrace (Alias_mentions_priv_type p)
                        | Vis_pub _ | Vis_default -> ()))
                | None -> (
                    match Global_env.All_types.find_type_by_path types p with
                    | None -> ()
                    | Some decl ->
                        if decl.ty_vis = Vis_priv then
                          raise_notrace (Alias_mentions_priv_type p)))
            | _ -> ()
              [@@inline]
          in
          match Stype.type_repr ty with
          | T_constr { type_constructor = p; tys } ->
              check_path p;
              Lst.iter tys check_priv
          | T_trait trait -> check_path trait
          | Tarrow { params_ty; ret_ty; err_ty } -> (
              Lst.iter params_ty check_priv;
              check_priv ret_ty;
              match err_ty with None -> () | Some err_ty -> check_priv err_ty)
          | Tvar _ | Tparam _ | T_blackhole | T_builtin _ -> ()
        in
        (match constr_id with
        | Lident name -> (
            if (not allow_private) && not alias.is_pub then
              try check_priv alias.alias
              with Alias_mentions_priv_type p ->
                errors :=
                  Errors.alias_with_priv_target_in_pub_sig ~alias:name
                    ~priv_type:(Type_path_util.name p) ~loc:loc_
                  :: !errors)
        | Ldot _ -> ());
        (Poly_type.instantiate_alias alias tys, true)
  in
  let _, typ, _ = go te in
  match !errors with [] -> Ok typ | errs -> Partial (typ, errs)

type type_name =
  | Tname_param of { index : int; name_ : string }
  | Tname_predef of Type_path.t
  | Tname_defined of Typedecl_info.t
  | Tname_trait of Trait_decl.t

let typing_type_name ~(allow_private : bool) (type_name : Syntax.type_name)
    ~(tvar_env : Tvar_env.t) ~(types : Global_env.All_types.t) :
    (type_name * Typedtree.type_name) option Local_diagnostics.partial_info =
  let loc = type_name.loc_ in
  let resolve_as_constr () : _ Local_diagnostics.partial_info =
    match Global_env.All_types.find_type_or_trait types type_name.name ~loc with
    | `Type ty_decl ->
        let type_name : Typedtree.type_name =
          Tname_path
            { name = ty_decl.ty_constr; kind = Type; loc_ = type_name.loc_ }
        in
        let result = Some (Tname_defined ty_decl, type_name) in
        if (not allow_private) && ty_decl.ty_vis = Vis_priv then
          Partial (result, [ Errors.cannot_depend_private ~entity:"type" ~loc ])
        else Ok result
    | `Trait trait ->
        let type_name : Typedtree.type_name =
          Tname_path { name = trait.name; kind = Trait; loc_ = type_name.loc_ }
        in
        let result = Some (Tname_trait trait, type_name) in
        if (not allow_private) && trait.vis_ = Vis_priv then
          Partial (result, [ Errors.cannot_depend_private ~entity:"trait" ~loc ])
        else Ok result
    | `Error err -> (
        let resolve_as_predef p : _ Local_diagnostics.partial_info =
          let type_name : Typedtree.type_name =
            Tname_path { name = p; kind = Type; loc_ = type_name.loc_ }
          in
          Ok (Some (Tname_predef p, type_name))
            [@@inline]
        in
        match type_name.name with
        | Lident "Unit" -> resolve_as_predef Type_path.Builtin.type_path_unit
        | Lident "Bool" -> resolve_as_predef Type_path.Builtin.type_path_bool
        | Lident "Byte" -> resolve_as_predef Type_path.Builtin.type_path_byte
        | Lident "Char" -> resolve_as_predef Type_path.Builtin.type_path_char
        | Lident "Int" -> resolve_as_predef Type_path.Builtin.type_path_int
        | Lident "Int64" -> resolve_as_predef Type_path.Builtin.type_path_int64
        | Lident "UInt" -> resolve_as_predef Type_path.Builtin.type_path_uint
        | Lident "UInt64" ->
            resolve_as_predef Type_path.Builtin.type_path_uint64
        | Lident "Float" -> resolve_as_predef Type_path.Builtin.type_path_float
        | Lident "Double" ->
            resolve_as_predef Type_path.Builtin.type_path_double
        | Lident "String" ->
            resolve_as_predef Type_path.Builtin.type_path_string
        | Lident "Bytes" -> resolve_as_predef Type_path.Builtin.type_path_bytes
        | _ -> Partial (None, [ err ]))
  in
  match Global_env.All_types.find_type_alias types type_name.name with
  | Some alias -> (
      match Stype.type_repr alias.alias with
      | T_constr { type_constructor = p; _ } -> (
          match Global_env.All_types.find_type_by_path types p with
          | Some decl ->
              let type_name : Typedtree.type_name =
                Tname_alias
                  {
                    name = p;
                    kind = Type;
                    alias_ = type_name.name;
                    loc_ = type_name.loc_;
                  }
              in
              let result = Some (Tname_defined decl, type_name) in
              if (not allow_private) && decl.ty_vis = Vis_priv then
                Partial
                  (result, [ Errors.cannot_depend_private ~entity:"type" ~loc ])
              else Ok result
          | None -> Ok None)
      | T_builtin predef ->
          let p = Stype.tpath_of_builtin predef in
          Ok
            (Some
               ( Tname_predef p,
                 Tname_alias
                   {
                     name = p;
                     kind = Type;
                     alias_ = type_name.name;
                     loc_ = type_name.loc_;
                   } ))
      | T_trait trait -> (
          match Global_env.All_types.find_trait_by_path types trait with
          | Some decl ->
              let type_name : Typedtree.type_name =
                Tname_alias
                  {
                    name = trait;
                    kind = Trait;
                    alias_ = type_name.name;
                    loc_ = type_name.loc_;
                  }
              in
              let result = Some (Tname_trait decl, type_name) in
              if (not allow_private) && decl.vis_ = Vis_priv then
                Partial
                  (result, [ Errors.cannot_depend_private ~entity:"trait" ~loc ])
              else Ok result
          | None -> Ok None)
      | Tarrow _ ->
          Partial
            ( None,
              [
                Errors.type_alias_not_a_constructor
                  ~alias_name:(Longident.to_string type_name.name)
                  ~loc:type_name.loc_;
              ] )
      | T_blackhole -> Ok None
      | Tparam _ | Tvar _ -> assert false)
  | None -> (
      match type_name.name with
      | Lident name -> (
          match Tvar_env.find_by_name tvar_env name with
          | Some tvar_info -> (
              match tvar_info.typ with
              | Tparam { index; name_ } ->
                  let type_name : Typedtree.type_name =
                    Tname_tvar { index; name_; loc_ = type_name.loc_ }
                  in
                  Ok (Some (Tname_param { index; name_ }, type_name))
              | _ -> assert false)
          | None -> resolve_as_constr ())
      | Ldot _ -> resolve_as_constr ())

let typing_trait_name ~types ~allow_private (trait : Syntax.type_name) :
    _ Local_diagnostics.partial_info =
  let fallback_type_name () : Typedtree.type_name =
    Tname_path
      {
        name = Type_path.toplevel_type ~pkg:!Basic_config.current_package "";
        kind = Trait;
        loc_ = trait.loc_;
      }
  in
  let error ?type_name err : _ Local_diagnostics.partial_info =
    let type_name =
      match type_name with
      | None -> fallback_type_name ()
      | Some type_name -> type_name
    in
    Partial ((None, type_name), [ err ])
      [@@local]
  in
  match Global_env.All_types.find_type_alias types trait.name with
  | Some alias -> (
      match Stype.type_repr alias.alias with
      | T_trait trait_path -> (
          match Global_env.All_types.find_trait_by_path types trait_path with
          | None -> Ok (None, fallback_type_name ())
          | Some decl ->
              let type_name : Typedtree.type_name =
                Tname_alias
                  {
                    name = trait_path;
                    kind = Trait;
                    alias_ = trait.name;
                    loc_ = trait.loc_;
                  }
              in
              let result = (Some decl, type_name) in
              if (not allow_private) && decl.vis_ = Vis_priv then
                Partial
                  ( result,
                    [
                      Errors.cannot_depend_private ~entity:"trait"
                        ~loc:trait.loc_;
                    ] )
              else Ok result)
      | T_constr { type_constructor = p; _ } ->
          let type_name : Typedtree.type_name =
            Tname_alias
              { name = p; kind = Type; alias_ = trait.name; loc_ = trait.loc_ }
          in
          error ~type_name
            (Errors.not_a_trait
               ~name:(Basic_longident.to_string trait.name)
               ~loc:trait.loc_)
      | _ ->
          error
            (Errors.not_a_trait
               ~name:(Basic_longident.to_string trait.name)
               ~loc:trait.loc_))
  | None -> (
      match
        Global_env.All_types.find_trait types trait.name ~loc:trait.loc_
      with
      | Ok decl ->
          let type_name : Typedtree.type_name =
            Tname_path { name = decl.name; kind = Trait; loc_ = trait.loc_ }
          in
          let result = (Some decl, type_name) in
          if (not allow_private) && decl.vis_ = Vis_priv then
            Partial
              ( result,
                [ Errors.cannot_depend_private ~entity:"trait" ~loc:trait.loc_ ]
              )
          else Ok result
      | Error err -> error err)

let typing_type_decl_binders (binders : Syntax.type_decl_binder list) :
    Tvar_env.t =
  let go index (tvb : Syntax.type_decl_binder) =
    let name =
      match tvb.tvar_name with Some tvar_name -> tvar_name | None -> "_"
    in
    let typ : Stype.t = Tparam { index; name_ = name } in
    Tvar_env.tparam_info ~name ~typ ~constraints:[] ~loc:tvb.loc_
  in
  Tvar_env.of_list_mapi binders go

let typing_func_def_tparam_binders ~(allow_private : bool)
    ~(types : Global_env.All_types.t) (binders : Syntax.tvar_binder list) :
    Tvar_env.t Local_diagnostics.partial_info =
  let errors : Local_diagnostics.report list ref = ref [] in
  let go index tvar_name tvar_constraints loc : Tvar_env.tparam_info =
    let name = tvar_name in
    let typ : Stype.t = Tparam { index; name_ = name } in
    let constraints =
      Lst.map tvar_constraints (fun tvc : Tvar_env.type_constraint ->
          let Syntax.{ tvc_trait; loc_ } = tvc in
          match tvc_trait with
          | Lident "Error" ->
              {
                trait = Type_path.Builtin.type_path_error;
                loc_;
                required_by_ = [];
              }
          | _ -> (
              let trait_decl =
                match
                  typing_trait_name ~types ~allow_private
                    { name = tvc_trait; loc_ }
                with
                | Ok (trait_decl, _) -> trait_decl
                | Partial ((trait_decl, _), errs) ->
                    errors := errs @ !errors;
                    trait_decl
              in
              match trait_decl with
              | Some trait -> { trait = trait.name; loc_; required_by_ = [] }
              | None ->
                  let trait =
                    match tvc_trait with
                    | Lident name ->
                        Type_path.toplevel_type
                          ~pkg:!Basic_config.current_package
                          name
                    | Ldot { pkg; id } -> Type_path.toplevel_type ~pkg id
                  in
                  { trait; loc_; required_by_ = [] }))
    in
    let constraints = Trait_closure.compute_closure ~types constraints in
    Tvar_env.tparam_info ~name ~typ ~constraints ~loc
  in
  let go_with_constraints index (tvb : Syntax.tvar_binder) =
    go index tvb.tvar_name tvb.tvar_constraints tvb.loc_
  in
  let tvar_env = Tvar_env.of_list_mapi binders go_with_constraints in
  match !errors with [] -> Ok tvar_env | errors -> Partial (tvar_env, errors)

let typing_constant ~(expect_ty : Stype.t option) (c : Syntax.constant) ~loc :
    (Stype.t * Constant.t) Local_diagnostics.partial_info =
  let overflow c = [ Errors.overflow ~value:c ~loc ] [@@inline] in
  let check_byte_literal v ~repr :
      (Stype.t * Constant.t) Local_diagnostics.partial_info =
    if v < 0 || v > 255 then
      Partial ((Stype.byte, C_int { v = 0l; repr = Some repr }), overflow repr)
    else Ok (Stype.byte, C_int { v = Int32.of_int v; repr = Some repr })
  in
  match expect_ty with
  | None -> (
      match c with
      | Const_bool b -> Ok (Stype.bool, C_bool b)
      | Const_char lit -> Ok (Stype.char, C_char lit.char_val)
      | Const_string lit -> Ok (Stype.string, C_string lit.string_val)
      | Const_bytes lit ->
          Ok
            ( Stype.bytes,
              C_bytes { v = lit.bytes_val; repr = Some lit.bytes_repr } )
      | Const_int c -> (
          match Int32.of_string_opt c with
          | Some v -> Ok (Stype.int, C_int { v; repr = Some c })
          | None ->
              Partial ((Stype.int, C_int { v = 0l; repr = Some c }), overflow c)
          )
      | Const_byte c -> check_byte_literal c.byte_val ~repr:c.byte_repr
      | Const_uint c -> (
          match UInt32.of_string_opt c with
          | Some v -> Ok (Stype.uint, C_uint { v; repr = Some c })
          | None ->
              Partial
                ( (Stype.uint, C_uint { v = UInt32.min_int; repr = Some c }),
                  overflow c ))
      | Const_int64 c -> (
          match Int64.of_string_opt c with
          | Some v -> Ok (Stype.int64, C_int64 { v; repr = Some c })
          | None ->
              Partial
                ((Stype.int64, C_int64 { v = 0L; repr = Some c }), overflow c))
      | Const_uint64 c -> (
          match UInt64.of_string_opt c with
          | Some v -> Ok (Stype.uint64, C_uint64 { v; repr = Some c })
          | None ->
              Partial
                ( (Stype.uint64, C_uint64 { v = UInt64.min_int; repr = Some c }),
                  overflow c ))
      | Const_bigint c ->
          Ok (Stype.bigint, C_bigint { v = Bigint.of_string c; repr = Some c })
      | Const_double c ->
          Ok (Stype.double, C_double { v = float_of_string c; repr = Some c }))
  | Some expect_ty -> (
      match (Stype.type_repr expect_ty, c) with
      | _, Const_bool b -> Ok (Stype.bool, C_bool b)
      | _, Const_char lit -> Ok (Stype.char, C_char lit.char_val)
      | _, Const_string lit -> Ok (Stype.string, C_string lit.string_val)
      | _, Const_bytes lit ->
          Ok
            ( Stype.bytes,
              C_bytes { v = lit.bytes_val; repr = Some lit.bytes_repr } )
      | T_builtin T_byte, Const_int c -> (
          match int_of_string_opt c with
          | Some v -> check_byte_literal v ~repr:c
          | None ->
              Partial ((Stype.byte, C_int { v = 0l; repr = Some c }), overflow c)
          )
      | _, Const_byte c -> check_byte_literal c.byte_val ~repr:c.byte_repr
      | T_builtin T_uint, Const_int c | _, Const_uint c -> (
          match UInt32.of_string_opt c with
          | Some v -> Ok (Stype.uint, C_uint { v; repr = Some c })
          | None ->
              Partial
                ( (Stype.uint, C_uint { v = UInt32.min_int; repr = Some c }),
                  overflow c ))
      | T_builtin T_int64, Const_int c | _, Const_int64 c -> (
          match Int64.of_string_opt c with
          | Some v -> Ok (Stype.int64, C_int64 { v; repr = Some c })
          | None ->
              Partial
                ((Stype.int64, C_int64 { v = 0L; repr = Some c }), overflow c))
      | T_builtin T_uint64, Const_int c | _, Const_uint64 c -> (
          match UInt64.of_string_opt c with
          | Some v -> Ok (Stype.uint64, C_uint64 { v; repr = Some c })
          | None ->
              Partial
                ( (Stype.uint64, C_uint64 { v = UInt64.min_int; repr = Some c }),
                  overflow c ))
      | ty, Const_int c when Type.same_type ty Stype.bigint ->
          Ok (Stype.bigint, C_bigint { v = Bigint.of_string c; repr = Some c })
      | _, Const_bigint c ->
          Ok (Stype.bigint, C_bigint { v = Bigint.of_string c; repr = Some c })
      | T_builtin T_float, (Const_double c | Const_int c) ->
          Ok (Stype.float, C_float { v = float_of_string c; repr = Some c })
      | T_builtin T_double, Const_int c | _, Const_double c ->
          Ok (Stype.double, C_double { v = float_of_string c; repr = Some c })
      | _, Const_int c -> (
          match Int32.of_string_opt c with
          | Some v -> Ok (Stype.int, C_int { v; repr = Some c })
          | None ->
              Partial ((Stype.int, C_int { v = 0l; repr = Some c }), overflow c)
          ))

let typed_constant_to_syntax_constant = function
  | Constant.C_bool b -> Syntax.Const_bool b
  | C_char c ->
      Const_char
        { char_val = c; char_repr = Basic_uchar_utils.uchar_to_string c }
  | C_int { repr; v } -> (
      match repr with
      | Some repr -> Const_int repr
      | None -> Const_int (Int32.to_string v))
  | C_int64 { repr; v } -> (
      match repr with
      | Some repr -> Const_int repr
      | None -> Const_int (Int64.to_string v))
  | C_uint { repr; v } -> (
      match repr with
      | Some repr -> Const_int repr
      | None -> Const_uint (UInt32.to_string v))
  | C_uint64 { repr; v } -> (
      match repr with
      | Some repr -> Const_int repr
      | None -> Const_uint (UInt64.to_string v))
  | C_float { repr; v } ->
      Const_double
        (match repr with Some repr -> repr | None -> string_of_float v)
  | C_double { repr; v } ->
      Const_double
        (match repr with Some repr -> repr | None -> string_of_float v)
  | C_string c -> Const_string { string_val = c; string_repr = c }
  | C_bytes { v; repr } ->
      Const_bytes
        {
          bytes_val = v;
          bytes_repr = (match repr with Some x -> x | None -> v);
        }
  | C_bigint { v; repr } ->
      Const_bigint
        (match repr with Some repr -> repr | None -> Bigint.to_string v)

type func_type = Method of Syntax.typ option | Regular_func

let classify_func (ps : Syntax.parameters) : func_type =
  match ps with
  | {
      param_binder = { binder_name = "self" };
      param_annot;
      param_kind = Positional;
    }
    :: _ ->
      Method param_annot
  | _ -> Regular_func

let check_stub_type ~(language : string option) (typ : Typedtree.typ)
    (global_env : Global_env.t) ~(allow_func : bool) ~(is_import_stub : bool) :
    Local_diagnostics.error_option =
  match language with
  | Some ("js" | "c" | "C") -> None
  | _ -> (
      let loc = Typedtree.loc_of_typ typ in
      match Typedtree_util.stype_of_typ typ with
      | T_builtin T_unit
      | T_builtin T_int
      | T_builtin T_bool
      | T_builtin T_byte
      | T_builtin T_char
      | T_builtin T_int64
      | T_builtin T_float
      | T_builtin T_double
      | T_builtin T_uint
      | T_builtin T_uint64 ->
          None
      | T_builtin T_string ->
          if is_import_stub && !Basic_config.target <> Wasm_gc then
            Some (Errors.invalid_stub_type loc)
          else None
      | T_builtin T_bytes ->
          if is_import_stub then Some (Errors.invalid_stub_type loc) else None
      | T_constr { type_constructor; tys = ty_elem :: [] }
        when Type_path.equal type_constructor
               Type_path.Builtin.type_path_fixedarray -> (
          if is_import_stub then Some (Errors.invalid_stub_type loc)
          else
            match language with
            | Some "wasm" -> (
                match ty_elem with
                | T_builtin _ -> None
                | _ -> Some (Errors.invalid_stub_type loc))
            | _ -> (
                match ty_elem with
                | T_builtin builtin -> (
                    match builtin with
                    | T_int | T_int64 | T_double -> None
                    | T_unit | T_byte | T_bytes | T_uint64 | T_string | T_uint
                    | T_bool | T_char | T_float ->
                        Some (Errors.invalid_stub_type loc))
                | _ -> Some (Errors.invalid_stub_type loc)))
      | T_constr { type_constructor = p; tys = _ } -> (
          match Global_env.find_type_by_path global_env p with
          | Some { ty_desc = Extern_type; _ } -> None
          | _ -> Some (Errors.invalid_stub_type loc))
      | Tarrow _ when not allow_func -> Some (Errors.invalid_stub_type loc)
      | Tarrow { params_ty; ret_ty; err_ty } ->
          let is_simple_stub (ty : Stype.t) =
            match Stype.type_repr ty with
            | T_builtin T_unit
            | T_builtin T_int
            | T_builtin T_bool
            | T_builtin T_byte
            | T_builtin T_char
            | T_builtin T_int64
            | T_builtin T_uint
            | T_builtin T_uint64
            | T_builtin T_float
            | T_builtin T_double ->
                true
            | T_builtin T_string | T_builtin T_bytes -> not is_import_stub
            | T_constr { type_constructor; tys = _ }
              when Type_path.equal type_constructor
                     Type_path.Builtin.type_path_fixedarray ->
                not is_import_stub
            | T_constr { type_constructor = p; tys = _ } -> (
                match Global_env.find_type_by_path global_env p with
                | Some { ty_desc = Extern_type; _ } -> true
                | _ -> false)
            | T_blackhole -> true
            | Tarrow _ | Tparam _ | T_trait _ -> false
            | Tvar _ -> assert false
              [@@inline]
          in
          if
            Lst.for_all params_ty is_simple_stub
            && is_simple_stub ret_ty && err_ty = None
          then None
          else Some (Errors.invalid_stub_type loc)
      | Tparam _ | T_trait _ | T_blackhole ->
          Some (Errors.invalid_stub_type loc)
      | Tvar _ -> assert false)

let is_raw_string = function Syntax.Multiline_string _ -> true | _ -> false

let is_tvar (typ : Stype.t) : bool =
  let typ = Stype.type_repr typ in
  match typ with Tvar _ -> true | _ -> false

let is_trait (typ : Stype.t) : bool =
  let typ = Stype.type_repr typ in
  match typ with T_trait _ -> true | _ -> false

let is_only_tag_enum (typ : Stype.t) : bool =
  let typ = Stype.type_repr typ in
  match typ with T_constr { only_tag_enum_; _ } -> only_tag_enum_ | _ -> false

let validate_record ~(context : [ `Pattern | `Creation ])
    ~(expected : Typedecl_info.fields) (fields : Syntax.label list)
    ~(record_ty : Stype.t) ~(is_strict : bool) ~loc :
    string list Local_diagnostics.partial_info =
  let seen_labels = Hashset_string.create 17 in
  let superfluous = ref [] in
  let errors = ref [] in
  Lst.iter fields (fun { label_name; loc_ = label_loc } ->
      if not (Hashset_string.check_add seen_labels label_name) then
        errors :=
          Errors.duplicate_record_field ~label:label_name ~context
            ~loc:label_loc
          :: !errors;
      if
        not
          (Lst.exists expected (fun { field_name; _ } ->
               field_name = label_name))
      then (
        superfluous := label_name :: !superfluous;
        errors :=
          Errors.superfluous_field ~label:label_name
            ~ty:(Printer.type_to_string record_ty)
            ~loc:label_loc
          :: !errors));
  let missing =
    Lst.fold_right expected [] (fun { field_name; _ } acc ->
        if Hashset_string.mem seen_labels field_name then acc
        else field_name :: acc)
  in
  if missing <> [] && is_strict then
    errors :=
      Errors.missing_fields_in_record ~labels:missing
        ~ty:(Printer.type_to_string record_ty)
        ~context ~loc
      :: !errors;
  match !errors with
  | [] -> Ok !superfluous
  | errs -> Partial (!superfluous, errs)

let add_binder (env : Local_env.t) (binder : Typedtree.binder) ~typ ~mut :
    Local_env.t =
  Local_env.add env binder.binder_id ~typ ~mut ~loc:binder.loc_

let fresh_binder (b : Syntax.binder) : Typedtree.binder =
  { binder_id = Ident.fresh b.binder_name; loc_ = b.loc_ }

let add_pat_binder (env : Local_env.t)
    ({ binder; binder_typ } : Typedtree.pat_binder) : Local_env.t =
  Local_env.add env binder.binder_id ~typ:binder_typ ~mut:false ~loc:binder.loc_

let add_pat_binders (env : Local_env.t) (bs : Typedtree.pat_binders) :
    Local_env.t =
  List.fold_left add_pat_binder env bs

let add_local_typing_error = Local_diagnostics.add_error

let store_error (error_option : Local_diagnostics.error_option) ~diagnostics =
  match error_option with
  | None -> ()
  | Some err -> add_local_typing_error diagnostics err

let take_info_partial (x : 'a Local_diagnostics.partial_info) ~diagnostics : 'a
    =
  match x with
  | Ok a -> a
  | Partial (a, errors) ->
      List.iter (add_local_typing_error diagnostics) errors;
      a
