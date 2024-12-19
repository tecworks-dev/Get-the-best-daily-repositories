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


module Ident = Basic_ident
module Type_path = Basic_type_path
module Constr_info = Basic_constr_info
module Lst = Basic_lst
module Vec = Basic_vec
module Longident = Basic_longident
module Syntax = Parsing_syntax
module Operators = Parsing_operators
module Syntax_util = Parsing_syntax_util
module Literal = Lex_literal

let add_error = Typeutil.add_local_typing_error
let store_error = Typeutil.store_error
let take_info_partial = Typeutil.take_info_partial

let typing_type ~diagnostics ~tvar_env ~global_env typ : Typedtree.typ =
  Typeutil.typing_type ~is_toplevel:false ~allow_private:true ~tvar_env
    ~types:(Global_env.get_all_types global_env)
    typ
  |> take_info_partial ~diagnostics

let typing_type_name ~diagnostics ~tvar_env ~global_env
    (type_name : Syntax.type_name) :
    (Typeutil.type_name * Typedtree.type_name) option =
  Typeutil.typing_type_name
    ~types:(Global_env.get_all_types global_env)
    ~tvar_env type_name ~allow_private:true
  |> take_info_partial ~diagnostics

let find_value ~(env : Local_env.t) ~(global_env : Global_env.t)
    (id : Longident.t) ~diagnostics ~loc : Value_info.t =
  match id with
  | Lident name -> (
      match Local_env.find_by_name_opt env name with
      | Some r -> r
      | None ->
          Global_env.find_value global_env id ~loc
          |> take_info_partial ~diagnostics)
  | Ldot _ ->
      Global_env.find_value global_env id ~loc |> take_info_partial ~diagnostics

type unify_result =
  | Ok
  | Error of Local_diagnostics.report
  | TraitUpCast of Type_path.t

let unify_expr_allow_trait_upcast ~global_env ~cenv ~expect_ty ~actual_ty loc :
    unify_result =
  let expect_ty' = Stype.type_repr expect_ty in
  let actual_ty' = Stype.type_repr actual_ty in
  let fallback () =
    match Ctype.unify_expr ~expect_ty ~actual_ty loc with
    | None -> Ok
    | Some err -> Error err
      [@@inline]
  in
  if Basic_prelude.phys_equal expect_ty' actual_ty' then Ok
  else
    match expect_ty' with
    | T_trait expect_trait -> (
        match actual_ty' with
        | T_trait actual_trait when Type_path.equal expect_trait actual_trait ->
            Ok
        | Tvar _ -> fallback ()
        | _ ->
            let closure =
              Trait_closure.compute_closure
                ~types:(Global_env.get_all_types global_env)
                [ { trait = expect_trait; loc_ = loc; required_by_ = [] } ]
            in
            List.iter (Poly_type.add_constraint cenv actual_ty') closure;
            TraitUpCast expect_trait)
    | _ -> fallback ()

let handle_unify_result (result : unify_result) ~diagnostics ~expr =
  match result with
  | Ok -> expr
  | Error e ->
      add_error diagnostics e;
      expr
  | TraitUpCast trait ->
      Typedtree.Texpr_as
        {
          expr;
          ty = T_trait trait;
          trait =
            Tname_path { name = trait; kind = Trait; loc_ = Rloc.no_location };
          is_implicit = true;
          loc_ = Typedtree.loc_of_typed_expr expr;
        }

type maybe_typed = Typechecked of Typedtree.expr | Not_yet of Syntax.expr

let wrap_newtype_constr (newtype_constr : Typedecl_info.constructor)
    (expr : Typedtree.expr) newtype_typ (loc : Rloc.t) =
  let ty = Typedtree_util.type_of_typed_expr expr in
  let constr_desc = newtype_constr in
  let syntax_constr : Syntax.constructor =
    {
      constr_name =
        { name = newtype_constr.constr_name; loc_ = Rloc.no_location };
      extra_info = No_extra_info;
      loc_ = Rloc.no_location;
    }
  in
  let constr_expr =
    Typedtree.Texpr_constr
      {
        constr = syntax_constr;
        tag = constr_desc.cs_tag;
        ty = Builtin.type_arrow [ ty ] newtype_typ ~err_ty:None;
        arity_ = constr_desc.cs_arity_;
        loc_ = loc;
      }
  in
  Typedtree.Texpr_apply
    {
      func = constr_expr;
      args = [ { arg_value = expr; arg_kind = Positional } ];
      ty = newtype_typ;
      kind_ = Normal;
      loc_ = loc;
    }

type expect_ty = Expect_type of Stype.t | Ignored

let get_expected_type = function Expect_type ty -> ty | Ignored -> Stype.unit

let desugar_multiline_string ~loc_ elems : Syntax.expr =
  if List.for_all Typeutil.is_raw_string elems then
    let lines =
      List.map
        (function
          | Syntax.Multiline_string s -> s
          | Syntax.Multiline_interp _ -> assert false)
        elems
    in
    let str = String.concat "\n" lines in
    Pexpr_constant
      { c = Const_string { string_val = str; string_repr = str }; loc_ }
  else
    let newline =
      Syntax.Interp_lit { str = "\n"; repr = ""; loc_ = Rloc.no_location }
    in
    let rec loop elems acc =
      match elems with
      | [] -> acc
      | x :: xs ->
          let acc =
            match x with
            | Syntax.Multiline_string str ->
                Syntax.Interp_lit { str; repr = ""; loc_ = Rloc.no_location }
                :: acc
            | Multiline_interp interps -> List.rev_append interps acc
          in
          if xs = [] then acc else loop xs (newline :: acc)
    in
    let rev_interps = loop elems [] in
    let elems =
      List.fold_left
        (fun acc interp ->
          match (interp, acc) with
          | _, [] -> [ interp ]
          | ( Syntax.Interp_lit { str = s1; repr = _ },
              Syntax.Interp_lit { str = s2; repr = _; loc_ } :: acc ) ->
              Interp_lit { str = s1 ^ s2; repr = ""; loc_ } :: acc
          | _, _ -> interp :: acc)
        [] rev_interps
    in
    Syntax.Pexpr_interp { elems; loc_ }

let rec infer_expr (env : Local_env.t) (expr : Syntax.expr)
    ~(control_ctx : Control_ctx.t) ~(tvar_env : Tvar_env.t)
    ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let check_go =
    check_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let infer_go =
    infer_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  match expr with
  | Pexpr_array_get _ | Pexpr_array_get_slice _ | Pexpr_array_set _ ->
      assert false
  | Pexpr_array_augmented_set _ -> assert false
  | Pexpr_unary _ -> assert false
  | Pexpr_constr { constr; loc_ } ->
      typing_constr_or_constant constr None ~expect_ty:None ~env ~loc:loc_
        ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_apply { func = Pexpr_constr { constr }; args; attr; loc_ } ->
      (if attr <> No_attr then
         let error = Errors.invalid_apply_attr ~kind:`Constructor ~loc:loc_ in
         add_error diagnostics error);
      typing_constr_or_constant constr (Some args) ~expect_ty:None ~env
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_apply { func; args; attr; loc_ } ->
      typing_application env (infer_go env func) args
        ~kind:(Normal : Typedtree.apply_kind)
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~attr ~diagnostics
  | Pexpr_infix
      { op = { var_name = Lident "&&"; loc_ = infix_loc }; lhs; rhs; loc_ } ->
      let fn =
        Typedtree.Texpr_ident
          {
            id =
              {
                var_id = Ident.of_qual_ident Basic_qual_ident.op_and;
                loc_ = infix_loc;
              };
            kind = Prim Primitive.Psequand;
            ty =
              Tarrow
                {
                  params_ty = [ Stype.bool; Stype.bool ];
                  ret_ty = Stype.bool;
                  err_ty = None;
                  generic_ = false;
                };
            ty_args_ = [||];
            arity_ = Some (Fn_arity.simple 2);
            loc_ = infix_loc;
          }
      in
      let tlhs = check_go env lhs (Expect_type Stype.bool) in
      let trhs = check_go env rhs (Expect_type Stype.bool) in
      Texpr_apply
        {
          func = fn;
          args =
            [
              { arg_kind = Positional; arg_value = tlhs };
              { arg_kind = Positional; arg_value = trhs };
            ];
          ty = Stype.bool;
          kind_ = Infix;
          loc_;
        }
  | Pexpr_infix
      { op = { var_name = Lident "||"; loc_ = infix_loc }; lhs; rhs; loc_ } ->
      let fn =
        Typedtree.Texpr_ident
          {
            id =
              {
                var_id = Ident.of_qual_ident Basic_qual_ident.op_or;
                loc_ = infix_loc;
              };
            kind = Prim Primitive.Psequor;
            ty =
              Tarrow
                {
                  params_ty = [ Stype.bool; Stype.bool ];
                  ret_ty = Stype.bool;
                  err_ty = None;
                  generic_ = false;
                };
            ty_args_ = [||];
            arity_ = Some (Fn_arity.simple 2);
            loc_ = infix_loc;
          }
      in
      let tlhs = check_go env lhs (Expect_type Stype.bool) in
      let trhs = check_go env rhs (Expect_type Stype.bool) in
      Texpr_apply
        {
          func = fn;
          args =
            [
              { arg_kind = Positional; arg_value = tlhs };
              { arg_kind = Positional; arg_value = trhs };
            ];
          ty = Stype.bool;
          kind_ = Infix;
          loc_;
        }
  | Pexpr_infix
      { op = { var_name = Lident ("..=" | "..<"); loc_ = infix_loc_ }; loc_; _ }
    ->
      add_error diagnostics (Errors.range_operator_only_in_for infix_loc_);
      Texpr_hole
        { ty = Stype.new_type_var Tvar_error; loc_; kind = Synthesized }
  | Pexpr_infix { op; lhs; rhs; loc_ } ->
      typing_infix_op env op (infer_go env lhs) rhs ~loc:loc_ ~global_env ~cenv
        ~tvar_env ~control_ctx ~diagnostics
  | Pexpr_array { exprs; loc_ } ->
      let ty = Stype.new_type_var Tvar_normal in
      let tes = Lst.map exprs (fun e -> check_go env e (Expect_type ty)) in
      Texpr_array
        {
          exprs = tes;
          ty =
            Stype.T_constr
              {
                type_constructor = Type_path.Builtin.type_path_array;
                tys = [ ty ];
                generic_ = false;
                only_tag_enum_ = false;
                is_suberror_ = false;
              };
          is_fixed_array = false;
          loc_;
        }
  | Pexpr_array_spread { elems; loc_ } ->
      typing_array_spread env elems ~loc:loc_ ~expect_ty:None ~control_ctx
        ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_constant { c; loc_ } ->
      let ty, c =
        Typeutil.typing_constant c ~expect_ty:None ~loc:loc_
        |> take_info_partial ~diagnostics
      in
      Texpr_constant { c; ty; name_ = None; loc_ }
  | Pexpr_multiline_string { elems; loc_ } ->
      infer_go env (desugar_multiline_string ~loc_ elems)
  | Pexpr_interp { elems; loc_ } ->
      typing_interp ~control_ctx ~global_env ~diagnostics ~tvar_env ~cenv env
        elems loc_
  | Pexpr_constraint { expr = e; ty; loc_ } ->
      let ty = typing_type ~tvar_env ~global_env ty ~diagnostics in
      let stype = Typedtree_util.stype_of_typ ty in
      Texpr_constraint
        {
          expr = check_go env e (Expect_type stype);
          konstraint = ty;
          ty = stype;
          loc_;
        }
  | Pexpr_while { loop_cond; loop_body; loc_; while_else } ->
      check_while env loop_cond loop_body while_else
        ~expect_ty:(Expect_type (Stype.new_type_var Tvar_normal))
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_function
      {
        func =
          Lambda
            { parameters; params_loc_; body; return_type; kind_; has_error };
        loc_;
      } ->
      let tfun, ty =
        infer_function env parameters params_loc_ body return_type ~has_error
          ~kind_ ~tvar_env ~cenv ~global_env ~diagnostics
      in
      Texpr_function { func = tfun; ty; loc_ }
  | Pexpr_function { func = Match _ } -> assert false
  | Pexpr_ident { id; loc_ } -> (
      let value_info =
        find_value id.var_name ~env ~global_env ~diagnostics ~loc:loc_
      in
      match value_info with
      | Local_imm { id = var_id; typ } ->
          let id : Typedtree.var = { var_id; loc_ = id.loc_ } in
          let ty, (kind : Typedtree.value_kind) =
            Type.deref_constr_type_to_local_value typ
          in
          Texpr_ident { id; kind; ty; ty_args_ = [||]; arity_ = None; loc_ }
      | Local_mut { id = var_id; typ } ->
          let id : Typedtree.var = { var_id; loc_ = id.loc_ } in
          Texpr_ident
            {
              id;
              kind = Mutable;
              ty = typ;
              ty_args_ = [||];
              arity_ = None;
              loc_;
            }
      | Toplevel_value
          { id = qid; typ; ty_params_; arity_; kind; doc_; direct_use_loc_; _ }
        ->
          Docstring.check_alerts ~diagnostics (Docstring.pragmas doc_) loc_;
          let qid =
            match direct_use_loc_ with
            | Explicit_import _ | Implicit_import_all _ ->
                Basic_qual_ident.make_implicit_pkg
                  ~pkg:(Basic_qual_ident.get_pkg qid)
                  ~name:(Basic_qual_ident.base_name qid)
            | Not_direct_use -> qid
          in
          let id : Typedtree.var =
            { var_id = Ident.of_qual_ident qid; loc_ = id.loc_ }
          in
          let kind : Typedtree.value_kind =
            match kind with
            | Normal -> Normal
            | Prim prim -> Prim prim
            | Const _ -> assert false
          in
          if Tvar_env.is_empty ty_params_ then
            Texpr_ident { id; kind; ty = typ; ty_args_ = [||]; arity_; loc_ }
          else
            let ty, ty_args_ =
              Poly_type.instantiate_value ~cenv ~loc:loc_ ty_params_ typ
            in
            Texpr_ident { id; kind; ty; ty_args_; arity_; loc_ })
  | Pexpr_if { cond; ifso; ifnot; loc_ } -> (
      let cond = check_go env cond (Expect_type Stype.bool) in
      match ifnot with
      | None ->
          let ifso = check_go env ifso Ignored in
          Texpr_if { cond; ifso; ifnot = None; ty = Stype.unit; loc_ }
      | Some ifnot ->
          let ifso = infer_go env ifso in
          let ty = Typedtree_util.type_of_typed_expr ifso in
          let ifnot = check_go env ifnot (Expect_type ty) in
          Texpr_if { cond; ifso; ifnot = Some ifnot; ty; loc_ })
  | Pexpr_guard { cond; otherwise; body; loc_ } ->
      let cond = check_go env cond (Expect_type Stype.bool) in
      let body = infer_go env body in
      let ty = Typedtree_util.type_of_typed_expr body in
      let otherwise =
        match otherwise with
        | None -> None
        | Some e2 -> Some (check_go env e2 (Expect_type ty))
      in
      Texpr_guard { cond; otherwise; body; ty; loc_ }
  | Pexpr_guard_let { pat; expr; otherwise; body; loc_ } ->
      let pat_binders, pat, rhs =
        typing_let env pat expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let body = infer_go (Typeutil.add_pat_binders env pat_binders) body in
      let ty = Typedtree_util.type_of_typed_expr body in
      let check_cases cases pat_ty action_ty =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p pat_ty ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action (Expect_type action_ty)
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
          [@@inline]
      in
      let otherwise =
        match otherwise with
        | None -> None
        | Some cases ->
            Some (check_cases cases (Typedtree_util.type_of_typed_expr rhs) ty)
      in
      Texpr_guard_let { pat; rhs; pat_binders; otherwise; body; ty; loc_ }
  | Pexpr_try
      {
        body;
        catch;
        catch_all;
        try_else;
        try_loc_;
        catch_loc_;
        else_loc_;
        loc_;
      } ->
      let error_ctx = ref (Control_ctx.Open_ctx Empty_ctx) in
      let new_ctx = Control_ctx.with_error_ctx ~error_ctx control_ctx in
      let body =
        infer_expr ~control_ctx:new_ctx ~tvar_env ~cenv ~global_env ~diagnostics
          env body
      in
      let body_ty = Typedtree_util.type_of_typed_expr body in
      let ty =
        if try_else = None then body_ty else Stype.new_type_var Tvar_normal
      in
      let check_cases cases pat_ty action_ty =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p pat_ty ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action (Expect_type action_ty)
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
          [@@inline]
      in
      let error_ty = Control_ctx.error_ctx_to_stype !error_ctx in
      let catch = check_cases catch error_ty ty in
      let try_else =
        match try_else with
        | None -> None
        | Some try_else -> Some (check_cases try_else body_ty ty)
      in
      if not !(new_ctx.has_error) then (
        Local_diagnostics.add_warning diagnostics
          { kind = Useless_try; loc = try_loc_ };
        match Stype.type_repr error_ty with
        | Tvar link -> link := Tlink Stype.unit
        | _ -> ());
      if catch_all then (
        control_ctx.has_error := true;
        match control_ctx.error_ctx with
        | Some error_ctx ->
            Control_ctx.check_error_in_ctx ~error_ty ~ctx:error_ctx catch_loc_
            |> store_error ~diagnostics
        | None ->
            add_error diagnostics
              (Errors.invalid_raise ~kind:`Catchall ~loc:catch_loc_));
      Texpr_try
        {
          body;
          catch;
          catch_all;
          try_else;
          ty;
          err_ty = error_ty;
          catch_loc_;
          else_loc_;
          loc_;
        }
  | Pexpr_letrec { bindings = funs; body; loc_ } ->
      let env_with_funs, tfuns =
        typing_letrec env funs ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let body = infer_go env_with_funs body in
      Texpr_letrec
        {
          bindings = tfuns;
          body;
          ty = Typedtree_util.type_of_typed_expr body;
          loc_;
        }
  | Pexpr_letfn { name; func; body; loc_ } -> (
      let env_with_funs, tfuns =
        typing_letrec env
          [ (name, func) ]
          ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let body = infer_go env_with_funs body in
      match tfuns with
      | (binder, fn) :: [] ->
          Texpr_letfn
            {
              binder;
              fn;
              body;
              ty = Typedtree_util.type_of_typed_expr body;
              is_rec = Typedtree_util.is_rec binder fn;
              loc_;
            }
      | _ -> assert false)
  | Pexpr_let { pattern = p; expr = e1; body = e2; loc_; _ } ->
      let pat_binders, tp, te1 =
        typing_let env p e1 ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let body = infer_go (Typeutil.add_pat_binders env pat_binders) e2 in
      Texpr_let
        {
          pat = tp;
          rhs = te1;
          body;
          ty = Typedtree_util.type_of_typed_expr body;
          pat_binders;
          loc_;
        }
  | Pexpr_sequence { expr1; expr2; loc_ } ->
      let expr1 = check_go env expr1 Ignored in
      let expr2 = infer_go env expr2 in
      Texpr_sequence
        { expr1; expr2; ty = Typedtree_util.type_of_typed_expr expr2; loc_ }
  | Pexpr_tuple { exprs; loc_ } ->
      let tes = Lst.map exprs (infer_go env) in
      Texpr_tuple
        {
          exprs = tes;
          ty =
            Builtin.type_product (Lst.map tes Typedtree_util.type_of_typed_expr);
          loc_;
        }
  | Pexpr_record { type_name; fields; loc_; trailing = _ } -> (
      let fallback type_name =
        infer_record env fields loc_ ~type_name ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
          [@@inline]
      in
      match type_name with
      | Some type_name -> (
          match
            typing_type_name ~global_env ~tvar_env type_name ~diagnostics
          with
          | Some
              ( Tname_defined
                  {
                    ty_constr;
                    ty_desc = Record_type { has_private_field_ = true; _ };
                    _;
                  },
                type_name )
            when Type_path_util.is_foreign ty_constr ->
              let name = Type_path_util.name ty_constr in
              add_error diagnostics
                (Errors.cannot_create_struct_with_priv_field ~name ~loc:loc_);
              fallback (Some type_name)
          | Some
              ( Tname_defined
                  ({ ty_desc = Record_type { fields = labels }; _ } as ty_decl),
                type_name ) ->
              Docstring.check_alerts ~diagnostics
                (Docstring.pragmas ty_decl.ty_doc_)
                loc_;
              let expect_ty : Stype.t =
                T_constr
                  {
                    type_constructor = ty_decl.ty_constr;
                    tys = Type.type_var_list ty_decl.ty_arity Tvar_normal;
                    generic_ = false;
                    only_tag_enum_ = false;
                    is_suberror_ = false;
                  }
              in
              let _, labels =
                Poly_type.instantiate_record ~ty_record:(`Known expect_ty)
                  labels
              in
              type_guided_record_check fields labels expect_ty
                ~type_name:(Some type_name) ~env ~control_ctx ~tvar_env ~cenv
                ~global_env ~diagnostics ~loc:loc_
          | Some (_, tast_type_name) ->
              add_error diagnostics
                (Errors.not_a_record_type
                   ~name:(Longident.to_string type_name.name)
                   ~loc:type_name.loc_);
              fallback (Some tast_type_name)
          | None -> fallback None)
      | None -> fallback None)
  | Pexpr_record_update { type_name; record; fields; loc_ } ->
      let type_name, expected_record =
        match type_name with
        | Some type_name -> (
            match
              typing_type_name ~global_env ~tvar_env type_name ~diagnostics
            with
            | Some
                ( Tname_defined ({ ty_desc = Record_type _; _ } as ty_decl),
                  type_name ) ->
                let expect_ty : Stype.t =
                  T_constr
                    {
                      type_constructor = ty_decl.ty_constr;
                      tys = Type.type_var_list ty_decl.ty_arity Tvar_normal;
                      generic_ = false;
                      only_tag_enum_ = false;
                      is_suberror_ = false;
                    }
                in
                (Some type_name, check_go env record (Expect_type expect_ty))
            | Some (_, tast_type_name) ->
                add_error diagnostics
                  (Errors.not_a_record_type
                     ~name:(Longident.to_string type_name.name)
                     ~loc:type_name.loc_);
                (Some tast_type_name, infer_go env record)
            | None -> (None, infer_go env record))
        | None -> (None, infer_go env record)
      in
      let expected_record_ty =
        Typedtree_util.type_of_typed_expr expected_record
      in
      let typed_fields, all_fields =
        match
          Global_env.labels_of_record global_env ~loc:loc_ ~context:`Update
            expected_record_ty
        with
        | Ok all_fields ->
            let _, all_fields =
              Poly_type.instantiate_record
                ~ty_record:(`Known expected_record_ty) all_fields
            in
            ( type_guided_record_update_check env all_fields fields
                expected_record_ty ~control_ctx ~tvar_env ~cenv ~global_env
                ~diagnostics,
              all_fields )
        | Error e ->
            add_error diagnostics e;
            let pseudo_typed =
              Lst.map fields (fun (Field_def { label; expr; _ }) ->
                  Typedtree.Field_def
                    {
                      label;
                      expr = infer_go env expr;
                      is_mut = false;
                      is_pun = false;
                      pos = 0;
                    })
            in
            (pseudo_typed, [])
      in
      Texpr_record_update
        {
          type_name;
          record = expected_record;
          all_fields;
          fields = typed_fields;
          ty = expected_record_ty;
          loc_;
        }
  | Pexpr_field
      {
        record = tuple;
        accessor = Index { tuple_index = index } as accessor;
        loc_;
      } -> (
      let tuple = infer_go env tuple in
      let ty_tuple =
        Stype.type_repr (Typedtree_util.type_of_typed_expr tuple)
      in
      let newtype_info = Global_env.get_newtype_info global_env ty_tuple in
      match newtype_info with
      | Some info -> (
          if index <> 0 then
            add_error diagnostics
              (Errors.invalid_newtype_index ~loc:loc_
                 ~ty:(Printer.type_to_string ty_tuple))
          else
            Local_diagnostics.add_warning diagnostics
              {
                kind =
                  Deprecated_syntax
                    {
                      old_usage = "`<expr>.0`";
                      purpose = "accessing underlying type of new type";
                      new_usage = Some "`<expr>._`";
                    };
                loc = loc_;
              };
          let ty_res, ty_args =
            Poly_type.instantiate_constr info.newtype_constr
          in
          Ctype.unify_exn ty_res ty_tuple;
          match ty_args with
          | ty :: [] ->
              Texpr_field
                { record = tuple; accessor = Newtype; ty; pos = 0; loc_ }
          | _ -> assert false)
      | None ->
          let ty =
            match
              Type.filter_product ~blame:Filtered_type ~arity:None ty_tuple loc_
            with
            | Ok tys -> (
                match Lst.nth_opt tys index with
                | Some ty -> ty
                | None ->
                    add_error diagnostics
                      (Errors.no_tuple_index ~required:index
                         ~actual:(List.length tys) ~loc:loc_);
                    Stype.new_type_var Tvar_error)
            | Partial (_, errs) ->
                List.iter (add_error diagnostics) errs;
                Stype.new_type_var Tvar_error
          in
          Texpr_field { record = tuple; accessor; ty; pos = index; loc_ })
  | Pexpr_field { record; accessor = Newtype; loc_ } -> (
      let record = infer_go env record in
      let ty_record = Typedtree_util.type_of_typed_expr record in
      match Global_env.get_newtype_info global_env ty_record with
      | None ->
          add_error diagnostics
            (Errors.not_a_newtype
               ~actual_ty:(Printer.type_to_string ty_record)
               ~loc:loc_);
          Texpr_field
            {
              record;
              accessor = Newtype;
              ty = Stype.new_type_var Tvar_error;
              pos = Typeutil.unknown_pos;
              loc_;
            }
      | Some info -> (
          let ty_res, ty_args =
            Poly_type.instantiate_constr info.newtype_constr
          in
          Ctype.unify_exn ty_res ty_record;
          match ty_args with
          | ty :: [] ->
              Texpr_field { record; accessor = Newtype; ty; pos = 0; loc_ }
          | _ -> assert false))
  | Pexpr_field { record; accessor = Label label as accessor; loc_ } ->
      let name = label.label_name in
      let record = infer_go env record in
      let ty_record =
        Stype.type_repr (Typedtree_util.type_of_typed_expr record)
      in
      let may_be_method () =
        match
          Type_constraint.resolve_method_by_type ty_record name ~tvar_env
            ~global_env ~src:Dot_src_direct ~loc:loc_
        with
        | Ok _ -> true
        | Error _ -> false
          [@@inline]
      in
      let actual_record = deref_newtype ~global_env ~loc:loc_ record in
      let ty_field, pos, _, _ =
        resolve_field ~global_env actual_record label ~may_be_method
          ~src_ty_record:ty_record ~diagnostics ~loc:loc_
      in
      Texpr_field { record = actual_record; accessor; ty = ty_field; pos; loc_ }
  | Pexpr_method { type_name; method_name; loc_ } -> (
      let fallback type_name err : Typedtree.expr =
        add_error diagnostics err;
        Texpr_method
          {
            type_name;
            meth =
              {
                var_id = Ident.fresh method_name.label_name;
                loc_ = method_name.loc_;
              };
            ty_args_ = [||];
            arity_ = None;
            prim = None;
            ty = Stype.new_type_var Tvar_error;
            loc_;
          }
          [@@inline]
      in
      let resolve_by_typename ~is_trait (type_name : Typedtree.type_name) :
          Typedtree.expr =
        match
          Type_constraint.resolve_method_by_type_name type_name
            method_name.label_name ~is_trait ~loc:loc_ ~tvar_env ~global_env
        with
        | Ok (Known_method method_info) ->
            Docstring.check_alerts ~diagnostics
              (Docstring.pragmas method_info.doc_)
              loc_;
            let ty, ty_args_ =
              Poly_type.instantiate_method ~cenv ~loc:loc_ method_info
            in
            Texpr_method
              {
                type_name;
                meth =
                  {
                    var_id = Ident.of_qual_ident method_info.id;
                    loc_ = method_name.loc_;
                  };
                prim = method_info.prim;
                ty;
                ty_args_;
                arity_ = Some method_info.arity_;
                loc_;
              }
        | Ok (Promised_method { method_id; method_ty; method_arity; prim }) ->
            Texpr_method
              {
                type_name;
                meth = { var_id = method_id; loc_ = method_name.loc_ };
                prim;
                ty = method_ty;
                ty_args_ = [||];
                arity_ = Some method_arity;
                loc_;
              }
        | Error err -> fallback type_name err
          [@@inline]
      in
      let type_name_loc = type_name.loc_ in
      match typing_type_name ~global_env ~tvar_env type_name ~diagnostics with
      | Some (Tname_trait { name; vis_ = Vis_default; _ }, type_name)
        when Type_path_util.is_foreign name ->
          fallback type_name
            (Errors.cannot_use_method_of_abstract_trait ~trait:name
               ~method_name:method_name.label_name ~loc:method_name.loc_)
      | Some (Tname_trait decl, type_name) -> (
          match
            Trait_decl.find_method decl method_name.label_name ~loc:loc_
          with
          | Error _ -> resolve_by_typename ~is_trait:true type_name
          | Ok meth_decl ->
              let trait_name = decl.name in
              let self_type = Stype.new_type_var Tvar_normal in
              let ty =
                Poly_type.instantiate_method_decl meth_decl ~self:self_type
              in
              Trait_closure.compute_closure
                ~types:(Global_env.get_all_types global_env)
                [ { trait = decl.name; loc_; required_by_ = [] } ]
              |> List.iter (Poly_type.add_constraint cenv self_type);
              let trait_name : Typedtree.type_path_loc =
                { name = trait_name; loc_ = type_name_loc }
              in
              Texpr_unresolved_method
                {
                  trait_name;
                  method_name = method_name.label_name;
                  self_type;
                  arity_ = Some meth_decl.method_arity;
                  ty;
                  loc_;
                })
      | Some (_, type_name) -> resolve_by_typename ~is_trait:false type_name
      | None ->
          Texpr_hole
            { ty = Stype.new_type_var Tvar_error; loc_; kind = Synthesized })
  | Pexpr_dot_apply { self; method_name; args; return_self; attr; loc_ } ->
      let self = infer_go env self in
      let ty_self = Typedtree_util.type_of_typed_expr self in
      let method_expr =
        typing_self_method ty_self method_name
          ~src:Type_constraint.Dot_src_direct ~loc:method_name.loc_ ~tvar_env
          ~cenv ~global_env ~diagnostics
      in
      typing_application env method_expr args ~self
        ~kind:(if return_self then Dot_return_self else Dot)
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~attr ~diagnostics
  | Pexpr_as { expr; trait; loc_ } -> (
      let expr = infer_go env expr in
      match typing_type_name trait ~tvar_env ~global_env ~diagnostics with
      | Some (Tname_trait trait_decl, type_name) ->
          Trait_decl.check_object_safety
            ~name:(Type_path_util.name trait_decl.name)
            ~loc:trait.loc_ trait_decl.object_safety_
          |> Option.iter (add_error diagnostics);
          let closure =
            Trait_closure.compute_closure
              ~types:(Global_env.get_all_types global_env)
              [ { trait = trait_decl.name; loc_; required_by_ = [] } ]
          in
          List.iter
            (Poly_type.add_constraint cenv
               (Typedtree_util.type_of_typed_expr expr))
            closure;
          Texpr_as
            {
              expr;
              trait = type_name;
              ty = T_trait trait_decl.name;
              is_implicit = false;
              loc_;
            }
      | Some (_, type_name) ->
          add_error diagnostics
            (Errors.not_a_trait
               ~name:(Longident.to_string trait.name)
               ~loc:trait.loc_);
          Texpr_as
            {
              expr;
              trait = type_name;
              ty = Stype.new_type_var Tvar_error;
              is_implicit = false;
              loc_;
            }
      | None ->
          let tpath =
            match trait.name with
            | Lident name ->
                Type_path.toplevel_type ~pkg:!Basic_config.current_package name
            | Ldot { pkg; id } -> Type_path.toplevel_type ~pkg id
          in
          Texpr_as
            {
              expr;
              trait =
                Tname_path { name = tpath; kind = Type; loc_ = trait.loc_ };
              ty = Stype.new_type_var Tvar_error;
              is_implicit = false;
              loc_;
            })
  | Pexpr_mutate { record; accessor; field; augmented_by; loc_ } -> (
      match accessor with
      | Label label ->
          typing_mutate env record label field augmented_by loc_ ~control_ctx
            ~tvar_env ~cenv ~global_env ~diagnostics
      | Index _ | Newtype ->
          add_error diagnostics (Errors.tuple_not_mutable loc_);
          Texpr_tuple { exprs = []; ty = Stype.unit; loc_ })
  | Pexpr_match { expr; cases; match_loc_; loc_ } ->
      let expr = infer_go env expr in
      let ty1 = Typedtree_util.type_of_typed_expr expr in
      let ty2 = Stype.new_type_var Tvar_normal in
      let trows =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p ty1 ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action (Expect_type ty2)
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
      in
      Texpr_match { expr; cases = trows; ty = ty2; match_loc_; loc_ }
  | Pexpr_pipe { lhs; rhs; loc_ } ->
      typing_pipe env lhs rhs ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env
        ~diagnostics
  | Pexpr_letmut { binder; ty; expr; body; loc_ } ->
      let env, binder, expr, konstraint =
        typing_letmut env binder ty expr ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
      in
      let body = infer_go env body in
      let ty_body = Typedtree_util.type_of_typed_expr body in
      Texpr_letmut { binder; konstraint; expr; body; ty = ty_body; loc_ }
  | Pexpr_assign { var; expr; augmented_by; loc_ } -> (
      let check_rhs var ty_of_var : Typedtree.expr =
        match augmented_by with
        | None ->
            let expr = check_go env expr (Expect_type ty_of_var) in
            Texpr_assign
              { var; expr; augmented_by = None; ty = Stype.unit; loc_ }
        | Some op ->
            let tlhs : Typedtree.expr =
              Texpr_ident
                {
                  id = var;
                  kind = Mutable;
                  ty = ty_of_var;
                  ty_args_ = [||];
                  arity_ = None;
                  loc_ = var.loc_;
                }
            in
            let infix_expr =
              typing_infix_op env op tlhs expr ~loc:loc_ ~global_env ~cenv
                ~tvar_env ~control_ctx ~diagnostics
            in
            let op_expr, expr =
              match infix_expr with
              | Texpr_apply
                  { func; args = [ _; { arg_value; _ } ]; ty = actual_ty; _ } ->
                  Ctype.unify_expr ~expect_ty:ty_of_var ~actual_ty loc_
                  |> store_error ~diagnostics;
                  (func, arg_value)
              | _ -> assert false
            in
            Texpr_assign
              { var; expr; augmented_by = Some op_expr; ty = Stype.unit; loc_ }
          [@@local]
      in
      let value_info =
        find_value var.var_name ~env ~global_env ~diagnostics ~loc:loc_
      in
      match value_info with
      | Local_mut { id = var_id; typ } ->
          let id : Typedtree.var = { var_id; loc_ = var.loc_ } in
          check_rhs id typ
      | Local_imm { id = var_id; typ } ->
          let id : Typedtree.var = { var_id; loc_ = var.loc_ } in
          add_error diagnostics
            (Errors.not_mutable ~id:var.var_name ~loc:var.loc_);
          check_rhs id typ
      | Toplevel_value { id; _ } ->
          add_error diagnostics
            (Errors.not_mutable ~id:var.var_name ~loc:var.loc_);
          let id : Typedtree.var =
            { var_id = Ident.of_qual_ident id; loc_ = var.loc_ }
          in
          check_rhs id (Stype.new_type_var Tvar_error))
  | Pexpr_hole { kind; loc_ } ->
      (match kind with
      | Incomplete -> add_error diagnostics (Errors.found_hole loc_)
      | Synthesized -> ()
      | Todo ->
          Local_diagnostics.add_warning diagnostics { kind = Todo; loc = loc_ });
      Texpr_hole { ty = Stype.new_type_var Tvar_normal; loc_; kind }
  | Pexpr_unit { loc_; _ } -> Texpr_unit { loc_ }
  | Pexpr_break { arg; loc_ } ->
      check_break env arg ~loc:loc_
        ~expect_ty:(Stype.new_type_var Tvar_normal)
        ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  | Pexpr_continue { args; loc_; _ } ->
      check_continue env args ~loc:loc_
        ~expect_ty:(Stype.new_type_var Tvar_normal)
        ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  | Pexpr_loop { args; body; loop_loc_; loc_ } ->
      check_loop ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics env args
        body ~loc:loc_ ~loop_loc_
        ~expect_ty:(Expect_type (Stype.new_type_var Tvar_normal))
  | Pexpr_for { binders; condition; continue_block; body; loc_; for_else } ->
      check_for ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
        ~expect_ty:(Expect_type (Stype.new_type_var Tvar_normal))
        env binders condition continue_block body for_else ~loc:loc_
  | Pexpr_foreach
      {
        binders;
        expr =
          Pexpr_infix
            {
              op = { var_name = Lident (("..<" | "..=") as op) };
              lhs;
              rhs;
              loc_ = operator_loc;
            };
        body;
        else_block;
        loc_;
      } ->
      let inclusive = op = "..=" in
      typing_range_for_in env binders lhs rhs body else_block ~inclusive
        ~operator_loc ~loc:loc_
        ~expect_ty:(Expect_type (Stype.new_type_var Tvar_normal))
        ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  | Pexpr_foreach { binders; expr; body; else_block; loc_ } ->
      typing_foreach ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
        ~expect_ty:(Expect_type (Stype.new_type_var Tvar_normal))
        env binders expr body else_block ~loc:loc_
  | Pexpr_return { return_value; loc_ } ->
      let ty = Stype.new_type_var Tvar_normal in
      check_return_value env return_value ty ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics ~loc:loc_
  | Pexpr_raise { err_value; loc_ } ->
      let ty = Stype.new_type_var Tvar_normal in
      typing_raise env err_value ty ~loc:loc_ ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics
  | Pexpr_map { elems; loc_ } ->
      typing_map_expr env elems ~expect_ty:None ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics ~loc:loc_
  | Pexpr_static_assert { asserts; body } ->
      typing_static_assert ~global_env ~tvar_env ~control_ctx ~cenv ~diagnostics
        ~expect_ty:None env asserts body
  | Pexpr_group _ -> assert false

and check_return_value (env : Local_env.t) (return_value : Syntax.expr option)
    (expected_ty : Stype.t) ~loc ~control_ctx ~tvar_env ~cenv ~global_env
    ~diagnostics : Typedtree.expr =
  match control_ctx.return with
  | Some ret_ty ->
      let return_value =
        match return_value with
        | Some ret_value ->
            Some
              (check_expr env ret_value (Expect_type ret_ty) ~control_ctx
                 ~tvar_env ~cenv ~global_env ~diagnostics)
        | None ->
            Ctype.unify_expr ~expect_ty:ret_ty ~actual_ty:Stype.unit loc
            |> store_error ~diagnostics;
            None
      in
      Texpr_return { return_value; ty = expected_ty; loc_ = loc }
  | None -> (
      add_error diagnostics (Errors.invalid_return loc);
      match return_value with
      | Some ret_value ->
          infer_expr env ret_value ~control_ctx ~tvar_env ~cenv ~global_env
            ~diagnostics
      | None -> Texpr_unit { loc_ = loc })

and typing_raise (env : Local_env.t) (err_value : Syntax.expr)
    (expect_ty : Stype.t) ~loc ~control_ctx ~tvar_env ~cenv ~global_env
    ~diagnostics : Typedtree.expr =
  let loc_ = loc in
  let make_raise error_value : Typedtree.expr =
    Texpr_raise { error_value; ty = expect_ty; loc_ }
      [@@local]
  in
  control_ctx.has_error := true;
  match control_ctx.error_ctx with
  | Some error_ctx ->
      let error_value =
        infer_expr env err_value ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let actual_ty = Typedtree_util.type_of_typed_expr error_value in
      Poly_type.add_constraint cenv actual_ty
        {
          trait = Type_path.Builtin.type_path_error;
          loc_ = loc;
          required_by_ = [];
        };
      Control_ctx.check_error_in_ctx ~error_ty:actual_ty ~ctx:error_ctx loc_
      |> store_error ~diagnostics;
      make_raise error_value
  | None ->
      add_error diagnostics (Errors.invalid_raise ~kind:`Raise ~loc);
      let error_value =
        infer_expr env err_value ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      Poly_type.add_constraint cenv
        (Typedtree_util.type_of_typed_expr error_value)
        {
          trait = Type_path.Builtin.type_path_error;
          loc_ = loc;
          required_by_ = [];
        };
      Texpr_raise { error_value; ty = expect_ty; loc_ }

and typing_infix_op (env : Local_env.t) (op : Syntax.var) (lhs : Typedtree.expr)
    (rhs : Syntax.expr) ~loc ~global_env ~cenv ~tvar_env ~control_ctx
    ~diagnostics : Typedtree.expr =
  let op_name =
    match op with { var_name = Lident name; _ } -> name | _ -> assert false
  in
  let op_info = Operators.find_exn op_name in
  let ty_lhs = Typedtree_util.type_of_typed_expr lhs in
  let op_expr, rhs =
    match
      ( Specialize_operator.try_specialize_op ~op_info ~ty_lhs ~loc:op.loc_,
        op_info )
    with
    | Some tmethod, _ -> (tmethod, Not_yet rhs)
    | None, { impl = Regular_function; method_name; _ } ->
        let op_expr =
          infer_expr env
            (Pexpr_ident
               {
                 id = { op with var_name = Lident method_name };
                 loc_ = op.loc_;
               })
            ~global_env ~cenv ~tvar_env ~control_ctx ~diagnostics
        in
        (op_expr, Not_yet rhs)
    | None, { impl = Trait_method trait; method_name; _ } ->
        let op : Typedtree.expr =
          match Global_env.find_trait_by_path global_env trait with
          | None ->
              add_error diagnostics
                (Errors.pkg_not_imported ~name:(Type_path.get_pkg trait)
                   ~action:("use operator `" ^ op_name ^ "`" : Stdlib.String.t)
                   ~loc:op.loc_);
              Texpr_hole
                {
                  ty = Stype.new_type_var Tvar_error;
                  kind = Synthesized;
                  loc_ = op.loc_;
                }
          | Some trait_decl -> (
              match
                Trait_decl.find_method trait_decl method_name ~loc:op.loc_
              with
              | Error err ->
                  add_error diagnostics err;
                  Texpr_hole
                    {
                      ty = Stype.new_type_var Tvar_error;
                      kind = Synthesized;
                      loc_ = op.loc_;
                    }
              | Ok meth_decl ->
                  let trait_name = trait_decl.name in
                  let self_type = Stype.new_type_var Tvar_normal in
                  let ty =
                    Poly_type.instantiate_method_decl meth_decl ~self:self_type
                  in
                  Trait_closure.compute_closure
                    ~types:(Global_env.get_all_types global_env)
                    [
                      { trait = trait_name; loc_ = op.loc_; required_by_ = [] };
                    ]
                  |> List.iter (Poly_type.add_constraint cenv self_type);
                  let trait_name : Typedtree.type_path_loc =
                    { name = trait_name; loc_ = Rloc.no_location }
                  in
                  Texpr_unresolved_method
                    {
                      trait_name;
                      method_name;
                      self_type;
                      arity_ = Some meth_decl.method_arity;
                      ty;
                      loc_ = op.loc_;
                    })
        in
        (op, Not_yet rhs)
    | None, { impl = Method | Method_with_default _; _ } ->
        let ty_self, rhs_maybe_typed =
          if not (Typeutil.is_tvar ty_lhs) then (ty_lhs, Not_yet rhs)
          else
            let trhs =
              infer_expr env rhs ~global_env ~cenv ~tvar_env ~control_ctx
                ~diagnostics
            in
            let ty_rhs = Typedtree_util.type_of_typed_expr trhs in
            if not (Typeutil.is_tvar ty_rhs) then (ty_rhs, Typechecked trhs)
            else
              match op_info.impl with
              | Method_with_default
                  (( T_unit | T_bool | T_byte | T_char | T_int | T_int64
                   | T_uint | T_uint64 | T_float | T_double | T_string ) as
                   default) ->
                  let self_ty : Stype.t =
                    T_constr
                      {
                        type_constructor = default;
                        tys = [];
                        generic_ = false;
                        only_tag_enum_ = false;
                        is_suberror_ = false;
                      }
                  in
                  (self_ty, Typechecked trhs)
              | Method_with_default
                  ( T_option | T_result | T_error_value_result | T_fixedarray
                  | T_bytes | T_ref | T_error | Toplevel _ | Tuple _ | Constr _
                    ) ->
                  assert false
              | Method | Regular_function | Trait_method _ ->
                  (ty_lhs, Typechecked trhs)
        in
        let label : Syntax.label =
          { label_name = op_info.method_name; loc_ = op.loc_ }
        in
        let tmethod =
          typing_self_method ty_self label ~global_env ~tvar_env ~cenv
            ~src:(Type_constraint.Dot_src_infix op_info)
            ~loc:(Rloc.merge (Typedtree.loc_of_typed_expr lhs) op.loc_)
            ~diagnostics
        in
        (tmethod, rhs_maybe_typed)
  in
  match[@warning "-fragile-match"]
    Type.filter_arrow ~blame:Filtered_type ~has_error:false 2
      (Typedtree_util.type_of_typed_expr op_expr)
      loc
    |> take_info_partial ~diagnostics
  with
  | [ ty_param1; ty_param2 ], ty_res, ty_err ->
      (match ty_err with
      | None -> ()
      | Some err_ty ->
          let err_ty_str = Printer.type_to_string err_ty in
          let err = Errors.unhandled_error ~err_ty:err_ty_str ~loc in
          add_error diagnostics err);
      let tlhs =
        unify_expr_allow_trait_upcast ~global_env ~cenv ~expect_ty:ty_param1
          ~actual_ty:(Typedtree_util.type_of_typed_expr lhs)
          (Typedtree.loc_of_typed_expr lhs)
        |> handle_unify_result ~diagnostics ~expr:lhs
      in
      let trhs =
        maybe_check env rhs ty_param2 ~tvar_env ~control_ctx ~cenv ~global_env
          ~diagnostics
      in
      Texpr_apply
        {
          func = op_expr;
          args =
            [
              { arg_value = tlhs; arg_kind = Positional };
              { arg_value = trhs; arg_kind = Positional };
            ];
          ty = ty_res;
          kind_ = Infix;
          loc_ = loc;
        }
  | _ -> assert false

and typing_application ?(expect_ty : Stype.t option)
    ?(self : Typedtree.expr option) (env : Local_env.t) (func : Typedtree.expr)
    (args : Syntax.argument list) ~(kind : Typedtree.apply_kind) ~loc
    ~control_ctx ~tvar_env ~cenv ~global_env ~(attr : Syntax.apply_attr)
    ~diagnostics : Typedtree.expr =
  let n = List.length args + match self with None -> 0 | _ -> 1 in
  let func_ty = Stype.type_repr (Typedtree_util.type_of_typed_expr func) in
  let err_ty = Stype.new_type_var Tvar_normal in
  let check_error_type actual_err_type =
    match attr with
    | Question -> (
        match actual_err_type with
        | None ->
            add_error diagnostics
              (Errors.invalid_apply_attr ~kind:`NoErrorType ~loc)
        | Some actual_err_ty -> Ctype.unify_exn err_ty actual_err_ty)
    | Exclamation -> (
        control_ctx.has_error := true;
        match actual_err_type with
        | None ->
            add_error diagnostics
              (Errors.invalid_apply_attr ~kind:`NoErrorType ~loc)
        | Some actual_err_type -> (
            match control_ctx.error_ctx with
            | None ->
                let error = Errors.invalid_raise ~kind:`Rethrow ~loc in
                add_error diagnostics error
            | Some error_ctx ->
                Control_ctx.check_error_in_ctx ~error_ty:actual_err_type
                  ~ctx:error_ctx loc
                |> store_error ~diagnostics))
    | No_attr -> (
        match actual_err_type with
        | None -> ()
        | Some actual_err_type ->
            let err_ty = Printer.type_to_string actual_err_type in
            add_error diagnostics (Errors.unhandled_error ~err_ty ~loc))
  in
  let delayed_error_type_check = ref None in
  let ty_params, ty_res, should_report_error =
    match func_ty with
    | Tarrow { params_ty; ret_ty; err_ty } ->
        (match err_ty with
        | Some t when Typeutil.is_tvar t ->
            delayed_error_type_check := Some (fun () -> check_error_type err_ty)
        | _ -> check_error_type err_ty);
        (params_ty, ret_ty, true)
    | Tvar ({ contents = Tnolink tvar_kind } as link) ->
        let params_ty = Type.type_var_list n tvar_kind in
        let ret_ty = Stype.new_type_var tvar_kind in
        link :=
          Tlink (Tarrow { params_ty; ret_ty; err_ty = None; generic_ = false });
        if attr <> No_attr then
          add_error diagnostics
            (Errors.invalid_apply_attr ~kind:`UnknownType ~loc);
        if
          tvar_kind = Tvar_normal
          && Lst.exists args (fun a -> a.arg_kind <> Positional)
        then
          add_error diagnostics
            (Errors.nontoplevel_func_cannot_have_labelled_arg ~loc);
        (params_ty, ret_ty, false)
    | _ ->
        let params_ty = Type.type_var_list n Tvar_error in
        let ret_ty = Stype.new_type_var Tvar_error in
        (if func_ty <> T_blackhole then
           let expected, actual =
             ("function type", Printer.type_to_string func_ty)
           in
           add_error diagnostics (Errors.type_mismatch ~expected ~actual ~loc));
        (params_ty, ret_ty, false)
  in
  let ty_res_application =
    match kind with
    | Dot_return_self ->
        if not (Ctype.try_unify Stype.unit ty_res) then
          add_error diagnostics
            (Errors.cascade_type_mismatch
               ~actual:(Printer.type_to_string ty_res)
               ~loc);
        Typedtree_util.type_of_typed_expr (Option.get self)
    | Infix | Dot | Normal -> ty_res
  in
  let ty_res_with_attr =
    if attr = Question then
      Stype.make_result_ty ~ok_ty:ty_res_application ~err_ty
    else ty_res_application
  in
  (match expect_ty with
  | Some expect_ty ->
      Ctype.unify_expr ~expect_ty ~actual_ty:ty_res_with_attr loc
      |> store_error ~diagnostics
  | None -> ());
  let arity, error_kind =
    match func with
    | Texpr_ident { arity_ = Some arity; _ } -> (arity, "function")
    | Texpr_method { arity_ = Some arity; _ } -> (arity, "method")
    | Texpr_unresolved_method { arity_ = Some arity; _ } -> (arity, "method")
    | Texpr_constr { arity_ = arity; _ } -> (arity, "constructor")
    | _ -> (Fn_arity.simple (List.length ty_params), "function")
  in
  let typ_of_args =
    Fn_arity.to_hashtbl arity ty_params (fun param_kind ty -> (param_kind, ty))
  in
  let seen_labels = Basic_hash_string.create 17 in
  let last_positional_index = ref (-1) in
  let lookup_positional_arg () =
    incr last_positional_index;
    match
      Fn_arity.Hash.find_opt typ_of_args (Positional !last_positional_index)
    with
    | Some (_, typ) -> typ
    | None -> Stype.new_type_var Tvar_error
  in
  let lookup_labelled_arg (label : Syntax.label) =
    (match Basic_hash_string.find_opt seen_labels label.label_name with
    | Some _first_loc ->
        add_error diagnostics
          (Errors.duplicated_fn_label ~label:label.label_name
             ~second_loc:label.loc_)
    | None -> Basic_hash_string.add seen_labels label.label_name label.loc_);
    match Fn_arity.Hash.find_opt typ_of_args (Labelled label.label_name) with
    | Some _ as result -> result
    | None ->
        if should_report_error then
          add_error diagnostics
            (Errors.superfluous_arg_label ~label:label.label_name
               ~kind:error_kind ~loc:label.loc_);
        None
  in
  let check_arg (arg : Syntax.argument) : Typedtree.argument =
    let ty =
      match arg.arg_kind with
      | Positional -> lookup_positional_arg ()
      | Labelled_option { label; question_loc }
      | Labelled_option_pun { label; question_loc } -> (
          match lookup_labelled_arg label with
          | None -> Stype.new_type_var Tvar_error
          | Some (param_kind, ty) ->
              (match param_kind with
              | Question_optional _ -> ()
              | _ ->
                  add_error diagnostics
                    (Errors.invalid_question_arg_application
                       ~label:label.label_name ~loc:question_loc));
              Builtin.type_option ty)
      | Labelled label | Labelled_pun label -> (
          match lookup_labelled_arg label with
          | None -> Stype.new_type_var Tvar_error
          | Some (_, ty) -> ty)
    in
    {
      arg_value =
        check_expr env arg.arg_value (Expect_type ty) ~control_ctx ~tvar_env
          ~cenv ~global_env ~diagnostics;
      arg_kind = arg.arg_kind;
    }
  in
  let targs : Typedtree.argument list =
    match self with
    | Some self ->
        let actual_ty_self = Typedtree_util.type_of_typed_expr self in
        let expected_ty_self = lookup_positional_arg () in
        Ctype.unify_expr ~actual_ty:actual_ty_self ~expect_ty:expected_ty_self
          (Typedtree.loc_of_typed_expr self)
        |> store_error ~diagnostics;
        { arg_value = self; arg_kind = Positional } :: Lst.map args check_arg
    | None -> Lst.map args check_arg
  in
  (match !delayed_error_type_check with None -> () | Some check -> check ());
  (if should_report_error then
     let actual = !last_positional_index + 1 in
     let expected = Fn_arity.count_positional arity in
     if actual <> expected then
       add_error diagnostics
         (Errors.fn_arity_mismatch
            ~func_ty:
              (Printer.toplevel_function_type_to_string ~arity
                 (Typedtree_util.type_of_typed_expr func))
            ~expected ~actual
            ~has_label:
              ((not (Fn_arity.is_simple arity))
              || Lst.exists args (fun arg ->
                     match arg.arg_kind with
                     | Positional -> false
                     | Labelled _ | Labelled_pun _ | Labelled_option _
                     | Labelled_option_pun _ ->
                         true))
            ~loc));
  if should_report_error then (
    let missing = Vec.empty () in
    Fn_arity.iter arity (fun param_kind ->
        match param_kind with
        | Positional _ | Optional _ | Autofill _ | Question_optional _ -> ()
        | Labelled { label; _ } ->
            if not (Basic_hash_string.mem seen_labels label) then
              Vec.push missing label);
    if not (Vec.is_empty missing) then
      add_error diagnostics
        (Errors.missing_fn_label ~labels:(Vec.to_list missing) ~loc));
  let apply : Typedtree.expr =
    Texpr_apply
      { func; args = targs; ty = ty_res_application; kind_ = kind; loc_ = loc }
  in
  match attr with
  | Exclamation ->
      Texpr_exclamation
        {
          expr = apply;
          loc_ = loc;
          ty = ty_res_with_attr;
          convert_to_result = false;
        }
  | Question ->
      if kind = Dot_return_self then
        add_error diagnostics (Errors.double_exclamation_with_cascade loc);
      Texpr_exclamation
        {
          expr = apply;
          loc_ = loc;
          ty = ty_res_with_attr;
          convert_to_result = true;
        }
  | No_attr -> apply

and typing_pipe ?(expect_ty : Stype.t option) (env : Local_env.t)
    (lhs : Syntax.expr) (rhs : Syntax.expr) ~loc ~control_ctx ~tvar_env ~cenv
    ~global_env ~diagnostics : Typedtree.expr =
  let infer_go =
    infer_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let rhs_loc = Syntax.loc_of_expression rhs in
  let from_typed_apply (tapply : Typedtree.expr) : Typedtree.expr =
    match tapply with
    | Texpr_apply { func; args = { arg_value = tlhs; _ } :: args; ty; _ } ->
        Texpr_pipe
          {
            lhs = tlhs;
            rhs = Pipe_partial_apply { func; args; loc_ = rhs_loc };
            ty;
            loc_ = loc;
          }
    | Texpr_exclamation
        {
          expr =
            Texpr_apply { func; args = { arg_value = tlhs; _ } :: args; ty; _ };
          loc_;
          ty = outer_ty;
          convert_to_result;
        } ->
        Texpr_exclamation
          {
            expr =
              Texpr_pipe
                {
                  lhs = tlhs;
                  rhs = Pipe_partial_apply { func; args; loc_ = rhs_loc };
                  ty;
                  loc_ = loc;
                };
            loc_;
            ty = outer_ty;
            convert_to_result;
          }
    | _ -> assert false
      [@@inline]
  in
  let fn_go func args ~attr =
    let func = infer_go env func in
    typing_application ?expect_ty env func
      ({ arg_value = lhs; arg_kind = Positional } :: args)
      ~kind:Normal ~loc ~control_ctx ~tvar_env ~cenv ~global_env ~attr
      ~diagnostics
    |> from_typed_apply
      [@@inline]
  in
  let constr_go constr args : Typedtree.expr =
    typing_constr_or_constant constr
      (Some ({ arg_value = lhs; arg_kind = Positional } :: args))
      ~expect_ty ~env ~loc ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
    |> from_typed_apply
      [@@inline]
  in
  match rhs with
  | Pexpr_apply { func = (Pexpr_ident _ | Pexpr_method _) as func; args; attr }
    ->
      fn_go func args ~attr
  | Pexpr_ident _ | Pexpr_method _ -> fn_go rhs [] ~attr:No_attr
  | Pexpr_constr { constr } -> constr_go constr []
  | Pexpr_apply { func = Pexpr_constr { constr }; attr; args } ->
      (if attr <> No_attr then
         let error = Errors.invalid_apply_attr ~kind:`Constructor ~loc in
         add_error diagnostics error);
      constr_go constr args
  | _ ->
      let lhs = infer_go env lhs in
      add_error diagnostics (Errors.unsupported_pipe_expr rhs_loc);
      let trhs = infer_go env rhs in
      Texpr_pipe
        {
          lhs;
          rhs =
            Pipe_invalid
              {
                expr = trhs;
                ty = Typedtree_util.type_of_typed_expr trhs;
                loc_ = rhs_loc;
              };
          ty =
            (match expect_ty with
            | Some ty -> ty
            | None -> Stype.new_type_var Tvar_error);
          loc_ = loc;
        }

and typing_self_method (ty_self : Stype.t) (method_name : Syntax.label) ~src
    ~loc ~global_env ~tvar_env ~cenv ~diagnostics : Typedtree.expr =
  let name = method_name.label_name in
  match
    Type_constraint.resolve_method_by_type ty_self name ~tvar_env ~global_env
      ~src ~loc
  with
  | Ok (Known_method method_info) ->
      Docstring.check_alerts ~diagnostics
        (Docstring.pragmas method_info.doc_)
        loc;
      let ty_method, ty_args_ =
        Poly_type.instantiate_method ~cenv ~loc method_info
      in
      Ctype.unify_exn ty_self ty_self;
      Texpr_ident
        {
          id =
            {
              var_id = Ident.of_qual_ident method_info.id;
              loc_ = method_name.loc_;
            };
          ty_args_;
          arity_ = Some method_info.arity_;
          kind =
            (match method_info.prim with None -> Normal | Some p -> Prim p);
          ty = ty_method;
          loc_ = method_name.loc_;
        }
  | Ok (Promised_method { method_id; method_ty; method_arity; prim }) ->
      Texpr_ident
        {
          id = { var_id = method_id; loc_ = method_name.loc_ };
          ty_args_ = [||];
          arity_ = Some method_arity;
          kind = (match prim with None -> Normal | Some p -> Prim p);
          ty = method_ty;
          loc_ = method_name.loc_;
        }
  | Error err ->
      add_error diagnostics err;
      Texpr_ident
        {
          id = { var_id = Ident.fresh name; loc_ = method_name.loc_ };
          ty_args_ = [||];
          arity_ = None;
          kind = Normal;
          ty = Stype.new_type_var Tvar_error;
          loc_ = method_name.loc_;
        }

and typing_constr_or_constant (constr : Syntax.constructor)
    (args : Syntax.argument list option) ~(expect_ty : Stype.t option) ~env ~loc
    ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics =
  let name = constr.constr_name.name in
  let check_args ~expect_ty (constr_expr : Typedtree.expr) ~ty_constr ~ty_res =
    match args with
    | None ->
        (match expect_ty with
        | None -> ()
        | Some expect_ty ->
            Ctype.unify_constr name ~expect_ty ~actual_ty:ty_constr loc
            |> store_error ~diagnostics);
        constr_expr
    | Some args ->
        (match expect_ty with
        | None -> ()
        | Some expect_ty ->
            Ctype.unify_constr name ~expect_ty ~actual_ty:ty_res loc
            |> store_error ~diagnostics);
        typing_application ?expect_ty:None env constr_expr args ~kind:Normal
          ~loc ~control_ctx ~tvar_env ~cenv ~global_env ~attr:No_attr
          ~diagnostics
      [@@inline]
  in
  match
    Global_env.resolve_constr_or_constant global_env ~expect_ty ~constr
      ~creating_value:true
  with
  | Ok (`Constr constr_desc) ->
      let ty_res, ty_args = Poly_type.instantiate_constr constr_desc in
      let ty_constr =
        match ty_args with
        | [] -> ty_res
        | _ -> Builtin.type_arrow ty_args ty_res ~err_ty:None
      in
      let constr_expr : Typedtree.expr =
        Texpr_constr
          {
            constr;
            tag = constr_desc.cs_tag;
            ty = ty_constr;
            arity_ = constr_desc.cs_arity_;
            loc_ = constr.loc_;
          }
      in
      check_args ~expect_ty constr_expr ~ty_constr ~ty_res
  | Ok (`Constant { id; kind; typ; _ }) ->
      let constant_expr : Typedtree.expr =
        match kind with
        | Const c ->
            Texpr_constant
              {
                c;
                ty = typ;
                name_ =
                  Some { var_id = Ident.of_qual_ident id; loc_ = constr.loc_ };
                loc_ = constr.loc_;
              }
        | Prim _ | Normal ->
            Texpr_hole { ty = typ; loc_ = constr.loc_; kind = Synthesized }
      in
      check_args ~expect_ty constant_expr ~ty_constr:typ ~ty_res:typ
  | Error err ->
      add_error diagnostics err;
      let ty = Stype.new_type_var Tvar_error in
      let constr_expr : Typedtree.expr =
        let arity_ =
          match args with
          | None -> Fn_arity.simple 0
          | Some args -> Fn_arity.simple (List.length args)
        in
        Texpr_constr
          { constr; tag = Typeutil.unknown_tag; ty; arity_; loc_ = loc }
      in
      check_args ~expect_ty constr_expr ~ty_constr:ty ~ty_res:T_blackhole

and typing_array_spread (env : Local_env.t)
    (elems : Syntax.spreadable_elem list) ~(loc : Rloc.t)
    ~(expect_ty : Stype.t option) ~(control_ctx : Control_ctx.t)
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let ty_elem = Stype.new_type_var Tvar_normal in
  let ty = Builtin.type_array ty_elem in
  (match expect_ty with
  | Some expect_ty ->
      Ctype.unify_expr ~expect_ty ~actual_ty:ty loc |> store_error ~diagnostics
  | None -> ());
  let self : Typedtree.expr =
    Texpr_array
      { exprs = []; ty; is_fixed_array = false; loc_ = Rloc.no_location }
  in
  let push : Typedtree.expr =
    typing_self_method ty
      { label_name = "push"; loc_ = Rloc.no_location }
      ~src:Dot_src_direct ~loc:Rloc.no_location ~global_env ~tvar_env ~cenv
      ~diagnostics
  in
  Ctype.unify_exn
    (Typedtree_util.type_of_typed_expr push)
    (Builtin.type_arrow [ ty; ty_elem ] Stype.unit ~err_ty:None);
  let push_iter : Typedtree.expr =
    typing_self_method ty
      { label_name = "push_iter"; loc_ = Rloc.no_location }
      ~src:Dot_src_direct ~loc:Rloc.no_location ~global_env ~tvar_env ~cenv
      ~diagnostics
  in
  let ty_elem_iter = Builtin.type_iter ty_elem in
  Ctype.unify_exn
    (Typedtree_util.type_of_typed_expr push_iter)
    (Builtin.type_arrow [ ty; ty_elem_iter ] Stype.unit ~err_ty:None);
  Lst.fold_left elems self (fun self elem ->
      match elem with
      | Elem_regular expr ->
          let texpr =
            check_expr env expr (Expect_type ty_elem) ~control_ctx ~tvar_env
              ~cenv ~global_env ~diagnostics
          in
          Texpr_apply
            {
              func = push;
              args =
                [
                  { arg_value = self; arg_kind = Positional };
                  { arg_value = texpr; arg_kind = Positional };
                ];
              kind_ = Dot_return_self;
              ty;
              loc_ = Rloc.no_location;
            }
      | Elem_spread { expr; loc_ } ->
          let iter_expr : Syntax.expr =
            Pexpr_dot_apply
              {
                self = expr;
                method_name = { label_name = "iter"; loc_ };
                args = [];
                return_self = false;
                attr = No_attr;
                loc_;
              }
          in
          let texpr =
            check_expr env iter_expr (Expect_type ty_elem_iter) ~control_ctx
              ~tvar_env ~cenv ~global_env ~diagnostics
          in
          Texpr_apply
            {
              func = push_iter;
              args =
                [
                  { arg_value = self; arg_kind = Positional };
                  { arg_value = texpr; arg_kind = Positional };
                ];
              kind_ = Dot_return_self;
              ty;
              loc_ = Rloc.no_location;
            })

and typing_map_expr (env : Local_env.t) (elems : Syntax.map_expr_elem list)
    ~(loc : Rloc.t) ~(expect_ty : Stype.t option) ~(control_ctx : Control_ctx.t)
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let map_tpath =
    Type_path.toplevel_type ~pkg:Basic_config.builtin_package "Map"
  in
  match
    Global_env.find_regular_method global_env ~type_name:map_tpath
      ~method_name:"from_array"
  with
  | None ->
      add_error diagnostics
        (Errors.pkg_not_imported ~name:Basic_config.builtin_package
           ~action:
             ("create value of type " ^ Type_path_util.name map_tpath
               : Stdlib.String.t)
           ~loc);
      Texpr_hole
        { ty = Stype.new_type_var Tvar_error; loc_ = loc; kind = Synthesized }
  | Some method_info ->
      let ty, ty_args_ = Poly_type.instantiate_method ~cenv ~loc method_info in
      let map_from_array : Typedtree.expr =
        Texpr_ident
          {
            id =
              {
                var_id = Ident.of_qual_ident method_info.id;
                loc_ = Rloc.no_location;
              };
            kind = Normal;
            arity_ = Some method_info.arity_;
            ty_args_;
            ty;
            loc_ = Rloc.no_location;
          }
      in
      let elems_array : Syntax.expr =
        Pexpr_array
          {
            exprs =
              Lst.map elems
                (fun (Map_expr_elem { key; expr; key_loc_; loc_ }) ->
                  let key_expr : Syntax.expr =
                    Pexpr_constant { c = key; loc_ = key_loc_ }
                  in
                  Syntax.Pexpr_tuple { exprs = [ key_expr; expr ]; loc_ });
            loc_ = Rloc.no_location;
          }
      in
      let is_json, expect_ty =
        match expect_ty with
        | Some expect_ty when Type.same_type expect_ty Stype.json ->
            (true, Some (Builtin.type_map Stype.string Stype.json))
        | _ -> (false, expect_ty)
      in
      let result =
        typing_application env map_from_array
          [ { arg_value = elems_array; arg_kind = Positional } ]
          ?expect_ty ~kind:Normal ~loc ~control_ctx ~tvar_env ~cenv ~global_env
          ~attr:No_attr ~diagnostics
      in
      if is_json then
        Json_literal.make_json_expr ~global_env ~diagnostics ~loc
          Json_literal.object_ result
      else result

and typing_static_assert ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
    ~expect_ty env asserts body =
  let has_error = ref false in
  Lst.iter asserts (fun { assert_type; assert_trait; assert_loc; assert_msg } ->
      let aux_diagnostics = Local_diagnostics.make ~base:Loc.no_location in
      let type_ =
        typing_type ~diagnostics:aux_diagnostics ~tvar_env ~global_env
          assert_type
        |> Typedtree_util.stype_of_typ
      in
      if Local_diagnostics.has_fatal_errors aux_diagnostics then
        has_error := true
      else
        match
          Global_env.All_types.find_trait
            (Global_env.get_all_types global_env)
            assert_trait ~loc:assert_loc
        with
        | Error _ -> has_error := true
        | Ok trait ->
            let aux_cenv = Poly_type.make () in
            Poly_type.add_constraint aux_cenv type_
              { trait = trait.name; required_by_ = []; loc_ = Rloc.no_location };
            Type_constraint.solve_constraints aux_cenv ~tvar_env ~global_env
              ~diagnostics:aux_diagnostics;
            if Local_diagnostics.has_fatal_errors aux_diagnostics then (
              add_error diagnostics
                (Errors.static_assert_failure
                   ~type_:(Printer.type_to_string type_)
                   ~trait:(Type_path_util.name trait.name)
                   ~required_by:assert_msg ~loc:assert_loc);
              has_error := true));
  if !has_error then
    Texpr_hole
      {
        ty = Stype.new_type_var Tvar_error;
        loc_ = Rloc.no_location;
        kind = Synthesized;
      }
  else
    match expect_ty with
    | None ->
        infer_expr env body ~global_env ~tvar_env ~control_ctx ~cenv
          ~diagnostics
    | Some expect_ty ->
        check_expr env body expect_ty ~global_env ~tvar_env ~control_ctx ~cenv
          ~diagnostics

and check_expr (env : Local_env.t) (expr : Syntax.expr) (expect_ty : expect_ty)
    ~(control_ctx : Control_ctx.t) ~(tvar_env : Tvar_env.t)
    ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let check_go =
    check_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let fallback expect_ty =
    check_expr_no_cast env expr expect_ty ~control_ctx ~tvar_env ~cenv
      ~global_env ~diagnostics
      [@@inline]
  in
  match (expect_ty, expr) with
  | Expect_type expect_ty, _ when Typeutil.is_trait expect_ty ->
      let expr =
        infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let actual_ty = Typedtree_util.type_of_typed_expr expr in
      let loc_ = Typedtree.loc_of_typed_expr expr in
      unify_expr_allow_trait_upcast ~global_env ~cenv ~expect_ty ~actual_ty loc_
      |> handle_unify_result ~diagnostics ~expr
  | _, Pexpr_sequence { expr1; expr2; loc_ } ->
      let expr1 = check_go env expr1 Ignored in
      let expr2 = check_go env expr2 expect_ty in
      Texpr_sequence
        { expr1; expr2; ty = Typedtree_util.type_of_typed_expr expr2; loc_ }
  | _, Pexpr_let { pattern = p; expr = e; body; loc_; _ } ->
      let pat_binders, tp, te1 =
        typing_let env p e ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let body =
        check_go (Typeutil.add_pat_binders env pat_binders) body expect_ty
      in
      let ty = Typedtree_util.type_of_typed_expr body in
      Texpr_let { pat = tp; rhs = te1; body; ty; pat_binders; loc_ }
  | _, Pexpr_letfn { name; func; body; loc_ } -> (
      match[@warning "-fragile-match"]
        typing_letrec env
          [ (name, func) ]
          ~tvar_env ~cenv ~global_env ~diagnostics
      with
      | env_with_funs, (binder, fn) :: [] ->
          let body = check_go env_with_funs body expect_ty in
          let ty = Typedtree_util.type_of_typed_expr body in
          Texpr_letfn
            {
              binder;
              fn;
              body;
              ty;
              loc_;
              is_rec = Typedtree_util.is_rec binder fn;
            }
      | _ -> assert false)
  | _, Pexpr_letrec { bindings = funs; body; loc_ } ->
      let env_with_funs, bindings =
        typing_letrec env funs ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let body = check_go env_with_funs body expect_ty in
      let ty = Typedtree_util.type_of_typed_expr body in
      Texpr_letrec { bindings; body; ty; loc_ }
  | _, Pexpr_letmut { binder; ty; expr; body; loc_ } ->
      let env, binder, expr, konstraint =
        typing_letmut env binder ty expr ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
      in
      let body = check_go env body expect_ty in
      let ty = Typedtree_util.type_of_typed_expr body in
      Texpr_letmut { binder; konstraint; expr; body; ty; loc_ }
  | _, Pexpr_if { cond; ifso; ifnot = None; loc_ } ->
      let cond = check_go env cond (Expect_type Stype.bool) in
      let ifso = check_go env ifso Ignored in
      (match expect_ty with
      | Ignored -> ()
      | Expect_type expect_ty ->
          Ctype.unify_expr ~expect_ty ~actual_ty:Stype.unit loc_
          |> store_error ~diagnostics);
      Texpr_if { cond; ifso; ifnot = None; ty = Stype.unit; loc_ }
  | _, Pexpr_if { cond; ifso; ifnot = Some ifnot; loc_ } ->
      let cond = check_go env cond (Expect_type Stype.bool) in
      let ifso = check_go env ifso expect_ty in
      let ifnot = check_go env ifnot expect_ty in
      let ty = Typedtree_util.type_of_typed_expr ifso in
      Texpr_if { cond; ifso; ifnot = Some ifnot; ty; loc_ }
  | _, Pexpr_guard { cond; otherwise; body; loc_ } ->
      let cond = check_go env cond (Expect_type Stype.bool) in
      let otherwise =
        match otherwise with
        | None -> None
        | Some otherwise -> Some (check_go env otherwise expect_ty)
      in
      let body = check_go env body expect_ty in
      let ty = Typedtree_util.type_of_typed_expr body in
      Texpr_guard { cond; otherwise; body; ty; loc_ }
  | _, Pexpr_guard_let { pat; expr; otherwise; body; loc_ } ->
      let pat_binders, pat, rhs =
        typing_let env pat expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let check_cases cases pat_ty action_ty =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p pat_ty ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action action_ty
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
          [@@inline]
      in
      let otherwise =
        match otherwise with
        | None -> None
        | Some otherwise ->
            Some
              (check_cases otherwise
                 (Typedtree_util.type_of_typed_expr rhs)
                 expect_ty)
      in
      let body =
        check_go (Typeutil.add_pat_binders env pat_binders) body expect_ty
      in
      let ty = Typedtree_util.type_of_typed_expr body in
      Texpr_guard_let { pat; rhs; pat_binders; otherwise; body; ty; loc_ }
  | ( _,
      Pexpr_try
        {
          body;
          catch;
          catch_all;
          try_else;
          try_loc_;
          catch_loc_;
          else_loc_;
          loc_;
        } ) ->
      let error_ctx = ref (Control_ctx.Open_ctx Empty_ctx) in
      let new_ctx = Control_ctx.with_error_ctx ~error_ctx control_ctx in
      let body_expect_ty =
        if try_else = None then expect_ty
        else Expect_type (Stype.new_type_var Tvar_normal)
      in
      let body =
        check_expr ~control_ctx:new_ctx ~tvar_env ~cenv ~global_env ~diagnostics
          env body body_expect_ty
      in
      let check_cases cases pat_ty action_ty =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p pat_ty ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action action_ty
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
          [@@inline]
      in
      let error_ty = Control_ctx.error_ctx_to_stype !error_ctx in
      let catch = check_cases catch error_ty expect_ty in
      let try_else =
        match try_else with
        | None -> None
        | Some try_else ->
            Some
              (check_cases try_else
                 (Typedtree_util.type_of_typed_expr body)
                 expect_ty)
      in
      if not !(new_ctx.has_error) then (
        Local_diagnostics.add_warning diagnostics
          { kind = Useless_try; loc = try_loc_ };
        match Stype.type_repr error_ty with
        | Tvar link -> link := Tlink Stype.unit
        | _ -> ());
      if catch_all then (
        control_ctx.has_error := true;
        match control_ctx.error_ctx with
        | Some error_ctx ->
            Control_ctx.check_error_in_ctx ~error_ty ~ctx:error_ctx catch_loc_
            |> store_error ~diagnostics
        | None ->
            add_error diagnostics
              (Errors.invalid_raise ~kind:`Catchall ~loc:catch_loc_));
      let ty =
        match try_else with
        | None -> Typedtree_util.type_of_typed_expr body
        | Some ({ action; _ } :: _) -> Typedtree_util.type_of_typed_expr action
        | Some [] -> (
            match expect_ty with Ignored -> Stype.unit | Expect_type ty -> ty)
      in
      Texpr_try
        {
          body;
          catch;
          catch_all;
          try_else;
          ty;
          err_ty = error_ty;
          catch_loc_;
          else_loc_;
          loc_;
        }
  | _, Pexpr_match { expr; cases; match_loc_; loc_ } ->
      let expr =
        infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let ty = Typedtree_util.type_of_typed_expr expr in
      let cases =
        Lst.map cases (fun (p, action) ->
            let pat_binders, tpat =
              Pattern_typer.check_pat p ty ~tvar_env ~cenv ~global_env
                ~diagnostics
            in
            let taction =
              check_go
                (Typeutil.add_pat_binders env pat_binders)
                action expect_ty
            in
            ({ pat = tpat; action = taction; pat_binders }
              : Typedtree.match_case))
      in
      let ty =
        match expect_ty with Ignored -> Stype.unit | Expect_type ty -> ty
      in
      Texpr_match { expr; cases; ty; match_loc_; loc_ }
  | _, Pexpr_loop { args; body; loop_loc_; loc_ } ->
      check_loop ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics env args
        body ~loc:loc_ ~loop_loc_ ~expect_ty
  | _, Pexpr_for { binders; condition; continue_block; body; for_else; loc_ } ->
      check_for ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics ~expect_ty
        env binders condition continue_block body for_else ~loc:loc_
  | ( _,
      Pexpr_foreach
        {
          binders;
          expr =
            Pexpr_infix
              {
                op = { var_name = Lident (("..<" | "..=") as op) };
                lhs;
                rhs;
                loc_ = operator_loc;
              };
          body;
          else_block;
          loc_;
        } ) ->
      let inclusive = op = "..=" in
      typing_range_for_in env binders lhs rhs body else_block ~inclusive
        ~operator_loc ~loc:loc_ ~expect_ty ~global_env ~tvar_env ~cenv
        ~control_ctx ~diagnostics
  | _, Pexpr_foreach { binders; expr; body; else_block; loc_ } ->
      typing_foreach ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
        ~expect_ty env binders expr body else_block ~loc:loc_
  | _, Pexpr_while { loop_cond; loop_body; loc_; while_else } ->
      check_while env loop_cond loop_body while_else ~expect_ty ~loc:loc_
        ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | _, Pexpr_static_assert { asserts; body } ->
      typing_static_assert ~global_env ~tvar_env ~control_ctx ~cenv ~diagnostics
        ~expect_ty:(Some expect_ty) env asserts body
  | Ignored, _ ->
      let expr =
        infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let actual_ty = Typedtree_util.type_of_typed_expr expr in
      let loc_ = Typedtree.loc_of_typed_expr expr in
      if not (Ctype.try_unify Stype.unit actual_ty) then
        add_error diagnostics
          (Errors.non_unit_cannot_be_ignored
             ~ty:(Printer.type_to_string actual_ty)
             ~loc:loc_);
      expr
  | Expect_type expect_ty, _ when Type.is_super_error expect_ty ->
      let error_value =
        infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      let error_value_ty = Typedtree_util.type_of_typed_expr error_value in
      let loc_ = Typedtree.loc_of_typed_expr error_value in
      if Typeutil.is_tvar error_value_ty then
        Ctype.unify_exn error_value_ty expect_ty
      else if
        not
          (Type.is_suberror error_value_ty || Type.is_super_error error_value_ty)
      then
        add_error diagnostics
          (Errors.not_error_subtype
             (Printer.type_to_string error_value_ty)
             loc_);
      error_value
  | Expect_type expect_ty, _ -> (
      match Global_env.get_newtype_info global_env expect_ty with
      | Some { newtype_constr; underlying_typ = _; recursive = _ } -> (
          match expr with
          | Pexpr_array_get _ | Pexpr_array_get_slice _ | Pexpr_array_set _
          | Pexpr_array_augmented_set _ | Pexpr_unary _ | Pexpr_group _ ->
              assert false
          | Pexpr_constr { constr }
          | Pexpr_apply { func = Pexpr_constr { constr }; args = _; attr = _ }
            when newtype_constr.constr_name = constr.constr_name.name ->
              fallback expect_ty
          | Pexpr_array _ | Pexpr_array_spread _ | Pexpr_constr _
          | Pexpr_apply { func = Pexpr_constr _; args = _; attr = _ }
          | Pexpr_tuple _ | Pexpr_constant _ | Pexpr_interp _ | Pexpr_function _
          | Pexpr_record _ | Pexpr_record_update _ | Pexpr_map _ -> (
              let loc = Syntax.loc_of_expression expr in
              match[@warning "-fragile-match"]
                Poly_type.instantiate_constr newtype_constr
              with
              | expect_ty', underlying_typ :: [] ->
                  Ctype.unify_exn expect_ty' expect_ty;
                  wrap_newtype_constr newtype_constr (fallback underlying_typ)
                    expect_ty loc
              | _ -> assert false)
          | Pexpr_apply _ | Pexpr_dot_apply _ | Pexpr_pipe _
          | Pexpr_constraint _ | Pexpr_field _ | Pexpr_method _ | Pexpr_infix _
          | Pexpr_ident _ | Pexpr_assign _ | Pexpr_as _ -> (
              let inferred_expr =
                infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
                  ~diagnostics
              in
              let inferred_ty =
                Typedtree_util.type_of_typed_expr inferred_expr
              in
              let loc = Syntax.loc_of_expression expr in
              match Ctype.unify_expr ~expect_ty ~actual_ty:inferred_ty loc with
              | None -> inferred_expr
              | Some err1 -> (
                  match[@warning "-fragile-match"]
                    Poly_type.instantiate_constr newtype_constr
                  with
                  | expect_ty', underlying_typ :: [] -> (
                      Ctype.unify_exn expect_ty' expect_ty;
                      match
                        Ctype.unify_expr ~expect_ty:underlying_typ
                          ~actual_ty:inferred_ty loc
                      with
                      | None ->
                          wrap_newtype_constr newtype_constr inferred_expr
                            expect_ty loc
                      | Some _err ->
                          add_error diagnostics err1;
                          inferred_expr)
                  | _ -> assert false))
          | Pexpr_while _ | Pexpr_if _ | Pexpr_try _ | Pexpr_letfn _
          | Pexpr_letrec _ | Pexpr_let _ | Pexpr_sequence _ | Pexpr_mutate _
          | Pexpr_match _ | Pexpr_letmut _ | Pexpr_hole _ | Pexpr_return _
          | Pexpr_raise _ | Pexpr_unit _ | Pexpr_break _ | Pexpr_continue _
          | Pexpr_loop _ | Pexpr_foreach _ | Pexpr_static_assert _
          | Pexpr_guard _ | Pexpr_guard_let _ | Pexpr_for _
          | Pexpr_multiline_string _ ->
              fallback expect_ty)
      | None -> fallback expect_ty)

and check_expr_no_cast (env : Local_env.t) (expr : Syntax.expr)
    (expect_ty : Stype.t) ~(control_ctx : Control_ctx.t)
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let check_go =
    check_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let infer_go =
    infer_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  match expr with
  | Pexpr_array_get _ | Pexpr_array_get_slice _ | Pexpr_array_set _
  | Pexpr_array_augmented_set _ | Pexpr_unary _ ->
      assert false
  | Pexpr_let _ | Pexpr_letfn _ | Pexpr_letrec _ | Pexpr_letmut _
  | Pexpr_sequence _ | Pexpr_guard _ | Pexpr_guard_let _ | Pexpr_match _
  | Pexpr_if _ | Pexpr_try _ | Pexpr_for _ | Pexpr_while _ | Pexpr_foreach _
  | Pexpr_loop _ | Pexpr_static_assert _ ->
      assert false
  | Pexpr_constr { constr; loc_ } ->
      typing_constr_or_constant constr None ~expect_ty:(Some expect_ty) ~env
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_apply { func = Pexpr_constr { constr }; args; attr; loc_ } ->
      (if attr <> No_attr then
         let error = Errors.invalid_apply_attr ~kind:`Constructor ~loc:loc_ in
         add_error diagnostics error);
      typing_constr_or_constant constr (Some args) ~expect_ty:(Some expect_ty)
        ~env ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  | Pexpr_record { type_name; fields; loc_; trailing = _ } -> (
      let type_name, expect_ty =
        match type_name with
        | None -> (None, expect_ty)
        | Some type_name -> (
            match
              typing_type_name ~global_env ~tvar_env type_name ~diagnostics
            with
            | Some
                ( Tname_defined ({ ty_desc = Record_type _; _ } as ty_decl),
                  type_name ) ->
                let another_expect_ty : Stype.t =
                  T_constr
                    {
                      type_constructor = ty_decl.ty_constr;
                      tys = Type.type_var_list ty_decl.ty_arity Tvar_normal;
                      generic_ = false;
                      only_tag_enum_ = false;
                      is_suberror_ = false;
                    }
                in
                Ctype.unify_expr ~expect_ty ~actual_ty:another_expect_ty loc_
                |> Option.iter (add_error diagnostics);
                (Some type_name, another_expect_ty)
            | Some (_, tast_type_name) ->
                add_error diagnostics
                  (Errors.not_a_record_type
                     ~name:(Longident.to_string type_name.name)
                     ~loc:type_name.loc_);
                (Some tast_type_name, expect_ty)
            | None -> (None, expect_ty))
      in
      if Typeutil.is_tvar expect_ty then (
        let typed_record =
          infer_record env fields loc_ ~type_name ~control_ctx ~tvar_env ~cenv
            ~global_env ~diagnostics
        in
        Ctype.unify_expr ~expect_ty
          ~actual_ty:(Typedtree_util.type_of_typed_expr typed_record)
          loc_
        |> store_error ~diagnostics;
        typed_record)
      else
        match
          Global_env.labels_of_record global_env expect_ty ~loc:loc_
            ~context:`Create
        with
        | Ok labels ->
            let _, labels =
              Poly_type.instantiate_record ~ty_record:(`Known expect_ty) labels
            in
            type_guided_record_check fields labels expect_ty ~type_name ~env
              ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics ~loc:loc_
        | Error error ->
            add_error diagnostics error;
            infer_record env fields loc_ ~type_name ~control_ctx ~tvar_env ~cenv
              ~global_env ~diagnostics)
  | Pexpr_record_update { type_name; record; fields; loc_ } -> (
      let type_name, expect_ty =
        match type_name with
        | None -> (None, expect_ty)
        | Some type_name -> (
            match
              typing_type_name ~global_env ~tvar_env type_name ~diagnostics
            with
            | Some
                ( Tname_defined ({ ty_desc = Record_type _; _ } as ty_decl),
                  type_name ) ->
                let another_expect_ty : Stype.t =
                  T_constr
                    {
                      type_constructor = ty_decl.ty_constr;
                      tys = Type.type_var_list ty_decl.ty_arity Tvar_normal;
                      generic_ = false;
                      only_tag_enum_ = false;
                      is_suberror_ = false;
                    }
                in
                Ctype.unify_expr ~expect_ty ~actual_ty:another_expect_ty loc_
                |> Option.iter (add_error diagnostics);
                (Some type_name, another_expect_ty)
            | Some (_, tast_type_name) ->
                add_error diagnostics
                  (Errors.not_a_record_type
                     ~name:(Longident.to_string type_name.name)
                     ~loc:type_name.loc_);
                (Some tast_type_name, expect_ty)
            | None -> (None, expect_ty))
      in
      let typed_old_record = check_go env record (Expect_type expect_ty) in
      match
        Global_env.labels_of_record global_env expect_ty ~loc:loc_
          ~context:`Update
      with
      | Ok all_fields ->
          let _, all_fields =
            Poly_type.instantiate_record ~ty_record:(`Known expect_ty)
              all_fields
          in
          let typed_fields, all_fields =
            ( type_guided_record_update_check env all_fields fields expect_ty
                ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics,
              all_fields )
          in
          Texpr_record_update
            {
              type_name;
              record = typed_old_record;
              all_fields;
              fields = typed_fields;
              ty = expect_ty;
              loc_;
            }
      | Error e ->
          add_error diagnostics e;
          infer_go env expr)
  | Pexpr_function { func = Match _ } -> assert false
  | Pexpr_function
      {
        func =
          Lambda
            { parameters; params_loc_; body; return_type; kind_; has_error };
        loc_;
      } ->
      let n = List.length parameters in
      let check_param_annotation (p : Syntax.parameter) expect_param_ty =
        match p.param_annot with
        | None -> (expect_param_ty, None)
        | Some syntax_ty ->
            let typedtree_ty =
              typing_type syntax_ty ~tvar_env ~global_env ~diagnostics
            in
            let sty = Typedtree_util.stype_of_typ typedtree_ty in
            Ctype.unify_param p.param_binder.binder_name
              ~expect_ty:expect_param_ty ~actual_ty:sty p.param_binder.loc_
            |> store_error ~diagnostics;
            (sty, Some typedtree_ty)
      in
      let check_return_annotation expect_return_ty expect_err_ty =
        match return_type with
        | None ->
            ( expect_return_ty,
              expect_err_ty,
              Toplevel_typer.No_error_type_annotated )
        | Some (syntax_ty, syntax_err_ty) ->
            let loc = Syntax.loc_of_type_expression syntax_ty in
            let unify expect_ty actual_ty =
              Ctype.unify_expr ~expect_ty ~actual_ty loc
              |> store_error ~diagnostics
                [@@inline]
            in
            let unify_err expect_ty actual_ty =
              match (expect_ty, actual_ty) with
              | Some expect_ty, Some actual_ty -> unify expect_ty actual_ty
              | None, Some actual_ty ->
                  let err =
                    Errors.error_type_mismatch ~loc ~expected_ty:"no error"
                      ~actual_ty:(Printer.type_to_string actual_ty)
                  in
                  add_error diagnostics err
              | Some _expect_ty, None ->
                  let err = Errors.anonymous_missing_error_annotation loc in
                  add_error diagnostics err
              | None, None -> ()
                [@@inline]
            in
            let ty = typing_type syntax_ty ~tvar_env ~global_env ~diagnostics in
            let ret_ty = Typedtree_util.stype_of_typ ty in
            unify expect_return_ty ret_ty;
            let err_sty, err_typ =
              match syntax_err_ty with
              | Error_typ { ty = err_ty } ->
                  let typ =
                    typing_type ~global_env ~tvar_env err_ty ~diagnostics
                  in
                  let sty = Typedtree_util.stype_of_typ typ in
                  (Some sty, Typedtree.Error_typ { ty = typ })
              | Default_error_typ { loc_ } ->
                  (Some Typeutil.default_err_type, Default_error_typ { loc_ })
              | No_error_typ ->
                  ( (if has_error then Some Typeutil.default_err_type else None),
                    No_error_typ )
            in
            unify_err expect_err_ty err_sty;
            (ret_ty, err_sty, Annotated (ty, err_typ))
      in
      let ({ params_ty; ret_ty; err_ty; ret_annotation }
            : Toplevel_typer.typed_fn_annotation) =
        match Stype.type_repr expect_ty with
        | Tarrow
            {
              params_ty = expect_params_ty;
              ret_ty = expect_ret_ty;
              err_ty = expect_err_ty;
            } as expect_ty ->
            let n_expect = List.length expect_params_ty in
            if n_expect = n then
              let params_ty =
                Lst.map2 parameters expect_params_ty check_param_annotation
              in
              let ret_ty, err_ty, ret_annotation =
                check_return_annotation expect_ret_ty expect_err_ty
              in
              { params_ty; ret_ty; err_ty; ret_annotation }
            else
              let ty = Printer.type_to_string expect_ty in
              let err =
                Errors.func_param_num_mismatch ~loc:loc_ ~expected:n_expect
                  ~actual:n ~ty
              in
              add_error diagnostics err;
              let expect_params_ty =
                if n > n_expect then
                  expect_params_ty
                  @ Type.type_var_list (n - n_expect) Tvar_error
                else Lst.take n expect_params_ty
              in
              let params_ty =
                Lst.map2 parameters expect_params_ty check_param_annotation
              in
              let ret_ty, err_ty, ret_annotation =
                check_return_annotation expect_ret_ty expect_err_ty
              in
              { params_ty; ret_ty; err_ty; ret_annotation }
        | Tvar ({ contents = Tnolink _ } as link) ->
            let typed_fn_annotation =
              typing_function_annotation parameters return_type ~has_error
                ~tvar_env ~global_env ~diagnostics
            in
            let params_ty = Lst.map typed_fn_annotation.params_ty fst in
            let ret_ty = typed_fn_annotation.ret_ty in
            let err_ty = typed_fn_annotation.err_ty in
            let arrow_ty =
              Stype.Tarrow { params_ty; ret_ty; err_ty; generic_ = false }
            in
            link := Tlink arrow_ty;
            typed_fn_annotation
        | expect_ty ->
            (if expect_ty <> T_blackhole then
               let expected, actual =
                 (Printer.type_to_string expect_ty, "function type")
               in
               add_error diagnostics
                 (Errors.type_mismatch ~expected ~actual ~loc:loc_));
            typing_function_annotation parameters return_type ~has_error
              ~tvar_env ~global_env ~diagnostics
      in
      let env, params =
        check_function_params env parameters params_ty ~is_global:false
          ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let error_ctx =
        match err_ty with
        | None -> None
        | Some t ->
            let ctx : Control_ctx.error_ctx =
              match Stype.type_repr t with
              | T_constr { type_constructor = Basic_type_path.T_error; _ } ->
                  Fixed_ctx Supererror
              | T_constr { type_constructor = p; is_suberror_ = true; _ } ->
                  Fixed_ctx (Suberror p)
              | Tparam { index; name_ } -> Fixed_ctx (Tparam { index; name_ })
              | Tvar { contents = Tnolink _ } as t ->
                  Ctype.unify_exn t Stype.error;
                  Open_ctx Empty_ctx
              | _ -> Open_ctx Empty_ctx
            in
            Some (ref ctx)
      in
      let control_ctx = Control_ctx.make_fn ~return:ret_ty ~error_ctx in
      let body =
        check_expr env body (Expect_type ret_ty) ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
      in
      (if not !(control_ctx.has_error) then
         match ret_annotation with
         | Annotated (_, Error_typ { ty }) ->
             let loc = Typedtree.loc_of_typ ty in
             Local_diagnostics.add_warning diagnostics
               { kind = Useless_error_type; loc }
         | Annotated (_, Default_error_typ { loc_ = loc }) | Has_error_type loc
           ->
             Local_diagnostics.add_warning diagnostics
               { kind = Useless_error_type; loc }
         | Annotated (_, No_error_typ) | No_error_type_annotated -> ());
      let ret_constraint =
        match ret_annotation with
        | Annotated r -> Some r
        | Has_error_type _ | No_error_type_annotated -> None
      in
      Texpr_function
        {
          func =
            { params; params_loc_; body; ty = expect_ty; ret_constraint; kind_ };
          ty = expect_ty;
          loc_;
        }
  | Pexpr_array { exprs; loc_ } -> (
      match Stype.type_repr expect_ty with
      | T_constr { type_constructor; tys = ty_elem :: [] }
        when Type_path.equal type_constructor
               Type_path.Builtin.type_path_fixedarray ->
          let texprs =
            Lst.map exprs (fun e -> check_go env e (Expect_type ty_elem))
          in
          Texpr_array
            { exprs = texprs; ty = expect_ty; is_fixed_array = true; loc_ }
      | T_constr { type_constructor; tys = ty_elem :: [] }
        when Type_path.equal type_constructor Type_path.Builtin.type_path_array
        ->
          let texprs =
            Lst.map exprs (fun e -> check_go env e (Expect_type ty_elem))
          in
          Texpr_array
            { exprs = texprs; ty = expect_ty; is_fixed_array = false; loc_ }
      | expect_ty when Type.same_type expect_ty Stype.json ->
          let texprs =
            Lst.map exprs (fun e -> check_go env e (Expect_type Stype.json))
          in
          Json_literal.make_json_expr ~global_env ~diagnostics ~loc:loc_
            Json_literal.array
            (Texpr_array
               {
                 exprs = texprs;
                 ty = Builtin.type_array Stype.json;
                 is_fixed_array = false;
                 loc_;
               })
      | _ ->
          let ty_elem = Stype.new_type_var Tvar_normal in
          let texprs =
            Lst.map exprs (fun e -> check_go env e (Expect_type ty_elem))
          in
          let ty = Builtin.type_array ty_elem in
          Ctype.unify_expr ~expect_ty ~actual_ty:ty loc_
          |> store_error ~diagnostics;
          Texpr_array { exprs = texprs; ty; is_fixed_array = false; loc_ })
  | Pexpr_array_spread { elems; loc_ } -> (
      match Stype.type_repr expect_ty with
      | expect_ty when Type.same_type expect_ty Stype.json ->
          typing_array_spread env elems ~loc:loc_
            ~expect_ty:(Some (Builtin.type_array Stype.json))
            ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
          |> Json_literal.make_json_expr ~global_env ~diagnostics ~loc:loc_
               Json_literal.array
      | T_constr { type_constructor; tys = elem_ty :: [] }
        when Type_path.equal type_constructor Type_path.Builtin.type_path_iter
        ->
          let array_ty = Builtin.type_array elem_ty in
          let array_iter =
            typing_self_method array_ty
              ({ label_name = "iter"; loc_ = Rloc.no_location } : Syntax.label)
              ~src:Dot_src_direct ~loc:loc_ ~global_env ~tvar_env ~cenv
              ~diagnostics
          in
          Ctype.unify_exn
            (Typedtree_util.type_of_typed_expr array_iter)
            (Builtin.type_arrow [ array_ty ] expect_ty ~err_ty:None);
          let array_expr =
            typing_array_spread env elems ~loc:loc_ ~expect_ty:(Some array_ty)
              ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
          in
          Texpr_apply
            {
              func = array_iter;
              args = [ { arg_value = array_expr; arg_kind = Positional } ];
              kind_ = Normal;
              ty = expect_ty;
              loc_;
            }
      | expect_ty ->
          typing_array_spread env elems ~loc:loc_ ~expect_ty:(Some expect_ty)
            ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics)
  | Pexpr_tuple { exprs; loc_ } ->
      let arity = List.length exprs in
      let tys =
        Type.filter_product ~blame:Filter_itself ~arity:(Some arity) expect_ty
          loc_
        |> take_info_partial ~diagnostics
      in
      let texprs =
        List.map2 (fun e ty -> check_go env e (Expect_type ty)) exprs tys
      in
      Texpr_tuple { exprs = texprs; ty = expect_ty; loc_ }
  | Pexpr_pipe { lhs; rhs; loc_ } ->
      typing_pipe env lhs rhs ~expect_ty ~loc:loc_ ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics
  | Pexpr_constant { c = Const_bool b; loc_ }
    when Type.same_type expect_ty Stype.json ->
      Json_literal.make_json_const_expr ~global_env ~diagnostics ~loc:loc_
        (if b then Json_literal.true_ else Json_literal.false_)
  | Pexpr_constant { c = Const_int rep | Const_double rep; loc_ }
    when Type.same_type expect_ty Stype.json ->
      let _, c =
        Typeutil.typing_constant (Const_double rep) ~expect_ty:None ~loc:loc_
        |> take_info_partial ~diagnostics
      in
      Json_literal.make_json_expr ~global_env ~diagnostics ~loc:loc_
        Json_literal.number
        (Texpr_constant { c; ty = Stype.double; name_ = None; loc_ })
  | Pexpr_constant { c = Const_string { string_val; string_repr = _ }; loc_ }
    when Type.same_type expect_ty Stype.json ->
      Json_literal.make_json_expr ~global_env ~diagnostics ~loc:loc_
        Json_literal.string
        (Texpr_constant
           { c = C_string string_val; ty = Stype.string; name_ = None; loc_ })
  | Pexpr_interp { elems; loc_ } when Type.same_type expect_ty Stype.json ->
      Json_literal.make_json_expr ~global_env ~diagnostics ~loc:loc_
        Json_literal.string
        (typing_interp ~expect_ty:Stype.json ~control_ctx ~global_env
           ~diagnostics ~tvar_env ~cenv env elems loc_)
  | Pexpr_constant { c; loc_ } ->
      let ty, c =
        Typeutil.typing_constant ~expect_ty:(Some expect_ty) c ~loc:loc_
        |> take_info_partial ~diagnostics
      in
      Ctype.unify_expr ~expect_ty ~actual_ty:ty loc_ |> store_error ~diagnostics;
      Texpr_constant { c; ty = expect_ty; name_ = None; loc_ }
  | Pexpr_interp { elems; loc_ } ->
      typing_interp ~expect_ty ~control_ctx ~global_env ~diagnostics ~tvar_env
        ~cenv env elems loc_
  | Pexpr_constraint { expr; ty = ty_expr; loc_ } ->
      let ty = typing_type ty_expr ~tvar_env ~global_env ~diagnostics in
      let stype = Typedtree_util.stype_of_typ ty in
      Ctype.unify_expr ~expect_ty ~actual_ty:stype loc_
      |> store_error ~diagnostics;
      Texpr_constraint
        {
          expr = check_go env expr (Expect_type stype);
          konstraint = ty;
          ty = stype;
          loc_;
        }
  | Pexpr_mutate { record; accessor; field; augmented_by; loc_ } -> (
      Ctype.unify_expr ~expect_ty ~actual_ty:Stype.unit loc_
      |> store_error ~diagnostics;
      match accessor with
      | Label label ->
          typing_mutate env record label field augmented_by loc_ ~control_ctx
            ~tvar_env ~cenv ~global_env ~diagnostics
      | Index _ | Newtype ->
          add_error diagnostics (Errors.tuple_not_mutable loc_);
          Texpr_tuple { exprs = []; ty = Stype.unit; loc_ })
  | Pexpr_return { return_value; loc_ } ->
      check_return_value env return_value expect_ty ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics ~loc:loc_
  | Pexpr_raise { err_value; loc_ } ->
      typing_raise env err_value expect_ty ~loc:loc_ ~control_ctx ~tvar_env
        ~cenv ~global_env ~diagnostics
  | Pexpr_apply { func; args; attr; loc_ } ->
      typing_application env (infer_go env func) args ~expect_ty ~kind:Normal
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~attr ~diagnostics
  | Pexpr_unit { loc_; _ } ->
      Ctype.unify_expr ~expect_ty ~actual_ty:Stype.unit loc_
      |> store_error ~diagnostics;
      Texpr_unit { loc_ }
  | Pexpr_dot_apply { self; method_name; args; return_self; attr; loc_ } ->
      let self = infer_go env self in
      let ty_self = Typedtree_util.type_of_typed_expr self in
      let method_expr =
        typing_self_method ty_self method_name
          ~src:Type_constraint.Dot_src_direct ~loc:method_name.loc_ ~tvar_env
          ~cenv ~global_env ~diagnostics
      in
      typing_application env method_expr args ~self ~expect_ty
        ~kind:(if return_self then Dot_return_self else Dot)
        ~loc:loc_ ~control_ctx ~tvar_env ~cenv ~global_env ~attr ~diagnostics
  | Pexpr_break { arg; loc_ } ->
      check_break env arg ~loc:loc_ ~expect_ty ~global_env ~tvar_env ~cenv
        ~control_ctx ~diagnostics
  | Pexpr_continue { args; loc_ } ->
      check_continue env args ~loc:loc_ ~expect_ty ~global_env ~tvar_env ~cenv
        ~control_ctx ~diagnostics
  | Pexpr_map { elems; loc_ } ->
      typing_map_expr env elems ~expect_ty:(Some expect_ty) ~control_ctx
        ~tvar_env ~cenv ~global_env ~diagnostics ~loc:loc_
  | Pexpr_hole { kind; loc_ } ->
      (match kind with
      | Incomplete -> add_error diagnostics (Errors.found_hole loc_)
      | Synthesized -> ()
      | Todo ->
          Local_diagnostics.add_warning diagnostics { kind = Todo; loc = loc_ });
      Texpr_hole { ty = expect_ty; loc_; kind }
  | Pexpr_multiline_string { elems; loc_ } ->
      check_go env
        (desugar_multiline_string ~loc_ elems)
        (Expect_type expect_ty)
  | Pexpr_field _ | Pexpr_method _ | Pexpr_infix _ | Pexpr_ident _
  | Pexpr_assign _ | Pexpr_as _ ->
      let texpr =
        infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      Ctype.unify_expr ~expect_ty
        ~actual_ty:(Typedtree_util.type_of_typed_expr texpr)
        (Syntax.loc_of_expression expr)
      |> store_error ~diagnostics;
      texpr
  | Pexpr_group _ -> assert false

and maybe_check (env : Local_env.t) (maybe : maybe_typed) (expect_ty : Stype.t)
    ~(control_ctx : Control_ctx.t) ~(tvar_env : Tvar_env.t)
    ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  match maybe with
  | Typechecked texpr ->
      unify_expr_allow_trait_upcast ~global_env ~cenv ~expect_ty
        ~actual_ty:(Typedtree_util.type_of_typed_expr texpr)
        (Typedtree.loc_of_typed_expr texpr)
      |> handle_unify_result ~diagnostics ~expr:texpr
  | Not_yet expr ->
      check_expr env expr (Expect_type expect_ty) ~control_ctx ~tvar_env ~cenv
        ~global_env ~diagnostics

and type_guided_record_check ~(type_name : Typedtree.type_name option)
    (fields : Syntax.field_def list) labels record_ty ~env ~control_ctx
    ~tvar_env ~cenv ~global_env ~diagnostics ~loc : Typedtree.expr =
  let superfluous =
    Typeutil.validate_record ~context:`Creation ~expected:labels
      (Lst.map fields (fun (Field_def { label; _ }) -> label))
      ~record_ty ~is_strict:true ~loc
    |> take_info_partial ~diagnostics
  in
  let check_label ty is_mut pos label_name =
    Lst.fold_right fields [] (fun (Field_def { label; expr; is_pun; _ }) acc ->
        if label.label_name = label_name then
          let expr =
            check_expr env expr (Expect_type ty) ~control_ctx ~tvar_env ~cenv
              ~global_env ~diagnostics
          in
          Typedtree.Field_def { label; expr; is_mut; is_pun; pos } :: acc
        else acc)
      [@@inline]
  in
  let check_superfluous label_name =
    check_label
      (Stype.new_type_var Tvar_error)
      false Typeutil.unknown_pos label_name
      [@@inline]
  in
  let fields =
    Lst.flat_map_append labels
      ~init:(Lst.concat_map superfluous check_superfluous)
      ~f:(fun ({ field_name; _ } as field_info) ->
        check_label field_info.ty_field field_info.mut field_info.pos field_name)
  in
  Texpr_record { type_name; fields; ty = record_ty; loc_ = loc }

and type_guided_record_update_check (env : Local_env.t)
    (old_fields : Typedecl_info.fields) (new_fields : Syntax.field_def list)
    (ty_record : Stype.t) ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
    : Typedtree.field_def list =
  let lookup_label (label : string) ~loc :
      Typedecl_info.field Local_diagnostics.info =
    match Lst.find_first old_fields (fun f -> f.field_name = label) with
    | Some f -> Ok f
    | None ->
        Error
          (Errors.field_not_found
             ~ty:(Printer.type_to_string ty_record)
             ~label ~loc)
  in
  let update (Field_def { label; expr; is_pun; loc_; _ } : Syntax.field_def) =
    let old = lookup_label label.label_name ~loc:loc_ in
    match old with
    | Error e ->
        add_error diagnostics e;
        Typedtree.Field_def
          {
            label;
            expr =
              infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
                ~diagnostics;
            is_mut = false;
            is_pun = false;
            pos = 0;
          }
    | Ok old ->
        let new_expr =
          check_expr env expr (Expect_type old.ty_field) ~control_ctx ~tvar_env
            ~cenv ~global_env ~diagnostics
        in
        Typedtree.Field_def
          { label; expr = new_expr; is_mut = old.mut; is_pun; pos = old.pos }
  in
  let is_dup =
    Lst.check_duplicate_opt
      (Lst.map new_fields (fun (Field_def { label; _ } : Syntax.field_def) ->
           label))
      ~equal:Syntax.equal_label
  in
  (match is_dup with
  | Some { label_name = label; loc_ = loc } ->
      add_error diagnostics
        (Errors.duplicate_record_field ~context:`Creation ~label ~loc)
  | None -> ());
  Lst.map new_fields update

and infer_record (env : Local_env.t) (fields : Syntax.field_def list) loc
    ~(type_name : Typedtree.type_name option) ~control_ctx ~tvar_env ~cenv
    ~global_env ~diagnostics : Typedtree.expr =
  let handle_error err : Typedtree.expr =
    add_error diagnostics err;
    let ty = Stype.new_type_var Tvar_error in
    let fields =
      Lst.map fields (fun (Field_def { label; expr; is_pun; _ }) ->
          Typedtree.Field_def
            {
              label;
              expr =
                infer_expr env expr ~control_ctx ~tvar_env ~cenv ~global_env
                  ~diagnostics;
              is_mut = false;
              is_pun;
              pos = 0;
            })
    in
    Texpr_record { type_name; fields; ty; loc_ = loc }
  in
  if fields = [] then handle_error (Errors.record_type_missing loc)
  else
    let labels = Lst.map fields (fun (Field_def { label; _ }) -> label) in
    match Lst.check_duplicate_opt labels ~equal:Syntax.equal_label with
    | Some { label_name = label; loc_ = loc } ->
        handle_error
          (Errors.duplicate_record_field ~context:`Creation ~label ~loc)
    | None -> (
        match Global_env.resolve_record global_env ~labels ~loc with
        | Error err -> handle_error err
        | Ok (ty_params, ty_record, labels) ->
            let ty_record, labels =
              Poly_type.instantiate_record
                ~ty_record:(`Generic (ty_params, ty_record))
                labels
            in
            type_guided_record_check fields labels ty_record ~type_name ~env
              ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics ~loc)

and infer_function (env : Local_env.t) (params : Syntax.parameters)
    (params_loc_ : Rloc.t) (body : Syntax.expr)
    (return_type : (Syntax.typ * Syntax.error_typ) option)
    ~(kind_ : Syntax.fn_kind) ~tvar_env ~cenv ~global_env ~has_error
    ~diagnostics : Typedtree.fn * Stype.t =
  let typed_fn_annotation =
    typing_function_annotation params return_type ~tvar_env ~global_env
      ~has_error ~diagnostics
  in
  check_function ~is_impl_method:false env params params_loc_ body
    typed_fn_annotation ~kind_ ~is_global:false ~is_in_test:false ~tvar_env
    ~cenv ~global_env ~diagnostics

and typing_function_annotation (params : Syntax.parameters)
    (return_type : (Syntax.typ * Syntax.error_typ) option) ~(has_error : bool)
    ~tvar_env ~global_env ~diagnostics : Toplevel_typer.typed_fn_annotation =
  let typing_annotation ann_opt =
    match ann_opt with
    | Some ann ->
        let ty = typing_type ann ~tvar_env ~global_env ~diagnostics in
        let stype = Typedtree_util.stype_of_typ ty in
        (stype, Some ty)
    | None -> (Stype.new_type_var Tvar_normal, None)
  in
  let ret_sty, err_sty, annotation =
    match return_type with
    | None ->
        ( Stype.new_type_var Tvar_normal,
          (if has_error then Some Typeutil.default_err_type else None),
          Toplevel_typer.No_error_type_annotated )
    | Some (res_ty, err_ty) ->
        let typ = typing_type res_ty ~tvar_env ~global_env ~diagnostics in
        let err_sty, err_typ =
          match err_ty with
          | Error_typ { ty = err_ty } ->
              let typ = typing_type ~global_env ~tvar_env err_ty ~diagnostics in
              ( Some (Typedtree_util.stype_of_typ typ),
                Typedtree.Error_typ { ty = typ } )
          | Default_error_typ { loc_ } ->
              (Some Typeutil.default_err_type, Default_error_typ { loc_ })
          | No_error_typ ->
              ( (if has_error then Some Typeutil.default_err_type else None),
                No_error_typ )
        in
        (Typedtree_util.stype_of_typ typ, err_sty, Annotated (typ, err_typ))
  in
  {
    params_ty =
      Lst.map params (fun (p : Syntax.parameter) ->
          typing_annotation p.param_annot);
    ret_ty = ret_sty;
    err_ty = err_sty;
    ret_annotation = annotation;
  }

and typing_interp ?expect_ty ~control_ctx ~global_env ~diagnostics ~tvar_env
    ~cenv env elems loc_ =
  let infer_go =
    infer_expr ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let elems =
    Lst.map elems (function
      | Syntax.Interp_source _ -> assert false
      | Interp_lit { str; repr = _ } -> Typedtree.Interp_lit str
      | Interp_expr { expr; loc_ } ->
          let expr = infer_go env expr in
          let to_string =
            typing_self_method
              (Typedtree_util.type_of_typed_expr expr)
              ({ label_name = "to_string"; loc_ = Rloc.no_location }
                : Syntax.label)
              ~src:Type_constraint.Dot_src_direct ~loc:loc_ ~global_env
              ~tvar_env ~cenv ~diagnostics
          in
          let self_ty = Typedtree_util.type_of_typed_expr expr in
          let actual = Typedtree_util.type_of_typed_expr to_string in
          (if
             not
               (Ctype.try_unify
                  (Builtin.type_arrow [ self_ty ] Stype.string ~err_ty:None)
                  actual)
           then
             let self_ty, actual = Printer.type_pair_to_string self_ty actual in
             add_error diagnostics
               (Errors.interp_to_string_incorrect_type ~self_ty ~actual
                  ~loc:loc_));
          Interp_expr { expr; to_string; loc_ })
  in
  let ty =
    match expect_ty with
    | Some ty when Type.same_type ty Stype.json -> Stype.json
    | Some expect_ty ->
        Ctype.unify_expr ~expect_ty ~actual_ty:Stype.string loc_
        |> store_error ~diagnostics;
        Stype.string
    | None -> Stype.string
  in
  Texpr_interp { elems; ty; loc_ }

and check_function_params (env : Local_env.t) (params : Syntax.parameters)
    (param_types : (Stype.t * Typedtree.typ option) list) ~(is_global : bool)
    ~tvar_env ~cenv ~global_env ~diagnostics : Local_env.t * Typedtree.params =
  let env = ref env in
  let params =
    Lst.map2 params param_types (fun (p : Syntax.parameter) (stype, ty) ->
        if (not is_global) && p.param_kind <> Positional then
          add_error diagnostics
            (Errors.no_local_labelled_function p.param_binder.loc_);
        let kind : Typedtree.param_kind =
          match p.param_kind with
          | Positional -> Positional
          | Labelled -> Labelled
          | Optional { default = Pexpr_hole { kind = Incomplete; loc_ } } ->
              if
                Type.same_type stype Stype.type_sourceloc
                || Type.same_type stype Stype.type_argsloc
              then Autofill
              else (
                add_error diagnostics
                  (Errors.unsupported_autofill
                     ~ty:(Printer.type_to_string stype)
                     ~loc:p.param_binder.loc_);
                Optional (Texpr_hole { loc_; ty = stype; kind = Incomplete }))
          | Optional { default } ->
              Optional
                (check_expr !env default (Expect_type stype) ~global_env
                   ~tvar_env ~cenv ~diagnostics ~control_ctx:Control_ctx.empty)
          | Question_optional -> Question_optional
        in
        let param_ty =
          match kind with
          | Question_optional -> Builtin.type_option stype
          | Positional | Labelled | Optional _ | Autofill -> stype
        in
        let binder = Typeutil.fresh_binder p.param_binder in
        env := Typeutil.add_binder !env binder ~typ:param_ty ~mut:false;
        Typedtree.Param { binder; ty = param_ty; konstraint = ty; kind })
  in
  (!env, params)

and check_function (env : Local_env.t) (params : Syntax.parameters)
    (params_loc_ : Rloc.t) (body : Syntax.expr)
    (typed_fn_annotation : Toplevel_typer.typed_fn_annotation)
    ~(kind_ : Syntax.fn_kind) ~(is_global : bool) ~tvar_env ~cenv ~global_env
    ~is_in_test ~is_impl_method ~diagnostics : Typedtree.fn * Stype.t =
  let ({
         params_ty = param_types;
         ret_ty = return_type;
         err_ty = error_type;
         ret_annotation = annotation;
       }
        : Toplevel_typer.typed_fn_annotation) =
    typed_fn_annotation
  in
  let env, params =
    check_function_params env params param_types ~is_global ~tvar_env ~cenv
      ~global_env ~diagnostics
  in
  let error_ctx =
    match error_type with
    | None -> None
    | Some t ->
        let ctx : Control_ctx.error_ctx =
          match Stype.type_repr t with
          | T_constr { type_constructor = Basic_type_path.T_error; _ } ->
              Fixed_ctx Supererror
          | T_constr { type_constructor = p; is_suberror_ = true; _ } ->
              Fixed_ctx (Suberror p)
          | Tparam { index; name_ } -> Fixed_ctx (Tparam { index; name_ })
          | _ -> Open_ctx Empty_ctx
        in
        Some (ref ctx)
  in
  let control_ctx = Control_ctx.make_fn ~return:return_type ~error_ctx in
  let body =
    check_expr env body (Expect_type return_type) ~control_ctx ~tvar_env ~cenv
      ~global_env ~diagnostics
  in
  (if (not !(control_ctx.has_error)) && (not is_in_test) && not is_impl_method
   then
     match annotation with
     | Annotated (_, Error_typ { ty }) ->
         let loc = Typedtree.loc_of_typ ty in
         Local_diagnostics.add_warning diagnostics
           { kind = Useless_error_type; loc }
     | Annotated (_, Default_error_typ { loc_ = loc }) | Has_error_type loc ->
         Local_diagnostics.add_warning diagnostics
           { kind = Useless_error_type; loc }
     | Annotated (_, No_error_typ) | No_error_type_annotated -> ());
  let ret_constraint =
    match annotation with
    | Annotated r -> Some r
    | No_error_type_annotated | Has_error_type _ -> None
  in
  let func_typ =
    Builtin.type_arrow (Lst.map param_types fst) return_type ~err_ty:error_type
  in
  ({ params; params_loc_; body; ty = func_typ; ret_constraint; kind_ }, func_typ)

and deref_newtype ~(global_env : Global_env.t) (expr : Typedtree.expr) ~loc :
    Typedtree.expr =
  let ty = Stype.type_repr (Typedtree_util.type_of_typed_expr expr) in
  match ty with
  | T_constr { type_constructor = p; _ } -> (
      match Global_env.find_type_by_path global_env p with
      | Some { ty_desc = New_type { newtype_constr; _ }; _ } -> (
          match[@warning "-fragile-match"]
            Poly_type.instantiate_constr newtype_constr
          with
          | ty_res, ty_arg :: [] ->
              Ctype.unify_exn ty ty_res;
              Texpr_field
                {
                  record = expr;
                  accessor = Newtype;
                  ty = ty_arg;
                  pos = 0;
                  loc_ = loc;
                }
          | _ -> assert false)
      | _ -> expr)
  | _ -> expr

and resolve_field ~(global_env : Global_env.t) (record : Typedtree.expr)
    (label : Syntax.label) ~(may_be_method : unit -> bool)
    ~(src_ty_record : Stype.t) ~(loc : Rloc.t)
    ~(diagnostics : Local_diagnostics.t) :
    Stype.t * int * bool * Typedecl_info.type_component_visibility =
  let name = label.label_name in
  let ty_record = Stype.type_repr (Typedtree_util.type_of_typed_expr record) in
  let error () =
    match Global_env.try_pick_field global_env name with
    | Some ({ pos; mut; vis; _ } as field_desc) ->
        let _, ty_field = Poly_type.instantiate_field field_desc in
        (ty_field, pos, mut, vis)
    | None ->
        (Stype.new_type_var Tvar_error, Typeutil.unknown_pos, true, Read_write)
      [@@inline]
  in
  let not_a_record () =
    let kind =
      match src_ty_record with
      | Tvar _ -> "unknown"
      | Tarrow _ -> "function"
      | Tparam _ -> "type parameter"
      | T_trait _ -> "trait"
      | T_builtin _ -> Printer.type_to_string ty_record
      | T_constr { type_constructor = p; _ } -> (
          match Global_env.find_type_by_path global_env p with
          | Some { ty_desc = Record_type _; _ } -> assert false
          | Some { ty_desc = Variant_type _ | ErrorEnum_type _; _ } -> "variant"
          | Some { ty_desc = New_type _; _ } -> "new type"
          | Some { ty_desc = Error_type _; _ } -> "error type"
          | Some { ty_desc = Extern_type | Abstract_type; _ } -> "abstract"
          | None -> Type_path_util.name p)
      | T_blackhole -> assert false
    in
    let ty = Printer.type_to_string src_ty_record in
    let may_be_method = if may_be_method () then Some name else None in
    add_error diagnostics (Errors.not_a_record ~ty ~may_be_method ~kind ~loc);
    error ()
      [@@inline]
  in
  match ty_record with
  | T_constr { type_constructor = p; _ } -> (
      match Global_env.find_type_by_path global_env p with
      | Some { ty_desc = Record_type { fields }; _ } -> (
          match Lst.find_first fields (fun f -> f.field_name = name) with
          | Some field_info ->
              let ty_record', ty_field =
                Poly_type.instantiate_field field_info
              in
              Ctype.unify_exn ty_record ty_record';
              (ty_field, field_info.pos, field_info.mut, field_info.vis)
          | _ ->
              let ty = Printer.type_to_string src_ty_record in
              let may_be_method = may_be_method () in
              add_error diagnostics
                (Errors.no_such_field ~ty ~field:name ~may_be_method ~loc);
              error ())
      | Some { ty_desc = Variant_type constrs | ErrorEnum_type constrs; _ } -> (
          match record with
          | Texpr_ident { kind = Value_constr tag; _ } -> (
              let constr =
                Lst.find_first constrs (fun constr ->
                    Constr_info.equal constr.cs_tag tag)
              in
              match constr with
              | None -> not_a_record ()
              | Some ({ constr_name; _ } as constr_info) -> (
                  let ty_res, ty_args =
                    Poly_type.instantiate_constr constr_info
                  in
                  Ctype.unify_exn ty_res ty_record;
                  match
                    Fn_arity.find_constr_label constr_info.cs_arity_ ty_args
                      ~label:label.label_name
                  with
                  | None ->
                      add_error diagnostics
                        (Errors.constr_no_such_field ~ty:p ~constr:constr_name
                           ~field:label.label_name ~loc);
                      error ()
                  | Some (ty, offset, mut) ->
                      (ty, offset, mut, constr_info.cs_vis)))
          | _ -> not_a_record ())
      | _ -> not_a_record ())
  | Tvar { contents = Tnolink Tvar_error } | T_blackhole -> error ()
  | _ -> not_a_record ()

and typing_mutate (env : Local_env.t) (record : Syntax.expr)
    (label : Syntax.label) (field : Syntax.expr)
    (augmented_by : Syntax.var option) loc ~control_ctx ~tvar_env ~cenv
    ~global_env ~(diagnostics : _) : Typedtree.expr =
  let name = label.label_name in
  let record =
    infer_expr env record ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
  in
  let ty_record = Stype.type_repr (Typedtree_util.type_of_typed_expr record) in
  let record =
    let loc = Rloc.merge (Typedtree.loc_of_typed_expr record) label.loc_ in
    deref_newtype ~global_env record ~loc
  in
  let ty_field, pos, is_mut, vis =
    resolve_field ~global_env record label
      ~may_be_method:(fun () -> false)
      ~src_ty_record:ty_record ~diagnostics ~loc
  in
  (if not is_mut then
     let error = Errors.immutable_field ~label:name ~loc in
     add_error diagnostics error);
  (if Stype.is_external ty_record && vis <> Read_write then
     let error = Errors.mutate_readonly_field ~label:name ~loc in
     add_error diagnostics error);
  match augmented_by with
  | None ->
      let field =
        check_expr env field (Expect_type ty_field) ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
      in
      Texpr_mutate
        {
          record;
          label;
          field;
          augmented_by = None;
          ty = Stype.unit;
          pos;
          loc_ = loc;
        }
  | Some op ->
      let lhs : Typedtree.expr =
        Texpr_field
          {
            record;
            accessor = Label label;
            ty = ty_field;
            pos;
            loc_ = Typedtree.loc_of_typed_expr record;
          }
      in
      let infix_expr =
        typing_infix_op env op lhs field ~loc ~global_env ~cenv ~tvar_env
          ~control_ctx ~diagnostics
      in
      let op_expr, field =
        match infix_expr with
        | Texpr_apply { func; args = [ _; { arg_value; _ } ]; ty; _ } ->
            Ctype.unify_expr ~expect_ty:ty_field ~actual_ty:ty loc
            |> store_error ~diagnostics;
            (func, arg_value)
        | _ -> assert false
      in
      Texpr_mutate
        {
          record;
          label;
          field;
          augmented_by = Some op_expr;
          ty = Stype.unit;
          pos;
          loc_ = loc;
        }

and typing_letrec (env : Local_env.t)
    (funs : (Syntax.binder * Syntax.func) list) ~(tvar_env : Tvar_env.t) ~cenv
    ~(global_env : Global_env.t) ~(diagnostics : Local_diagnostics.t) :
    Local_env.t * (Typedtree.binder * Typedtree.fn) list =
  let func_types =
    Lst.map funs (fun (f, func) ->
        match func with
        | Lambda { parameters; body = _; return_type; has_error } ->
            let typed_fn_annotation =
              typing_function_annotation parameters return_type ~has_error
                ~tvar_env ~global_env ~diagnostics
            in
            (Typeutil.fresh_binder f, typed_fn_annotation)
        | Match _ -> assert false)
  in
  let env_with_funs =
    Lst.fold_left func_types env (fun env (binder, typed_fn_ann) ->
        let typ =
          Builtin.type_arrow
            (Lst.map typed_fn_ann.params_ty fst)
            typed_fn_ann.ret_ty ~err_ty:typed_fn_ann.err_ty
        in
        Typeutil.add_binder env binder ~typ ~mut:false)
  in
  let tfuns =
    Lst.map2 funs func_types (fun (_, func) (binder, typed_fn_annotation) ->
        match func with
        | Lambda
            {
              parameters = ps;
              params_loc_;
              body = funbody;
              return_type = _;
              kind_;
              has_error = _;
            } ->
            let tfun, _ =
              check_function ~is_impl_method:false env_with_funs ps params_loc_
                funbody typed_fn_annotation ~kind_ ~is_global:false
                ~is_in_test:false ~tvar_env ~cenv ~global_env ~diagnostics
            in
            (binder, tfun)
        | Match _ -> assert false)
  in
  (env_with_funs, tfuns)

and typing_let (env : Local_env.t) (p : Syntax.pattern) (e : Syntax.expr)
    ~control_ctx ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t)
    ~(global_env : Global_env.t) ~(diagnostics : Local_diagnostics.t) :
    Typedtree.pat_binders * Typedtree.pat * Typedtree.expr =
  match (p, e) with
  | (Ppat_constraint { ty = ty_expr; _ } as pc), _ ->
      let ty = typing_type ty_expr ~tvar_env ~global_env ~diagnostics in
      let stype = Typedtree_util.stype_of_typ ty in
      let te =
        check_expr env e (Expect_type stype) ~control_ctx ~tvar_env ~cenv
          ~global_env ~diagnostics
      in
      let pat_binders, tpat =
        Pattern_typer.check_pat pc stype ~tvar_env ~cenv ~global_env
          ~diagnostics
      in
      (pat_binders, tpat, te)
  | _ ->
      let te =
        infer_expr env e ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
      in
      let ty = Typedtree_util.type_of_typed_expr te in
      let pat_binders, tpat =
        Pattern_typer.check_pat p ty ~tvar_env ~cenv ~global_env ~diagnostics
      in
      (pat_binders, tpat, te)

and typing_letmut (env : Local_env.t) (binder : Syntax.binder)
    (ty : Syntax.typ option) (e : Syntax.expr) ~control_ctx
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) :
    Local_env.t * Typedtree.binder * Typedtree.expr * Typedtree.typ option =
  let binder = Typeutil.fresh_binder binder in
  let var_ty, expr, konstraint =
    match ty with
    | Some ty_expr ->
        let ty = typing_type ty_expr ~tvar_env ~global_env ~diagnostics in
        let stype = Typedtree_util.stype_of_typ ty in
        ( stype,
          check_expr env e (Expect_type stype) ~control_ctx ~tvar_env ~cenv
            ~global_env ~diagnostics,
          Some ty )
    | None ->
        let expr =
          infer_expr env e ~control_ctx ~tvar_env ~cenv ~global_env ~diagnostics
        in
        (Typedtree_util.type_of_typed_expr expr, expr, None)
  in
  let env = Typeutil.add_binder env binder ~typ:var_ty ~mut:true in
  (env, binder, expr, konstraint)

and check_loop (env : Local_env.t) (args : Syntax.expr list)
    (body : (Syntax.pattern list * Syntax.expr) list) ~(expect_ty : expect_ty)
    ~loop_loc_ ~loc ~control_ctx ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t)
    ~(global_env : Global_env.t) ~(diagnostics : Local_diagnostics.t) :
    Typedtree.expr =
  let targs, params =
    Lst.map_split args (fun arg ->
        let targ =
          infer_expr ~global_env ~tvar_env ~cenv ~diagnostics ~control_ctx env
            arg
        in
        let param : Typedtree.param =
          Param
            {
              binder =
                { binder_id = Ident.fresh "*param"; loc_ = Rloc.no_location };
              konstraint = None;
              ty = Typedtree_util.type_of_typed_expr targ;
              kind = Positional;
            }
        in
        (targ, param))
  in
  let tys = List.map Typedtree_util.type_of_typed_expr targs in
  let ty_tuple = Builtin.type_product tys in
  let n = List.length tys in
  let control_ctx =
    Control_ctx.with_loop control_ctx ~arg_typs:tys
      ~result_typ:(get_expected_type expect_ty)
  in
  let cases =
    Lst.map body (fun (pats, action) : Typedtree.match_case ->
        let actual = List.length pats in
        let ty_pat =
          if n = actual then ty_tuple
          else
            let loc =
              Rloc.merge
                (Syntax.loc_of_pattern (List.hd pats))
                (Syntax.loc_of_pattern (Lst.last pats))
            in
            let () =
              add_error diagnostics
                (Errors.loop_pat_arity_mismatch ~expected:n ~actual ~loc)
            in
            Builtin.type_product
              (if n < actual then
                 tys @ Type.type_var_list (actual - n) Tvar_error
               else Lst.take actual tys)
        in
        let tuple_pat : Syntax.pattern =
          Ppat_tuple { pats; loc_ = Rloc.no_location }
        in
        let pat_binders, tpat =
          Pattern_typer.check_pat tuple_pat ty_pat ~tvar_env ~cenv ~global_env
            ~diagnostics
        in
        let taction =
          check_expr ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
            (Typeutil.add_pat_binders env pat_binders)
            action expect_ty
        in
        { pat = tpat; pat_binders; action = taction })
  in
  if not !(control_ctx.has_continue) then
    Local_diagnostics.add_warning diagnostics { kind = Useless_loop; loc };
  let tuple_expr : Typedtree.expr =
    let loc_ =
      Rloc.merge
        (Syntax.loc_of_expression (List.hd args))
        (Syntax.loc_of_expression (Lst.last args))
    in
    let exprs =
      Lst.map params (fun p ->
          match p with
          | Typedtree.Param { binder; konstraint = _; ty; kind = _ } ->
              Typedtree.Texpr_ident
                {
                  id = { var_id = binder.binder_id; loc_ = binder.loc_ };
                  ty_args_ = [||];
                  arity_ = None;
                  kind = Normal;
                  ty;
                  loc_ = Rloc.no_location;
                })
    in
    Texpr_tuple { exprs; ty = ty_tuple; loc_ }
  in
  let ty =
    match cases with
    | { action; _ } :: _ -> Typedtree_util.type_of_typed_expr action
    | [] -> get_expected_type expect_ty
  in
  let body : Typedtree.expr =
    Texpr_match
      { expr = tuple_expr; cases; ty; match_loc_ = loop_loc_; loc_ = loc }
  in
  Texpr_loop { params; body; args = targs; ty; loc_ = loc }

and check_while (env : Local_env.t) (cond : Syntax.expr) (body : Syntax.expr)
    (while_else : Syntax.expr option) ~(expect_ty : expect_ty) ~loc ~control_ctx
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let typed_cond =
    check_expr ~global_env ~tvar_env ~cenv ~diagnostics
      ~control_ctx:(Control_ctx.with_ambiguous_position control_ctx)
      env cond (Expect_type Stype.bool)
  in
  let new_control_ctx =
    match while_else with
    | Some _ ->
        Control_ctx.with_while
          ~break_typ:(Some (get_expected_type expect_ty))
          control_ctx
    | None -> Control_ctx.with_while ~break_typ:None control_ctx
  in
  let typed_body =
    check_expr ~global_env ~tvar_env ~cenv ~diagnostics
      ~control_ctx:new_control_ctx env body Ignored
  in
  let typed_while_else =
    match while_else with
    | None ->
        (match expect_ty with
        | Expect_type expect_ty when not (Ctype.try_unify expect_ty Stype.unit)
          ->
            add_error diagnostics
              (Errors.need_else_branch ~loop_kind:`While ~loc
                 ~ty:(Printer.type_to_string expect_ty))
        | Ignored | Expect_type _ -> ());
        None
    | Some while_else ->
        Some
          (check_expr ~global_env ~tvar_env ~cenv ~diagnostics ~control_ctx env
             while_else expect_ty)
  in
  Texpr_while
    {
      loop_cond = typed_cond;
      loop_body = typed_body;
      ty =
        (match typed_while_else with
        | Some expr -> Typedtree_util.type_of_typed_expr expr
        | None -> Stype.unit);
      while_else = typed_while_else;
      loc_ = loc;
    }

and check_for (env : Local_env.t) (binders : (Syntax.binder * Syntax.expr) list)
    (condition : Syntax.expr option)
    (steps : (Syntax.binder * Syntax.expr) list) (body : Syntax.expr)
    (for_else : Syntax.expr option) ~(expect_ty : expect_ty) ~loc ~control_ctx
    ~(tvar_env : Tvar_env.t) ~(cenv : Poly_type.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.expr =
  let is_relaxed_for = condition = None in
  let ambiguous_control_ctx = Control_ctx.with_ambiguous_position control_ctx in
  let typed_binders_rev =
    Lst.fold_left binders [] (fun acc (binder, init) ->
        let name = binder.binder_name in
        let binder_loc = binder.loc_ in
        (if
           Lst.exists_fst acc (fun b ->
               Ident.base_name b.Typedtree.binder_id = name)
         then
           let error =
             Errors.duplicate_for_binder ~name:binder.binder_name
               ~loc:binder_loc
           in
           add_error diagnostics error);
        let binder = Typeutil.fresh_binder binder in
        let init =
          infer_expr ~global_env ~tvar_env ~cenv ~diagnostics
            ~control_ctx:ambiguous_control_ctx env init
        in
        (binder, init) :: acc)
  in
  let typed_binders = List.rev typed_binders_rev in
  let env_with_binders =
    Lst.fold_left typed_binders env (fun env (binder, init) ->
        Typeutil.add_binder env binder
          ~typ:(Typedtree_util.type_of_typed_expr init)
          ~mut:false)
  in
  let typed_condition =
    match condition with
    | None -> None
    | Some cond ->
        Some
          (check_expr ~global_env ~tvar_env ~cenv
             ~control_ctx:ambiguous_control_ctx ~diagnostics env_with_binders
             cond (Expect_type Stype.bool))
  in
  let typed_steps_rev =
    Lst.fold_left steps [] (fun acc (binder, step) : (Typedtree.var * _) list ->
        let name = binder.binder_name in
        let binder_loc = binder.loc_ in
        (if
           Lst.exists_fst acc (fun b ->
               Ident.base_name b.Typedtree.var_id = name)
         then
           let error =
             Errors.duplicate_for_binder ~name:binder.binder_name
               ~loc:binder_loc
           in
           add_error diagnostics error);
        match
          Lst.find_opt typed_binders (fun (binder, expr) ->
              if Ident.base_name binder.binder_id = name then
                Some (binder, Typedtree_util.type_of_typed_expr expr)
              else None)
        with
        | Some (id, typ) ->
            let typed_step =
              check_expr ~global_env ~tvar_env ~cenv ~diagnostics
                ~control_ctx:ambiguous_control_ctx env_with_binders step
                (Expect_type typ)
            in
            ({ var_id = id.binder_id; loc_ = binder_loc }, typed_step) :: acc
        | None ->
            let error =
              Errors.unknown_binder_in_for_steps ~name ~loc:binder_loc
            in
            add_error diagnostics error;
            ( ({ var_id = Ident.fresh name; loc_ = binder_loc } : Typedtree.var),
              infer_expr env_with_binders step
                ~control_ctx:ambiguous_control_ctx ~tvar_env ~cenv ~global_env
                ~diagnostics )
            :: acc)
  in
  let typed_steps = List.rev typed_steps_rev in
  let binder_types =
    Lst.map typed_binders (fun (_, expr) ->
        Typedtree_util.type_of_typed_expr expr)
  in
  let new_control_ctx =
    if for_else <> None || is_relaxed_for then
      Control_ctx.with_for ~arg_typs:binder_types
        ~break_typ:(Some (get_expected_type expect_ty))
        control_ctx
    else Control_ctx.with_for ~arg_typs:binder_types ~break_typ:None control_ctx
  in
  let typed_body =
    check_expr ~global_env ~tvar_env ~cenv ~control_ctx:new_control_ctx
      ~diagnostics env_with_binders body Ignored
  in
  let typed_for_else =
    match for_else with
    | None when is_relaxed_for -> None
    | None ->
        (match expect_ty with
        | Expect_type expect_ty when not (Ctype.try_unify expect_ty Stype.unit)
          ->
            add_error diagnostics
              (Errors.need_else_branch ~loop_kind:`For ~loc
                 ~ty:(Printer.type_to_string expect_ty))
        | Ignored | Expect_type _ -> ());
        None
    | Some for_else ->
        Some
          (check_expr ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
             env_with_binders for_else expect_ty)
  in
  Texpr_for
    {
      binders = typed_binders;
      condition = typed_condition;
      steps = typed_steps;
      body = typed_body;
      for_else = typed_for_else;
      ty = get_expected_type expect_ty;
      loc_ = loc;
    }

and typing_foreach env binders expr body else_block ~loc ~expect_ty ~global_env
    ~tvar_env ~cenv ~control_ctx ~diagnostics =
  let loc_of_expr = Syntax.loc_of_expression expr in
  let n_binders = List.length binders in
  let expr =
    infer_expr env expr ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  in
  if Type.is_array_like (Typedtree_util.type_of_typed_expr expr) then
    typing_foreach_transform_arraylike env binders expr body else_block ~loc
      ~expect_ty ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  else
    let expr, elem_tys =
      match Stype.type_repr (Typedtree_util.type_of_typed_expr expr) with
      | T_constr { type_constructor; tys }
        when Type_path.equal type_constructor Type_path.Builtin.type_path_iter
             || Type_path.equal type_constructor
                  Type_path.Builtin.type_path_iter2 ->
          let n_tys = List.length tys in
          if n_tys <> n_binders then (
            add_error diagnostics
              (Errors.foreach_loop_variable_count_mismatch ~actual:n_binders
                 ~expected:(Some n_tys) ~loc);
            let tys =
              if n_tys < n_binders then
                tys
                @ List.init (n_binders - n_tys) (fun _ ->
                      Stype.new_type_var Tvar_error)
              else Lst.take n_binders tys
            in
            (expr, tys))
          else (expr, tys)
      | ty ->
          let elem_tys =
            Lst.map binders (fun _ -> Stype.new_type_var Tvar_normal)
          in
          let method_name, iter_ty =
            match elem_tys with
            | [ ty1; ty2 ] -> ("iter2", Builtin.type_iter2 ty1 ty2)
            | ty :: rest ->
                if rest <> [] then
                  add_error diagnostics
                    (Errors.foreach_loop_variable_count_mismatch
                       ~actual:(List.length elem_tys) ~expected:None ~loc);
                ("iter", Builtin.type_iter ty)
            | [] -> ("iter", Builtin.type_iter (Stype.new_type_var Tvar_error))
          in
          let iter =
            typing_self_method ty
              { label_name = method_name; loc_ = loc_of_expr }
              ~src:Dot_src_direct ~loc:loc_of_expr ~global_env ~tvar_env ~cenv
              ~diagnostics
          in
          let iter_method_ty = Typedtree_util.type_of_typed_expr iter in
          let actual_ty_str, has_correct_arity =
            match iter with
            | Texpr_ident { arity_ = Some arity; _ } ->
                ( (fun () ->
                    Printer.toplevel_function_type_to_string ~arity
                      iter_method_ty),
                  Fn_arity.equal arity (Fn_arity.simple 1) )
            | _ -> ((fun () -> Printer.type_to_string iter_method_ty), true)
          in
          let expected = Builtin.type_arrow [ ty ] iter_ty ~err_ty:None in
          (if not (Ctype.try_unify expected iter_method_ty && has_correct_arity)
           then
             let expected = Printer.type_to_string expected in
             let actual = actual_ty_str () in
             add_error diagnostics
               (Errors.generic_type_mismatch
                  ~header:
                    ("`foreach` loop: `" ^ method_name
                     ^ "` method of iterated expression has wrong type"
                      : Stdlib.String.t)
                  ~expected ~actual ~loc:loc_of_expr));
          ( Texpr_apply
              {
                func = iter;
                args = [ { arg_value = expr; arg_kind = Positional } ];
                kind_ = Dot;
                ty = iter_ty;
                loc_ = Rloc.no_location;
              },
            elem_tys )
    in
    let else_block =
      match else_block with
      | None ->
          (match expect_ty with
          | Expect_type expect_ty
            when not (Ctype.try_unify expect_ty Stype.unit) ->
              add_error diagnostics
                (Errors.need_else_branch ~loop_kind:`For ~loc
                   ~ty:(Printer.type_to_string expect_ty))
          | Ignored | Expect_type _ -> ());
          None
      | Some else_block ->
          check_expr env else_block expect_ty ~global_env ~tvar_env ~cenv
            ~control_ctx ~diagnostics
          |> Option.some
    in
    let new_control_ctx =
      Control_ctx.with_foreach
        ~break_typ:(Option.map Typedtree_util.type_of_typed_expr else_block)
        control_ctx
    in
    let binders =
      Lst.map binders (fun binder : Typedtree.binder option ->
          match binder with
          | None -> None
          | Some { binder_name; loc_ } ->
              Some { binder_id = Ident.fresh binder_name; loc_ })
    in
    let new_env =
      Lst.fold_left2 binders elem_tys env (fun binder elem_ty env ->
          match binder with
          | None -> env
          | Some binder ->
              Typeutil.add_binder env binder ~typ:elem_ty ~mut:false)
    in
    let body =
      check_expr new_env body Ignored ~global_env ~tvar_env ~cenv
        ~control_ctx:new_control_ctx ~diagnostics
    in
    let ty =
      match else_block with
      | Some expr -> Typedtree_util.type_of_typed_expr expr
      | None -> Stype.unit
    in
    Texpr_foreach { binders; elem_tys; expr; body; else_block; ty; loc_ = loc }

and typing_foreach_transform_arraylike env binders expr_typed body else_block
    ~loc ~expect_ty ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics =
  let arr_name_raw = "*arr" in
  let arr_name = Ident.fresh arr_name_raw in
  let arr_binder : Typedtree.binder =
    { binder_id = arr_name; loc_ = Rloc.no_location }
  in
  let let_array body =
    let ty = Typedtree_util.type_of_typed_expr body in
    Typedtree.Texpr_let
      {
        pat = Tpat_var { binder = arr_binder; ty; loc_ = Rloc.no_location };
        pat_binders = [ { binder = arr_binder; binder_typ = ty } ];
        ty = Typedtree_util.type_of_typed_expr body;
        rhs = expr_typed;
        loc_ = Typedtree.loc_of_typed_expr expr_typed;
        body;
      }
  in
  let transformed_ast =
    let arr_len_name = "*len" in
    let arr_len_var : Syntax.var =
      { var_name = Lident arr_len_name; loc_ = Rloc.no_location }
    in
    let let_length body =
      Syntax.Pexpr_let
        {
          pattern =
            Syntax.Ppat_var
              { binder_name = arr_len_name; loc_ = Rloc.no_location };
          expr =
            Syntax.Pexpr_dot_apply
              {
                self =
                  Syntax.Pexpr_ident
                    {
                      id =
                        {
                          var_name = Longident.Lident arr_name_raw;
                          loc_ = Rloc.no_location;
                        };
                      loc_ = Rloc.no_location;
                    };
                method_name = { label_name = "length"; loc_ = Rloc.no_location };
                args = [];
                return_self = false;
                attr = No_attr;
                loc_ = Rloc.no_location;
              };
          body;
          loc_ = Rloc.no_location;
        }
    in
    let iter_var_name = "*i" in
    let iter_var : Syntax.var =
      { var_name = Lident iter_var_name; loc_ = Rloc.no_location }
    in
    let for_loop body =
      Syntax.Pexpr_for
        {
          binders =
            [
              ( { binder_name = iter_var_name; loc_ = Rloc.no_location },
                Syntax.Pexpr_constant
                  { c = Syntax.Const_int "0"; loc_ = Rloc.no_location } );
            ];
          condition =
            Some
              (Syntax.Pexpr_infix
                 {
                   op = { var_name = Lident "<"; loc_ = Rloc.no_location };
                   lhs =
                     Syntax.Pexpr_ident
                       { id = iter_var; loc_ = Rloc.no_location };
                   rhs =
                     Syntax.Pexpr_ident
                       { id = arr_len_var; loc_ = Rloc.no_location };
                   loc_ = Rloc.no_location;
                 });
          continue_block =
            [
              ( { binder_name = iter_var_name; loc_ = Rloc.no_location },
                Syntax.Pexpr_infix
                  {
                    op = { var_name = Lident "+"; loc_ = Rloc.no_location };
                    lhs =
                      Syntax.Pexpr_ident
                        { id = iter_var; loc_ = Rloc.no_location };
                    rhs =
                      Syntax.Pexpr_constant
                        { c = Const_int "1"; loc_ = Rloc.no_location };
                    loc_ = Rloc.no_location;
                  } );
            ];
          for_else = else_block;
          body;
          loc_ = loc;
        }
    in
    let el_binder, it_binder =
      match binders with
      | el_binder :: [] -> (el_binder, None)
      | [ it_binder; el_binder ] -> (el_binder, Some it_binder)
      | _ -> assert false
    in
    let binder_to_pat : Syntax.binder option -> Syntax.pattern = function
      | None -> Syntax.Ppat_any { loc_ = Rloc.no_location }
      | Some { binder_name; loc_ } -> Syntax.Ppat_var { binder_name; loc_ }
    in
    let let_el body =
      Syntax.Pexpr_let
        {
          pattern = binder_to_pat el_binder;
          expr =
            Syntax.Pexpr_dot_apply
              {
                self =
                  Syntax.Pexpr_ident
                    {
                      id =
                        {
                          var_name = Longident.Lident arr_name_raw;
                          loc_ = Rloc.no_location;
                        };
                      loc_ = Rloc.no_location;
                    };
                method_name = { label_name = "op_get"; loc_ = Rloc.no_location };
                args =
                  [
                    {
                      arg_value =
                        Syntax.Pexpr_ident
                          { id = iter_var; loc_ = Rloc.no_location };
                      arg_kind = Positional;
                    };
                  ];
                return_self = false;
                attr = No_attr;
                loc_ = Rloc.no_location;
              };
          body;
          loc_ = Rloc.no_location;
        }
    in
    let let_iter body =
      match it_binder with
      | None -> body
      | Some it_binder ->
          Syntax.Pexpr_let
            {
              pattern = binder_to_pat it_binder;
              expr =
                Syntax.Pexpr_ident { id = iter_var; loc_ = Rloc.no_location };
              body;
              loc_ = Rloc.no_location;
            }
    in
    let for_loop_body = let_iter (let_el body) in
    let for_loop = for_loop for_loop_body in
    let_length for_loop
  in
  let tast =
    check_expr
      (Typeutil.add_binder env arr_binder
         ~typ:(Typedtree_util.type_of_typed_expr expr_typed)
         ~mut:false)
      transformed_ast expect_ty ~control_ctx ~tvar_env ~cenv ~global_env
      ~diagnostics
  in
  let_array tast

and typing_range_for_in env (binders : Syntax.binder option list)
    (lhs : Syntax.expr) (rhs : Syntax.expr) (body : Syntax.expr)
    (else_block : Syntax.expr option) ~(inclusive : bool) ~operator_loc ~loc
    ~expect_ty ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics :
    Typedtree.expr =
  let tlhs =
    infer_expr env lhs ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  in
  let lhs_ty = Stype.type_repr (Typedtree_util.type_of_typed_expr tlhs) in
  let trhs, elem_ty =
    let check_rhs expect_ty =
      check_expr env rhs (Expect_type expect_ty) ~global_env ~tvar_env ~cenv
        ~control_ctx ~diagnostics
        [@@inline]
    in
    let is_supported (b : Stype.builtin) =
      match b with
      | T_int | T_uint | T_int64 | T_uint64 -> true
      | T_unit | T_bool | T_byte | T_char | T_float | T_double | T_string
      | T_bytes ->
          false
        [@@inline]
    in
    match lhs_ty with
    | T_builtin b when is_supported b -> (check_rhs lhs_ty, lhs_ty)
    | Tvar _ -> (
        let trhs = check_rhs lhs_ty in
        match Stype.type_repr (Typedtree_util.type_of_typed_expr trhs) with
        | T_builtin b when is_supported b -> (trhs, lhs_ty)
        | Tvar _ as rhs_ty ->
            Ctype.unify_exn rhs_ty Stype.int;
            (trhs, lhs_ty)
        | _ ->
            add_error diagnostics
              (Errors.range_operator_unsupported_type
                 ~actual_ty:(Printer.type_to_string lhs_ty)
                 ~loc:operator_loc);
            (trhs, Stype.int))
    | _ ->
        add_error diagnostics
          (Errors.range_operator_unsupported_type
             ~actual_ty:(Printer.type_to_string lhs_ty)
             ~loc:operator_loc);
        (check_rhs (Stype.new_type_var Tvar_error), Stype.int)
  in
  let lhs_var =
    ("*start" ^ Int.to_string (Basic_uuid.next ()) : Stdlib.String.t)
  in
  let rhs_var =
    ("*end" ^ Int.to_string (Basic_uuid.next ()) : Stdlib.String.t)
  in
  let hole () : Syntax.expr =
    Pexpr_hole { kind = Synthesized; loc_ = Rloc.no_location }
  in
  let loop_binders =
    match binders with
    | [] -> []
    | first_binder :: rest ->
        let first_binder =
          match first_binder with
          | Some binder -> binder
          | None -> { binder_name = "_"; loc_ = Rloc.no_location }
        in
        if rest <> [] then
          add_error diagnostics
            (Errors.foreach_loop_variable_count_mismatch
               ~actual:(List.length binders) ~expected:(Some 1) ~loc);
        ( first_binder,
          Syntax.Pexpr_ident
            {
              id = { var_name = Lident lhs_var; loc_ = Rloc.no_location };
              loc_ = Rloc.no_location;
            } )
        :: Lst.fold_right rest [] (fun binder_opt acc ->
               match binder_opt with
               | Some binder -> (binder, hole ()) :: acc
               | None -> acc)
  in
  let loop_variable : Syntax.expr =
    match loop_binders with
    | (binder, _) :: _ ->
        Pexpr_ident
          {
            id =
              { var_name = Lident binder.binder_name; loc_ = Rloc.no_location };
            loc_ = Rloc.no_location;
          }
    | [] -> hole ()
  in
  let cond_op : Syntax.var =
    if inclusive then { var_name = Lident "<="; loc_ = Rloc.no_location }
    else { var_name = Lident "<"; loc_ = Rloc.no_location }
  in
  let condition : Syntax.expr =
    Pexpr_infix
      {
        op = cond_op;
        lhs = loop_variable;
        rhs =
          Pexpr_ident
            {
              id = { var_name = Lident rhs_var; loc_ = Rloc.no_location };
              loc_ = Rloc.no_location;
            };
        loc_ = Rloc.no_location;
      }
  in
  let loop_update : (Syntax.binder * Syntax.expr) list =
    match loop_binders with
    | (binder, _) :: _ ->
        [
          ( binder,
            Pexpr_infix
              {
                op = { var_name = Lident "+"; loc_ = Rloc.no_location };
                lhs = loop_variable;
                rhs =
                  Pexpr_constant { c = Const_int "1"; loc_ = Rloc.no_location };
                loc_ = Rloc.no_location;
              } );
        ]
    | [] -> []
  in
  let lhs_binder : Typedtree.binder =
    { binder_id = Ident.fresh lhs_var; loc_ = Rloc.no_location }
  in
  let rhs_binder : Typedtree.binder =
    { binder_id = Ident.fresh rhs_var; loc_ = Rloc.no_location }
  in
  let lhs_pat_binder : Typedtree.pat_binder =
    { binder = lhs_binder; binder_typ = elem_ty }
  in
  let rhs_pat_binder : Typedtree.pat_binder =
    { binder = rhs_binder; binder_typ = elem_ty }
  in
  let loop : Syntax.expr =
    Pexpr_for
      {
        binders = loop_binders;
        condition = Some condition;
        continue_block = loop_update;
        for_else = else_block;
        body;
        loc_ = loc;
      }
  in
  let tloop =
    check_expr
      (Typeutil.add_pat_binders env [ lhs_pat_binder; rhs_pat_binder ])
      loop expect_ty ~global_env ~tvar_env ~cenv ~control_ctx ~diagnostics
  in
  let ty = Stype.type_repr (Typedtree_util.type_of_typed_expr tloop) in
  Texpr_let
    {
      pat =
        Tpat_var { binder = lhs_binder; ty = elem_ty; loc_ = Rloc.no_location };
      pat_binders = [ lhs_pat_binder ];
      rhs = tlhs;
      ty;
      loc_ = Rloc.no_location;
      body =
        Texpr_let
          {
            pat =
              Tpat_var
                { binder = rhs_binder; ty = elem_ty; loc_ = Rloc.no_location };
            pat_binders = [ rhs_pat_binder ];
            rhs = trhs;
            ty;
            loc_ = Rloc.no_location;
            body = tloop;
          };
    }

and check_break env arg ~expect_ty ~loc ~global_env ~tvar_env ~cenv ~control_ctx
    ~diagnostics : Typedtree.expr =
  let check_break_arg break_arg arg =
    let arg =
      check_expr env arg (Expect_type break_arg) ~global_env ~tvar_env ~cenv
        ~control_ctx ~diagnostics
    in
    Typedtree.Texpr_break { arg = Some arg; ty = expect_ty; loc_ = loc }
  in
  let fallback () =
    let arg =
      arg
      |> Option.map (fun arg ->
             infer_expr env arg ~global_env ~tvar_env ~cenv ~control_ctx
               ~diagnostics)
    in
    Typedtree.Texpr_break { arg; ty = expect_ty; loc_ = loc }
      [@@inline]
  in
  match control_ctx.control_info with
  | Not_in_loop ->
      add_error diagnostics (Errors.outside_loop ~msg:"break" ~loc);
      fallback ()
  | Ambiguous_position ->
      add_error diagnostics (Errors.ambiguous_break loc);
      fallback ()
  | In_while
  | In_for { break = None; continue = _ }
  | In_foreach { break = None } -> (
      match arg with
      | None -> Texpr_break { arg = None; ty = expect_ty; loc_ = loc }
      | Some arg ->
          let kind =
            match control_ctx.control_info with
            | In_while | In_while_else _ -> `While
            | In_for _ -> `For
            | In_foreach _ -> `For
            | _ -> assert false
          in
          add_error diagnostics (Errors.invalid_break ~loop_kind:kind ~loc);
          let arg =
            infer_expr env arg ~global_env ~tvar_env ~cenv ~control_ctx
              ~diagnostics
          in
          Texpr_break { arg = Some arg; ty = expect_ty; loc_ = loc })
  | In_loop { break = break_arg; continue = _ }
  | In_while_else { break = break_arg }
  | In_for { break = Some break_arg; continue = _ }
  | In_foreach { break = Some break_arg } -> (
      match arg with
      | None ->
          (try Ctype.unify_exn break_arg Stype.unit
           with _ ->
             add_error diagnostics
               (Errors.break_type_mismatch
                  ~expected:(Printer.type_to_string break_arg)
                  ~actual:"no arguments" ~loc));
          Texpr_break { arg = None; ty = expect_ty; loc_ = loc }
      | Some arg -> check_break_arg break_arg arg)

and check_continue env args ~expect_ty ~loc ~global_env ~tvar_env ~cenv
    ~control_ctx ~diagnostics : Typedtree.expr =
  let check_continue_args continue_args =
    let expected = List.length continue_args in
    let actual = List.length args in
    if expected <> actual then
      add_error diagnostics
        (Errors.continue_arity_mismatch ~expected ~actual ~loc);
    let rec check_args expected_tys args =
      match (expected_tys, args) with
      | [], args ->
          Lst.map args (fun arg ->
              infer_expr env arg ~global_env ~tvar_env ~cenv ~control_ctx
                ~diagnostics)
      | _, [] -> []
      | expected_ty :: expected_tys, arg :: args ->
          check_expr env arg (Expect_type expected_ty) ~global_env ~tvar_env
            ~cenv ~control_ctx ~diagnostics
          :: check_args expected_tys args
    in
    let args = check_args continue_args args in
    Typedtree.Texpr_continue { args; ty = expect_ty; loc_ = loc }
      [@@inline]
  in
  let fallback () =
    let args =
      Lst.map args (fun arg ->
          infer_expr env arg ~global_env ~tvar_env ~cenv ~control_ctx
            ~diagnostics)
    in
    Typedtree.Texpr_continue { args; ty = expect_ty; loc_ = loc }
      [@@inline]
  in
  match control_ctx.control_info with
  | Not_in_loop ->
      add_error diagnostics (Errors.outside_loop ~msg:"continue" ~loc);
      fallback ()
  | Ambiguous_position ->
      add_error diagnostics (Errors.ambiguous_continue loc);
      fallback ()
  | In_while | In_while_else _ | In_foreach _ ->
      if args <> [] then
        add_error diagnostics
          (Errors.continue_arity_mismatch ~expected:0 ~actual:(List.length args)
             ~loc);
      let args =
        Lst.map args (fun arg ->
            infer_expr env arg ~global_env ~tvar_env ~cenv ~control_ctx
              ~diagnostics)
      in
      Texpr_continue { args; ty = expect_ty; loc_ = loc }
  | In_for { continue = continue_args; break = _ } ->
      if args = [] then Texpr_continue { args = []; ty = expect_ty; loc_ = loc }
      else check_continue_args continue_args
  | In_loop { continue = continue_args; _ } ->
      control_ctx.has_continue := true;
      check_continue_args continue_args

let typing_impl_expr env expr ~(global_env : Global_env.t) ~diagnostics =
  let cenv = Poly_type.make () in
  let tvar_env = Tvar_env.empty in
  let te =
    check_expr env expr Ignored
      ~control_ctx:(Control_ctx.make_fn ~return:Stype.unit ~error_ctx:None)
      ~tvar_env ~cenv ~global_env ~diagnostics
  in
  Type_lint.type_lint te ~diagnostics;
  Type_constraint.solve_constraints cenv ~tvar_env ~global_env ~diagnostics;
  te

let typing_impl_let (binder : Syntax.binder)
    (expr : Toplevel_typer.top_let_body) ~(id : Ident.t) ~(typ : Stype.t)
    (env : Local_env.t) ~(global_env : Global_env.t)
    ~(diagnostics : Local_diagnostics.t) : Typedtree.binder * Typedtree.expr =
  let cenv = Poly_type.make () in
  let tvar_env = Tvar_env.empty in
  let te =
    match expr with
    | Wl_toplet_const c -> c
    | Wl_toplet_normal expr ->
        check_expr env expr (Expect_type typ) ~control_ctx:Control_ctx.empty
          ~tvar_env ~cenv ~global_env ~diagnostics
  in
  Type_lint.type_lint te ~diagnostics;
  Type_constraint.solve_constraints cenv ~tvar_env ~global_env ~diagnostics;
  let binder : Typedtree.binder = { binder_id = id; loc_ = binder.loc_ } in
  (binder, te)

let derive_by_ast ~global_env (directive : Syntax.deriving_directive)
    (decl : Syntax.type_decl) (method_name : string) ~trait_path
    ~(deriver : Ast_derive.deriver) ~(diagnostics : Local_diagnostics.t) :
    Toplevel_typer.value_worklist_item =
  let trait = directive.type_name_ in
  let tpath =
    Type_path.toplevel_type ~pkg:!Basic_config.current_package decl.tycon
  in
  let meth_info =
    Ext_method_env.find_method
      (Global_env.get_ext_method_env global_env)
      ~trait:trait_path ~self_type:tpath ~method_name
    |> Option.get
  in
  match[@warning "-fragile-match"] meth_info.typ with
  | (Tarrow { params_ty; ret_ty; err_ty } : Stype.t) ->
      let param_names, params =
        Lst.map_split params_ty (fun _ ->
            let param_name =
              ("*x_" ^ Int.to_string (Basic_uuid.next ()) : Stdlib.String.t)
            in
            let param : Syntax.parameter =
              {
                param_binder =
                  { binder_name = param_name; loc_ = Rloc.no_location };
                param_annot = None;
                param_kind = Positional;
              }
            in
            (param_name, param))
      in
      let assertions = Vec.empty () in
      let body =
        deriver directive decl ~params:param_names ~assertions ~diagnostics
      in
      Wl_top_funcdef
        {
          fun_binder = { binder_name = method_name; loc_ = trait.loc_ };
          decl_params = params;
          params_loc = Rloc.no_location;
          is_pub = meth_info.pub;
          doc = meth_info.doc_;
          decl_body =
            Decl_body
              {
                expr =
                  Pexpr_static_assert { asserts = Vec.to_list assertions; body };
                local_types = [];
              };
          loc_ = decl.loc_;
          id = meth_info.id;
          kind = Fun_kind_method None;
          arity = meth_info.arity_;
          tvar_env = meth_info.ty_params_;
          typed_fn_annotation =
            {
              params_ty = Lst.map params_ty (fun ty -> (ty, None));
              ret_ty;
              err_ty;
              ret_annotation = No_error_type_annotated;
            };
        }
  | _ -> assert false

let rec typing_worklist_item (item : Toplevel_typer.value_worklist_item)
    ~global_env ~diagnostics ~(acc : Typedtree.impls) : Typedtree.impls =
  match item with
  | Wl_top_expr { expr; is_main; id; loc_ } ->
      let local_diagnostics = Local_diagnostics.make ~base:loc_ in
      let te =
        typing_impl_expr Local_env.empty expr ~global_env
          ~diagnostics:local_diagnostics
      in
      Local_diagnostics.add_to_global local_diagnostics diagnostics;
      Typedtree.Timpl_expr
        { expr = te; is_main; expr_id = id; loc_; is_generated_ = false }
      :: acc
  | Wl_top_letdef { binder; expr; loc_; is_pub; doc_; konstraint; id; typ } ->
      let local_diagnostics = Local_diagnostics.make ~base:loc_ in
      let binder, te =
        typing_impl_let binder expr ~id:(Ident.of_qual_ident id) ~typ
          Local_env.empty ~global_env ~diagnostics:local_diagnostics
      in
      Local_diagnostics.add_to_global local_diagnostics diagnostics;
      Typedtree.Timpl_letdef
        {
          binder;
          expr = te;
          is_pub;
          loc_;
          konstraint;
          doc_;
          is_generated_ = false;
        }
      :: acc
  | Wl_top_funcdef
      {
        fun_binder;
        decl_params;
        params_loc;
        is_pub;
        doc;
        decl_body;
        loc_;
        id;
        kind;
        arity;
        tvar_env;
        typed_fn_annotation;
      } -> (
      let local_diagnostics = Local_diagnostics.make ~base:loc_ in
      let binder : Typedtree.binder =
        { binder_id = Ident.of_qual_ident id; loc_ = fun_binder.loc_ }
      in
      match decl_body with
      | Decl_stubs func_stubs ->
          let is_import_stub, language =
            match func_stubs with
            | Import _ -> (true, None)
            | Embedded { language; _ } -> (false, language)
          in
          let ({ params_ty; ret_ty = _; err_ty = _; ret_annotation }
                : Toplevel_typer.typed_fn_annotation) =
            typed_fn_annotation
          in
          let stub_body : Typedtree.stub_body =
            if Syntax_util.is_intrinsic func_stubs then Intrinsic
            else
              let language =
                Option.map (fun s -> s.Literal.string_val) language
              in
              let check_type ~allow_func ty =
                Typeutil.check_stub_type ~language ty global_env ~allow_func
                  ~is_import_stub
                |> store_error ~diagnostics:local_diagnostics
              in
              Lst.iter params_ty (fun (_, ty_opt) ->
                  match ty_opt with
                  | Some ty -> check_type ~allow_func:true ty
                  | None -> ());
              (match ret_annotation with
              | Annotated (ret_annot, _) ->
                  check_type ~allow_func:false ret_annot
              | No_error_type_annotated | Has_error_type _ -> ());
              let func_stubs =
                Stub_type.from_syntax func_stubs ~loc:loc_
                |> Parsing_partial_info.take_info ~diagnostics
              in
              Func_stub func_stubs
          in
          let cenv = Poly_type.make () in
          let _, params =
            check_function_params Local_env.empty decl_params params_ty
              ~is_global:true ~tvar_env ~cenv ~global_env
              ~diagnostics:local_diagnostics
          in
          Lst.iter params (fun (Param { kind; _ }) ->
              match kind with
              | Optional default ->
                  Type_lint.type_lint default ~diagnostics:local_diagnostics
              | Positional | Labelled | Autofill | Question_optional -> ());
          Type_constraint.solve_constraints cenv ~tvar_env ~global_env
            ~diagnostics:local_diagnostics;
          let ret =
            match ret_annotation with
            | Annotated (ret, _) -> Some ret
            | Has_error_type _ | No_error_type_annotated -> None
          in
          Local_diagnostics.add_to_global local_diagnostics diagnostics;
          Timpl_stub_decl
            {
              binder;
              params;
              ret;
              func_stubs = stub_body;
              is_pub;
              arity_ = arity;
              kind_ = kind;
              loc_;
              doc_ = doc;
            }
          :: acc
      | Decl_body { expr = decl_body; local_types = _ } ->
          let cenv = Poly_type.make () in
          let is_in_test =
            String.starts_with ~prefix:"__test_" fun_binder.binder_name
          in
          let is_impl_method =
            match id with
            | Qext_method _ -> true
            | Qregular _ | Qmethod _ | Qregular_implicit_pkg _ -> false
          in
          let tfuns, _ =
            check_function ~is_impl_method Local_env.empty decl_params
              params_loc decl_body typed_fn_annotation ~kind_:Lambda
              ~is_global:true ~is_in_test ~tvar_env ~cenv ~global_env
              ~diagnostics:local_diagnostics
          in
          Type_lint.type_lint_fn tfuns ~diagnostics:local_diagnostics;
          Type_constraint.solve_constraints cenv ~tvar_env ~global_env
            ~diagnostics:local_diagnostics;
          Local_diagnostics.add_to_global local_diagnostics diagnostics;
          Timpl_fun_decl
            {
              fun_decl =
                {
                  kind;
                  fn_binder = binder;
                  fn = tfuns;
                  is_pub;
                  ty_params_ = tvar_env;
                  doc_ = doc;
                };
              arity_ = arity;
              loc_;
              is_generated_ = false;
            }
          :: acc)
  | Wl_derive { ty_decl; directive; syn_decl; trait_path; _ } -> (
      let local_diagnostics = Local_diagnostics.make ~base:syn_decl.loc_ in
      match Type_path.Hash.find_opt Ast_derive.derivers trait_path with
      | Some derivers ->
          let res =
            Lst.fold_right derivers acc (fun (method_name, deriver) acc ->
                derive_by_ast ~global_env directive syn_decl method_name
                  ~trait_path ~deriver ~diagnostics:local_diagnostics
                |> typing_worklist_item ~acc ~global_env ~diagnostics)
          in
          Local_diagnostics.add_to_global local_diagnostics diagnostics;
          res
      | None ->
          Local_diagnostics.add_error local_diagnostics
            (Errors.derive_unsupported_trait ~tycon:ty_decl.ty_constr
               ~trait:directive.type_name_.name ~loc:directive.loc_);
          Local_diagnostics.add_to_global local_diagnostics diagnostics;
          acc)

let type_check ~diagnostics (input : Toplevel_typer.output) : Typedtree.output =
  let global_env = input.global_env in
  let value_defs =
    Vec.fold_right input.values [] ~f:(fun item acc ->
        typing_worklist_item ~global_env ~diagnostics item ~acc)
  in
  Typedtree.Output
    { value_defs; type_defs = input.type_decls; trait_defs = input.trait_decls }
