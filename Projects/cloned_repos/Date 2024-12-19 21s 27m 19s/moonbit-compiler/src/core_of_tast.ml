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
module Type_path = Basic_type_path
module Lst = Basic_lst
module Vec = Basic_vec
module Hash_string = Basic_hash_string
module Syntax = Parsing_syntax

let ghost_loc_ = Rloc.no_location
let mutable_var_label : Syntax.label = { label_name = "val"; loc_ = ghost_loc_ }
let mutable_var_type (ty : Core.typ) : Core.typ = Builtin.type_ref ty

type foreach_context = {
  result_var : Ident.t;
  exit_join : Ident.t;
  mutable has_early_exit : bool;
  foreach_result_ty : Stype.t;
}

type return_context =
  | No_return
  | Normal_return of { return_join : Ident.t; mutable need_return_join : bool }
  | Foreach_return of foreach_context

type error_context = { raise_join : Ident.t; mutable need_raise_join : bool }

type loop_context =
  | Loop_label of Label.t
  | For_loop_info of {
      label : Label.t;
      continue_join : Ident.t;
      mutable need_for_loop_join : bool;
    }
  | Foreach of foreach_context
  | No_loop

type transl_context = {
  return_ctx : return_context;
  error_ctx : error_context option;
  loop_ctx : loop_context;
  error_ty : Stype.t option;
  return_ty : Stype.t;
  wrapper_info : Stype.t option;
  base : Loc.t;
}

let wrap_ok_prim expr ~result_ty =
  let rec tail_is_optimizable (expr : Core.expr) =
    match expr with
    | Cexpr_let { body; _ }
    | Cexpr_letfn { body; kind = Rec | Nonrec; _ }
    | Cexpr_letrec { body; _ }
    | Cexpr_sequence { expr2 = body; _ } ->
        tail_is_optimizable body
    | Cexpr_letfn { body; kind = Tail_join | Nontail_join; fn; _ } ->
        tail_is_optimizable body || tail_is_optimizable fn.body
    | Cexpr_if { ifso; ifnot; _ } -> (
        tail_is_optimizable ifso
        ||
        match ifnot with
        | Some ifnot -> tail_is_optimizable ifnot
        | None -> false)
    | Cexpr_switch_constr { cases; _ } ->
        Lst.exists cases (fun (_, _, action) -> tail_is_optimizable action)
    | Cexpr_switch_constant { cases; _ } ->
        Lst.exists cases (fun (_, action) -> tail_is_optimizable action)
    | Cexpr_handle_error { handle_kind = Joinapply _ | Return_err _; _ }
    | Cexpr_return _
    | Cexpr_apply { kind = Join; _ } ->
        true
    | Cexpr_loop _ | Cexpr_break _ | Cexpr_const _ | Cexpr_unit _ | Cexpr_var _
    | Cexpr_prim _ | Cexpr_function _ | Cexpr_apply _ | Cexpr_constr _
    | Cexpr_tuple _ | Cexpr_record _ | Cexpr_record_update _ | Cexpr_field _
    | Cexpr_mutate _ | Cexpr_array _ | Cexpr_assign _ | Cexpr_continue _
    | Cexpr_as _ | Cexpr_handle_error _ ->
        false
  in
  let ok_tag = Builtin.constr_ok.cs_tag in
  let wrap_ok_expr expr =
    Core.prim ~ty:result_ty
      (Primitive.Pmake_value_or_error { tag = ok_tag })
      [ expr ]
  in
  let rec push_ok_to_tail (expr : Core.expr) : Core.expr =
    match expr with
    | Cexpr_let { body; loc_; name; rhs; ty = _ } ->
        Core.let_ ~loc:loc_ name rhs (push_ok_to_tail body)
    | Cexpr_letfn
        { name; body; kind = (Rec | Nonrec) as kind; fn; loc_; ty = _ } ->
        Core.letfn ~loc:loc_ ~kind name fn (push_ok_to_tail body)
    | Cexpr_letfn
        {
          name;
          body;
          kind = (Tail_join | Nontail_join) as kind;
          fn;
          loc_;
          ty = _;
        } ->
        let fn = { fn with body = push_ok_to_tail fn.body } in
        Core.letfn ~loc:loc_ ~kind name fn (push_ok_to_tail body)
    | Cexpr_letrec { body; bindings; loc_; ty = _ } ->
        Core.letrec ~loc:loc_ bindings (push_ok_to_tail body)
    | Cexpr_sequence { expr2 = body; expr1; ty = _ } ->
        Core.sequence expr1 (push_ok_to_tail body)
    | Cexpr_if { ifso; ifnot; cond; loc_; ty = _ } ->
        let ifnot =
          match ifnot with
          | Some ifnot -> push_ok_to_tail ifnot
          | None -> push_ok_to_tail (Core.unit ())
        in
        Core.if_ ~loc:loc_ cond ~ifso:(push_ok_to_tail ifso) ~ifnot
    | Cexpr_switch_constr { cases; default; obj; loc_; ty = _ } ->
        let cases =
          Lst.map cases (fun (tag, arg, action) ->
              (tag, arg, push_ok_to_tail action))
        in
        let default = Option.map push_ok_to_tail default in
        Core.switch_constr ~loc:loc_ obj cases ~default
    | Cexpr_switch_constant { cases; default; obj; loc_; ty = _ } ->
        let cases =
          Lst.map cases (fun (c, action) -> (c, push_ok_to_tail action))
        in
        let default = push_ok_to_tail default in
        Core.switch_constant ~loc:loc_ ~default obj cases
    | Cexpr_handle_error { obj; handle_kind = Joinapply _ | Return_err _; _ } ->
        obj
    | Cexpr_return { expr; return_kind; loc_; ty = _ } ->
        Core.return ~loc:loc_ ~return_kind expr ~ty:result_ty
    | Cexpr_apply { kind = Join; func; args; ty = _; ty_args_; prim; loc_ } ->
        Core.apply ~loc:loc_ ~ty_args_ ~prim ~kind:Join func args ~ty:result_ty
    | Cexpr_loop _ | Cexpr_break _ | Cexpr_const _ | Cexpr_unit _ | Cexpr_var _
    | Cexpr_prim _ | Cexpr_function _ | Cexpr_apply _ | Cexpr_constr _
    | Cexpr_tuple _ | Cexpr_record _ | Cexpr_record_update _ | Cexpr_field _
    | Cexpr_mutate _ | Cexpr_array _ | Cexpr_assign _ | Cexpr_continue _
    | Cexpr_as _ | Cexpr_handle_error _ ->
        wrap_ok_expr expr
  in
  if tail_is_optimizable expr then push_ok_to_tail expr else wrap_ok_expr expr

let name_of_default_arg ~label (name_of_fn : Basic_ident.t) =
  match name_of_fn with
  | Pident _ | Plocal_method _ | Pdyntrait_method _ -> assert false
  | Pdot (Qregular { pkg; name }) | Pdot (Qregular_implicit_pkg { pkg; name })
    ->
      Ident.of_qual_ident
        (Qregular
           {
             pkg;
             name = Stdlib.String.concat "" [ name; "."; label; ".default" ];
           })
  | Pdot (Qmethod { self_typ; name }) ->
      Ident.of_qual_ident
        (Qmethod
           {
             self_typ;
             name = Stdlib.String.concat "" [ name; "."; label; ".default" ];
           })
  | Pdot (Qext_method _) -> assert false

let rec transl_expr ~(is_tail : bool) ~(need_wrap_ok : bool)
    ~(global_env : Global_env.t) (ctx : transl_context) (texpr : Typedtree.expr)
    : Core.expr =
  let go_tail ~need_wrap_ok texpr =
    transl_expr ~is_tail ~need_wrap_ok ~global_env ctx texpr
      [@@inline]
  in
  let go_nontail texpr =
    transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx texpr
      [@@inline]
  in
  let wrap_ok core_expr =
    if need_wrap_ok then
      match ctx.wrapper_info with
      | Some result_ty -> wrap_ok_prim core_expr ~result_ty
      | None -> core_expr
    else core_expr
      [@@inline]
  in
  match texpr with
  | Texpr_constant { c; ty = _; loc_ } -> (
      match c with
      | C_bigint { v; _ } ->
          let func =
            match[@warning "-fragile-match"]
              Global_env.find_dot_method global_env
                ~type_name:Type_path.Builtin.type_path_bigint
                ~method_name:"from_string"
            with
            | method_info :: [] ->
                Typedtree.Texpr_method
                  {
                    type_name =
                      Tname_path
                        {
                          name = Type_path.Builtin.type_path_bigint;
                          kind = Type;
                          loc_;
                        };
                    meth =
                      {
                        var_id = Basic_ident.of_qual_ident method_info.id;
                        loc_;
                      };
                    prim = method_info.prim;
                    ty = method_info.typ;
                    ty_args_ = [||];
                    arity_ = Some method_info.arity_;
                    loc_;
                  }
            | _ -> assert false
          in
          let arg : Typedtree.argument =
            {
              arg_value =
                Texpr_constant
                  {
                    c = Constant.C_string (Basic_bigint.to_string v);
                    ty = Stype.string;
                    name_ = None;
                    loc_;
                  };
              arg_kind = Positional;
            }
          in
          transl_apply ~global_env ~ctx ~kind:`Normal ~loc_ ~ty:Stype.bigint
            func [ arg ]
          |> wrap_ok
      | _ -> Core.const ~loc:loc_ c |> wrap_ok)
  | Texpr_unit { loc_ } -> Core.unit ~loc:loc_ () |> wrap_ok
  | Texpr_ident { id = _; kind = Prim prim; ty; loc_ }
  | Texpr_method { type_name = _; meth = _; prim = Some prim; ty; loc_ }
    when not (Primitive.is_intrinsic prim) ->
      Core.unsaturated_prim ~loc:loc_ ~ty prim |> wrap_ok
  | Texpr_unresolved_method
      { trait_name; method_name; self_type = self_ty; ty = expect_ty; loc_ }
    -> (
      let trait = trait_name.name in
      match
        transl_trait_method ~global_env ~trait ~self_ty ~expect_ty ~method_name
      with
      | `Prim prim ->
          Core.unsaturated_prim ~loc:loc_ ~ty:expect_ty prim |> wrap_ok
      | `Regular (id, ty_args_, prim) ->
          Core.var ~loc:loc_ ~ty:expect_ty ~ty_args_ ~prim id |> wrap_ok)
  | Texpr_ident { id = { var_id }; kind = Mutable; ty; ty_args_; loc_ } ->
      Core.field ~ty ~pos:0
        (Core.var ~loc:loc_ ~ty:(mutable_var_type ty) ~ty_args_
           (Ident.of_ident var_id))
        (Label mutable_var_label)
      |> wrap_ok
  | Texpr_ident { id = { var_id }; kind = Normal; ty; ty_args_; loc_ } ->
      Core.var ~loc:loc_ ~ty ~ty_args_ (Ident.of_ident var_id) |> wrap_ok
  | Texpr_ident { id = { var_id }; kind = Value_constr tag; ty; ty_args_; loc_ }
    ->
      let constr_ty = Type.make_constr_type ty ~tag in
      Core.prim ~ty
        (Pcast { kind = Constr_to_enum })
        [ Core.var ~loc:loc_ ~ty:constr_ty ~ty_args_ (Ident.of_ident var_id) ]
      |> wrap_ok
  | Texpr_ident { id = { var_id }; kind = Prim prim; ty; ty_args_; loc_ } ->
      Core.var ~loc:loc_ ~ty ~ty_args_ ~prim:(Some prim) (Ident.of_ident var_id)
      |> wrap_ok
  | Texpr_method { type_name = _; meth; prim; ty; ty_args_; loc_ } ->
      Core.var ~loc:loc_ ~ty ~ty_args_ ~prim (Ident.of_ident meth.var_id)
      |> wrap_ok
  | Texpr_as { expr; trait; ty = _; is_implicit = _; loc_ } -> (
      match trait with
      | Tname_path { kind = Trait; name = trait }
      | Tname_alias { kind = Trait; name = trait } ->
          let obj_type =
            Stype.type_repr (Typedtree_util.type_of_typed_expr expr)
          in
          let expr = go_nontail expr in
          let expr_ty = Stype.type_repr (Core.type_of_expr expr) in
          (match expr_ty with
          | T_trait expr_trait ->
              if Type_path.equal expr_trait trait then expr
              else Core.as_ ~loc:loc_ ~trait ~obj_type expr
          | _ -> Core.as_ ~loc:loc_ ~trait ~obj_type expr)
          |> wrap_ok
      | Tname_tvar _ | Tname_path _ | Tname_alias _ -> assert false)
  | Texpr_let { pat; pat_binders; rhs; body; ty; loc_ } ->
      Transl_match.transl_let ~global_env pat ~pat_binders (go_nontail rhs)
        (go_tail ~need_wrap_ok body)
        ~ty ~loc:loc_
  | Texpr_letmut { binder; konstraint = _; expr; body; ty = _; loc_ } ->
      let expr = go_nontail expr in
      let name = Ident.of_ident binder.binder_id in
      Core.let_ ~loc:loc_ name
        (Core.record
           ~ty:(mutable_var_type (Core.type_of_expr expr))
           [ { label = mutable_var_label; pos = 0; expr; is_mut = true } ])
        (go_tail ~need_wrap_ok body)
  | Texpr_function { func; ty; loc_ } ->
      let ({ params; body } : Core.fn) =
        transl_fn ~base:ctx.base func ~global_env
      in
      Core.function_ ~loc:loc_ ~ty params body |> wrap_ok
  | Texpr_apply { kind_ = Dot_return_self; _ }
  | Texpr_exclamation { expr = Texpr_apply { kind_ = Dot_return_self; _ }; _ }
    ->
      let rec desugar_cascade (expr : Typedtree.expr)
          (k : Typedtree.expr -> Core.expr) =
        match expr with
        | Texpr_apply
            {
              func;
              args = { arg_value = self; arg_kind = Positional } :: args;
              ty = _;
              kind_ = Dot_return_self;
              loc_;
            } ->
            desugar_cascade self (fun actual_self ->
                let args : Typedtree.argument list =
                  { arg_value = actual_self; arg_kind = Positional } :: args
                in
                let apply =
                  transl_apply ~global_env ~ctx ~kind:`Normal ~loc_
                    ~ty:Stype.unit func args
                in
                Core.sequence apply (k actual_self))
        | Texpr_exclamation
            {
              expr =
                Texpr_apply
                  {
                    func;
                    args = { arg_value = self; arg_kind = Positional } :: args;
                    ty = _;
                    kind_ = Dot_return_self;
                    loc_ = apply_loc;
                  };
              convert_to_result;
              ty = _;
              loc_;
            } ->
            assert (not convert_to_result);
            desugar_cascade self (fun actual_self ->
                let args : Typedtree.argument list =
                  { arg_value = actual_self; arg_kind = Positional } :: args
                in
                let desugared_expr : Typedtree.expr =
                  Texpr_exclamation
                    {
                      expr =
                        Texpr_apply
                          {
                            func;
                            args;
                            ty = Stype.unit;
                            kind_ = Dot;
                            loc_ = apply_loc;
                          };
                      convert_to_result;
                      ty = Stype.unit;
                      loc_;
                    }
                in
                let apply = go_nontail desugared_expr in
                Core.sequence apply (k actual_self))
        | Texpr_ident { kind = Normal | Prim _ | Value_constr _; _ } -> k expr
        | _ ->
            let ty = Typedtree_util.type_of_typed_expr expr in
            let tast_id = Basic_ident.fresh "self" in
            let tast_self : Typedtree.expr =
              Texpr_ident
                {
                  id = { var_id = tast_id; loc_ = ghost_loc_ };
                  ty_args_ = [||];
                  arity_ = None;
                  kind = Normal;
                  ty;
                  loc_ = ghost_loc_;
                }
            in
            let id = Ident.of_ident tast_id in
            Core.let_ id (go_nontail expr) (k tast_self)
      in
      desugar_cascade texpr (fun self -> go_tail ~need_wrap_ok self)
  | Texpr_apply { func = Texpr_constr _; args; ty }
    when Global_env.is_newtype global_env ty -> (
      match[@warning "-fragile-match"] args with
      | arg :: [] ->
          Core.prim ~ty
            (Pcast { kind = Make_newtype })
            [ go_nontail arg.arg_value ]
          |> wrap_ok
      | _ -> assert false)
  | Texpr_apply { func; args; ty; loc_ } ->
      transl_apply ~global_env ~ctx ~kind:`Normal ~loc_ ~ty func args |> wrap_ok
  | Texpr_letrec { bindings; body; ty = _; loc_ } ->
      let bindings =
        Lst.map bindings (fun (binder, fn) ->
            ( Ident.of_ident binder.binder_id,
              transl_fn ~global_env ~base:ctx.base fn ))
      in
      let body = go_tail ~need_wrap_ok body in
      let fn_groups = Core_util.group_local_fn_bindings bindings in
      Lst.fold_right fn_groups body (fun fn_group body ->
          match fn_group with
          | Non_rec (name, fn) -> Core.letfn ~loc:loc_ ~kind:Nonrec name fn body
          | Rec ((name, fn) :: []) ->
              Core.letfn ~loc:loc_ ~kind:Rec name fn body
          | Rec bindings -> Core.letrec ~loc:loc_ bindings body)
  | Texpr_letfn { binder; fn; body; is_rec; ty = _; loc_ } ->
      let name = Ident.of_ident binder.binder_id in
      let ({ params; body = fn_body } : Core.fn) =
        transl_fn ~base:ctx.base ~global_env fn
      in
      let body = go_tail ~need_wrap_ok body in
      let kind = if is_rec then Core.Rec else Nonrec in
      Core.letfn ~loc:loc_ ~kind name { params; body = fn_body } body
  | Texpr_constr { constr; tag; ty; loc_ } ->
      (match Stype.type_repr ty with
      | Tarrow { params_ty; ret_ty; err_ty = _ } ->
          let params, args =
            Lst.map_split params_ty (fun ty ->
                let id = Ident.fresh "*x" in
                let param : Core.param =
                  { binder = id; ty; loc_ = ghost_loc_ }
                in
                let arg = Core.var ~ty id in
                (param, arg))
          in
          let body : Core.expr =
            if Global_env.is_newtype global_env ret_ty then
              Core.prim ~ty:ret_ty (Pcast { kind = Make_newtype }) args
            else Core.constr ~ty:ret_ty constr tag args
          in
          Core.function_ ~loc:loc_ ~ty params body
      | ty ->
          if Typeutil.is_only_tag_enum ty then
            let tag =
              match tag with
              | Constr_tag_regular { index = tag; _ } -> tag
              | Extensible_tag _ -> failwith "unimplemented"
            in
            let c = Constant.C_int { v = Int32.of_int tag; repr = None } in
            Core.const ~loc:loc_ c
          else Core.constr ~loc:loc_ ~ty constr tag [])
      |> wrap_ok
  | Texpr_tuple { exprs; ty; loc_ } ->
      Core.tuple ~loc:loc_ ~ty (List.map go_nontail exprs) |> wrap_ok
  | Texpr_record { type_name = _; fields; ty; loc_ } ->
      let fields = List.map (transl_field_def ~global_env ctx) fields in
      Core.record ~loc:loc_ ~ty fields |> wrap_ok
  | Texpr_record_update { type_name = _; record; fields; all_fields; ty; loc_ }
    ->
      let fields_num = List.length all_fields in
      if fields_num <= 6 then
        Core.bind
          (transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx record)
          (fun record_id ->
            let record_var = Core.var ~ty record_id in
            let fields =
              Lst.map all_fields (fun f ->
                  let pos = f.pos in
                  match
                    Lst.find_first fields (fun (Field_def { pos = p; _ }) ->
                        p = pos)
                  with
                  | Some field -> transl_field_def ~global_env ctx field
                  | None ->
                      let label : Parsing_syntax.label =
                        { label_name = f.field_name; loc_ = ghost_loc_ }
                      in
                      let accessor : Parsing_syntax.accessor = Label label in
                      let field =
                        Core.field ~ty:f.ty_field record_var ~pos accessor
                      in
                      let is_mut = f.mut in
                      { label; expr = field; pos; is_mut })
            in
            Core.record ~loc:loc_ ~ty fields |> wrap_ok)
      else
        let record = go_nontail record in
        let fields = List.map (transl_field_def ~global_env ctx) fields in
        Core.record_update ~loc:loc_ record fields fields_num |> wrap_ok
  | Texpr_field { record; accessor = Newtype; pos = _; ty; loc_ } ->
      let core_record = go_nontail record in
      let newtype_info =
        Global_env.get_newtype_info global_env
          (Typedtree_util.type_of_typed_expr record)
        |> Option.get
      in
      (if newtype_info.recursive then
         Core.prim ~loc:loc_ ~ty
           (Pcast { kind = Unfold_rec_newtype })
           [ core_record ]
       else Core.field ~loc:loc_ ~ty ~pos:0 core_record Newtype)
      |> wrap_ok
  | Texpr_field { record; accessor; pos; ty; loc_ } ->
      (match record with
      | Texpr_ident
          {
            id = { var_id };
            kind = Value_constr tag;
            ty = var_ty;
            ty_args_;
            loc_;
          } ->
          Core.prim ~ty
            (Penum_field { index = pos; tag })
            [
              Core.var
                ~ty:(Type.make_constr_type var_ty ~tag)
                ~ty_args_ ~loc:loc_ (Ident.of_ident var_id);
            ]
      | _ ->
          let core_record = go_nontail record in
          Core.field ~loc:loc_ ~ty ~pos core_record accessor)
      |> wrap_ok
  | Texpr_mutate { record; label; field; pos; augmented_by; ty; loc_ } -> (
      let ty_field = Typedtree_util.type_of_typed_expr field in
      match record with
      | Texpr_ident
          {
            id = { var_id };
            kind = Value_constr tag;
            ty = var_ty;
            ty_args_;
            loc_;
          } ->
          let core_record =
            Core.var
              ~ty:(Type.make_constr_type var_ty ~tag)
              ~ty_args_ ~loc:loc_ (Ident.of_ident var_id)
          in
          let core_field = go_nontail field in
          let rhs =
            match augmented_by with
            | None -> core_field
            | Some fn ->
                let lhs =
                  Core.prim ~ty:ty_field
                    (Penum_field { index = pos; tag })
                    [ core_record ]
                in
                make_apply ~global_env ~loc:ghost_loc_ ~ty:ty_field ctx fn
                  [ lhs; core_field ]
          in
          Core.prim ~loc:loc_ ~ty
            (Pset_enum_field { index = pos; tag })
            [ core_record; rhs ]
          |> wrap_ok
      | _ -> (
          let core_record = go_nontail record in
          let core_field = go_nontail field in
          match augmented_by with
          | None ->
              Core.mutate ~loc:loc_ ~pos core_record label core_field |> wrap_ok
          | Some fn ->
              Core.bind core_record (fun var ->
                  let core_record =
                    Core.var ~ty:(Core.type_of_expr core_record) var
                  in
                  let lhs =
                    Core.field ~loc:loc_ ~ty:ty_field ~pos core_record
                      (Label label)
                  in
                  let rhs =
                    make_apply ~loc:ghost_loc_ ~ty:ty_field ~global_env ctx fn
                      [ lhs; core_field ]
                  in
                  Core.mutate ~loc:loc_ ~pos core_record label rhs |> wrap_ok)))
  | Texpr_array { exprs; ty; is_fixed_array = true; loc_ } ->
      Core.array ~loc:loc_ ~ty (List.map go_nontail exprs) |> wrap_ok
  | Texpr_array { exprs; ty; is_fixed_array = false; loc_ } ->
      Core.prim ~loc:loc_ ~ty Parray_make (List.map go_nontail exprs) |> wrap_ok
  | Texpr_assign { var; expr; augmented_by; ty = _; loc_ } ->
      let expr = go_nontail expr in
      let id = Ident.of_ident var.var_id in
      let ty = Core.type_of_expr expr in
      let mut_var_ty = mutable_var_type ty in
      let var : Core.expr = Core.var ~loc:var.loc_ ~ty:mut_var_ty id in
      let rhs =
        match augmented_by with
        | None -> expr
        | Some fn ->
            let lhs =
              Core.field ~ty ~pos:0
                (Core.var ~loc:loc_ ~ty:mut_var_ty id)
                (Label mutable_var_label)
            in
            make_apply ~loc:ghost_loc_ ~ty ~global_env ctx fn [ lhs; expr ]
      in
      Core.mutate ~loc:loc_ ~pos:0 var mutable_var_label rhs |> wrap_ok
  | Texpr_sequence { expr1; expr2; ty = _; loc_ } ->
      let expr1 = go_nontail expr1 in
      let expr2 = go_tail ~need_wrap_ok expr2 in
      Core.sequence ~loc:loc_ expr1 expr2
  | Texpr_if { cond; ifso; ifnot; ty = _; loc_ } ->
      let cond = go_nontail cond in
      let ifso = go_tail ~need_wrap_ok:false ifso in
      let ifnot = Option.map (go_tail ~need_wrap_ok:false) ifnot in
      Core.if_ ~loc:loc_ cond ~ifso ?ifnot |> wrap_ok
  | Texpr_match { expr; cases; ty; loc_ } ->
      let cases =
        Lst.map cases (fun { pat; pat_binders; action } ->
            (pat, pat_binders, go_tail ~need_wrap_ok:false action))
      in
      Transl_match.transl_match ~global_env (go_nontail expr) cases ~ty
        ~loc:loc_
      |> wrap_ok
  | Texpr_try { body; catch; catch_all; try_else; ty; err_ty; loc_ } ->
      let try_join = Ident.fresh "*try" in
      let try_join_param_id = Ident.fresh "*try_err" in
      let join_param_var = Core.var ~ty:err_ty try_join_param_id in
      let error_ctx = { raise_join = try_join; need_raise_join = false } in
      let try_ctx =
        { ctx with error_ctx = Some error_ctx; error_ty = Some err_ty }
      in
      let body =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env try_ctx body
      in
      let catch_cases =
        if catch_all then
          let binder_id = Basic_ident.fresh "*catchall" in
          let binder : Typedtree.binder = { binder_id; loc_ = ghost_loc_ } in
          let var : Typedtree.var = { var_id = binder_id; loc_ = ghost_loc_ } in
          let pat : Typedtree.pat =
            Tpat_var { binder; ty = err_ty; loc_ = ghost_loc_ }
          in
          let pat_binders : Typedtree.pat_binders =
            [ { binder; binder_typ = err_ty } ]
          in
          let error_value : Typedtree.expr =
            Texpr_ident
              {
                id = var;
                ty_args_ = [||];
                arity_ = None;
                kind = Normal;
                ty = err_ty;
                loc_ = ghost_loc_;
              }
          in
          let action : Typedtree.expr =
            Texpr_raise { error_value; ty; loc_ = ghost_loc_ }
          in
          let catch_all_case : Typedtree.match_case =
            { pat; pat_binders; action }
          in
          Lst.map (catch @ [ catch_all_case ])
            (fun { pat; pat_binders; action } ->
              (pat, pat_binders, go_tail ~need_wrap_ok:false action))
        else
          Lst.map catch (fun { pat; pat_binders; action } ->
              (pat, pat_binders, go_tail ~need_wrap_ok:false action))
      in
      let catch_body =
        Transl_match.transl_match ~global_env join_param_var catch_cases ~ty
          ~loc:loc_
      in
      let body =
        match try_else with
        | None -> body
        | Some try_else ->
            let try_else_cases =
              Lst.map try_else (fun { pat; pat_binders; action } ->
                  (pat, pat_binders, go_tail ~need_wrap_ok:false action))
            in
            Core.bind ~loc:loc_ body (fun tmp ->
                let tmp_var = Core.var ~ty:(Core.type_of_expr body) tmp in
                Transl_match.transl_match ~global_env tmp_var try_else_cases ~ty
                  ~loc:loc_)
      in
      Core.letfn ~kind:Nontail_join try_join
        {
          params =
            [ { binder = try_join_param_id; ty = err_ty; loc_ = ghost_loc_ } ];
          body = catch_body;
        }
        body
      |> wrap_ok
  | Texpr_while { loop_cond; loop_body; while_else; ty; loc_ } ->
      let label = Label.fresh "*while" in
      let ctx = { ctx with loop_ctx = Loop_label label } in
      let cond = go_nontail loop_cond in
      let body =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx loop_body
      in
      let body_with_continue = Core.sequence body (Core.continue [] label ty) in
      let ifnot = Option.map (go_tail ~need_wrap_ok) while_else in
      Core.loop ~loc:loc_ []
        (Core.if_ cond ~ifso:body_with_continue ?ifnot)
        [] label
      |> wrap_ok
  | Texpr_break { arg; ty; loc_ } -> (
      if is_tail then
        match arg with
        | None -> Core.unit ()
        | Some arg -> go_tail ~need_wrap_ok:false arg
      else
        match ctx.loop_ctx with
        | Loop_label label | For_loop_info { label; _ } ->
            Core.break ~loc_
              (Option.map (go_tail ~need_wrap_ok:false) arg)
              label ty
        | Foreach loop_ctx ->
            let arg =
              match arg with
              | None -> Core.unit ()
              | Some arg -> go_tail ~need_wrap_ok:false arg
            in
            let break_value =
              Core.constr ~ty:loop_ctx.foreach_result_ty
                {
                  extra_info = No_extra_info;
                  constr_name = { name = "Break"; loc_ = ghost_loc_ };
                  loc_ = ghost_loc_;
                }
                Foreach_util.break.cs_tag [ arg ]
            in
            loop_ctx.has_early_exit <- true;
            Core.sequence
              (Core.mutate ~pos:0
                 (Core.var
                    ~ty:(mutable_var_type loop_ctx.foreach_result_ty)
                    loop_ctx.result_var)
                 mutable_var_label break_value)
              (Core.join_apply ~ty loop_ctx.exit_join
                 [ Foreach_util.iter_result_end ])
        | No_loop -> assert false)
  | Texpr_continue { args; ty; loc_ } -> (
      match ctx.loop_ctx with
      | No_loop -> assert false
      | Loop_label label ->
          Core.continue ~loc:loc_ (Lst.map args go_nontail) label ty
      | For_loop_info for_loop_ctx ->
          if args = [] then (
            for_loop_ctx.need_for_loop_join <- true;
            Core.join_apply ~loc:loc_ ~ty for_loop_ctx.continue_join [])
          else
            Core.continue ~loc:loc_ (Lst.map args go_nontail) for_loop_ctx.label
              ty
      | Foreach loop_ctx ->
          loop_ctx.has_early_exit <- true;
          Core.join_apply ~ty loop_ctx.exit_join
            [ Foreach_util.iter_result_continue ])
  | Texpr_loop { params; body; args; ty = _; loc_ } ->
      let params =
        Lst.map params
          (fun (Param { binder; ty; konstraint = _; kind = _ }) : Core.param ->
            { binder = Ident.of_ident binder.binder_id; ty; loc_ = binder.loc_ })
      in
      let label = Label.fresh "*loop" in
      let ctx = { ctx with loop_ctx = Loop_label label } in
      let body =
        transl_expr ~is_tail ~need_wrap_ok:false ~global_env ctx body
      in
      Core.loop ~loc:loc_ params body (Lst.map args go_nontail) label |> wrap_ok
  | Texpr_for { binders; condition; steps; body; for_else; ty; loc_ } ->
      let params, init_values =
        Lst.split_map binders (fun (binder, init) ->
            ( ({
                 binder = Ident.of_ident binder.binder_id;
                 ty = Typedtree_util.type_of_typed_expr init;
                 loc_ = binder.loc_;
               }
                : Core.param),
              go_nontail init ))
      in
      let steps =
        Lst.map params (fun p ->
            match
              Lst.find_opt steps (fun (step_binder, step_expr) ->
                  if Ident.equal p.binder (Ident.of_ident step_binder.var_id)
                  then Some step_expr
                  else None)
            with
            | None -> Core.var ~ty:p.ty p.binder
            | Some step -> go_nontail step)
      in
      let body_loc = Typedtree.loc_of_typed_expr body in
      let continue_join = Ident.fresh "*continue" in
      let label = Label.fresh "*for" in
      let ctx =
        {
          ctx with
          loop_ctx =
            For_loop_info { continue_join; need_for_loop_join = false; label };
        }
      in
      let body =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx body
      in
      let continue = Core.continue steps label ty in
      let body_with_continue =
        match ctx.loop_ctx with
        | For_loop_info { need_for_loop_join = true; _ } ->
            let apply = Core.join_apply ~ty continue_join [] in
            Core.letfn ~kind:Nontail_join continue_join
              { params = []; body = continue }
              (Core.sequence ~loc:body_loc body apply)
        | For_loop_info { need_for_loop_join = false; _ } ->
            Core.sequence ~loc:body_loc body continue
        | No_loop | Loop_label _ | Foreach _ -> assert false
      in
      let ifnot = Option.map (go_tail ~need_wrap_ok:false) for_else in
      let loop_body =
        match condition with
        | None -> body_with_continue
        | Some cond ->
            Core.if_ ~loc:body_loc ~ifso:body_with_continue ?ifnot
              (go_nontail cond)
      in
      Core.loop ~loc:loc_ params loop_body init_values label |> wrap_ok
  | Texpr_foreach { binders; elem_tys; expr; body; else_block; ty; loc_ } ->
      let expr = go_nontail expr in
      let result_var = Ident.fresh "*foreach_result" in
      let foreach_result_ty =
        Foreach_util.type_foreach_result ty ctx.return_ty
          (match ctx.error_ty with None -> Stype.unit | Some err_ty -> err_ty)
      in
      let exit_join = Ident.fresh "*foreach_exit" in
      let body_err_ctx =
        match ctx.error_ty with
        | None -> None
        | Some _ ->
            Some
              {
                raise_join = Ident.fresh "*foreach_raise";
                need_raise_join = false;
              }
      in
      let loop_ctx =
        { result_var; exit_join; has_early_exit = false; foreach_result_ty }
      in
      let body =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env
          {
            ctx with
            loop_ctx = Foreach loop_ctx;
            return_ctx = Foreach_return loop_ctx;
            error_ctx = body_err_ctx;
          }
          body
      in
      let body = Core.sequence body Foreach_util.iter_result_continue in
      let body =
        match body_err_ctx with
        | Some { need_raise_join = true; raise_join } -> (
            match[@warning "-fragile-match"] ctx.error_ty with
            | Some error_ty ->
                loop_ctx.has_early_exit <- true;
                (match ctx.error_ctx with
                | None -> ()
                | Some error_ctx -> error_ctx.need_raise_join <- true);
                let err_value_param = Ident.fresh "*foreach_error" in
                let err_value =
                  Core.constr ~ty:foreach_result_ty
                    {
                      extra_info = No_extra_info;
                      constr_name = { name = "Error"; loc_ = ghost_loc_ };
                      loc_ = ghost_loc_;
                    }
                    Foreach_util.error.cs_tag
                    [ Core.var ~ty:error_ty err_value_param ]
                in
                Core.letfn ~kind:Nontail_join raise_join
                  {
                    params =
                      [
                        {
                          binder = err_value_param;
                          ty = error_ty;
                          loc_ = ghost_loc_;
                        };
                      ];
                    body =
                      Core.sequence
                        (Core.mutate ~pos:0
                           (Core.var
                              ~ty:(mutable_var_type loop_ctx.foreach_result_ty)
                              loop_ctx.result_var)
                           mutable_var_label err_value)
                        Foreach_util.iter_result_end;
                  }
                  body
            | _ -> assert false)
        | _ -> body
      in
      let body =
        if loop_ctx.has_early_exit then
          (let exit_join_param = Ident.fresh "*foreach_body_result" in
           Core.letfn ~kind:Nontail_join exit_join
             {
               params =
                 [
                   {
                     binder = exit_join_param;
                     ty = Stype.type_iter_result;
                     loc_ = ghost_loc_;
                   };
                 ];
               body = Core.var ~ty:Stype.type_iter_result exit_join_param;
             })
            body
        else body
      in
      let callback =
        let params =
          Lst.map2 binders elem_tys (fun binder ty : Core.param ->
              match binder with
              | None -> { binder = Ident.fresh "*_"; ty; loc_ = ghost_loc_ }
              | Some { binder_id; loc_ } ->
                  { binder = Ident.of_ident binder_id; ty; loc_ })
        in
        Core.function_
          ~ty:(Builtin.type_arrow elem_tys Stype.type_iter_result ~err_ty:None)
          params body
      in
      let main_loop =
        Core.bind ~loc:loc_ expr (fun expr_id ->
            Core.apply expr_id [ callback ] ~ty:Stype.type_iter_result ~loc:loc_
              ~kind:
                (Normal
                   {
                     func_ty =
                       Builtin.type_arrow
                         [ Core.type_of_expr callback ]
                         Stype.type_iter_result ~err_ty:None;
                   }))
      in
      let main_loop = Core.prim ~ty:Stype.unit Pignore [ main_loop ] in
      if not loop_ctx.has_early_exit then
        match else_block with
        | None -> main_loop |> wrap_ok
        | Some else_block ->
            Core.sequence main_loop (go_tail ~need_wrap_ok else_block)
      else
        let postprocess =
          let break_value = Ident.fresh "*break" in
          let return_value = Ident.fresh "*return" in
          let error_case =
            match body_err_ctx with
            | Some _ -> (
                match[@warning "-fragile-match"] ctx.error_ty with
                | Some error_ty ->
                    let error_value = Ident.fresh "*error" in
                    ( Foreach_util.error.cs_tag,
                      Some error_value,
                      transl_raise ctx
                        (Foreach_util.get_first_enum_field error_value
                           Foreach_util.error ~ty:error_ty
                           ~constr_ty:foreach_result_ty)
                        ~ty ~loc_:ghost_loc_ )
                | _ -> assert false)
            | None -> (Foreach_util.error.cs_tag, None, Core.prim ~ty Ppanic [])
          in
          Core.switch_constr
            (Core.field ~ty:foreach_result_ty ~pos:0
               (Core.var ~ty:(mutable_var_type foreach_result_ty) result_var)
               (Label mutable_var_label))
            ~default:None
            [
              ( Foreach_util.continue.cs_tag,
                None,
                match else_block with
                | None -> Core.unit ()
                | Some else_block -> go_nontail else_block );
              ( Foreach_util.break.cs_tag,
                Some break_value,
                Foreach_util.get_first_enum_field break_value Foreach_util.break
                  ~ty ~constr_ty:foreach_result_ty );
              ( Foreach_util.return.cs_tag,
                Some return_value,
                transl_return ~is_tail ctx
                  (Foreach_util.get_first_enum_field return_value
                     Foreach_util.return ~ty:ctx.return_ty
                     ~constr_ty:foreach_result_ty)
                  ~ty ~loc_:ghost_loc_ );
              error_case;
            ]
          |> wrap_ok
        in
        Core.let_ result_var
          (Core.record
             ~ty:(mutable_var_type foreach_result_ty)
             [
               {
                 label = mutable_var_label;
                 pos = 0;
                 is_mut = true;
                 expr =
                   Core.constr ~ty:foreach_result_ty
                     {
                       extra_info = No_extra_info;
                       constr_name = { name = "Continue"; loc_ = ghost_loc_ };
                       loc_ = ghost_loc_;
                     }
                     Foreach_util.continue.cs_tag [];
               };
             ])
          (Core.sequence main_loop postprocess)
  | Texpr_return { return_value; ty; loc_ } ->
      let return_value =
        match return_value with
        | Some rt ->
            transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx rt
        | None -> Core.unit ()
      in
      if need_wrap_ok then
        match ctx.wrapper_info with
        | Some result_ty -> wrap_ok_prim return_value ~result_ty
        | None -> assert false
      else transl_return ctx ~is_tail return_value ~ty ~loc_
  | Texpr_raise { error_value; ty; loc_ } ->
      let error_value =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx
          error_value
      in
      let ty =
        if need_wrap_ok then
          match ctx.wrapper_info with Some result_ty -> result_ty | None -> ty
        else ty
      in
      transl_raise ctx error_value ~ty ~loc_
  | Texpr_exclamation { expr; ty; loc_; convert_to_result } -> (
      let expr = go_nontail expr in
      let result_ty = Core.type_of_expr expr in
      let ok_ty, _ =
        match result_ty with
        | T_constr { type_constructor; tys = [ ok_ty; err_ty ] }
          when Type_path.equal type_constructor
                 Type_path.Builtin.type_path_multi_value_result ->
            (ok_ty, err_ty)
        | _ -> assert false
      in
      if convert_to_result then
        Core.handle_error ~loc:loc_ expr To_result ~ty |> wrap_ok
      else if need_wrap_ok then expr
      else
        match ctx.error_ctx with
        | Some error_ctx ->
            error_ctx.need_raise_join <- true;
            Core.handle_error ~loc:loc_ expr (Joinapply error_ctx.raise_join)
              ~ty:ok_ty
        | None ->
            let ctx_ok_ty = ctx.return_ty in
            Core.handle_error ~loc:loc_ expr
              (Return_err { ok_ty = ctx_ok_ty })
              ~ty:ok_ty)
  | Texpr_hole { ty; loc_; kind } ->
      let msg =
        match kind with
        | Synthesized | Incomplete -> "hole in expr"
        | Todo -> "todo in expr"
      in
      Core.raise ~loc:loc_ ~ty msg
  | Texpr_constraint { expr; _ } -> go_tail ~need_wrap_ok expr
  | Texpr_pipe
      {
        lhs = expr;
        ty;
        rhs =
          Pipe_partial_apply
            {
              args;
              func =
                ( Texpr_ident _ | Texpr_method _ | Texpr_unresolved_method _
                | Texpr_constr { constr = _; tag = _; ty = _; loc_ = _ } ) as
                func;
              loc_ = rhs_loc_;
            };
        loc_ = _;
      }
    when Lst.for_all args (fun { arg_kind; _ } -> arg_kind = Positional) ->
      transl_apply ~global_env ~ctx ~ty ~loc_:rhs_loc_
        ~kind:(`Pipe (Typedtree.loc_of_typed_expr expr))
        func
        ({ arg_value = expr; arg_kind = Positional } :: args)
      |> wrap_ok
  | Texpr_pipe { lhs = expr; rhs = body; ty; loc_ } ->
      let var_id = Basic_ident.fresh "*lhs" in
      let body =
        let arg =
          Typedtree.Texpr_ident
            {
              id = { var_id; loc_ = ghost_loc_ };
              kind = Normal;
              ty = Typedtree_util.type_of_typed_expr expr;
              ty_args_ = [||];
              arity_ = None;
              loc_ = ghost_loc_;
            }
        in
        match body with
        | Typedtree.Pipe_partial_apply { func; args; loc_ } ->
            transl_apply ~global_env ~ctx ~ty ~loc_
              ~kind:(`Pipe (Typedtree.loc_of_typed_expr expr))
              func
              ({ arg_value = arg; arg_kind = Positional } :: args)
            |> wrap_ok
        | _ -> assert false
      in
      Core.let_ ~loc:loc_ (Ident.of_ident var_id) (go_nontail expr) body
  | Texpr_interp { elems; ty = _ } ->
      let core_elems =
        Lst.map elems (function
          | Interp_lit s -> Core.const (C_string s)
          | Interp_expr { expr; to_string } ->
              transl_apply ~global_env ~ctx ~kind:`Normal ~ty:Stype.string
                ~loc_:(Typedtree.loc_of_typed_expr expr)
                to_string
                [ { arg_value = expr; arg_kind = Positional } ])
      in
      (match core_elems with
      | [] -> Core.const (C_string "")
      | x :: xs ->
          Lst.fold_left xs x (fun acc x ->
              Core.prim ~ty:Stype.string
                (Pccall { func_name = "add_string"; arity = 2 })
                [ acc; x ]))
      |> wrap_ok
  | Texpr_guard { cond; otherwise; body; loc_ = loc; ty } ->
      let cond = go_nontail cond in
      let body = go_tail ~need_wrap_ok:false body in
      let ifnot =
        Some
          (match otherwise with
          | Some expr -> go_tail ~need_wrap_ok:false expr
          | None -> Core.prim ~loc ~ty Primitive.Ppanic [])
      in
      Core.if_ ~loc cond ~ifso:body ?ifnot |> wrap_ok
  | Texpr_guard_let { pat; rhs; pat_binders; otherwise; body; loc_; ty } ->
      let ok_case = { Typedtree.pat; pat_binders; action = body } in
      let fail_cases =
        match otherwise with Some cases -> cases | None -> []
      in
      let expr =
        Typedtree.Texpr_match
          {
            expr = rhs;
            cases = ok_case :: fail_cases;
            ty;
            loc_;
            match_loc_ = loc_;
          }
      in
      transl_expr ~global_env ~is_tail ~need_wrap_ok ctx expr

and transl_apply ~global_env ~ctx ~(kind : [ `Normal | `Pipe of Rloc.t ]) ~ty
    ~loc_ func args =
  let go_nontail expr =
    transl_expr ~global_env ~is_tail:false ~need_wrap_ok:false ctx expr
      [@@inline]
  in
  let labelled_args = Hash_string.create 17 in
  let positional_args = Vec.empty () in
  Lst.iter args (fun arg ->
      let carg = go_nontail arg.arg_value in
      match arg.arg_kind with
      | Positional -> Vec.push positional_args carg
      | Labelled label | Labelled_pun label ->
          Hash_string.add labelled_args label.label_name (false, carg)
      | Labelled_option { label; question_loc = _ }
      | Labelled_option_pun { label; question_loc = _ } ->
          Hash_string.add labelled_args label.label_name (true, carg));
  let arg_bindings : (Ident.t * Core.expr * Core.expr) Vec.t = Vec.empty () in
  let need_let = ref false in
  let process_labelled_args ~func_ty ~make_default_arg arity =
    match[@warning "-fragile-match"] Stype.type_repr func_ty with
    | (Tarrow { params_ty; ret_ty = _; err_ty = _ } : Stype.t) ->
        let loc_of_args =
          Fn_arity.to_list_map arity (fun param_kind ->
              match (kind, param_kind) with
              | `Pipe lhs_loc, Positional 0 -> lhs_loc
              | _, Positional index ->
                  Core.loc_of_expr (Vec.get positional_args index)
              | ( _,
                  ( Labelled { label; _ }
                  | Optional { label; _ }
                  | Autofill { label }
                  | Question_optional { label } ) ) -> (
                  match Hash_string.find_opt labelled_args label with
                  | None -> ghost_loc_
                  | Some (_, expr) -> Core.loc_of_expr expr))
        in
        Fn_arity.iter2 arity params_ty (fun param_kind param_ty ->
            let arg_expr =
              match param_kind with
              | Optional { label; depends_on } -> (
                  match Hash_string.find_opt labelled_args label with
                  | Some (_, arg_expr) -> arg_expr
                  | None ->
                      let args_of_default_fn =
                        Lst.map depends_on (fun index ->
                            need_let := true;
                            let _, arg_id_expr, _ =
                              Vec.get arg_bindings index
                            in
                            arg_id_expr)
                      in
                      let default_arg =
                        make_default_arg label param_ty args_of_default_fn
                      in
                      Hash_string.add labelled_args label (false, default_arg);
                      default_arg)
              | Question_optional { label } -> (
                  match Hash_string.find_opt labelled_args label with
                  | None ->
                      Core.constr
                        {
                          extra_info = No_extra_info;
                          constr_name = { name = "None"; loc_ = ghost_loc_ };
                          loc_ = ghost_loc_;
                        }
                        Builtin.constr_none.cs_tag []
                        ~ty:(Builtin.type_option param_ty)
                  | Some (true, arg_expr) -> arg_expr
                  | Some (false, arg_expr) ->
                      Core.constr
                        {
                          extra_info = No_extra_info;
                          constr_name = { name = "Some"; loc_ = ghost_loc_ };
                          loc_ = ghost_loc_;
                        }
                        Builtin.constr_some.cs_tag [ arg_expr ]
                        ~ty:(Builtin.type_option param_ty))
              | Autofill { label } -> (
                  match Hash_string.find_opt labelled_args label with
                  | Some (_, arg_expr) -> arg_expr
                  | None when Type.same_type param_ty Stype.type_sourceloc ->
                      let arg_expr =
                        Core.const
                          (C_string (Rloc.loc_range_string ~base:ctx.base loc_))
                      in
                      Hash_string.add labelled_args label (false, arg_expr);
                      arg_expr
                  | None when Type.same_type param_ty Stype.type_argsloc ->
                      let loc_of_args_expr =
                        Lst.map loc_of_args (fun loc ->
                            if Rloc.is_no_location loc then
                              Core.constr
                                ~ty:(Builtin.type_option Stype.type_sourceloc)
                                {
                                  constr_name =
                                    { name = "None"; loc_ = ghost_loc_ };
                                  extra_info = No_extra_info;
                                  loc_ = ghost_loc_;
                                }
                                Builtin.constr_none.cs_tag []
                            else
                              Core.constr
                                ~ty:(Builtin.type_option Stype.type_sourceloc)
                                {
                                  constr_name =
                                    { name = "Some"; loc_ = ghost_loc_ };
                                  extra_info = No_extra_info;
                                  loc_ = ghost_loc_;
                                }
                                Builtin.constr_some.cs_tag
                                [
                                  Core.const
                                    (C_string
                                       (Rloc.loc_range_string ~base:ctx.base loc));
                                ])
                      in
                      let loc_of_args_array =
                        Core.prim
                          ~ty:
                            (Builtin.type_array
                               (Builtin.type_option Stype.type_sourceloc))
                          Parray_make loc_of_args_expr
                      in
                      Hash_string.add labelled_args label
                        (false, loc_of_args_array);
                      loc_of_args_array
                  | None -> assert false)
              | Positional index -> Vec.get positional_args index
              | Labelled { label; _ } ->
                  snd (Hash_string.find_exn labelled_args label)
            in
            let arg_id = Ident.fresh "arg" in
            let arg_id_expr =
              Core.var ~ty:(Core.type_of_expr arg_expr) arg_id
            in
            Vec.push arg_bindings (arg_id, arg_id_expr, arg_expr))
    | _ -> assert false
      [@@inline]
  in
  (match func with
  | Texpr_ident
      { arity_ = Some arity; id = { var_id = id }; ty_args_; ty = func_ty; _ }
  | Texpr_method
      { arity_ = Some arity; meth = { var_id = id }; ty_args_; ty = func_ty; _ }
    when not (Fn_arity.is_simple arity) ->
      let make_default_arg label param_ty args =
        Core.apply ~loc:ghost_loc_
          ~kind:
            (Normal
               {
                 func_ty =
                   Builtin.type_arrow
                     (Lst.map args Core.type_of_expr)
                     param_ty ~err_ty:None;
               })
          ~ty:param_ty ~ty_args_
          (name_of_default_arg ~label id)
          args
      in
      process_labelled_args ~func_ty ~make_default_arg arity
  | Texpr_constr { arity_; ty = func_ty; _ }
  | Texpr_unresolved_method { arity_ = Some arity_; ty = func_ty; _ }
    when not (Fn_arity.is_simple arity_) ->
      process_labelled_args ~func_ty
        ~make_default_arg:(fun _ _ _ -> assert false)
        arity_
  | _ -> ());
  let args =
    if Vec.is_empty arg_bindings then (
      assert (Hash_string.length labelled_args = 0);
      Vec.to_list positional_args)
    else if !need_let then
      Vec.map_into_list arg_bindings (fun (_, arg_id_expr, _) -> arg_id_expr)
    else Vec.map_into_list arg_bindings (fun (_, _, arg_expr) -> arg_expr)
  in
  let apply_expr =
    match Stype.type_repr (Typedtree_util.type_of_typed_expr func) with
    | Tarrow { err_ty = Some err_ty; _ } ->
        let result_ty = Stype.make_multi_value_result_ty ~ok_ty:ty ~err_ty in
        make_apply ~global_env ctx func args ~loc:loc_ ~ty:result_ty
    | _ -> make_apply ~global_env ctx func args ~loc:loc_ ~ty
  in
  if !need_let then
    Vec.fold_right arg_bindings apply_expr ~f:(fun (id, _, rhs) body ->
        Core.let_ id rhs body)
  else apply_expr

and make_apply ~global_env ctx (func : Typedtree.expr) (args : Core.expr list)
    ~loc ~ty =
  let make_apply () =
    match
      transl_expr ~global_env ~is_tail:false ~need_wrap_ok:false ctx func
    with
    | Cexpr_var { id = func; ty = func_ty; ty_args_; prim } ->
        Core.apply ~loc ~ty_args_ ~kind:(Normal { func_ty }) ~ty ~prim func args
    | func ->
        let id = Ident.fresh "*func" in
        let func_ty = Core.type_of_expr func in
        let app : Core.expr =
          Core.apply ~loc ~kind:(Normal { func_ty }) ~ty id args
        in
        Core.let_ id func app
      [@@inline]
  in
  match func with
  | Texpr_ident { id = _; kind = Prim prim; ty = _ }
  | Texpr_method { type_name = _; meth = _; prim = Some prim; ty = _ } -> (
      match prim with
      | Pintrinsic intrinsic when not !Basic_config.debug -> (
          match Core_util.try_apply_intrinsic intrinsic args ~loc ~ty with
          | Some result -> result
          | None -> make_apply ())
      | Pintrinsic _ -> make_apply ()
      | _ -> Core.prim ~loc ~ty prim args)
  | Texpr_unresolved_method
      { trait_name; method_name; self_type = self_ty; ty = func_ty } -> (
      let trait = trait_name.name in
      match
        transl_trait_method ~global_env ~trait ~self_ty ~expect_ty:func_ty
          ~method_name
      with
      | `Prim prim -> Core.prim ~loc ~ty prim args
      | `Regular (id, ty_args_, prim) ->
          Core.apply ~loc ~ty ~kind:(Normal { func_ty }) ~ty_args_ ~prim id args
      )
  | Texpr_ident { id = { var_id = Pdot qual_name }; kind = _; ty = _; ty_args_ }
    -> (
      match Core_util.specialize qual_name ty_args_ with
      | Some prim -> Core.prim ~loc ~ty prim args
      | None -> make_apply ())
  | Texpr_constr { constr; tag; ty = _ } ->
      if Global_env.is_newtype global_env ty then
        Core.prim ~ty (Pcast { kind = Make_newtype }) args
      else Core.constr ~loc ~ty constr tag args
  | _ -> make_apply ()

and transl_fn ~global_env ~base (fn : Typedtree.fn) : Core.fn =
  let params =
    Lst.map fn.params
      (fun (Param { binder; ty; konstraint = _; kind = _ }) : Core.param ->
        { binder = Ident.of_ident binder.binder_id; ty; loc_ = binder.loc_ })
  in
  let err_type =
    match Stype.type_repr fn.ty with
    | Tarrow { err_ty; _ } -> err_ty
    | _ -> assert false
  in
  let return_value_ty = Typedtree_util.type_of_typed_expr fn.body in
  let wrapper_info =
    match err_type with
    | None -> None
    | Some err_ty ->
        Some (Stype.make_multi_value_result_ty ~ok_ty:return_value_ty ~err_ty)
  in
  let ctx =
    {
      return_ctx = No_return;
      error_ctx = None;
      loop_ctx = No_loop;
      return_ty = return_value_ty;
      error_ty = err_type;
      wrapper_info;
      base;
    }
  in
  let need_wrap_ok = match err_type with Some _ -> true | None -> false in
  let body = transl_expr ~is_tail:true ~need_wrap_ok ~global_env ctx fn.body in
  { params; body }

and transl_field_def ~global_env (ctx : transl_context)
    (field : Typedtree.field_def) : Core.field_def =
  let (Field_def { label; expr; pos; is_mut; is_pun = _ }) = field in
  {
    label;
    expr = transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env ctx expr;
    pos;
    is_mut;
  }

and transl_return ctx ~is_tail return_value ~ty ~loc_ =
  match ctx.return_ctx with
  | Foreach_return loop_ctx ->
      let return_value =
        Core.constr ~ty:loop_ctx.foreach_result_ty
          {
            extra_info = No_extra_info;
            constr_name = { name = "Return"; loc_ = ghost_loc_ };
            loc_ = ghost_loc_;
          }
          Foreach_util.return.cs_tag [ return_value ]
      in
      loop_ctx.has_early_exit <- true;
      Core.sequence
        (Core.mutate ~pos:0
           (Core.var
              ~ty:(mutable_var_type loop_ctx.foreach_result_ty)
              loop_ctx.result_var)
           mutable_var_label return_value)
        (Core.join_apply ~ty loop_ctx.exit_join
           [ Foreach_util.iter_result_end ])
  | No_return -> (
      match ctx.error_ty with
      | None ->
          if is_tail then return_value
          else Core.return return_value ~return_kind:Single_value ~ty ~loc:loc_
      | Some err_ty ->
          let ok_ty = Core.type_of_expr return_value in
          let result_ty = Stype.make_multi_value_result_ty ~ok_ty ~err_ty in
          Core.return return_value
            ~return_kind:
              (Error_result { is_error = false; return_ty = result_ty })
            ~ty ~loc:loc_)
  | Normal_return return_ctx ->
      if is_tail then return_value
      else (
        return_ctx.need_return_join <- true;
        Core.join_apply ~loc:loc_ ~ty return_ctx.return_join [ return_value ])

and transl_raise ctx error_value ~ty ~loc_ =
  match ctx.error_ctx with
  | Some error_ctx ->
      error_ctx.need_raise_join <- true;
      Core.join_apply ~loc:loc_ ~ty error_ctx.raise_join [ error_value ]
  | None ->
      let ok_ty = ctx.return_ty in
      let err_ty = Core.type_of_expr error_value in
      let result_ty = Stype.make_multi_value_result_ty ~ok_ty ~err_ty in
      Core.return error_value
        ~return_kind:(Error_result { is_error = true; return_ty = result_ty })
        ~ty ~loc:loc_

and transl_trait_method ~global_env ~trait ~method_name ~(self_ty : Stype.t)
    ~(expect_ty : Stype.t) =
  let resolve_by_path (type_name : Type_path.t) =
    match
      Global_env.find_trait_method global_env ~trait ~type_name ~method_name
    with
    | None -> assert false
    | Some mi -> (
        let actual_ty, ty_args =
          Poly_type.instantiate_method_no_constraint mi
        in
        Ctype.unify_exn expect_ty actual_ty;
        match mi.prim with
        | None | Some (Pintrinsic _) ->
            `Regular (Ident.of_qual_ident mi.id, ty_args, mi.prim)
        | Some p -> `Prim p)
      [@@inline]
  in
  let self_ty = Stype.type_repr self_ty in
  match self_ty with
  | Tparam { index; name_ } ->
      let id =
        Ident.of_local_method ~index ~tvar_name:name_ ~trait method_name
      in
      `Regular (id, [||], None)
  | T_constr { type_constructor; _ } -> resolve_by_path type_constructor
  | T_builtin builtin -> resolve_by_path (Stype.tpath_of_builtin builtin)
  | T_trait object_trait -> (
      let trait_decl =
        Global_env.find_trait_by_path global_env object_trait |> Option.get
      in
      match
        Lst.find_first_with_index trait_decl.closure_methods
          (fun (actual_trait, method_decl) ->
            Type_path.equal actual_trait trait
            && method_decl.method_name = method_name)
      with
      | Some (method_index, _) ->
          `Prim (Pcall_object_method { method_index; method_name })
      | None -> resolve_by_path object_trait)
  | Tvar _ | Tarrow _ | T_blackhole -> assert false

let dummy_ctx ~base =
  {
    return_ctx = No_return;
    error_ctx = None;
    loop_ctx = No_loop;
    return_ty = Stype.unit;
    error_ty = None;
    wrapper_info = None;
    base;
  }

let transl_top_expr expr ~global_env ~base =
  let return_join = Ident.fresh "*return" in
  let return_value_id = Ident.fresh "*return_value" in
  let return_ctx = Normal_return { return_join; need_return_join = false } in
  let ctx =
    {
      return_ctx;
      error_ctx = None;
      loop_ctx = No_loop;
      return_ty = Typedtree_util.type_of_typed_expr expr;
      error_ty = None;
      wrapper_info = None;
      base;
    }
  in
  let body =
    transl_expr ~is_tail:true ~need_wrap_ok:false ~global_env ctx expr
  in
  match[@warning "-fragile-match"] return_ctx with
  | Normal_return return_ctx ->
      if return_ctx.need_return_join then
        Core.letfn ~kind:Nontail_join return_join
          {
            params =
              [
                { binder = return_value_id; ty = Stype.unit; loc_ = ghost_loc_ };
              ];
            body = Core.var ~ty_args_:[||] ~ty:Stype.unit return_value_id;
          }
          body
      else body
  | _ -> assert false

let generate_default_exprs ~global_env ~(fn_binder : Typedtree.binder) ~is_pub
    ~ty_params_ ~(params : Typedtree.params) ~arity ~base =
  let prev_params : Core.param Vec.t = Vec.empty () in
  let default_exprs = Vec.empty () in
  Fn_arity.iter2 arity params (fun param_kind (Param { binder; kind; ty; _ }) ->
      match (param_kind, kind) with
      | Optional { label = _; depends_on }, Optional expr ->
          let subst = Ident.Hash.create 16 in
          let params =
            Lst.map depends_on (fun index ->
                let param = Vec.get prev_params index in
                let new_binder = Ident.rename param.binder in
                Ident.Hash.add subst param.binder new_binder;
                { param with binder = new_binder })
          in
          Vec.push prev_params
            { binder = Ident.of_ident binder.binder_id; ty; loc_ = binder.loc_ };
          let cexpr =
            transl_expr ~is_tail:true ~need_wrap_ok:false ~global_env
              (dummy_ctx ~base) expr
            |> Core_util.substitute ~subst
          in
          Vec.push default_exprs
            (Core.Ctop_fn
               {
                 binder =
                   name_of_default_arg
                     ~label:(Basic_ident.base_name binder.binder_id)
                     fn_binder.binder_id;
                 ty_params_;
                 is_pub_ = is_pub;
                 loc_ = base;
                 func = { params; body = cexpr };
                 subtops = [];
               })
      | _, Optional _ -> assert false
      | _, (Positional | Labelled | Autofill | Question_optional) ->
          Vec.push prev_params
            { binder = Ident.of_ident binder.binder_id; ty; loc_ = binder.loc_ });
  Vec.to_list default_exprs
[@@inline]

let transl_impl ~global_env (impl : Typedtree.impl) : Core.top_item list =
  match impl with
  | Timpl_expr { expr; is_main; expr_id = _; loc_ } ->
      [
        Ctop_expr
          { expr = transl_top_expr ~global_env ~base:loc_ expr; is_main; loc_ };
      ]
  | Timpl_letdef { binder; konstraint = _; expr; is_pub; loc_ } ->
      let binder = Ident.of_ident binder.binder_id in
      let expr =
        transl_expr ~is_tail:false ~need_wrap_ok:false ~global_env
          (dummy_ctx ~base:loc_) expr
      in
      [ Ctop_let { binder; expr; is_pub_ = is_pub; loc_ } ]
  | Timpl_fun_decl
      {
        fun_decl = { kind = _; fn_binder; fn; is_pub; ty_params_ };
        arity_;
        loc_;
      } ->
      let binder = Ident.of_ident fn_binder.binder_id in
      let func = transl_fn ~base:loc_ ~global_env fn in
      Ctop_fn { binder; func; subtops = []; ty_params_; is_pub_ = is_pub; loc_ }
      :: generate_default_exprs ~fn_binder ~global_env ~is_pub ~ty_params_
           ~arity:arity_ ~params:fn.params ~base:loc_
  | Timpl_stub_decl { func_stubs; binder; is_pub; loc_; params; arity_; ret }
    -> (
      let default_exprs =
        generate_default_exprs ~fn_binder:binder ~global_env ~is_pub
          ~ty_params_:Tvar_env.empty ~arity:arity_ ~params ~base:loc_
      in
      let binder = Ident.of_ident binder.binder_id in
      let params_ty = Lst.map params (fun (Param { ty; _ }) -> ty) in
      let return_ty = Option.map Typedtree_util.stype_of_typ ret in
      match func_stubs with
      | Intrinsic -> default_exprs
      | Func_stub func_stubs ->
          Ctop_stub
            { binder; func_stubs; params_ty; return_ty; is_pub_ = is_pub; loc_ }
          :: default_exprs)

let transl ~(global_env : Global_env.t) (output : Typedtree.output) :
    Core.program =
  let (Output { value_defs; type_defs = _; trait_defs = _ }) = output in
  Lst.concat_map value_defs (transl_impl ~global_env)
