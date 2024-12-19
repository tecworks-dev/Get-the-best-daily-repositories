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
module Lst = Basic_lst
module Arr = Basic_arr
module Path = Pat_path
module Db = Patmatch_db

let ghost_loc_ = Rloc.no_location

type dotdot_info =
  | No_dotdot
  | Dotdot_with_binder of {
      binder : Typedtree.binder;
      len_pats1 : int;
      len_pats2 : int;
      binder_ty : Stype.t;
    }
  | Dotdot_no_binder

let make_constr_switch tag arg ~head ~ifso ~ifnot : Core.expr =
  let new_case = (tag, arg, ifso) in
  let cases, default =
    match (ifnot : Core.expr option) with
    | Some (Cexpr_switch_constr { obj; cases; default; ty = _ })
      when Basic_prelude.phys_equal obj head ->
        (new_case :: cases, default)
    | _ -> ([ new_case ], ifnot)
  in
  Core.switch_constr ~default head cases

let make_constant_switch ~global_env c ~head ~ifso ~ifnot : Core.expr =
  match Core.type_of_expr head with
  | ty when Type.same_type ty Stype.bigint ->
      let constant =
        let str =
          match c with
          | Constant.C_bigint { v; _ } ->
              Constant.C_string (Basic_bigint.to_string v)
          | _ -> assert false
        in
        match[@warning "-fragile-match"]
          Global_env.find_dot_method global_env
            ~type_name:Basic_type_path.Builtin.type_path_bigint
            ~method_name:"from_string"
        with
        | method_info :: [] ->
            Core.apply ~loc:Rloc.no_location
              ~kind:(Normal { func_ty = method_info.typ })
              ~ty:Stype.bigint
              (Basic_core_ident.of_qual_ident method_info.id)
              [ Core.const str ]
        | _ -> assert false
      in
      let cond =
        match[@warning "-fragile-match"]
          Global_env.find_dot_method global_env
            ~type_name:Basic_type_path.Builtin.type_path_bigint
            ~method_name:"op_equal"
        with
        | method_info :: [] ->
            let id = Basic_core_ident.of_qual_ident method_info.id in
            Core.apply ~loc:Rloc.no_location
              ~kind:(Normal { func_ty = method_info.typ })
              ~ty:Stype.bool id [ head; constant ]
        | _ -> assert false
      in
      Core.if_ cond ~ifso ~ifnot
  | _ ->
      let new_case = (c, ifso) in
      let cases, default =
        match (ifnot : Core.expr) with
        | Cexpr_switch_constant { obj; cases; default; ty = _ }
          when Basic_prelude.phys_equal obj head ->
            (new_case :: cases, default)
        | _ -> ([ new_case ], ifnot)
      in
      Core.switch_constant ~default head cases

let make_range_if obj lo hi ~inclusive ~ifso ~ifnot =
  let constant_to_operand_type (c : Constant.t) : Primitive.operand_type =
    match c with
    | C_int _ | C_char _ -> I32
    | C_uint _ -> U32
    | C_int64 _ -> I64
    | C_uint64 _ -> U64
    | C_float _ -> F32
    | C_double _ -> F64
    | C_bool _ | C_string _ | C_bytes _ | C_bigint _ -> assert false
  in
  let lo_satisfied =
    match lo with
    | None -> None
    | Some lo ->
        Core.prim ~ty:Stype.bool
          (Pcomparison
             { operand_type = constant_to_operand_type lo; operator = Le })
          [ Core.const lo; obj ]
        |> Option.some
  in
  let hi_satisfied =
    match hi with
    | None -> None
    | Some hi ->
        let operator : Primitive.comparison = if inclusive then Le else Lt in
        Core.prim ~ty:Stype.bool
          (Pcomparison { operand_type = constant_to_operand_type hi; operator })
          [ obj; Core.const hi ]
        |> Option.some
  in
  let predicate =
    match (lo_satisfied, hi_satisfied) with
    | None, None -> assert false
    | Some pred, None | None, Some pred -> pred
    | Some lo_satisfied, Some hi_satisfied ->
        Core.prim ~ty:Stype.bool Psequand [ lo_satisfied; hi_satisfied ]
  in
  Core.if_ predicate ~ifso ~ifnot

type bindings = (Ident.t * Path.t) list

type transl_context = { db : Db.t; path_cache : path_value Path.Map.t }
and path_value = { expr : Core.expr; mutable used : bool }

let set_db ctx db = { ctx with db } [@@inline]

let expr_of_path ctx path =
  let value = Path.Map.find_exn ctx.path_cache path in
  value.used <- true;
  value.expr
[@@inline]

let add_path ?(name_hint = "*x") ctx path ~ty :
    Core.binder * path_value * transl_context =
  let id = Ident.fresh name_hint in
  let expr : Core.expr = Core.var ~ty id in
  let value = { expr; used = false } in
  (id, value, { ctx with path_cache = Path.Map.add ctx.path_cache path value })

let add_path_and_insert_let_if_necessary ctx path ~ty ~(rhs : unit -> Core.expr)
    ~(body : transl_context -> Core.expr) =
  if Path.Map.mem ctx.path_cache path then body ctx
  else
    let binder, value, ctx = add_path ctx path ~ty in
    let body = body ctx in
    if value.used then
      let rhs = rhs () in
      Core.let_ ~loc:ghost_loc_ binder rhs body
    else body

let rec process_pat ~(global_env : Global_env.t) ~(ctx : transl_context)
    ~(bindings : bindings) ~(path : Path.t) ~(pat : Typedtree.pat)
    ~(ok_cont : transl_context -> bindings -> Core.expr)
    ~(fail_cont : transl_context -> Core.expr) =
  match (pat : Typedtree.pat) with
  | Tpat_any _ -> ok_cont ctx bindings
  | Tpat_var { binder; _ } ->
      ok_cont ctx ((Ident.of_ident binder.binder_id, path) :: bindings)
  | Tpat_constraint { pat; _ } ->
      process_pat ~global_env ~ctx ~bindings ~path ~pat ~ok_cont ~fail_cont
  | Tpat_alias { pat; alias; _ } ->
      let target_path : Path.t =
        match pat with
        | Tpat_constr { tag = Constr_tag_regular { index; total = _ }; _ } ->
            Casted_constr { tag_index = index } :: path
        | Tpat_constr { tag = Extensible_tag tag; _ } ->
            Casted_constr { tag_index = tag.index } :: path
        | _ -> path
      in
      process_pat ~global_env ~ctx
        ~bindings:((Ident.of_ident alias.binder_id, target_path) :: bindings)
        ~path ~pat ~ok_cont ~fail_cont
  | Tpat_or { pat1; pat2; _ } ->
      process_pat ~global_env ~ctx ~bindings ~path ~pat:pat1 ~ok_cont
        ~fail_cont:(fun ctx ->
          process_pat ~global_env ~ctx ~bindings ~path ~pat:pat2 ~ok_cont
            ~fail_cont)
  | Tpat_constant { c; _ } -> (
      let r = Db.eval_constant ctx.db path c in
      match r with
      | For_sure_yes r -> ok_cont (set_db ctx r.ok_db) bindings
      | For_sure_no r -> fail_cont (set_db ctx r.fail_db)
      | Uncertain r ->
          let head = expr_of_path ctx path in
          let ifso = ok_cont (set_db ctx r.ok_db) bindings in
          let ifnot = fail_cont (set_db ctx r.fail_db) in
          make_constant_switch ~global_env c ~head ~ifso ~ifnot)
  | Tpat_range { lhs; rhs; inclusive; ty = _ } -> (
      let lo =
        match lhs with
        | Tpat_constant { c; _ } -> Some c
        | Tpat_any _ -> None
        | _ -> assert false
      in
      let hi =
        match rhs with
        | Tpat_constant { c; _ } -> Some c
        | Tpat_any _ -> None
        | _ -> assert false
      in
      match Db.eval_range ctx.db path lo hi ~inclusive with
      | For_sure_yes r -> ok_cont (set_db ctx r.ok_db) bindings
      | For_sure_no r -> fail_cont (set_db ctx r.fail_db)
      | Uncertain r ->
          let head = expr_of_path ctx path in
          let ifso = ok_cont (set_db ctx r.ok_db) bindings in
          let ifnot = fail_cont (set_db ctx r.fail_db) in
          make_range_if head lo hi ~inclusive ~ifso ~ifnot)
  | Tpat_tuple { pats; _ } ->
      let rec go ctx bindings pats index =
        match pats with
        | [] -> ok_cont ctx bindings
        | pat :: pats ->
            let this_path : Path.t = Field index :: path in
            let ty = Typedtree_util.type_of_pat pat in
            let rhs () =
              let head = expr_of_path ctx path in
              Core.field ~ty ~pos:index head
                (Index { tuple_index = index; loc_ = ghost_loc_ })
            in
            let body ctx =
              process_pat ~global_env ~ctx ~bindings ~path:this_path ~pat
                ~ok_cont:(fun ctx bindings -> go ctx bindings pats (index + 1))
                ~fail_cont
            in
            add_path_and_insert_let_if_necessary ctx this_path ~ty ~rhs ~body
      in
      go ctx bindings pats 0
  | Tpat_record { fields = field_pats; _ } ->
      let head = expr_of_path ctx path in
      let rec go ctx bindings fields =
        match fields with
        | [] -> ok_cont ctx bindings
        | Typedtree.Field_pat { label; pat; pos; _ } :: fields ->
            let this_path : Path.t = Field pos :: path in
            let ty = Typedtree_util.type_of_pat pat in
            let rhs () = Core.field ~pos ~ty head (Label label) in
            let body ctx =
              process_pat ~global_env ~ctx ~bindings ~path:this_path ~pat
                ~ok_cont:(fun ctx bindings -> go ctx bindings fields)
                ~fail_cont
            in
            add_path_and_insert_let_if_necessary ctx this_path ~ty ~rhs ~body
      in
      go ctx bindings field_pats
  | Tpat_constr { constr; tag; args; ty; used_error_subtyping } -> (
      let tag_index =
        match tag with
        | Constr_tag_regular { total = _; index } -> index
        | Extensible_tag { index; _ } -> index
      in
      let redirect_casted_path ctx =
        {
          ctx with
          path_cache =
            Path.Map.add ctx.path_cache
              (Casted_constr { tag_index } :: path)
              (Path.Map.find_exn ctx.path_cache path);
        }
          [@@inline]
      in
      match Global_env.get_newtype_info global_env ty with
      | Some { recursive; _ } -> (
          let ctx = redirect_casted_path ctx in
          match args with
          | [] -> ok_cont ctx bindings
          | _ :: _ :: _ -> assert false
          | Constr_pat_arg { pat; _ } :: _ ->
              let head = expr_of_path ctx path in
              let this_path : Path.t =
                match tag with
                | Constr_tag_regular { total = _; index } ->
                    Constr_field { tag_index = index; arg_index = 0 } :: path
                | Extensible_tag _ -> assert false
              in
              let ty = Typedtree_util.type_of_pat pat in
              let rhs () =
                if recursive then
                  Core.prim ~ty (Pcast { kind = Unfold_rec_newtype }) [ head ]
                else Core.field ~ty ~pos:0 head Newtype
              in
              let body ctx =
                process_pat ~global_env ~ctx ~bindings ~path:this_path ~pat
                  ~ok_cont ~fail_cont
              in
              add_path_and_insert_let_if_necessary ctx this_path ~ty ~rhs ~body)
      | None -> (
          let head = expr_of_path ctx path in
          let go ctx (args : Typedtree.constr_pat_arg list) ~must_destruct
              ~ifnot =
            match[@warning "-fragile-match"]
              Result.get_ok
                (Global_env.resolve_constr_or_constant global_env
                   ~expect_ty:(Some ty) ~constr ~creating_value:false)
            with
            | `Constr constr_desc -> (
                if constr_desc.cs_args = [] then
                  let ctx = redirect_casted_path ctx in
                  let ifso = ok_cont ctx bindings in
                  if must_destruct then
                    make_constr_switch tag None ~head ~ifso ~ifnot
                  else ifso
                else
                  let casted_path : Path.t =
                    Casted_constr { tag_index } :: path
                  in
                  let constr_binder, constr_value, ctx =
                    match Path.Map.find_opt ctx.path_cache casted_path with
                    | None ->
                        let b, v, ctx =
                          add_path
                            ~name_hint:("*" ^ constr.constr_name.name)
                            ctx casted_path
                            ~ty:(Type.make_constr_type ty ~tag)
                        in
                        (Some b, v, ctx)
                    | Some constr_value -> (None, constr_value, ctx)
                  in
                  let rec go ctx bindings (args : Typedtree.constr_pat_arg list)
                      =
                    match args with
                    | [] -> ok_cont ctx bindings
                    | Constr_pat_arg { pat; pos; _ } :: args ->
                        let path_item : Path.access =
                          if used_error_subtyping then
                            Error_constr_field { tag; arg_index = pos }
                          else Constr_field { tag_index; arg_index = pos }
                        in
                        let path = path_item :: path in
                        let arg_ty = Typedtree_util.type_of_pat pat in
                        add_path_and_insert_let_if_necessary ctx path ~ty:arg_ty
                          ~rhs:(fun () ->
                            constr_value.used <- true;
                            Core.prim ~ty:arg_ty
                              (Penum_field { index = pos; tag })
                              [ constr_value.expr ])
                          ~body:(fun ctx ->
                            process_pat ~global_env ~ctx ~bindings ~pat ~path
                              ~fail_cont ~ok_cont:(fun ctx bindings ->
                                go ctx bindings args))
                  in
                  let ifso = go ctx bindings args in
                  match constr_binder with
                  | Some _ when constr_value.used ->
                      make_constr_switch tag constr_binder ~head ~ifso ~ifnot
                  | _ when must_destruct ->
                      make_constr_switch tag None ~head ~ifso ~ifnot
                  | _ -> ifso)
            | _ -> assert false
          in
          let r = Db.eval_constructor ctx.db path tag ~used_error_subtyping in
          if Typeutil.is_only_tag_enum ty then
            let ok_cont ctx bindings =
              ok_cont (redirect_casted_path ctx) bindings
            in
            match r with
            | For_sure_yes r -> ok_cont (set_db ctx r.ok_db) bindings
            | For_sure_no r -> fail_cont (set_db ctx r.fail_db)
            | Uncertain r ->
                let head = expr_of_path ctx path in
                let ifso = ok_cont (set_db ctx r.ok_db) bindings in
                let ifnot = fail_cont (set_db ctx r.fail_db) in
                let tag =
                  match tag with
                  | Constr_tag_regular { index = tag; _ } -> tag
                  | Extensible_tag _ -> assert false
                in
                let c = Constant.C_int { v = Int32.of_int tag; repr = None } in
                make_constant_switch ~global_env c ~head ~ifso ~ifnot
          else
            match r with
            | For_sure_yes r ->
                go (set_db ctx r.ok_db) args ~must_destruct:false ~ifnot:None
            | For_sure_no r -> fail_cont (set_db ctx r.fail_db)
            | Uncertain r ->
                let ifnot = Some (fail_cont (set_db ctx r.fail_db)) in
                go (set_db ctx r.ok_db) args ~must_destruct:true ~ifnot))
  | Tpat_array { pats; _ } -> (
      let dotdot_binder, pats_with_path, num_pats =
        match pats with
        | Closed pats ->
            ( No_dotdot,
              Lst.mapi pats (fun i pat -> (pat, Path.Field i)),
              List.length pats )
        | Open (pats1, pats2, dotdot_binder) ->
            let len1 = List.length pats1 in
            let len2 = List.length pats2 in
            let pats2 =
              Lst.mapi pats2 (fun i pat ->
                  (pat, Path.Last_field (len2 - 1 - i)))
            in
            let dotdot_binder =
              match dotdot_binder with
              | Some (binder, binder_ty) ->
                  Dotdot_with_binder
                    {
                      binder;
                      len_pats1 = List.length pats1;
                      len_pats2 = List.length pats2;
                      binder_ty;
                    }
              | None -> Dotdot_no_binder
            in
            ( dotdot_binder,
              Lst.mapi_append pats1 (fun i pat -> (pat, Path.Field i)) pats2,
              len1 + len2 )
      in
      let head = expr_of_path ctx path in
      let rec go ctx bindings pats =
        match pats with
        | [] -> (
            match dotdot_binder with
            | Dotdot_no_binder | No_dotdot -> ok_cont ctx bindings
            | Dotdot_with_binder { binder = b; len_pats1; len_pats2; binder_ty }
              ->
                let rhs () =
                  Core_util.make_op_as_view global_env head len_pats1 len_pats2
                in
                let this_path = Path.Field_slice :: path in
                let body ctx =
                  ok_cont ctx
                    ((Ident.of_ident b.binder_id, this_path) :: bindings)
                in
                add_path_and_insert_let_if_necessary ctx this_path ~ty:binder_ty
                  ~rhs ~body)
        | (pat, field) :: pats ->
            let index, rev =
              match field with
              | Path.Field i -> (i, false)
              | Last_field i -> (i, true)
              | _ -> assert false
            in
            let this_path = field :: path in
            let ty = Typedtree_util.type_of_pat pat in
            let rhs () = Core_util.make_op_get global_env head index ~rev in
            let body ctx =
              process_pat ~global_env ~ctx ~bindings ~path:this_path ~pat
                ~ok_cont:(fun ctx bindings -> go ctx bindings pats)
                ~fail_cont
            in
            add_path_and_insert_let_if_necessary ctx this_path ~ty ~rhs ~body
      in
      let r =
        if dotdot_binder = No_dotdot then
          Db.eval_eq_array_len ctx.db path num_pats
        else Db.eval_geq_array_len ctx.db path num_pats
      in
      match r with
      | For_sure_yes r -> go (set_db ctx r.ok_db) bindings pats_with_path
      | For_sure_no r -> fail_cont (set_db ctx r.fail_db)
      | Uncertain r ->
          let ifso = go (set_db ctx r.ok_db) bindings pats_with_path in
          let ifnot = fail_cont (set_db ctx r.fail_db) in
          let arr_len_expr = Core_util.make_length global_env head in
          let num_pats_expr =
            Core.const (C_int { v = Int32.of_int num_pats; repr = None })
          in
          let test_expr =
            let prim =
              if dotdot_binder = No_dotdot then Primitive.equal_int
              else Primitive.ge_int
            in
            Core.prim prim [ arr_len_expr; num_pats_expr ] ~ty:Stype.bool
          in
          Core.if_ test_expr ~ifso ~ifnot)
  | Tpat_map
      { elems; op_get_info_ = op_get_id, op_get_ty, op_get_ty_args; ty = _ } ->
      let ty_op_get_result =
        match Stype.type_repr op_get_ty with
        | Tarrow { params_ty = _; ret_ty; err_ty = _ } -> ret_ty
        | _ -> assert false
      in
      let rec go ctx bindings elems =
        match elems with
        | [] -> ok_cont ctx bindings
        | (key, pat) :: elems ->
            let elem_path : Path.t = Map_elem { key } :: path in
            let rhs () =
              Core.apply ~ty:ty_op_get_result ~ty_args_:op_get_ty_args
                ~kind:(Normal { func_ty = op_get_ty })
                (Ident.of_ident op_get_id)
                [ expr_of_path ctx path; Core.const key ]
            in
            add_path_and_insert_let_if_necessary
              (set_db ctx
                 (Db.eval_map_elem ctx.db path key ~elem_ty:ty_op_get_result))
              elem_path ~ty:ty_op_get_result ~rhs
              ~body:(fun ctx ->
                process_pat ~global_env ~ctx ~bindings ~path:elem_path ~pat
                  ~ok_cont:(fun ctx bindings -> go ctx bindings elems)
                  ~fail_cont)
      in
      go ctx bindings elems

let transl_match ~(global_env : Global_env.t) (head : Core.expr)
    (cases : (Typedtree.pat * Typedtree.pat_binders * Core.expr) list)
    ~(ty : Stype.t) ~(loc : Rloc.t) : Core.expr =
  let arm_fns : (Ident.t * Core.param list * Core.expr) array =
    Arr.of_list_map cases (fun (_, pat_binders, action) ->
        let params =
          Lst.map pat_binders (fun { binder; binder_typ } : Core.param ->
              {
                binder = Ident.of_ident binder.binder_id;
                ty = binder_typ;
                loc_ = binder.loc_;
              })
        in
        (Ident.fresh "*arm", params, action))
  in
  let rec process_cases ctx
      (cases : (Typedtree.pat * Typedtree.pat_binders * Core.expr) list)
      (index : int) : Core.expr =
    match cases with
    | [] -> Core.raise ~ty "match_failure"
    | (pat, _, action) :: cases ->
        let ok_cont ctx bindings : Core.expr =
          let binding_table = Ident.Hash.of_list bindings in
          let arm_id, params, _ = arm_fns.(index) in
          let args =
            Lst.map params (fun { binder; ty = _ } ->
                let path = Ident.Hash.find_exn binding_table binder in
                expr_of_path ctx path)
          in
          let ty = Core.type_of_expr action in
          Core.join_apply arm_id args ~ty
        in
        let fail_cont ctx = process_cases ctx cases (index + 1) in
        process_pat ~global_env ~ctx ~bindings:[] ~path:[] ~pat ~ok_cont
          ~fail_cont
  in
  let main_code =
    match head with
    | Cexpr_tuple { exprs; ty; loc_ } ->
        let rec bind_exprs ctx index acc exprs =
          match exprs with
          | [] ->
              add_path_and_insert_let_if_necessary ctx [] ~ty
                ~rhs:(fun () -> Core.tuple ~loc:loc_ ~ty (List.rev acc))
                ~body:(fun ctx -> process_cases ctx cases 0)
          | expr0 :: exprs ->
              let ty0 = Core.type_of_expr expr0 in
              Core.bind expr0 (fun expr_id ->
                  let expr0 = Core.var ~ty:ty0 expr_id in
                  let ctx =
                    {
                      ctx with
                      path_cache =
                        Path.Map.add ctx.path_cache [ Field index ]
                          { expr = expr0; used = false };
                    }
                  in
                  bind_exprs ctx (index + 1) (expr0 :: acc) exprs)
        in
        bind_exprs { db = Db.empty; path_cache = Path.Map.empty } 0 [] exprs
    | _ ->
        let head_ty = Core.type_of_expr head in
        Core.bind ~loc head (fun head_id ->
            let expr = Core.var ~ty:head_ty head_id in
            let ctx =
              {
                db = Db.empty;
                path_cache = Path.Map.singleton [] { expr; used = false };
              }
            in
            process_cases ctx cases 0)
  in
  Arr.fold_left arm_fns main_code (fun body (id, params, join_body) ->
      Core.joinlet_tail id params join_body body)
  |> Core.update_expr_loc ~loc

let rec transl_let ~(global_env : Global_env.t) (pat : Typedtree.pat)
    ~(pat_binders : Typedtree.pat_binders) (rhs : Core.expr) (body : Core.expr)
    ~ty ~loc : Core.expr =
  match pat with
  | Tpat_var { binder; _ } ->
      Core.let_ ~loc (Ident.of_ident binder.binder_id) rhs body
  | Tpat_any _ -> Core.sequence ~loc rhs body
  | Tpat_constraint { pat; _ } ->
      transl_let ~global_env pat ~pat_binders rhs body ~loc ~ty
  | Tpat_tuple { pats; _ } -> (
      let exception Cannot_simplify in
      let rec pat_to_param (pat : Typedtree.pat) : Core.param =
        match pat with
        | Tpat_var { binder; ty; loc_ } ->
            let binder = Ident.of_ident binder.binder_id in
            { binder; ty; loc_ }
        | Tpat_constraint { pat; _ } -> pat_to_param pat
        | _ -> raise_notrace Cannot_simplify
      in
      match Lst.map pats pat_to_param with
      | exception Cannot_simplify ->
          transl_match ~global_env rhs [ (pat, pat_binders, body) ] ~loc ~ty
      | params ->
          let join = Ident.fresh "*join" in
          let ty_body = Core.type_of_expr body in
          let join_body = body in
          Core.tuple_map rhs
            ~ok:(fun body -> Core.joinlet_tail ~loc join params join_body body)
            ~err:(fun _ ->
              transl_match ~global_env rhs [ (pat, pat_binders, body) ] ~loc ~ty)
            ~join ~join_ty:ty_body)
  | _ -> transl_match ~global_env rhs [ (pat, pat_binders, body) ] ~loc ~ty
