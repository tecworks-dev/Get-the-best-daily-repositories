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
module Ident = Basic_core_ident

let not_in (ident : Ident.t) (expr : Core.expr) =
  let rec go : Core.expr -> bool = function
    | Cexpr_break { arg; _ } -> (
        match arg with None -> true | Some arg -> go arg)
    | Cexpr_continue { args; _ } -> Lst.for_all args go
    | Cexpr_loop { body; args; _ } -> Lst.for_all args go && go body
    | Cexpr_const _ | Cexpr_unit _ -> true
    | Cexpr_var { id; _ } -> not (Ident.equal id ident)
    | Cexpr_as { expr; _ }
    | Cexpr_assign { expr; _ }
    | Cexpr_field { record = expr; _ } ->
        go expr
    | Cexpr_array { exprs = args; _ }
    | Cexpr_tuple { exprs = args; _ }
    | Cexpr_prim { args; _ } ->
        Lst.for_all args go
    | Cexpr_sequence { expr1; expr2; _ }
    | Cexpr_mutate { record = expr1; field = expr2; _ } ->
        go expr1 && go expr2
    | Cexpr_let { name; rhs; body; _ } ->
        Ident.equal name ident || (go rhs && go body)
    | Cexpr_function { func; _ } -> go func.body
    | Cexpr_apply { func; args; _ } ->
        (not (Ident.equal func ident)) && Lst.for_all args go
    | Cexpr_letrec { bindings; body; _ } ->
        Lst.for_all bindings (fun (binder, fn) ->
            Ident.equal binder ident || go fn.body)
        && go body
    | Cexpr_letfn { name; fn; body; _ } ->
        Ident.equal name ident || (go fn.body && go body)
    | Cexpr_constr { args; _ } -> Lst.for_all args go
    | Cexpr_record { fields; _ } -> Lst.for_all fields (fun def -> go def.expr)
    | Cexpr_record_update { record; fields; _ } ->
        go record && Lst.for_all fields (fun def -> go def.expr)
    | Cexpr_if { cond; ifso; ifnot; _ } -> (
        go cond && go ifso && match ifnot with None -> true | Some e -> go e)
    | Cexpr_switch_constr { obj; cases; default; _ } -> (
        go obj
        && Lst.for_all cases (fun (_, binder, expr) ->
               (match binder with
               | None -> false
               | Some binder -> Ident.equal binder ident)
               || go expr)
        && match default with None -> true | Some default -> go default)
    | Cexpr_switch_constant { obj; cases; default; _ } ->
        go obj && Lst.for_all cases (fun (_, expr) -> go expr) && go default
    | Cexpr_handle_error { obj; handle_kind = _; ty = _ } -> go obj
    | Cexpr_return { expr; _ } -> go expr
  in
  go expr

type contify_result =
  | Never_called
  | Contifiable
  | Not_contifiable of { tail_called : bool }

let is_contifiable (ident : Ident.t) (expr : Core.expr) =
  let has_other_occurence = ref false in
  let tail_called = ref false in
  let rec analyze_usage (expr : Core.expr) =
    let got expr =
      if (not !has_other_occurence) || not !tail_called then analyze_usage expr
        [@@inline]
    in
    let gon expr =
      if not !has_other_occurence then
        has_other_occurence := not (not_in ident expr)
        [@@inline]
    in
    match expr with
    | Cexpr_apply { func; args; _ } when Ident.equal func ident ->
        Lst.iter args gon;
        tail_called := true
    | Cexpr_let { name = _; rhs; body; _ } ->
        gon rhs;
        got body
    | Cexpr_letrec { bindings; body; _ } ->
        Lst.iter bindings (fun (_, fn) -> gon fn.body);
        got body
    | Cexpr_letfn { name = _; fn; body; kind; _ } -> (
        match kind with
        | Tail_join | Nontail_join ->
            got fn.body;
            got body
        | Nonrec | Rec ->
            gon fn.body;
            got body)
    | Cexpr_if { cond; ifso; ifnot; _ } -> (
        gon cond;
        got ifso;
        match ifnot with None -> () | Some e -> got e)
    | Cexpr_sequence { expr1; expr2; _ } ->
        gon expr1;
        got expr2
    | Cexpr_switch_constr { obj; cases; default; _ } -> (
        gon obj;
        Lst.iter cases (fun (_, _, expr) -> got expr);
        match default with None -> () | Some default -> got default)
    | Cexpr_switch_constant { obj; cases; default; _ } ->
        gon obj;
        Lst.iter cases (fun (_, expr) -> got expr);
        got default
    | Cexpr_prim { prim = Psequand | Psequor; args = [ expr1; expr2 ]; ty = _ }
      ->
        gon expr1;
        got expr2
    | Cexpr_loop { body; args; _ } ->
        Lst.iter args gon;
        got body
    | Cexpr_return _ | Cexpr_break _ | Cexpr_tuple _ | Cexpr_constr _
    | Cexpr_record _ | Cexpr_record_update _ | Cexpr_field _ | Cexpr_mutate _
    | Cexpr_array _ | Cexpr_assign _ | Cexpr_continue _ | Cexpr_const _
    | Cexpr_unit _ | Cexpr_var _ | Cexpr_function _ | Cexpr_prim _
    | Cexpr_apply _ | Cexpr_as _ | Cexpr_handle_error _ ->
        gon expr
  in
  analyze_usage expr;
  if !has_other_occurence then Not_contifiable { tail_called = !tail_called }
  else if !tail_called then Contifiable
  else Never_called

let apply2joinapply ident join body =
  let map =
    object (self)
      inherit [_] Core.Map.map as super

      method! visit_Cexpr_apply () func args kind ty ty_args_ prim loc_ =
        match kind with
        | Normal _ when Ident.equal func ident ->
            Core.apply ~loc:loc_ ~ty_args_ ~kind:Join ~ty ~prim join
              (List.map (self#visit_expr ()) args)
        | _ -> super#visit_Cexpr_apply () func args kind ty ty_args_ prim loc_
    end
  in
  map#visit_expr () body

let rec tail_apply2continue ident label (expr : Core.expr) =
  let go expr = tail_apply2continue ident label expr [@@inline] in
  match expr with
  | Cexpr_apply { func; args; ty; loc_; kind = _; prim = _ }
    when Ident.equal func ident ->
      Core.continue ~loc:loc_ args label ty
  | Cexpr_let { name; rhs; body; ty = _; loc_ } ->
      Core.let_ ~loc:loc_ name rhs (go body)
  | Cexpr_letrec { bindings; body; ty = _; loc_ } ->
      Core.letrec ~loc:loc_ bindings body
  | Cexpr_letfn { name; fn; body; kind; ty = _; loc_ } -> (
      match kind with
      | Tail_join | Nontail_join ->
          Core.letfn ~loc:loc_ ~kind name
            { fn with body = go fn.body }
            (go body)
      | Nonrec | Rec -> Core.letfn ~loc:loc_ ~kind name fn (go body))
  | Cexpr_if { cond; ifso; ifnot; ty = _; loc_ } ->
      Core.if_ ~loc:loc_ cond ~ifso:(go ifso) ?ifnot:(Option.map go ifnot)
  | Cexpr_sequence { expr1; expr2; ty = _; loc_ } ->
      Core.sequence ~loc:loc_ expr1 (go expr2)
  | Cexpr_switch_constr { obj; cases; default; ty = _; loc_ } ->
      Core.switch_constr ~loc:loc_ ~default:(Option.map go default) obj
        (Lst.map cases (fun (tag, binder, action) -> (tag, binder, go action)))
  | Cexpr_switch_constant { obj; cases; default; ty = _; loc_ } ->
      Core.switch_constant ~loc:loc_ ~default:(go default) obj
        (Lst.map cases (fun (c, action) -> (c, go action)))
  | Cexpr_prim
      { prim = (Psequand | Psequor) as prim; args = [ expr1; expr2 ]; ty; loc_ }
    ->
      Core.prim ~loc:loc_ ~ty prim [ expr1; go expr2 ]
  | Cexpr_loop { params; body; args; label; ty = _; loc_ } ->
      Core.loop ~loc:loc_ params (go body) args label
  | Cexpr_return _ | Cexpr_break _ | Cexpr_tuple _ | Cexpr_constr _
  | Cexpr_record _ | Cexpr_record_update _ | Cexpr_field _ | Cexpr_mutate _
  | Cexpr_array _ | Cexpr_assign _ | Cexpr_continue _ | Cexpr_const _
  | Cexpr_unit _ | Cexpr_var _ | Cexpr_function _ | Cexpr_prim _ | Cexpr_apply _
  | Cexpr_as _ | Cexpr_handle_error _ ->
      expr

let loopify (ident : Ident.t) (params : Core.param list) (body : Core.expr) :
    Core.fn =
  let label = Label.fresh (Ident.base_name ident) in
  let fresh_params =
    Lst.map params (fun p -> { p with binder = Ident.rename p.binder })
  in
  let args =
    Lst.map fresh_params (fun { binder = id; ty; loc_ } ->
        Core.var ~loc:loc_ ~ty id)
  in
  {
    params = fresh_params;
    body =
      Core.loop ~loc:(Core.loc_of_expr body) params
        (tail_apply2continue ident label body)
        args label;
  }

let contifier =
  (object (self)
     inherit [_] Core.Map.map as super

     method! visit_Ctop_fn () ({ binder; func; _ } as decl) =
       match is_contifiable binder func.body with
       | Contifiable | Not_contifiable { tail_called = true } ->
           super#visit_Ctop_fn ()
             { decl with func = loopify binder func.params func.body }
       | Never_called | Not_contifiable { tail_called = false } ->
           super#visit_Ctop_fn () decl

     method! visit_Cexpr_letfn () name fn body _ty kind loc_ =
       let fn = self#visit_fn () fn in
       let body = self#visit_expr () body in
       match kind with
       | Nonrec -> (
           match is_contifiable name body with
           | Contifiable ->
               let body = apply2joinapply name name body in
               Core.joinlet_tail ~loc:loc_ name fn.params
                 (Core_util.transform_return_in_fn_body fn.body)
                 body
           | Never_called -> body
           | Not_contifiable _ -> Core.letfn ~loc:loc_ ~kind name fn body)
       | Rec -> (
           match is_contifiable name fn.body with
           | Contifiable -> (
               match is_contifiable name body with
               | Contifiable ->
                   let fn =
                     loopify name fn.params
                       (Core_util.transform_return_in_fn_body fn.body)
                   in
                   let body = apply2joinapply name name body in
                   Core.letfn ~loc:loc_ ~kind:Tail_join name fn body
               | Never_called -> body
               | Not_contifiable { tail_called = true } ->
                   Core.letfn ~loc:loc_ ~kind:Rec name
                     (loopify name fn.params fn.body)
                     body
               | Not_contifiable { tail_called = false } ->
                   Core.letfn ~loc:loc_ ~kind:Rec name fn body)
           | Not_contifiable { tail_called = true } ->
               Core.letfn ~loc:loc_ ~kind:Rec name
                 (loopify name fn.params fn.body)
                 body
           | Never_called -> Core.letfn ~loc:loc_ ~kind:Nonrec name fn body
           | Not_contifiable { tail_called = false } ->
               Core.letfn ~loc:loc_ ~kind name fn body)
       | Tail_join | Nontail_join -> Core.letfn ~loc:loc_ ~kind name fn body
   end
    : < visit_top_item : unit -> Core.top_item -> Core.top_item ; .. >)

let contify (prog : Core.program) =
  Lst.map prog (fun top_item -> contifier#visit_top_item () top_item)
