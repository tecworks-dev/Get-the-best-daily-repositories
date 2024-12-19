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

type scope = int

type analyze_ctx = {
  depth_of_vars : scope Ident.Hash.t;
  loop_info : Label.Hashset.t;
  need_transform : bool ref;
  depth : int;
}

type transform_ctx = {
  depth_of_vars : scope Ident.Hash.t;
  fields_of_vars : (Ident.t * Stype.t) array Ident.Hash.t;
  loop_info : Label.Hashset.t;
}

let analyze_stack_vars =
  object (self)
    inherit [_] Core.Iter.iter as super

    method! visit_fn ctx fn =
      super#visit_fn { ctx with depth = ctx.depth + 1 } fn

    method! visit_Cexpr_let ctx name rhs body ty loc_ =
      match rhs with
      | Cexpr_record _ ->
          Ident.Hash.add ctx.depth_of_vars name ctx.depth;
          super#visit_Cexpr_let ctx name rhs body ty loc_;
          ctx.need_transform :=
            !(ctx.need_transform) || Ident.Hash.mem ctx.depth_of_vars name
      | _ -> super#visit_Cexpr_let ctx name rhs body ty loc_

    method! visit_Cexpr_field ctx record accessor pos ty loc =
      match record with
      | Cexpr_var { id; _ } -> (
          match Ident.Hash.find_opt ctx.depth_of_vars id with
          | Some depth' when depth' < ctx.depth ->
              Ident.Hash.remove ctx.depth_of_vars id
          | _ -> ())
      | _ -> super#visit_Cexpr_field ctx record accessor pos ty loc

    method! visit_Cexpr_mutate ctx record label (field : Core.expr) pos ty loc =
      match record with
      | Cexpr_var { id; _ } -> (
          self#visit_expr ctx field;
          match Ident.Hash.find_opt ctx.depth_of_vars id with
          | Some depth' when depth' < ctx.depth ->
              Ident.Hash.remove ctx.depth_of_vars id
          | _ -> ())
      | _ -> super#visit_Cexpr_mutate ctx record label field pos ty loc

    method! visit_var ctx var = Ident.Hash.remove ctx.depth_of_vars var

    method! visit_Cexpr_loop ctx params body args label ty loc_ =
      match params with
      | { ty = T_constr { type_constructor = Tuple _; _ }; binder } :: [] ->
          Label.Hashset.add ctx.loop_info label;
          Ident.Hash.add ctx.depth_of_vars binder Int.max_int;
          super#visit_Cexpr_loop ctx params body args label ty loc_;
          if
            Label.Hashset.mem ctx.loop_info label
            && Ident.Hash.mem ctx.depth_of_vars binder
          then ctx.need_transform := true
          else Label.Hashset.remove ctx.loop_info label
      | _ -> super#visit_Cexpr_loop ctx params body args label ty loc_

    method! visit_Cexpr_continue ctx args label ty loc_ =
      (if Label.Hashset.mem ctx.loop_info label then
         match args with
         | Cexpr_tuple _ :: [] -> ()
         | _ -> Label.Hashset.remove ctx.loop_info label);
      super#visit_Cexpr_continue ctx args label ty loc_
  end

let transform =
  object (self)
    inherit [_] Core.Map.map as super

    method! visit_Cexpr_let ctx name rhs body ty loc =
      match rhs with
      | Cexpr_record { fields; _ } when Ident.Hash.mem ctx.depth_of_vars name ->
          let base_name = Ident.base_name name in
          let field_vars = Array.make (List.length fields) (name, Stype.unit) in
          let new_vars =
            Lst.map fields (fun { is_mut; pos; expr; label } ->
                let name =
                  if Array.length field_vars = 1 then base_name
                  else (base_name ^ "_" ^ label.label_name : Stdlib.String.t)
                in
                let id =
                  if is_mut then Ident.fresh_mut name else Ident.fresh name
                in
                field_vars.(pos) <- (id, Core.type_of_expr expr);
                id)
          in
          Ident.Hash.add ctx.fields_of_vars name field_vars;
          let new_body = self#visit_expr ctx body in
          Lst.fold_right2 new_vars fields new_body (fun var { expr; _ } body ->
              let expr = self#visit_expr ctx expr in
              Core.let_ var expr body)
      | _ -> super#visit_Cexpr_let ctx name rhs body ty loc

    method! visit_Cexpr_field ctx record accessor pos ty loc =
      match record with
      | Cexpr_var { id; _ } -> (
          match Ident.Hash.find_opt ctx.fields_of_vars id with
          | Some fields ->
              let id, ty = fields.(pos) in
              Core.var ~loc ~ty id
          | None -> super#visit_Cexpr_field ctx record accessor pos ty loc)
      | _ -> super#visit_Cexpr_field ctx record accessor pos ty loc

    method! visit_Cexpr_mutate ctx record label field pos ty loc =
      match record with
      | Cexpr_var { id; _ } -> (
          match Ident.Hash.find_opt ctx.fields_of_vars id with
          | Some fields ->
              Core.assign ~loc (fst fields.(pos)) (self#visit_expr ctx field)
          | None -> super#visit_Cexpr_mutate ctx record label field pos ty loc)
      | _ -> super#visit_Cexpr_mutate ctx record label field pos ty loc

    method! visit_Cexpr_loop ctx params body args label ty loc_ =
      match params with
      | {
          ty = T_constr { type_constructor = Tuple n; tys } as ty_tuple;
          binder;
        }
        :: [] ->
          if
            Label.Hashset.mem ctx.loop_info label
            && Ident.Hash.mem ctx.depth_of_vars binder
          then (
            let base_name = Ident.base_name binder in
            let field_vars = Array.make n (binder, Stype.unit) in
            let new_params =
              Lst.mapi tys (fun i ty : Core.param ->
                  let name =
                    (base_name ^ "_" ^ Int.to_string i : Stdlib.String.t)
                  in
                  let binder = Ident.fresh name in
                  field_vars.(i) <- (binder, ty);
                  { binder; ty; loc_ = Rloc.no_location })
            in
            Ident.Hash.add ctx.fields_of_vars binder field_vars;
            match args with
            | Cexpr_tuple { exprs; _ } :: [] ->
                let args = Lst.map exprs (self#visit_expr ctx) in
                Core.loop new_params (self#visit_expr ctx body) args label
                  ~loc:loc_
            | arg :: [] ->
                Core.bind (self#visit_expr ctx arg) (fun id ->
                    let args =
                      Lst.mapi tys (fun i ty ->
                          let var : Core.expr = Core.var id ~ty:ty_tuple in
                          let index : Parsing_syntax.accessor =
                            Index { tuple_index = i; loc_ = Rloc.no_location }
                          in
                          Core.field var ~ty ~pos:i index)
                    in
                    Core.loop new_params (self#visit_expr ctx body) args label
                      ~loc:loc_)
            | _ -> assert false)
          else super#visit_Cexpr_loop ctx params body args label ty loc_
      | _ -> super#visit_Cexpr_loop ctx params body args label ty loc_

    method! visit_Cexpr_continue ctx args label ty loc_ =
      if Label.Hashset.mem ctx.loop_info label then
        match args with
        | Cexpr_tuple { exprs; _ } :: [] ->
            let new_args = Lst.map exprs (self#visit_expr ctx) in
            Core.continue new_args label ty
        | _ -> super#visit_Cexpr_continue ctx args label ty loc_
      else super#visit_Cexpr_continue ctx args label ty loc_
  end

let unbox_mut_records (prog : Core.program) =
  Lst.map prog (fun top ->
      let analyze_ctx =
        {
          depth_of_vars = Ident.Hash.create 16;
          loop_info = Label.Hashset.create 16;
          need_transform = ref false;
          depth = 0;
        }
      in
      analyze_stack_vars#visit_top_item analyze_ctx top;
      if !(analyze_ctx.need_transform) then
        let transform_ctx =
          {
            depth_of_vars = analyze_ctx.depth_of_vars;
            fields_of_vars = Ident.Hash.create 16;
            loop_info = analyze_ctx.loop_info;
          }
        in
        transform#visit_top_item transform_ctx top
      else top)
