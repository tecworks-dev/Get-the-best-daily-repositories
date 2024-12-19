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


module Type_path = Basic_type_path

type control_info =
  | Not_in_loop
  | Ambiguous_position
  | In_while
  | In_while_else of { break : Stype.t }
  | In_loop of { break : Stype.t; continue : Stype.t list }
  | In_for of { break : Stype.t option; continue : Stype.t list }
  | In_foreach of { break : Stype.t option }

type open_error_ctx =
  | Empty_ctx
  | Tparam of { index : int; name_ : string }
  | Suberrors of Basic_type_path.t list

type fixed_error_ctx =
  | Supererror
  | Tparam of { index : int; name_ : string }
  | Suberror of Basic_type_path.t

type error_ctx = Fixed_ctx of fixed_error_ctx | Open_ctx of open_error_ctx

let error_ctx_to_stype ctx : Stype.t =
  match ctx with
  | Open_ctx Empty_ctx -> Stype.new_type_var Stype.Tvar_error
  | Open_ctx (Suberrors es) -> (
      match es with
      | [] -> assert false
      | p :: [] ->
          T_constr
            {
              type_constructor = p;
              tys = [];
              generic_ = false;
              only_tag_enum_ = false;
              is_suberror_ = true;
            }
      | _ -> Stype.error)
  | Open_ctx (Tparam { index; name_ }) | Fixed_ctx (Tparam { index; name_ }) ->
      Stype.Tparam { index; name_ }
  | Fixed_ctx Supererror -> Stype.error
  | Fixed_ctx (Suberror p) ->
      T_constr
        {
          type_constructor = p;
          tys = [];
          generic_ = false;
          only_tag_enum_ = false;
          is_suberror_ = false;
        }

type t = {
  return : Stype.t option;
  error_ctx : error_ctx ref option;
  control_info : control_info;
  has_continue : bool ref;
  has_error : bool ref;
}

let empty =
  {
    return = None;
    error_ctx = None;
    control_info = Not_in_loop;
    has_continue = ref false;
    has_error = ref false;
  }

let make_fn ~return ~error_ctx =
  {
    return = Some return;
    error_ctx;
    control_info = Not_in_loop;
    has_continue = ref false;
    has_error = ref false;
  }

let with_while ~break_typ parent =
  {
    return = parent.return;
    error_ctx = parent.error_ctx;
    control_info =
      (match break_typ with
      | None -> In_while
      | Some break_typ -> In_while_else { break = break_typ });
    has_continue = ref false;
    has_error = parent.has_error;
  }

let with_loop ~arg_typs ~result_typ parent =
  {
    return = parent.return;
    error_ctx = parent.error_ctx;
    control_info = In_loop { break = result_typ; continue = arg_typs };
    has_continue = ref false;
    has_error = parent.has_error;
  }

let with_for ~break_typ ~arg_typs parent =
  {
    return = parent.return;
    error_ctx = parent.error_ctx;
    control_info = In_for { continue = arg_typs; break = break_typ };
    has_continue = ref false;
    has_error = parent.has_error;
  }

let with_foreach ~break_typ parent =
  {
    return = parent.return;
    error_ctx = parent.error_ctx;
    has_error = parent.has_error;
    control_info = In_foreach { break = break_typ };
    has_continue = ref false;
  }

let with_ambiguous_position parent =
  {
    return = parent.return;
    error_ctx = parent.error_ctx;
    control_info = Ambiguous_position;
    has_continue = ref false;
    has_error = parent.has_error;
  }

let with_error_ctx ~error_ctx parent =
  { parent with error_ctx = Some error_ctx; has_error = ref false }

let check_error_in_ctx ~error_ty ~(ctx : error_ctx ref) loc :
    Local_diagnostics.error_option =
  let is_tvar (typ : Stype.t) : bool =
    let typ = Stype.type_repr typ in
    match typ with Tvar _ -> true | _ -> false
      [@@local]
  in
  let error_ty = Stype.type_repr error_ty in
  let make_error expect_ty =
    let expected_ty, error_ty =
      Printer.type_pair_to_string expect_ty error_ty
    in
    Some (Errors.error_type_mismatch ~expected_ty ~actual_ty:error_ty ~loc)
      [@@local]
  in
  if is_tvar error_ty then Ctype.unify_exn error_ty Stype.error;
  match !ctx with
  | Fixed_ctx expect_ty -> (
      match expect_ty with
      | Supererror -> None
      | Tparam { index; name_ } -> (
          match error_ty with
          | Tparam { index = index' } when index = index' -> None
          | _ ->
              let expect_ty : Stype.t = Tparam { index; name_ } in
              make_error expect_ty)
      | Suberror p -> (
          match error_ty with
          | T_constr { type_constructor = p'; _ } when Type_path.equal p p' ->
              None
          | _ ->
              let expect_ty : Stype.t =
                T_constr
                  {
                    type_constructor = p;
                    tys = [];
                    generic_ = false;
                    only_tag_enum_ = false;
                    is_suberror_ = true;
                  }
              in
              make_error expect_ty))
  | Open_ctx open_ctx -> (
      match open_ctx with
      | Empty_ctx -> (
          match error_ty with
          | T_constr { type_constructor = p; is_suberror_ = true; _ } ->
              ctx := Open_ctx (Suberrors [ p ]);
              None
          | Tparam { index; name_ } ->
              ctx := Open_ctx (Tparam { index; name_ });
              None
          | T_constr { type_constructor = Type_path.T_error; _ } ->
              ctx := Fixed_ctx Supererror;
              None
          | _ -> None)
      | Tparam { index } -> (
          match error_ty with
          | Tparam { index = index' } ->
              if index <> index' then ctx := Fixed_ctx Supererror;
              None
          | T_constr { type_constructor = _; is_suberror_ = true; _ }
          | T_constr { type_constructor = Type_path.T_error; _ } ->
              ctx := Fixed_ctx Supererror;
              None
          | _ -> None)
      | Suberrors ps -> (
          match error_ty with
          | T_constr { type_constructor = p; is_suberror_ = true; _ } ->
              if not (Basic_lst.exists ps (Type_path.equal p)) then
                ctx := Open_ctx (Suberrors (p :: ps));
              None
          | Tparam _ | T_constr { type_constructor = Type_path.T_error; _ } ->
              ctx := Fixed_ctx Supererror;
              None
          | _ -> None))
