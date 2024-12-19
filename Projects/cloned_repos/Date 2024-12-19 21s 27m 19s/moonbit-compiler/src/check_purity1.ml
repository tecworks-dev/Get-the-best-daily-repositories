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


let rec check (lam : Clam1.lambda) : bool =
  match lam with
  | Levent { expr; loc_ = _ } | Lcast { expr; target_type = _ } -> check expr
  | Lvar _ | Lconst _ | Lallocate _ | Lclosure _ | Lmake_multi_result _
  | Lget_field _ | Lclosure_field _ | Lmake_array _ | Larray_get_item _ ->
      true
  | Lloop _ | Lapply _ | Lstub_call _ | Ljoinlet _ | Ljoinapply _ | Lbreak _
  | Lcontinue _ | Lreturn _ | Lcatch _ | Lset_field _ | Larray_set_item _
  | Lassign _ ->
      false
  | Lif { pred; ifso; ifnot } -> check pred && check ifso && check ifnot
  | Lhandle_error { obj; ok_branch = _, ok_branch; err_branch = _, err_branch }
    ->
      check obj && check ok_branch && check err_branch
  | Llet { name = _; e; body } | Llet_multi { names = _; e; body } ->
      check e && check body
  | Lletrec { body; _ } -> check body
  | Lprim { fn; args = _ } -> Primitive.is_pure fn
  | Lsequence { expr1; expr2 } -> check expr1 && check expr2
  | Lswitch { obj = _; cases; default } ->
      List.for_all (fun (_, arm) -> check arm) cases && check default
  | Lswitchstring { obj = _; cases; default } ->
      List.for_all (fun (_, arm) -> check arm) cases && check default
  | Lswitchint { obj = _; cases; default } ->
      List.for_all (fun (_, arm) -> check arm) cases && check default
