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
module Tid = Basic_ty_ident
module Hash_tid = Basic_ty_ident.Hash

let size_of (t : Ltype.t) =
  match t with
  | F64 | I64 | U64 -> 8
  | F32 | I32_Int | U32 | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
  | I32_Option_Char | Ref_extern | Ref _ | Ref_nullable _ | Ref_lazy_init _
  | Ref_string | Ref_bytes | Ref_func | Ref_any ->
      4

let get_fields (tid : Tid.t) (type_defs : Ltype.type_defs) =
  match Hash_tid.find_exn type_defs tid with
  | Ref_struct { fields } -> Lst.map fields fst
  | Ref_late_init_struct { fields } -> fields
  | Ref_constructor { args } -> Lst.map args fst
  | Ref_closure { fn_sig_tid = _; captures } -> captures
  | Ref_array _ | Ref_closure_abstract _ | Ref_object _ -> assert false

let get_arr_elem (tid : Tid.t) (type_defs : Ltype.type_defs) : Ltype.t =
  if Tid.equal tid Ltype.tid_bytes then I32_Byte
  else
    match Hash_tid.find_exn type_defs tid with
    | Ref_array { elem } -> elem
    | Ref_struct _ | Ref_late_init_struct _ | Ref_closure _
    | Ref_closure_abstract _ | Ref_object _ | Ref_constructor _ ->
        assert false

let result_payload_type (ok_ty : Ltype.t) : Ltype.t =
  match ok_ty with F64 | I64 | U64 -> I64 | _ -> I32_Int

let convert_prim ~(from : Ltype.t) ~(to_ : Ltype.t) : Primitive.prim =
  let to_operand_ty (ty : Ltype.t) =
    match ty with
    | I64 -> Primitive.I64
    | U64 -> Primitive.U64
    | F64 -> Primitive.F64
    | F32 -> Primitive.F32
    | _ -> Primitive.I32
  in
  let from = to_operand_ty from in
  let to_ = to_operand_ty to_ in
  if from = to_ then Primitive.Pidentity
  else if
    from = F64 || to_ = F64 || from = F32 || to_ = F32 || from = U64
    || to_ = U64
  then Pconvert { kind = Reinterpret; from; to_ }
  else Pconvert { kind = Convert; from; to_ }

let is_gc_ref (ty : Ltype.t) =
  match ty with
  | I32_Int | U32 | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
  | I32_Option_Char | I64 | U64 | F32 | F64 | Ref_extern | Ref_func ->
      false
  | Ref _ | Ref_nullable _ | Ref_lazy_init _ | Ref_string | Ref_bytes | Ref_any
    ->
      true
