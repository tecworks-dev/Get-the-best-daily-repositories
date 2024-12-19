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


module Ltype = Ltype_gc
module Tid = Basic_ty_ident
module Hash_tid = Basic_ty_ident.Hash

let get_arr_elem (tid : Tid.t) (type_defs : Ltype.type_defs) : Ltype.t =
  if Tid.equal tid Ltype.tid_bytes then I32_Byte
  else
    match Hash_tid.find_exn type_defs tid with
    | Ref_array { elem } -> elem
    | Ref_struct _ | Ref_late_init_struct _ | Ref_closure _
    | Ref_closure_abstract _ | Ref_object _ | Ref_constructor _ ->
        assert false

let is_non_nullable_ref_type (ty : Ltype.t) =
  match ty with
  | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
  | I32_Option_Char | I64 | F32 | F64 ->
      false
  | Ref_nullable _ -> false
  | Ref_lazy_init _ | Ref _ | Ref_extern | Ref_string | Ref_bytes | Ref_func
  | Ref_any ->
      true
