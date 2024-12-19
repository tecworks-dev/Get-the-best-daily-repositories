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


module Itype = Dwarfsm_itype

type var = { var_name : string option; mutable index : int }
type index = var
type binder = { id : string option; mutable index : int }
type label = binder
type typeidx = index
type funcidx = index
type tableidx = index
type memidx = index
type globalidx = index
type elemidx = index
type dataidx = index
type localidx = index
type labelidx = index
type fieldidx = index
type tagidx = index
type numtype = I32 | I64 | F32 | F64
type vectype = V128

type absheaptype =
  | Any
  | Eq
  | I31
  | Struct
  | Array
  | None
  | Func
  | NoFunc
  | Extern
  | NoExtern

type heaptype = Type of typeidx | Absheaptype of absheaptype
type null = Nullable | NonNull
type reftype = Ref of null * heaptype
type valtype = Numtype of numtype | Vectype of vectype | Reftype of reftype

type local = {
  id : binder;
  source_name : string option;
  type_ : valtype;
  source_type : Itype.t option;
}

type param = local
type result = valtype
type functype = Func of param list * result list
type packedtype = I8 | I16
type storagetype = Valtype of valtype | Packedtype of packedtype
type mut = Const | Var
type fieldtype = { mut : mut; type_ : storagetype }
type field = binder * fieldtype
type arraytype = Array of fieldtype
type structtype = Struct of field list

type comptype =
  | Arraytype of arraytype
  | Structtype of structtype
  | Functype of functype

type subtype = { final : bool; super : typeidx list; type_ : comptype }
type typedef = binder * subtype
type rectype = typedef list
type limits = { min : int32; max : int32 option }
type memtype = { limits : limits }
type tabletype = { limits : limits; element_type : reftype }
type globaltype = { mut : mut; type_ : valtype }

type typeuse =
  | Use of typeidx * param list * valtype list
  | Inline of param list * valtype list

type importdesc =
  | Func of binder * typeuse
  | Table of binder * tabletype
  | Memory of binder * memtype
  | Global of binder * globaltype
  | Tag of binder * typeuse

type import = { module_ : string; name : string; desc : importdesc }
type memarg = { align : int32; offset : int32 }
type source_pos = { pkg : string; file : string; line : int; col : int }
type catch = Catch of tagidx * labelidx

class ['a] mapbase =
  object
    method visit_typeidx : 'a -> typeidx -> typeidx = fun _ e -> e
    method visit_dataidx : 'a -> dataidx -> dataidx = fun _ e -> e
    method visit_labelidx : 'a -> labelidx -> labelidx = fun _ e -> e
    method visit_funcidx : 'a -> funcidx -> funcidx = fun _ e -> e
    method visit_tableidx : 'a -> tableidx -> tableidx = fun _ e -> e
    method visit_globalidx : 'a -> globalidx -> globalidx = fun _ e -> e
    method visit_localidx : 'a -> localidx -> localidx = fun _ e -> e
    method visit_fieldidx : 'a -> fieldidx -> fieldidx = fun _ e -> e
    method visit_tagidx : 'a -> tagidx -> tagidx = fun _ e -> e
    method visit_memarg : 'a -> memarg -> memarg = fun _ e -> e
    method visit_label : 'a -> label -> label = fun _ e -> e
    method visit_reftype : 'a -> reftype -> reftype = fun _ e -> e
    method visit_heaptype : 'a -> heaptype -> heaptype = fun _ e -> e
    method visit_typeuse : 'a -> typeuse -> typeuse = fun _ e -> e
    method visit_catch : 'a -> catch -> catch = fun _ e -> e
  end

type instr =
  | Any_convert_extern
  | Array_copy of typeidx * typeidx
  | Array_fill of typeidx
  | Array_get of typeidx
  | Array_get_u of typeidx
  | Array_len
  | Array_new of typeidx
  | Array_new_data of typeidx * dataidx
  | Array_new_default of typeidx
  | Array_new_fixed of typeidx * int32
  | Array_set of typeidx
  | Block of label * typeuse * instr list
  | Br of labelidx
  | Br_if of labelidx
  | Br_table of labelidx list * labelidx
  | Call of funcidx
  | Call_indirect of tableidx * typeuse
  | Call_ref of typeidx
  | Drop
  | Extern_convert_any
  | F64_add
  | F64_const of string * float
  | F64_convert_i32_s
  | F64_convert_i32_u
  | F64_convert_i64_s
  | F64_convert_i64_u
  | F64_div
  | F64_eq
  | F64_ge
  | F64_gt
  | F64_le
  | F64_load of memarg
  | F64_lt
  | F64_mul
  | F64_ne
  | F64_neg
  | F64_reinterpret_i64
  | F64_store of memarg
  | F64_sub
  | F64_sqrt
  | F64_abs
  | F64_trunc
  | F64_floor
  | F64_ceil
  | F64_nearest
  | F32_add
  | F32_const of string * float
  | F32_convert_i32_s
  | F32_convert_i32_u
  | F32_convert_i64_s
  | F32_convert_i64_u
  | F32_demote_f64
  | F32_div
  | F32_eq
  | F32_ge
  | F32_gt
  | F32_le
  | F32_load of memarg
  | F32_lt
  | F32_mul
  | F32_ne
  | F32_neg
  | F32_reinterpret_i32
  | F32_sqrt
  | F32_store of memarg
  | F32_sub
  | F32_abs
  | F32_trunc
  | F32_floor
  | F32_ceil
  | F32_nearest
  | F64_promote_f32
  | I32_reinterpret_f32
  | I32_trunc_f32_s
  | I32_trunc_f32_u
  | I64_trunc_f32_s
  | I64_trunc_f32_u
  | Global_get of globalidx
  | Global_set of globalidx
  | I32_add
  | I32_and
  | I32_clz
  | I32_const of int32
  | I32_ctz
  | I32_div_s
  | I32_div_u
  | I32_eq
  | I32_eqz
  | I32_ge_s
  | I32_ge_u
  | I32_gt_s
  | I32_gt_u
  | I32_le_s
  | I32_le_u
  | I32_load of memarg
  | I32_load16_u of memarg
  | I32_load16_s of memarg
  | I32_load8_u of memarg
  | I32_load8_s of memarg
  | I32_lt_s
  | I32_lt_u
  | I32_mul
  | I32_ne
  | I32_or
  | I32_popcnt
  | I32_rem_s
  | I32_rem_u
  | I32_shl
  | I32_shr_s
  | I32_shr_u
  | I32_rotl
  | I32_store of memarg
  | I32_store16 of memarg
  | I32_store8 of memarg
  | I32_sub
  | I32_trunc_f64_s
  | I32_trunc_f64_u
  | I32_wrap_i64
  | I32_xor
  | I32_extend_8_s
  | I32_extend_16_s
  | I32_trunc_sat_f32_s
  | I32_trunc_sat_f32_u
  | I32_trunc_sat_f64_s
  | I32_trunc_sat_f64_u
  | I64_add
  | I64_and
  | I64_clz
  | I64_const of int64
  | I64_ctz
  | I64_div_s
  | I64_div_u
  | I64_eq
  | I64_extend_i32_s
  | I64_extend_i32_u
  | I64_ge_s
  | I64_gt_s
  | I64_le_s
  | I64_ge_u
  | I64_gt_u
  | I64_le_u
  | I64_load of memarg
  | I64_load32_u of memarg
  | I64_load32_s of memarg
  | I64_load16_u of memarg
  | I64_load16_s of memarg
  | I64_load8_u of memarg
  | I64_load8_s of memarg
  | I64_lt_s
  | I64_lt_u
  | I64_mul
  | I64_ne
  | I64_or
  | I64_popcnt
  | I64_reinterpret_f64
  | I64_rem_s
  | I64_rem_u
  | I64_shl
  | I64_shr_s
  | I64_shr_u
  | I64_store of memarg
  | I64_store32 of memarg
  | I64_store16 of memarg
  | I64_store8 of memarg
  | I64_sub
  | I64_trunc_f64_s
  | I64_trunc_f64_u
  | I64_xor
  | I64_extend_8_s
  | I64_extend_16_s
  | I64_extend_32_s
  | I64_trunc_sat_f32_s
  | I64_trunc_sat_f32_u
  | I64_trunc_sat_f64_s
  | I64_trunc_sat_f64_u
  | V128_load of memarg
  | V128_store of memarg
  | F64x2_add
  | F64x2_mul
  | F32x4_add
  | F32x4_mul
  | If of label * typeuse * instr list * instr list
  | Local_get of localidx
  | Local_set of localidx
  | Local_tee of localidx
  | Loop of label * typeuse * instr list
  | Memory_init of typeidx
  | Memory_copy
  | Memory_grow
  | Memory_size
  | Memory_fill
  | Ref_eq
  | Ref_as_non_null
  | Ref_cast of reftype
  | Ref_func of funcidx
  | Ref_is_null
  | Ref_null of heaptype
  | Return
  | Struct_get of typeidx * fieldidx
  | Struct_new of typeidx
  | Struct_new_default of typeidx
  | Struct_set of typeidx * fieldidx
  | Table_get of tableidx
  | Unreachable
  | Throw of tagidx
  | Try_table of label * typeuse * catch list * instr list
  | Select
  | No_op
  | Source_pos of (source_pos[@visitors.opaque])
  | Prologue_end

include struct
  [@@@ocaml.warning "-4-26-27"]
  [@@@VISITORS.BEGIN]

  class virtual ['self] map =
    object (self : 'self)
      inherit [_] mapbase

      method visit_Any_convert_extern : _ -> instr =
        fun env -> Any_convert_extern

      method visit_Array_copy : _ -> typeidx -> typeidx -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          let _visitors_r1 = self#visit_typeidx env _visitors_c1 in
          Array_copy (_visitors_r0, _visitors_r1)

      method visit_Array_fill : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_fill _visitors_r0

      method visit_Array_get : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_get _visitors_r0

      method visit_Array_get_u : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_get_u _visitors_r0

      method visit_Array_len : _ -> instr = fun env -> Array_len

      method visit_Array_new : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_new _visitors_r0

      method visit_Array_new_data : _ -> typeidx -> dataidx -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          let _visitors_r1 = self#visit_dataidx env _visitors_c1 in
          Array_new_data (_visitors_r0, _visitors_r1)

      method visit_Array_new_default : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_new_default _visitors_r0

      method visit_Array_new_fixed : _ -> typeidx -> int32 -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          let _visitors_r1 =
            (fun _visitors_this -> _visitors_this) _visitors_c1
          in
          Array_new_fixed (_visitors_r0, _visitors_r1)

      method visit_Array_set : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Array_set _visitors_r0

      method visit_Block : _ -> label -> typeuse -> instr list -> instr =
        fun env _visitors_c0 _visitors_c1 _visitors_c2 ->
          let _visitors_r0 = self#visit_label env _visitors_c0 in
          let _visitors_r1 = self#visit_typeuse env _visitors_c1 in
          let _visitors_r2 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_instr env))
              _visitors_c2
          in
          Block (_visitors_r0, _visitors_r1, _visitors_r2)

      method visit_Br : _ -> labelidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_labelidx env _visitors_c0 in
          Br _visitors_r0

      method visit_Br_if : _ -> labelidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_labelidx env _visitors_c0 in
          Br_if _visitors_r0

      method visit_Br_table : _ -> labelidx list -> labelidx -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_labelidx env))
              _visitors_c0
          in
          let _visitors_r1 = self#visit_labelidx env _visitors_c1 in
          Br_table (_visitors_r0, _visitors_r1)

      method visit_Call : _ -> funcidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_funcidx env _visitors_c0 in
          Call _visitors_r0

      method visit_Call_indirect : _ -> tableidx -> typeuse -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_tableidx env _visitors_c0 in
          let _visitors_r1 = self#visit_typeuse env _visitors_c1 in
          Call_indirect (_visitors_r0, _visitors_r1)

      method visit_Call_ref : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Call_ref _visitors_r0

      method visit_Drop : _ -> instr = fun env -> Drop

      method visit_Extern_convert_any : _ -> instr =
        fun env -> Extern_convert_any

      method visit_F64_add : _ -> instr = fun env -> F64_add

      method visit_F64_const : _ -> string -> float -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 =
            (fun _visitors_this -> _visitors_this) _visitors_c0
          in
          let _visitors_r1 =
            (fun _visitors_this -> _visitors_this) _visitors_c1
          in
          F64_const (_visitors_r0, _visitors_r1)

      method visit_F64_convert_i32_s : _ -> instr = fun env -> F64_convert_i32_s
      method visit_F64_convert_i32_u : _ -> instr = fun env -> F64_convert_i32_u
      method visit_F64_convert_i64_s : _ -> instr = fun env -> F64_convert_i64_s
      method visit_F64_convert_i64_u : _ -> instr = fun env -> F64_convert_i64_u
      method visit_F64_div : _ -> instr = fun env -> F64_div
      method visit_F64_eq : _ -> instr = fun env -> F64_eq
      method visit_F64_ge : _ -> instr = fun env -> F64_ge
      method visit_F64_gt : _ -> instr = fun env -> F64_gt
      method visit_F64_le : _ -> instr = fun env -> F64_le

      method visit_F64_load : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          F64_load _visitors_r0

      method visit_F64_lt : _ -> instr = fun env -> F64_lt
      method visit_F64_mul : _ -> instr = fun env -> F64_mul
      method visit_F64_ne : _ -> instr = fun env -> F64_ne
      method visit_F64_neg : _ -> instr = fun env -> F64_neg

      method visit_F64_reinterpret_i64 : _ -> instr =
        fun env -> F64_reinterpret_i64

      method visit_F64_store : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          F64_store _visitors_r0

      method visit_F64_sub : _ -> instr = fun env -> F64_sub
      method visit_F64_sqrt : _ -> instr = fun env -> F64_sqrt
      method visit_F64_abs : _ -> instr = fun env -> F64_abs
      method visit_F64_trunc : _ -> instr = fun env -> F64_trunc
      method visit_F64_floor : _ -> instr = fun env -> F64_floor
      method visit_F64_ceil : _ -> instr = fun env -> F64_ceil
      method visit_F64_nearest : _ -> instr = fun env -> F64_nearest
      method visit_F32_add : _ -> instr = fun env -> F32_add

      method visit_F32_const : _ -> string -> float -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 =
            (fun _visitors_this -> _visitors_this) _visitors_c0
          in
          let _visitors_r1 =
            (fun _visitors_this -> _visitors_this) _visitors_c1
          in
          F32_const (_visitors_r0, _visitors_r1)

      method visit_F32_convert_i32_s : _ -> instr = fun env -> F32_convert_i32_s
      method visit_F32_convert_i32_u : _ -> instr = fun env -> F32_convert_i32_u
      method visit_F32_convert_i64_s : _ -> instr = fun env -> F32_convert_i64_s
      method visit_F32_convert_i64_u : _ -> instr = fun env -> F32_convert_i64_u
      method visit_F32_demote_f64 : _ -> instr = fun env -> F32_demote_f64
      method visit_F32_div : _ -> instr = fun env -> F32_div
      method visit_F32_eq : _ -> instr = fun env -> F32_eq
      method visit_F32_ge : _ -> instr = fun env -> F32_ge
      method visit_F32_gt : _ -> instr = fun env -> F32_gt
      method visit_F32_le : _ -> instr = fun env -> F32_le

      method visit_F32_load : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          F32_load _visitors_r0

      method visit_F32_lt : _ -> instr = fun env -> F32_lt
      method visit_F32_mul : _ -> instr = fun env -> F32_mul
      method visit_F32_ne : _ -> instr = fun env -> F32_ne
      method visit_F32_neg : _ -> instr = fun env -> F32_neg

      method visit_F32_reinterpret_i32 : _ -> instr =
        fun env -> F32_reinterpret_i32

      method visit_F32_sqrt : _ -> instr = fun env -> F32_sqrt

      method visit_F32_store : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          F32_store _visitors_r0

      method visit_F32_sub : _ -> instr = fun env -> F32_sub
      method visit_F32_abs : _ -> instr = fun env -> F32_abs
      method visit_F32_trunc : _ -> instr = fun env -> F32_trunc
      method visit_F32_floor : _ -> instr = fun env -> F32_floor
      method visit_F32_ceil : _ -> instr = fun env -> F32_ceil
      method visit_F32_nearest : _ -> instr = fun env -> F32_nearest
      method visit_F64_promote_f32 : _ -> instr = fun env -> F64_promote_f32

      method visit_I32_reinterpret_f32 : _ -> instr =
        fun env -> I32_reinterpret_f32

      method visit_I32_trunc_f32_s : _ -> instr = fun env -> I32_trunc_f32_s
      method visit_I32_trunc_f32_u : _ -> instr = fun env -> I32_trunc_f32_u
      method visit_I64_trunc_f32_s : _ -> instr = fun env -> I64_trunc_f32_s
      method visit_I64_trunc_f32_u : _ -> instr = fun env -> I64_trunc_f32_u

      method visit_Global_get : _ -> globalidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_globalidx env _visitors_c0 in
          Global_get _visitors_r0

      method visit_Global_set : _ -> globalidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_globalidx env _visitors_c0 in
          Global_set _visitors_r0

      method visit_I32_add : _ -> instr = fun env -> I32_add
      method visit_I32_and : _ -> instr = fun env -> I32_and
      method visit_I32_clz : _ -> instr = fun env -> I32_clz

      method visit_I32_const : _ -> int32 -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 =
            (fun _visitors_this -> _visitors_this) _visitors_c0
          in
          I32_const _visitors_r0

      method visit_I32_ctz : _ -> instr = fun env -> I32_ctz
      method visit_I32_div_s : _ -> instr = fun env -> I32_div_s
      method visit_I32_div_u : _ -> instr = fun env -> I32_div_u
      method visit_I32_eq : _ -> instr = fun env -> I32_eq
      method visit_I32_eqz : _ -> instr = fun env -> I32_eqz
      method visit_I32_ge_s : _ -> instr = fun env -> I32_ge_s
      method visit_I32_ge_u : _ -> instr = fun env -> I32_ge_u
      method visit_I32_gt_s : _ -> instr = fun env -> I32_gt_s
      method visit_I32_gt_u : _ -> instr = fun env -> I32_gt_u
      method visit_I32_le_s : _ -> instr = fun env -> I32_le_s
      method visit_I32_le_u : _ -> instr = fun env -> I32_le_u

      method visit_I32_load : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_load _visitors_r0

      method visit_I32_load16_u : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_load16_u _visitors_r0

      method visit_I32_load16_s : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_load16_s _visitors_r0

      method visit_I32_load8_u : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_load8_u _visitors_r0

      method visit_I32_load8_s : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_load8_s _visitors_r0

      method visit_I32_lt_s : _ -> instr = fun env -> I32_lt_s
      method visit_I32_lt_u : _ -> instr = fun env -> I32_lt_u
      method visit_I32_mul : _ -> instr = fun env -> I32_mul
      method visit_I32_ne : _ -> instr = fun env -> I32_ne
      method visit_I32_or : _ -> instr = fun env -> I32_or
      method visit_I32_popcnt : _ -> instr = fun env -> I32_popcnt
      method visit_I32_rem_s : _ -> instr = fun env -> I32_rem_s
      method visit_I32_rem_u : _ -> instr = fun env -> I32_rem_u
      method visit_I32_shl : _ -> instr = fun env -> I32_shl
      method visit_I32_shr_s : _ -> instr = fun env -> I32_shr_s
      method visit_I32_shr_u : _ -> instr = fun env -> I32_shr_u
      method visit_I32_rotl : _ -> instr = fun env -> I32_rotl

      method visit_I32_store : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_store _visitors_r0

      method visit_I32_store16 : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_store16 _visitors_r0

      method visit_I32_store8 : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I32_store8 _visitors_r0

      method visit_I32_sub : _ -> instr = fun env -> I32_sub
      method visit_I32_trunc_f64_s : _ -> instr = fun env -> I32_trunc_f64_s
      method visit_I32_trunc_f64_u : _ -> instr = fun env -> I32_trunc_f64_u
      method visit_I32_wrap_i64 : _ -> instr = fun env -> I32_wrap_i64
      method visit_I32_xor : _ -> instr = fun env -> I32_xor
      method visit_I32_extend_8_s : _ -> instr = fun env -> I32_extend_8_s
      method visit_I32_extend_16_s : _ -> instr = fun env -> I32_extend_16_s

      method visit_I32_trunc_sat_f32_s : _ -> instr =
        fun env -> I32_trunc_sat_f32_s

      method visit_I32_trunc_sat_f32_u : _ -> instr =
        fun env -> I32_trunc_sat_f32_u

      method visit_I32_trunc_sat_f64_s : _ -> instr =
        fun env -> I32_trunc_sat_f64_s

      method visit_I32_trunc_sat_f64_u : _ -> instr =
        fun env -> I32_trunc_sat_f64_u

      method visit_I64_add : _ -> instr = fun env -> I64_add
      method visit_I64_and : _ -> instr = fun env -> I64_and
      method visit_I64_clz : _ -> instr = fun env -> I64_clz

      method visit_I64_const : _ -> int64 -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 =
            (fun _visitors_this -> _visitors_this) _visitors_c0
          in
          I64_const _visitors_r0

      method visit_I64_ctz : _ -> instr = fun env -> I64_ctz
      method visit_I64_div_s : _ -> instr = fun env -> I64_div_s
      method visit_I64_div_u : _ -> instr = fun env -> I64_div_u
      method visit_I64_eq : _ -> instr = fun env -> I64_eq
      method visit_I64_extend_i32_s : _ -> instr = fun env -> I64_extend_i32_s
      method visit_I64_extend_i32_u : _ -> instr = fun env -> I64_extend_i32_u
      method visit_I64_ge_s : _ -> instr = fun env -> I64_ge_s
      method visit_I64_gt_s : _ -> instr = fun env -> I64_gt_s
      method visit_I64_le_s : _ -> instr = fun env -> I64_le_s
      method visit_I64_ge_u : _ -> instr = fun env -> I64_ge_u
      method visit_I64_gt_u : _ -> instr = fun env -> I64_gt_u
      method visit_I64_le_u : _ -> instr = fun env -> I64_le_u

      method visit_I64_load : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load _visitors_r0

      method visit_I64_load32_u : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load32_u _visitors_r0

      method visit_I64_load32_s : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load32_s _visitors_r0

      method visit_I64_load16_u : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load16_u _visitors_r0

      method visit_I64_load16_s : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load16_s _visitors_r0

      method visit_I64_load8_u : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load8_u _visitors_r0

      method visit_I64_load8_s : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_load8_s _visitors_r0

      method visit_I64_lt_s : _ -> instr = fun env -> I64_lt_s
      method visit_I64_lt_u : _ -> instr = fun env -> I64_lt_u
      method visit_I64_mul : _ -> instr = fun env -> I64_mul
      method visit_I64_ne : _ -> instr = fun env -> I64_ne
      method visit_I64_or : _ -> instr = fun env -> I64_or
      method visit_I64_popcnt : _ -> instr = fun env -> I64_popcnt

      method visit_I64_reinterpret_f64 : _ -> instr =
        fun env -> I64_reinterpret_f64

      method visit_I64_rem_s : _ -> instr = fun env -> I64_rem_s
      method visit_I64_rem_u : _ -> instr = fun env -> I64_rem_u
      method visit_I64_shl : _ -> instr = fun env -> I64_shl
      method visit_I64_shr_s : _ -> instr = fun env -> I64_shr_s
      method visit_I64_shr_u : _ -> instr = fun env -> I64_shr_u

      method visit_I64_store : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_store _visitors_r0

      method visit_I64_store32 : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_store32 _visitors_r0

      method visit_I64_store16 : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_store16 _visitors_r0

      method visit_I64_store8 : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          I64_store8 _visitors_r0

      method visit_I64_sub : _ -> instr = fun env -> I64_sub
      method visit_I64_trunc_f64_s : _ -> instr = fun env -> I64_trunc_f64_s
      method visit_I64_trunc_f64_u : _ -> instr = fun env -> I64_trunc_f64_u
      method visit_I64_xor : _ -> instr = fun env -> I64_xor
      method visit_I64_extend_8_s : _ -> instr = fun env -> I64_extend_8_s
      method visit_I64_extend_16_s : _ -> instr = fun env -> I64_extend_16_s
      method visit_I64_extend_32_s : _ -> instr = fun env -> I64_extend_32_s

      method visit_I64_trunc_sat_f32_s : _ -> instr =
        fun env -> I64_trunc_sat_f32_s

      method visit_I64_trunc_sat_f32_u : _ -> instr =
        fun env -> I64_trunc_sat_f32_u

      method visit_I64_trunc_sat_f64_s : _ -> instr =
        fun env -> I64_trunc_sat_f64_s

      method visit_I64_trunc_sat_f64_u : _ -> instr =
        fun env -> I64_trunc_sat_f64_u

      method visit_V128_load : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          V128_load _visitors_r0

      method visit_V128_store : _ -> memarg -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_memarg env _visitors_c0 in
          V128_store _visitors_r0

      method visit_F64x2_add : _ -> instr = fun env -> F64x2_add
      method visit_F64x2_mul : _ -> instr = fun env -> F64x2_mul
      method visit_F32x4_add : _ -> instr = fun env -> F32x4_add
      method visit_F32x4_mul : _ -> instr = fun env -> F32x4_mul

      method visit_If
          : _ -> label -> typeuse -> instr list -> instr list -> instr =
        fun env _visitors_c0 _visitors_c1 _visitors_c2 _visitors_c3 ->
          let _visitors_r0 = self#visit_label env _visitors_c0 in
          let _visitors_r1 = self#visit_typeuse env _visitors_c1 in
          let _visitors_r2 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_instr env))
              _visitors_c2
          in
          let _visitors_r3 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_instr env))
              _visitors_c3
          in
          If (_visitors_r0, _visitors_r1, _visitors_r2, _visitors_r3)

      method visit_Local_get : _ -> localidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_localidx env _visitors_c0 in
          Local_get _visitors_r0

      method visit_Local_set : _ -> localidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_localidx env _visitors_c0 in
          Local_set _visitors_r0

      method visit_Local_tee : _ -> localidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_localidx env _visitors_c0 in
          Local_tee _visitors_r0

      method visit_Loop : _ -> label -> typeuse -> instr list -> instr =
        fun env _visitors_c0 _visitors_c1 _visitors_c2 ->
          let _visitors_r0 = self#visit_label env _visitors_c0 in
          let _visitors_r1 = self#visit_typeuse env _visitors_c1 in
          let _visitors_r2 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_instr env))
              _visitors_c2
          in
          Loop (_visitors_r0, _visitors_r1, _visitors_r2)

      method visit_Memory_init : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Memory_init _visitors_r0

      method visit_Memory_copy : _ -> instr = fun env -> Memory_copy
      method visit_Memory_grow : _ -> instr = fun env -> Memory_grow
      method visit_Memory_size : _ -> instr = fun env -> Memory_size
      method visit_Memory_fill : _ -> instr = fun env -> Memory_fill
      method visit_Ref_eq : _ -> instr = fun env -> Ref_eq
      method visit_Ref_as_non_null : _ -> instr = fun env -> Ref_as_non_null

      method visit_Ref_cast : _ -> reftype -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_reftype env _visitors_c0 in
          Ref_cast _visitors_r0

      method visit_Ref_func : _ -> funcidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_funcidx env _visitors_c0 in
          Ref_func _visitors_r0

      method visit_Ref_is_null : _ -> instr = fun env -> Ref_is_null

      method visit_Ref_null : _ -> heaptype -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_heaptype env _visitors_c0 in
          Ref_null _visitors_r0

      method visit_Return : _ -> instr = fun env -> Return

      method visit_Struct_get : _ -> typeidx -> fieldidx -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          let _visitors_r1 = self#visit_fieldidx env _visitors_c1 in
          Struct_get (_visitors_r0, _visitors_r1)

      method visit_Struct_new : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Struct_new _visitors_r0

      method visit_Struct_new_default : _ -> typeidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          Struct_new_default _visitors_r0

      method visit_Struct_set : _ -> typeidx -> fieldidx -> instr =
        fun env _visitors_c0 _visitors_c1 ->
          let _visitors_r0 = self#visit_typeidx env _visitors_c0 in
          let _visitors_r1 = self#visit_fieldidx env _visitors_c1 in
          Struct_set (_visitors_r0, _visitors_r1)

      method visit_Table_get : _ -> tableidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_tableidx env _visitors_c0 in
          Table_get _visitors_r0

      method visit_Unreachable : _ -> instr = fun env -> Unreachable

      method visit_Throw : _ -> tagidx -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_tagidx env _visitors_c0 in
          Throw _visitors_r0

      method visit_Try_table
          : _ -> label -> typeuse -> catch list -> instr list -> instr =
        fun env _visitors_c0 _visitors_c1 _visitors_c2 _visitors_c3 ->
          let _visitors_r0 = self#visit_label env _visitors_c0 in
          let _visitors_r1 = self#visit_typeuse env _visitors_c1 in
          let _visitors_r2 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_catch env))
              _visitors_c2
          in
          let _visitors_r3 =
            (fun _visitors_this ->
              Basic_lst.map _visitors_this (self#visit_instr env))
              _visitors_c3
          in
          Try_table (_visitors_r0, _visitors_r1, _visitors_r2, _visitors_r3)

      method visit_Select : _ -> instr = fun env -> Select
      method visit_No_op : _ -> instr = fun env -> No_op

      method visit_Source_pos : _ -> _ -> instr =
        fun env _visitors_c0 ->
          let _visitors_r0 =
            (fun _visitors_this -> _visitors_this) _visitors_c0
          in
          Source_pos _visitors_r0

      method visit_Prologue_end : _ -> instr = fun env -> Prologue_end

      method visit_instr : _ -> instr -> instr =
        fun env _visitors_this ->
          match _visitors_this with
          | Any_convert_extern -> self#visit_Any_convert_extern env
          | Array_copy (_visitors_c0, _visitors_c1) ->
              self#visit_Array_copy env _visitors_c0 _visitors_c1
          | Array_fill _visitors_c0 -> self#visit_Array_fill env _visitors_c0
          | Array_get _visitors_c0 -> self#visit_Array_get env _visitors_c0
          | Array_get_u _visitors_c0 -> self#visit_Array_get_u env _visitors_c0
          | Array_len -> self#visit_Array_len env
          | Array_new _visitors_c0 -> self#visit_Array_new env _visitors_c0
          | Array_new_data (_visitors_c0, _visitors_c1) ->
              self#visit_Array_new_data env _visitors_c0 _visitors_c1
          | Array_new_default _visitors_c0 ->
              self#visit_Array_new_default env _visitors_c0
          | Array_new_fixed (_visitors_c0, _visitors_c1) ->
              self#visit_Array_new_fixed env _visitors_c0 _visitors_c1
          | Array_set _visitors_c0 -> self#visit_Array_set env _visitors_c0
          | Block (_visitors_c0, _visitors_c1, _visitors_c2) ->
              self#visit_Block env _visitors_c0 _visitors_c1 _visitors_c2
          | Br _visitors_c0 -> self#visit_Br env _visitors_c0
          | Br_if _visitors_c0 -> self#visit_Br_if env _visitors_c0
          | Br_table (_visitors_c0, _visitors_c1) ->
              self#visit_Br_table env _visitors_c0 _visitors_c1
          | Call _visitors_c0 -> self#visit_Call env _visitors_c0
          | Call_indirect (_visitors_c0, _visitors_c1) ->
              self#visit_Call_indirect env _visitors_c0 _visitors_c1
          | Call_ref _visitors_c0 -> self#visit_Call_ref env _visitors_c0
          | Drop -> self#visit_Drop env
          | Extern_convert_any -> self#visit_Extern_convert_any env
          | F64_add -> self#visit_F64_add env
          | F64_const (_visitors_c0, _visitors_c1) ->
              self#visit_F64_const env _visitors_c0 _visitors_c1
          | F64_convert_i32_s -> self#visit_F64_convert_i32_s env
          | F64_convert_i32_u -> self#visit_F64_convert_i32_u env
          | F64_convert_i64_s -> self#visit_F64_convert_i64_s env
          | F64_convert_i64_u -> self#visit_F64_convert_i64_u env
          | F64_div -> self#visit_F64_div env
          | F64_eq -> self#visit_F64_eq env
          | F64_ge -> self#visit_F64_ge env
          | F64_gt -> self#visit_F64_gt env
          | F64_le -> self#visit_F64_le env
          | F64_load _visitors_c0 -> self#visit_F64_load env _visitors_c0
          | F64_lt -> self#visit_F64_lt env
          | F64_mul -> self#visit_F64_mul env
          | F64_ne -> self#visit_F64_ne env
          | F64_neg -> self#visit_F64_neg env
          | F64_reinterpret_i64 -> self#visit_F64_reinterpret_i64 env
          | F64_store _visitors_c0 -> self#visit_F64_store env _visitors_c0
          | F64_sub -> self#visit_F64_sub env
          | F64_sqrt -> self#visit_F64_sqrt env
          | F64_abs -> self#visit_F64_abs env
          | F64_trunc -> self#visit_F64_trunc env
          | F64_floor -> self#visit_F64_floor env
          | F64_ceil -> self#visit_F64_ceil env
          | F64_nearest -> self#visit_F64_nearest env
          | F32_add -> self#visit_F32_add env
          | F32_const (_visitors_c0, _visitors_c1) ->
              self#visit_F32_const env _visitors_c0 _visitors_c1
          | F32_convert_i32_s -> self#visit_F32_convert_i32_s env
          | F32_convert_i32_u -> self#visit_F32_convert_i32_u env
          | F32_convert_i64_s -> self#visit_F32_convert_i64_s env
          | F32_convert_i64_u -> self#visit_F32_convert_i64_u env
          | F32_demote_f64 -> self#visit_F32_demote_f64 env
          | F32_div -> self#visit_F32_div env
          | F32_eq -> self#visit_F32_eq env
          | F32_ge -> self#visit_F32_ge env
          | F32_gt -> self#visit_F32_gt env
          | F32_le -> self#visit_F32_le env
          | F32_load _visitors_c0 -> self#visit_F32_load env _visitors_c0
          | F32_lt -> self#visit_F32_lt env
          | F32_mul -> self#visit_F32_mul env
          | F32_ne -> self#visit_F32_ne env
          | F32_neg -> self#visit_F32_neg env
          | F32_reinterpret_i32 -> self#visit_F32_reinterpret_i32 env
          | F32_sqrt -> self#visit_F32_sqrt env
          | F32_store _visitors_c0 -> self#visit_F32_store env _visitors_c0
          | F32_sub -> self#visit_F32_sub env
          | F32_abs -> self#visit_F32_abs env
          | F32_trunc -> self#visit_F32_trunc env
          | F32_floor -> self#visit_F32_floor env
          | F32_ceil -> self#visit_F32_ceil env
          | F32_nearest -> self#visit_F32_nearest env
          | F64_promote_f32 -> self#visit_F64_promote_f32 env
          | I32_reinterpret_f32 -> self#visit_I32_reinterpret_f32 env
          | I32_trunc_f32_s -> self#visit_I32_trunc_f32_s env
          | I32_trunc_f32_u -> self#visit_I32_trunc_f32_u env
          | I64_trunc_f32_s -> self#visit_I64_trunc_f32_s env
          | I64_trunc_f32_u -> self#visit_I64_trunc_f32_u env
          | Global_get _visitors_c0 -> self#visit_Global_get env _visitors_c0
          | Global_set _visitors_c0 -> self#visit_Global_set env _visitors_c0
          | I32_add -> self#visit_I32_add env
          | I32_and -> self#visit_I32_and env
          | I32_clz -> self#visit_I32_clz env
          | I32_const _visitors_c0 -> self#visit_I32_const env _visitors_c0
          | I32_ctz -> self#visit_I32_ctz env
          | I32_div_s -> self#visit_I32_div_s env
          | I32_div_u -> self#visit_I32_div_u env
          | I32_eq -> self#visit_I32_eq env
          | I32_eqz -> self#visit_I32_eqz env
          | I32_ge_s -> self#visit_I32_ge_s env
          | I32_ge_u -> self#visit_I32_ge_u env
          | I32_gt_s -> self#visit_I32_gt_s env
          | I32_gt_u -> self#visit_I32_gt_u env
          | I32_le_s -> self#visit_I32_le_s env
          | I32_le_u -> self#visit_I32_le_u env
          | I32_load _visitors_c0 -> self#visit_I32_load env _visitors_c0
          | I32_load16_u _visitors_c0 ->
              self#visit_I32_load16_u env _visitors_c0
          | I32_load16_s _visitors_c0 ->
              self#visit_I32_load16_s env _visitors_c0
          | I32_load8_u _visitors_c0 -> self#visit_I32_load8_u env _visitors_c0
          | I32_load8_s _visitors_c0 -> self#visit_I32_load8_s env _visitors_c0
          | I32_lt_s -> self#visit_I32_lt_s env
          | I32_lt_u -> self#visit_I32_lt_u env
          | I32_mul -> self#visit_I32_mul env
          | I32_ne -> self#visit_I32_ne env
          | I32_or -> self#visit_I32_or env
          | I32_popcnt -> self#visit_I32_popcnt env
          | I32_rem_s -> self#visit_I32_rem_s env
          | I32_rem_u -> self#visit_I32_rem_u env
          | I32_shl -> self#visit_I32_shl env
          | I32_shr_s -> self#visit_I32_shr_s env
          | I32_shr_u -> self#visit_I32_shr_u env
          | I32_rotl -> self#visit_I32_rotl env
          | I32_store _visitors_c0 -> self#visit_I32_store env _visitors_c0
          | I32_store16 _visitors_c0 -> self#visit_I32_store16 env _visitors_c0
          | I32_store8 _visitors_c0 -> self#visit_I32_store8 env _visitors_c0
          | I32_sub -> self#visit_I32_sub env
          | I32_trunc_f64_s -> self#visit_I32_trunc_f64_s env
          | I32_trunc_f64_u -> self#visit_I32_trunc_f64_u env
          | I32_wrap_i64 -> self#visit_I32_wrap_i64 env
          | I32_xor -> self#visit_I32_xor env
          | I32_extend_8_s -> self#visit_I32_extend_8_s env
          | I32_extend_16_s -> self#visit_I32_extend_16_s env
          | I32_trunc_sat_f32_s -> self#visit_I32_trunc_sat_f32_s env
          | I32_trunc_sat_f32_u -> self#visit_I32_trunc_sat_f32_u env
          | I32_trunc_sat_f64_s -> self#visit_I32_trunc_sat_f64_s env
          | I32_trunc_sat_f64_u -> self#visit_I32_trunc_sat_f64_u env
          | I64_add -> self#visit_I64_add env
          | I64_and -> self#visit_I64_and env
          | I64_clz -> self#visit_I64_clz env
          | I64_const _visitors_c0 -> self#visit_I64_const env _visitors_c0
          | I64_ctz -> self#visit_I64_ctz env
          | I64_div_s -> self#visit_I64_div_s env
          | I64_div_u -> self#visit_I64_div_u env
          | I64_eq -> self#visit_I64_eq env
          | I64_extend_i32_s -> self#visit_I64_extend_i32_s env
          | I64_extend_i32_u -> self#visit_I64_extend_i32_u env
          | I64_ge_s -> self#visit_I64_ge_s env
          | I64_gt_s -> self#visit_I64_gt_s env
          | I64_le_s -> self#visit_I64_le_s env
          | I64_ge_u -> self#visit_I64_ge_u env
          | I64_gt_u -> self#visit_I64_gt_u env
          | I64_le_u -> self#visit_I64_le_u env
          | I64_load _visitors_c0 -> self#visit_I64_load env _visitors_c0
          | I64_load32_u _visitors_c0 ->
              self#visit_I64_load32_u env _visitors_c0
          | I64_load32_s _visitors_c0 ->
              self#visit_I64_load32_s env _visitors_c0
          | I64_load16_u _visitors_c0 ->
              self#visit_I64_load16_u env _visitors_c0
          | I64_load16_s _visitors_c0 ->
              self#visit_I64_load16_s env _visitors_c0
          | I64_load8_u _visitors_c0 -> self#visit_I64_load8_u env _visitors_c0
          | I64_load8_s _visitors_c0 -> self#visit_I64_load8_s env _visitors_c0
          | I64_lt_s -> self#visit_I64_lt_s env
          | I64_lt_u -> self#visit_I64_lt_u env
          | I64_mul -> self#visit_I64_mul env
          | I64_ne -> self#visit_I64_ne env
          | I64_or -> self#visit_I64_or env
          | I64_popcnt -> self#visit_I64_popcnt env
          | I64_reinterpret_f64 -> self#visit_I64_reinterpret_f64 env
          | I64_rem_s -> self#visit_I64_rem_s env
          | I64_rem_u -> self#visit_I64_rem_u env
          | I64_shl -> self#visit_I64_shl env
          | I64_shr_s -> self#visit_I64_shr_s env
          | I64_shr_u -> self#visit_I64_shr_u env
          | I64_store _visitors_c0 -> self#visit_I64_store env _visitors_c0
          | I64_store32 _visitors_c0 -> self#visit_I64_store32 env _visitors_c0
          | I64_store16 _visitors_c0 -> self#visit_I64_store16 env _visitors_c0
          | I64_store8 _visitors_c0 -> self#visit_I64_store8 env _visitors_c0
          | I64_sub -> self#visit_I64_sub env
          | I64_trunc_f64_s -> self#visit_I64_trunc_f64_s env
          | I64_trunc_f64_u -> self#visit_I64_trunc_f64_u env
          | I64_xor -> self#visit_I64_xor env
          | I64_extend_8_s -> self#visit_I64_extend_8_s env
          | I64_extend_16_s -> self#visit_I64_extend_16_s env
          | I64_extend_32_s -> self#visit_I64_extend_32_s env
          | I64_trunc_sat_f32_s -> self#visit_I64_trunc_sat_f32_s env
          | I64_trunc_sat_f32_u -> self#visit_I64_trunc_sat_f32_u env
          | I64_trunc_sat_f64_s -> self#visit_I64_trunc_sat_f64_s env
          | I64_trunc_sat_f64_u -> self#visit_I64_trunc_sat_f64_u env
          | V128_load _visitors_c0 -> self#visit_V128_load env _visitors_c0
          | V128_store _visitors_c0 -> self#visit_V128_store env _visitors_c0
          | F64x2_add -> self#visit_F64x2_add env
          | F64x2_mul -> self#visit_F64x2_mul env
          | F32x4_add -> self#visit_F32x4_add env
          | F32x4_mul -> self#visit_F32x4_mul env
          | If (_visitors_c0, _visitors_c1, _visitors_c2, _visitors_c3) ->
              self#visit_If env _visitors_c0 _visitors_c1 _visitors_c2
                _visitors_c3
          | Local_get _visitors_c0 -> self#visit_Local_get env _visitors_c0
          | Local_set _visitors_c0 -> self#visit_Local_set env _visitors_c0
          | Local_tee _visitors_c0 -> self#visit_Local_tee env _visitors_c0
          | Loop (_visitors_c0, _visitors_c1, _visitors_c2) ->
              self#visit_Loop env _visitors_c0 _visitors_c1 _visitors_c2
          | Memory_init _visitors_c0 -> self#visit_Memory_init env _visitors_c0
          | Memory_copy -> self#visit_Memory_copy env
          | Memory_grow -> self#visit_Memory_grow env
          | Memory_size -> self#visit_Memory_size env
          | Memory_fill -> self#visit_Memory_fill env
          | Ref_eq -> self#visit_Ref_eq env
          | Ref_as_non_null -> self#visit_Ref_as_non_null env
          | Ref_cast _visitors_c0 -> self#visit_Ref_cast env _visitors_c0
          | Ref_func _visitors_c0 -> self#visit_Ref_func env _visitors_c0
          | Ref_is_null -> self#visit_Ref_is_null env
          | Ref_null _visitors_c0 -> self#visit_Ref_null env _visitors_c0
          | Return -> self#visit_Return env
          | Struct_get (_visitors_c0, _visitors_c1) ->
              self#visit_Struct_get env _visitors_c0 _visitors_c1
          | Struct_new _visitors_c0 -> self#visit_Struct_new env _visitors_c0
          | Struct_new_default _visitors_c0 ->
              self#visit_Struct_new_default env _visitors_c0
          | Struct_set (_visitors_c0, _visitors_c1) ->
              self#visit_Struct_set env _visitors_c0 _visitors_c1
          | Table_get _visitors_c0 -> self#visit_Table_get env _visitors_c0
          | Unreachable -> self#visit_Unreachable env
          | Throw _visitors_c0 -> self#visit_Throw env _visitors_c0
          | Try_table (_visitors_c0, _visitors_c1, _visitors_c2, _visitors_c3)
            ->
              self#visit_Try_table env _visitors_c0 _visitors_c1 _visitors_c2
                _visitors_c3
          | Select -> self#visit_Select env
          | No_op -> self#visit_No_op env
          | Source_pos _visitors_c0 -> self#visit_Source_pos env _visitors_c0
          | Prologue_end -> self#visit_Prologue_end env
    end

  [@@@VISITORS.END]
end

include struct
  let _ = fun (_ : instr) -> ()
end

type expr = instr list
type extra_info = { mutable low_pc : int; mutable high_pc : int }

type func = {
  id : binder;
  type_ : typeuse;
  locals : local list;
  code : instr list;
  source_name : string option;
  aux : extra_info;
}

type table = { id : binder; type_ : tabletype; init : expr }
type mem = { id : binder; type_ : memtype }
type global = { id : binder; type_ : globaltype; init : expr }

type exportdesc =
  | Func of funcidx
  | Table of tableidx
  | Memory of memidx
  | Global of globalidx

type export = { name : string; desc : exportdesc }
type start = funcidx
type elemmode = EMPassive | EMActive of tableidx * expr | EMDeclarative
type elem = { id : binder; type_ : reftype; list : expr list; mode : elemmode }
type datamode = DMPassive | DMActive of memidx * expr
type data = { id : binder; data_str : string; mode : datamode }
type tag = { id : binder; type_ : typeuse }

type 'a mk_modulefield =
  | Rectype of rectype
  | Import of import
  | Func of 'a
  | Table of table
  | Mem of mem
  | Global of global
  | Export of export
  | Start of start
  | Elem of elem
  | Data of data
  | Tag of tag

type modulefield = func mk_modulefield
type modulefield_new = W.t mk_modulefield
type 'a mk_module = { id : binder; fields : 'a mk_modulefield list }
type module_ = func mk_module
type module_new = W.t mk_module
