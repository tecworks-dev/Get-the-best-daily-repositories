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


module Config = Basic_config
module Fn_address = Basic_fn_address
module UInt32 = Basic_uint32
module UInt64 = Basic_uint64
module Lst = Basic_lst
module Hash_int = Basic_hash_int

type instr = W.t

let cons_peephole instr rest =
  match (instr, rest) with
  | ( ( (List [ Atom "i32.const"; _ ] : W.t)
      | (List [ Atom "local.get"; _ ] : W.t) ),
      (List (Atom "drop" :: []) : W.t) :: rest ) ->
      rest
  | ( (List [ Atom "local.set"; id ] : W.t),
      (List [ Atom "local.get"; id2 ] : W.t) :: rest )
    when W.equal id id2 ->
      List.cons
        (List (List.cons (Atom "local.tee" : W.t) ([ id ] : W.t list)) : W.t)
        (rest : W.t list)
  | (List [ Atom "br"; _ ] : W.t), _ -> [ instr ]
  | _ -> List.cons instr rest

module List = struct
  include List

  let cons = cons_peephole
end

let int32_to_sexp (i : int32) : instr =
  let int32_to_string = Int32.to_string in
  (List
     (List.cons
        (Atom "i32.const" : W.t)
        ([ Atom (int32_to_string i) ] : W.t list))
    : W.t)

let i32_to_sexp (i : int) : instr =
  let int_to_string = string_of_int in
  (List
     (List.cons
        (Atom "i32.const" : W.t)
        ([ Atom (int_to_string i) ] : W.t list))
    : W.t)

let i64_to_sexp (i : int64) : instr =
  let int64_to_string = Int64.to_string in
  (List
     (List.cons
        (Atom "i64.const" : W.t)
        ([ Atom (int64_to_string i) ] : W.t list))
    : W.t)

let f32_to_sexp (f : float) : instr =
  if f = Float.infinity then
    (List (List.cons (Atom "f32.const" : W.t) ([ Atom "inf" ] : W.t list))
      : W.t)
  else if f = Float.neg_infinity then
    (List (List.cons (Atom "f32.const" : W.t) ([ Atom "-inf" ] : W.t list))
      : W.t)
  else
    let float_to_string = Printf.sprintf "%h" in
    (List
       (List.cons
          (Atom "f32.const" : W.t)
          ([ Atom (float_to_string f) ] : W.t list))
      : W.t)

let f64_to_sexp (f : float) : instr =
  if f = Float.infinity then
    (List (List.cons (Atom "f64.const" : W.t) ([ Atom "inf" ] : W.t list))
      : W.t)
  else if f = Float.neg_infinity then
    (List (List.cons (Atom "f64.const" : W.t) ([ Atom "-inf" ] : W.t list))
      : W.t)
  else
    let float_to_string = Printf.sprintf "%h" in
    (List
       (List.cons
          (Atom "f64.const" : W.t)
          ([ Atom (float_to_string f) ] : W.t list))
      : W.t)

let u32_to_sexp i =
  let uint32_to_string = UInt32.to_string in
  (List
     (List.cons
        (Atom "i32.const" : W.t)
        ([ Atom (uint32_to_string i) ] : W.t list))
    : W.t)

let u64_to_sexp i =
  let uint64_to_string = UInt64.to_string in
  (List
     (List.cons
        (Atom "i64.const" : W.t)
        ([ Atom (uint64_to_string i) ] : W.t list))
    : W.t)

let str_to_sexp (s : string) : instr = W.Atom ("\"" ^ W.escaped s ^ "\"")
let offset_to_string : int -> string = Printf.sprintf "offset=%d"

let add_cst i rest =
  match (i, rest) with
  | _, (List (Atom "drop" :: []) : W.t) :: rest -> rest
  | _, _ -> List.cons (i32_to_sexp i : W.t) (rest : W.t list)

let compile_arith (ty : Primitive.operand_type) (op : Primitive.arith_operator)
    =
  match ty with
  | I32 -> (
      match op with
      | Add -> (List ([ Atom "i32.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "i32.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "i32.mul" ] : W.t list) : W.t)
      | Div -> (List ([ Atom "i32.div_s" ] : W.t list) : W.t)
      | Mod -> (List ([ Atom "i32.rem_s" ] : W.t list) : W.t)
      | Neg | Sqrt -> assert false)
  | I64 -> (
      match op with
      | Add -> (List ([ Atom "i64.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "i64.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "i64.mul" ] : W.t list) : W.t)
      | Div -> (List ([ Atom "i64.div_s" ] : W.t list) : W.t)
      | Mod -> (List ([ Atom "i64.rem_s" ] : W.t list) : W.t)
      | Neg | Sqrt -> assert false)
  | U32 -> (
      match op with
      | Div -> (List ([ Atom "i32.div_u" ] : W.t list) : W.t)
      | Mod -> (List ([ Atom "i32.rem_u" ] : W.t list) : W.t)
      | Add -> (List ([ Atom "i32.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "i32.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "i32.mul" ] : W.t list) : W.t)
      | Neg | Sqrt -> assert false)
  | U64 -> (
      match op with
      | Div -> (List ([ Atom "i64.div_u" ] : W.t list) : W.t)
      | Mod -> (List ([ Atom "i64.rem_u" ] : W.t list) : W.t)
      | Add -> (List ([ Atom "i64.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "i64.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "i64.mul" ] : W.t list) : W.t)
      | Neg | Sqrt -> assert false)
  | F32 -> (
      match op with
      | Add -> (List ([ Atom "f32.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "f32.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "f32.mul" ] : W.t list) : W.t)
      | Div -> (List ([ Atom "f32.div" ] : W.t list) : W.t)
      | Mod -> assert false
      | Sqrt -> (List ([ Atom "f32.sqrt" ] : W.t list) : W.t)
      | Neg -> (List ([ Atom "f32.neg" ] : W.t list) : W.t))
  | F64 -> (
      match op with
      | Add -> (List ([ Atom "f64.add" ] : W.t list) : W.t)
      | Sub -> (List ([ Atom "f64.sub" ] : W.t list) : W.t)
      | Mul -> (List ([ Atom "f64.mul" ] : W.t list) : W.t)
      | Div -> (List ([ Atom "f64.div" ] : W.t list) : W.t)
      | Mod -> assert false
      | Sqrt -> (List ([ Atom "f64.sqrt" ] : W.t list) : W.t)
      | Neg -> (List ([ Atom "f64.neg" ] : W.t list) : W.t))
  | U8 -> assert false

let compile_convert (kind : Primitive.convert_kind)
    (from : Primitive.operand_type) (to_ : Primitive.operand_type)
    (rest : W.t list) =
  match kind with
  | Convert -> (
      match from with
      | I32 -> (
          match to_ with
          | I32 -> assert false
          | I64 -> (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t) :: rest
          | F32 ->
              (List ([ Atom "f32.convert_i32_s" ] : W.t list) : W.t) :: rest
          | F64 ->
              (List ([ Atom "f64.convert_i32_s" ] : W.t list) : W.t) :: rest
          | U8 ->
              List.cons
                (List
                   (List.cons
                      (Atom "i32.const" : W.t)
                      ([ Atom "255" ] : W.t list))
                  : W.t)
                (List.cons
                   (List ([ Atom "i32.and" ] : W.t list) : W.t)
                   (rest : W.t list))
          | U64 | U32 -> assert false)
      | I64 -> (
          match to_ with
          | I32 -> (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t) :: rest
          | I64 -> assert false
          | F32 ->
              (List ([ Atom "f32.convert_i64_s" ] : W.t list) : W.t) :: rest
          | F64 ->
              (List ([ Atom "f64.convert_i64_s" ] : W.t list) : W.t) :: rest
          | U8 ->
              List.cons
                (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
                (List.cons
                   (List
                      (List.cons
                         (Atom "i32.const" : W.t)
                         ([ Atom "255" ] : W.t list))
                     : W.t)
                   (List.cons
                      (List ([ Atom "i32.and" ] : W.t list) : W.t)
                      (rest : W.t list)))
          | U64 | U32 -> assert false)
      | F32 -> (
          match to_ with
          | I32 -> (List ([ Atom "i32.trunc_f32_s" ] : W.t list) : W.t) :: rest
          | I64 -> (List ([ Atom "i64.trunc_f32_s" ] : W.t list) : W.t) :: rest
          | U32 -> (List ([ Atom "i32.trunc_f32_u" ] : W.t list) : W.t) :: rest
          | U64 -> (List ([ Atom "i64.trunc_f32_u" ] : W.t list) : W.t) :: rest
          | F32 -> assert false
          | F64 -> (List ([ Atom "f64.promote_f32" ] : W.t list) : W.t) :: rest
          | U8 ->
              List.cons
                (List ([ Atom "i32.trunc_f32_s" ] : W.t list) : W.t)
                (List.cons
                   (List
                      (List.cons
                         (Atom "i32.const" : W.t)
                         ([ Atom "255" ] : W.t list))
                     : W.t)
                   (List.cons
                      (List ([ Atom "i32.and" ] : W.t list) : W.t)
                      (rest : W.t list))))
      | F64 -> (
          match to_ with
          | I32 -> (List ([ Atom "i32.trunc_f64_s" ] : W.t list) : W.t) :: rest
          | I64 -> (List ([ Atom "i64.trunc_f64_s" ] : W.t list) : W.t) :: rest
          | U32 -> (List ([ Atom "i32.trunc_f64_u" ] : W.t list) : W.t) :: rest
          | U64 -> (List ([ Atom "i64.trunc_f64_u" ] : W.t list) : W.t) :: rest
          | F32 -> (List ([ Atom "f32.demote_f64" ] : W.t list) : W.t) :: rest
          | F64 -> assert false
          | U8 ->
              List.cons
                (List ([ Atom "i32.trunc_f64_s" ] : W.t list) : W.t)
                (List.cons
                   (List
                      (List.cons
                         (Atom "i32.const" : W.t)
                         ([ Atom "255" ] : W.t list))
                     : W.t)
                   (List.cons
                      (List ([ Atom "i32.and" ] : W.t list) : W.t)
                      (rest : W.t list))))
      | U8 -> (
          match to_ with
          | I32 -> rest
          | I64 -> (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t) :: rest
          | F64 ->
              (List ([ Atom "f64.convert_i32_s" ] : W.t list) : W.t) :: rest
          | F32 ->
              (List ([ Atom "f32.convert_i32_s" ] : W.t list) : W.t) :: rest
          | U64 | U32 | U8 -> assert false)
      | U32 -> (
          match to_ with
          | U32 -> assert false
          | F32 ->
              (List ([ Atom "f32.convert_i32_u" ] : W.t list) : W.t) :: rest
          | F64 ->
              (List ([ Atom "f64.convert_i32_u" ] : W.t list) : W.t) :: rest
          | U64 | I64 ->
              (List ([ Atom "i64.extend_i32_u" ] : W.t list) : W.t) :: rest
          | I32 | U8 -> assert false)
      | U64 -> (
          match to_ with
          | F32 ->
              (List ([ Atom "f32.convert_i64_u" ] : W.t list) : W.t) :: rest
          | F64 ->
              (List ([ Atom "f64.convert_i64_u" ] : W.t list) : W.t) :: rest
          | U32 | I32 ->
              (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t) :: rest
          | U64 | I64 | U8 -> assert false))
  | Reinterpret -> (
      match from with
      | F32 -> (
          match to_ with
          | I32 ->
              (List ([ Atom "i32.reinterpret_f32" ] : W.t list) : W.t) :: rest
          | I64 | F32 | F64 | U64 | U32 | U8 -> assert false)
      | F64 -> (
          match to_ with
          | I64 ->
              (List ([ Atom "i64.reinterpret_f64" ] : W.t list) : W.t) :: rest
          | I32 | F32 | F64 | U64 | U32 | U8 -> assert false)
      | I64 -> (
          match to_ with
          | F64 ->
              (List ([ Atom "f64.reinterpret_i64" ] : W.t list) : W.t) :: rest
          | U64 -> rest
          | I32 | I64 | F32 | U32 | U8 -> assert false)
      | U64 -> (
          match to_ with
          | I64 -> rest
          | I32 | F32 | F64 | U64 | U32 | U8 -> assert false)
      | I32 -> (
          match to_ with
          | F32 ->
              (List ([ Atom "f32.reinterpret_i32" ] : W.t list) : W.t) :: rest
          | U32 -> rest
          | I32 | I64 | U8 | F64 | U64 -> assert false)
      | U32 -> (
          match to_ with
          | I32 -> rest
          | U32 | I64 | U8 | F32 | F64 | U64 -> assert false)
      | U8 -> assert false)

let compile_bitwise (ty : Primitive.operand_type)
    (op : Primitive.bitwise_operator) (rest : W.t list) =
  match ty with
  | I32 -> (
      match op with
      | Not ->
          List.cons
            (List
               (List.cons
                  (Atom "i32.xor" : W.t)
                  ([
                     List
                       (List.cons
                          (Atom "i32.const" : W.t)
                          ([ Atom "-1" ] : W.t list));
                   ]
                    : W.t list))
              : W.t)
            (rest : W.t list)
      | And ->
          List.cons
            (List ([ Atom "i32.and" ] : W.t list) : W.t)
            (rest : W.t list)
      | Or ->
          List.cons
            (List ([ Atom "i32.or" ] : W.t list) : W.t)
            (rest : W.t list)
      | Xor ->
          List.cons
            (List ([ Atom "i32.xor" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shl ->
          List.cons
            (List ([ Atom "i32.shl" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shr ->
          List.cons
            (List ([ Atom "i32.shr_s" ] : W.t list) : W.t)
            (rest : W.t list)
      | Clz ->
          List.cons
            (List ([ Atom "i32.clz" ] : W.t list) : W.t)
            (rest : W.t list)
      | Ctz ->
          List.cons
            (List ([ Atom "i32.ctz" ] : W.t list) : W.t)
            (rest : W.t list)
      | Popcnt ->
          List.cons
            (List ([ Atom "i32.popcnt" ] : W.t list) : W.t)
            (rest : W.t list))
  | I64 -> (
      match op with
      | Not ->
          List.cons
            (List
               (List.cons
                  (Atom "i64.xor" : W.t)
                  ([
                     List
                       (List.cons
                          (Atom "i64.const" : W.t)
                          ([ Atom "-1" ] : W.t list));
                   ]
                    : W.t list))
              : W.t)
            (rest : W.t list)
      | And ->
          List.cons
            (List ([ Atom "i64.and" ] : W.t list) : W.t)
            (rest : W.t list)
      | Or ->
          List.cons
            (List ([ Atom "i64.or" ] : W.t list) : W.t)
            (rest : W.t list)
      | Xor ->
          List.cons
            (List ([ Atom "i64.xor" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shl ->
          List.cons
            (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i64.shl" ] : W.t list) : W.t)
               (rest : W.t list))
      | Shr ->
          List.cons
            (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i64.shr_s" ] : W.t list) : W.t)
               (rest : W.t list))
      | Clz ->
          List.cons
            (List ([ Atom "i64.clz" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list))
      | Ctz ->
          List.cons
            (List ([ Atom "i64.ctz" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list))
      | Popcnt ->
          List.cons
            (List ([ Atom "i64.popcnt" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list)))
  | F32 -> assert false
  | F64 -> assert false
  | U8 -> assert false
  | U32 -> (
      match op with
      | Not ->
          List.cons
            (List
               (List.cons
                  (Atom "i32.xor" : W.t)
                  ([
                     List
                       (List.cons
                          (Atom "i32.const" : W.t)
                          ([ Atom "-1" ] : W.t list));
                   ]
                    : W.t list))
              : W.t)
            (rest : W.t list)
      | And ->
          List.cons
            (List ([ Atom "i32.and" ] : W.t list) : W.t)
            (rest : W.t list)
      | Or ->
          List.cons
            (List ([ Atom "i32.or" ] : W.t list) : W.t)
            (rest : W.t list)
      | Xor ->
          List.cons
            (List ([ Atom "i32.xor" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shl ->
          List.cons
            (List ([ Atom "i32.shl" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shr ->
          List.cons
            (List ([ Atom "i32.shr_u" ] : W.t list) : W.t)
            (rest : W.t list)
      | Clz ->
          List.cons
            (List ([ Atom "i32.clz" ] : W.t list) : W.t)
            (rest : W.t list)
      | Ctz ->
          List.cons
            (List ([ Atom "i32.ctz" ] : W.t list) : W.t)
            (rest : W.t list)
      | Popcnt ->
          List.cons
            (List ([ Atom "i32.popcnt" ] : W.t list) : W.t)
            (rest : W.t list))
  | U64 -> (
      match op with
      | Not ->
          List.cons
            (List
               (List.cons
                  (Atom "i64.xor" : W.t)
                  ([
                     List
                       (List.cons
                          (Atom "i64.const" : W.t)
                          ([ Atom "-1" ] : W.t list));
                   ]
                    : W.t list))
              : W.t)
            (rest : W.t list)
      | And ->
          List.cons
            (List ([ Atom "i64.and" ] : W.t list) : W.t)
            (rest : W.t list)
      | Or ->
          List.cons
            (List ([ Atom "i64.or" ] : W.t list) : W.t)
            (rest : W.t list)
      | Xor ->
          List.cons
            (List ([ Atom "i64.xor" ] : W.t list) : W.t)
            (rest : W.t list)
      | Shl ->
          List.cons
            (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i64.shl" ] : W.t list) : W.t)
               (rest : W.t list))
      | Shr ->
          List.cons
            (List ([ Atom "i64.extend_i32_s" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i64.shr_u" ] : W.t list) : W.t)
               (rest : W.t list))
      | Clz ->
          List.cons
            (List ([ Atom "i64.clz" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list))
      | Ctz ->
          List.cons
            (List ([ Atom "i64.ctz" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list))
      | Popcnt ->
          List.cons
            (List ([ Atom "i64.popcnt" ] : W.t list) : W.t)
            (List.cons
               (List ([ Atom "i32.wrap_i64" ] : W.t list) : W.t)
               (rest : W.t list)))

let compile_comparison (ty : Primitive.operand_type) (op : Primitive.comparison)
    =
  match ty with
  | I32 -> (
      match op with
      | Lt -> (List ([ Atom "i32.lt_s" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "i32.le_s" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "i32.gt_s" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "i32.ge_s" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "i32.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "i32.ne" ] : W.t list) : W.t))
  | I64 -> (
      match op with
      | Lt -> (List ([ Atom "i64.lt_s" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "i64.le_s" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "i64.gt_s" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "i64.ge_s" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "i64.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "i64.ne" ] : W.t list) : W.t))
  | F32 -> (
      match op with
      | Lt -> (List ([ Atom "f32.lt" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "f32.le" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "f32.gt" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "f32.ge" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "f32.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "f32.ne" ] : W.t list) : W.t))
  | F64 -> (
      match op with
      | Lt -> (List ([ Atom "f64.lt" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "f64.le" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "f64.gt" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "f64.ge" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "f64.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "f64.ne" ] : W.t list) : W.t))
  | U8 -> (
      match op with
      | Lt -> (List ([ Atom "i32.lt_s" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "i32.le_s" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "i32.gt_s" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "i32.ge_s" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "i32.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "i32.ne" ] : W.t list) : W.t))
  | U32 -> (
      match op with
      | Lt -> (List ([ Atom "i32.lt_u" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "i32.le_u" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "i32.gt_u" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "i32.ge_u" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "i32.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "i32.ne" ] : W.t list) : W.t))
  | U64 -> (
      match op with
      | Lt -> (List ([ Atom "i64.lt_u" ] : W.t list) : W.t)
      | Le -> (List ([ Atom "i64.le_u" ] : W.t list) : W.t)
      | Gt -> (List ([ Atom "i64.gt_u" ] : W.t list) : W.t)
      | Ge -> (List ([ Atom "i64.ge_u" ] : W.t list) : W.t)
      | Eq -> (List ([ Atom "i64.eq" ] : W.t list) : W.t)
      | Ne -> (List ([ Atom "i64.ne" ] : W.t list) : W.t))

let compile_compare (ty : Primitive.operand_type) x y =
  match ty with
  | I32 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "i32.ge_s" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "i32.le_s" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | U32 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "i32.ge_u" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "i32.le_u" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | I64 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "i64.ge_s" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "i64.le_s" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | U64 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "i64.ge_u" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "i64.le_u" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | F32 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "f32.ge" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "f32.le" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | F64 ->
      (List
         (List.cons
            (Atom "i32.sub" : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "f64.ge" : W.t)
                     (List.cons (x : W.t) ([ y ] : W.t list)))
                 : W.t)
               ([
                  List
                    (List.cons
                       (Atom "f64.le" : W.t)
                       (List.cons (x : W.t) ([ y ] : W.t list)));
                ]
                 : W.t list)))
        : W.t)
  | U8 -> assert false

let compile_int_switch ~(result_ty : W.t list) ~(obj : W.t)
    ~(cases : (int * W.t list) list) ~(default : W.t list) (rest : W.t list) =
  let label_to_string = Label.to_wasm_name in
  let cases_len = List.length cases in
  let min_index, max_index =
    Lst.fold_left cases (Int.max_int, Int.min_int) (fun (min, max) (tag, _) ->
        (Int.min min tag, Int.max max tag))
  in
  if cases_len >= 3 && max_index - min_index + 1 = cases_len then
    let exit_label = Label.fresh "switch_int" in
    let default_label = Label.fresh "switch_default" in
    let label_table = Hash_int.create 17 in
    let labels =
      List.init cases_len (fun i ->
          let label =
            Label.fresh ("switch_int_" ^ Int.to_string i : Stdlib.String.t)
          in
          Hash_int.add label_table i label;
          (Atom (label_to_string label) : W.t))
    in
    let br_table =
      let br_table =
        ([
           List
             (List.cons
                (Atom "br_table" : W.t)
                (List.append
                   (labels : W.t list)
                   ([ Atom (label_to_string default_label) ] : W.t list)));
         ]
          : W.t list)
      in
      if min_index = 0 then List.cons (obj : W.t) (br_table : W.t list)
      else
        List.cons
          (obj : W.t)
          (List.cons
             (i32_to_sexp min_index : W.t)
             (List.cons
                (List ([ Atom "i32.sub" ] : W.t list) : W.t)
                (br_table : W.t list)))
    in
    let blocks =
      Lst.fold_left cases br_table (fun acc (lhs, act) ->
          let label = Hash_int.find_exn label_table (lhs - min_index) in
          List.cons
            (List
               (List.cons
                  (Atom "block" : W.t)
                  (List.cons
                     (Atom (label_to_string label) : W.t)
                     (acc : W.t list)))
              : W.t)
            (List.append
               (act : W.t list)
               ([
                  List
                    (List.cons
                       (Atom "br" : W.t)
                       ([ Atom (label_to_string exit_label) ] : W.t list));
                ]
                 : W.t list)))
    in
    let blocks_with_default =
      List.cons
        (List
           (List.cons
              (Atom "block" : W.t)
              (List.cons
                 (Atom (label_to_string default_label) : W.t)
                 (blocks : W.t list)))
          : W.t)
        (List.append
           (default : W.t list)
           ([
              List
                (List.cons
                   (Atom "br" : W.t)
                   ([ Atom (label_to_string exit_label) ] : W.t list));
            ]
             : W.t list))
    in
    List.cons
      (List
         (List.cons
            (Atom "block" : W.t)
            (List.cons
               (Atom (label_to_string exit_label) : W.t)
               (List.append
                  (result_ty : W.t list)
                  (blocks_with_default : W.t list))))
        : W.t)
      (rest : W.t list)
  else
    let switches =
      List.fold_right
        (fun (lhs, act) acc : W.t list ->
          [
            List
              (List.cons
                 (Atom "if" : W.t)
                 (List.append
                    (result_ty : W.t list)
                    (List.cons
                       (List
                          (List.cons
                             (Atom "i32.eq" : W.t)
                             (List.cons
                                (obj : W.t)
                                ([ i32_to_sexp lhs ] : W.t list)))
                         : W.t)
                       (List.cons
                          (List (List.cons (Atom "then" : W.t) (act : W.t list))
                            : W.t)
                          ([
                             List
                               (List.cons (Atom "else" : W.t) (acc : W.t list));
                           ]
                            : W.t list)))));
          ])
        cases default
    in
    List.append (switches : W.t list) (rest : W.t list)

let compile_source_pos ~pkg (pos : Lexing.position) =
  let file = pos.pos_fname in
  let line = pos.pos_lnum in
  let column = pos.pos_cnum - pos.pos_bol in
  let string_to_sexp s = W.Atom s in
  let int_to_sexp x = W.Atom (string_of_int x) in
  (List
     (List.cons
        (Atom "source_pos" : W.t)
        (List.cons
           (string_to_sexp pkg : W.t)
           (List.cons
              (string_to_sexp file : W.t)
              (List.cons
                 (int_to_sexp line : W.t)
                 ([ int_to_sexp column ] : W.t list)))))
    : W.t)

let prepend_source_pos ~get_pos (loc : Loc.t option) (v : W.t list) =
  if !Config.debug then
    match loc with
    | None -> v
    | Some loc ->
        let pos : Lexing.position = get_pos loc in
        if pos.pos_cnum = -1 then v
        else compile_source_pos ~pkg:(Loc.package loc) pos :: v
  else v

let prepend_start_source_pos (loc : Loc.t option) (v : W.t list) =
  prepend_source_pos ~get_pos:Loc.get_start loc v

let prepend_end_source_pos (loc : Loc.t option) (v : W.t list) =
  prepend_source_pos ~get_pos:Loc.get_end loc v

let compile_source_name (source_name : string) =
  if !Config.debug then
    ([
       List
         (List.cons
            (Atom "source_name" : W.t)
            ([ str_to_sexp source_name ] : W.t list));
     ]
      : W.t list)
  else []

let compile_fn_source_name (fn_addr : Fn_address.t) =
  if !Config.debug then
    let name = Fn_address.source_name fn_addr in
    ([
       List
         (List.cons
            (Atom "source_name" : W.t)
            ([ str_to_sexp name ] : W.t list));
     ]
      : W.t list)
  else []

let compile_source_type (typ : string) =
  if !Config.debug then
    ([ List (List.cons (Atom "source_type" : W.t) ([ Atom typ ] : W.t list)) ]
      : W.t list)
  else []

let compile_prologue_end () =
  if !Config.debug then
    ([ List ([ Atom "prologue_end" ] : W.t list) ] : W.t list)
  else []
