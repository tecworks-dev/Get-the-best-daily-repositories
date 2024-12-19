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


module UInt32 = Basic_uint32
module UInt64 = Basic_uint64
module BigInt = Basic_bigint
module Utf8 = Basic_utf8
module Vec_int = Basic_vec_int

type t =
  | C_bool of bool
  | C_char of Uchar.t
  | C_int of { v : int32; repr : string option [@ceh.ignore] }
  | C_int64 of { v : int64; repr : string option [@ceh.ignore] }
  | C_uint of { v : UInt32.t; repr : string option [@ceh.ignore] }
  | C_uint64 of { v : UInt64.t; repr : string option [@ceh.ignore] }
  | C_float of { v : float; repr : string option [@ceh.ignore] }
  | C_double of { v : float; repr : string option [@ceh.ignore] }
  | C_string of string
  | C_bytes of { v : string; repr : string option [@ceh.ignore] }
  | C_bigint of { v : BigInt.t; repr : string option [@ceh.ignore] }

include struct
  let _ = fun (_ : t) -> ()

  let compare =
    (fun a__001_ b__002_ ->
       if Stdlib.( == ) a__001_ b__002_ then 0
       else
         match (a__001_, b__002_) with
         | C_bool _a__003_, C_bool _b__004_ ->
             Stdlib.compare (_a__003_ : bool) _b__004_
         | C_bool _, _ -> -1
         | _, C_bool _ -> 1
         | C_char _a__005_, C_char _b__006_ -> Uchar.compare _a__005_ _b__006_
         | C_char _, _ -> -1
         | _, C_char _ -> 1
         | C_int _a__007_, C_int _b__008_ ->
             Stdlib.Int32.compare (_a__007_.v : int32) _b__008_.v
         | C_int _, _ -> -1
         | _, C_int _ -> 1
         | C_int64 _a__009_, C_int64 _b__010_ ->
             Stdlib.Int64.compare (_a__009_.v : int64) _b__010_.v
         | C_int64 _, _ -> -1
         | _, C_int64 _ -> 1
         | C_uint _a__011_, C_uint _b__012_ ->
             UInt32.compare _a__011_.v _b__012_.v
         | C_uint _, _ -> -1
         | _, C_uint _ -> 1
         | C_uint64 _a__013_, C_uint64 _b__014_ ->
             UInt64.compare _a__013_.v _b__014_.v
         | C_uint64 _, _ -> -1
         | _, C_uint64 _ -> 1
         | C_float _a__015_, C_float _b__016_ ->
             Stdlib.compare (_a__015_.v : float) _b__016_.v
         | C_float _, _ -> -1
         | _, C_float _ -> 1
         | C_double _a__017_, C_double _b__018_ ->
             Stdlib.compare (_a__017_.v : float) _b__018_.v
         | C_double _, _ -> -1
         | _, C_double _ -> 1
         | C_string _a__019_, C_string _b__020_ ->
             Stdlib.compare (_a__019_ : string) _b__020_
         | C_string _, _ -> -1
         | _, C_string _ -> 1
         | C_bytes _a__021_, C_bytes _b__022_ ->
             Stdlib.compare (_a__021_.v : string) _b__022_.v
         | C_bytes _, _ -> -1
         | _, C_bytes _ -> 1
         | C_bigint _a__023_, C_bigint _b__024_ ->
             BigInt.compare _a__023_.v _b__024_.v
      : t -> t -> int)

  let _ = compare

  let equal =
    (fun a__025_ b__026_ ->
       if Stdlib.( == ) a__025_ b__026_ then true
       else
         match (a__025_, b__026_) with
         | C_bool _a__027_, C_bool _b__028_ ->
             Stdlib.( = ) (_a__027_ : bool) _b__028_
         | C_bool _, _ -> false
         | _, C_bool _ -> false
         | C_char _a__029_, C_char _b__030_ -> Uchar.equal _a__029_ _b__030_
         | C_char _, _ -> false
         | _, C_char _ -> false
         | C_int _a__031_, C_int _b__032_ ->
             Stdlib.( = ) (_a__031_.v : int32) _b__032_.v
         | C_int _, _ -> false
         | _, C_int _ -> false
         | C_int64 _a__033_, C_int64 _b__034_ ->
             Stdlib.( = ) (_a__033_.v : int64) _b__034_.v
         | C_int64 _, _ -> false
         | _, C_int64 _ -> false
         | C_uint _a__035_, C_uint _b__036_ ->
             UInt32.equal _a__035_.v _b__036_.v
         | C_uint _, _ -> false
         | _, C_uint _ -> false
         | C_uint64 _a__037_, C_uint64 _b__038_ ->
             UInt64.equal _a__037_.v _b__038_.v
         | C_uint64 _, _ -> false
         | _, C_uint64 _ -> false
         | C_float _a__039_, C_float _b__040_ ->
             Stdlib.( = ) (_a__039_.v : float) _b__040_.v
         | C_float _, _ -> false
         | _, C_float _ -> false
         | C_double _a__041_, C_double _b__042_ ->
             Stdlib.( = ) (_a__041_.v : float) _b__042_.v
         | C_double _, _ -> false
         | _, C_double _ -> false
         | C_string _a__043_, C_string _b__044_ ->
             Stdlib.( = ) (_a__043_ : string) _b__044_
         | C_string _, _ -> false
         | _, C_string _ -> false
         | C_bytes _a__045_, C_bytes _b__046_ ->
             Stdlib.( = ) (_a__045_.v : string) _b__046_.v
         | C_bytes _, _ -> false
         | _, C_bytes _ -> false
         | C_bigint _a__047_, C_bigint _b__048_ ->
             BigInt.equal _a__047_.v _b__048_.v
      : t -> t -> bool)

  let _ = equal
end

let sexp_of_t = function
  | C_int { v; repr } -> (
      match repr with
      | Some s -> S.Atom s
      | None -> Moon_sexp_conv.sexp_of_int32 v)
  | C_int64 { v; repr } -> (
      match repr with
      | Some s -> S.Atom s
      | None -> Moon_sexp_conv.sexp_of_int64 v)
  | C_uint { v; repr } -> (
      match repr with Some s -> S.Atom s | None -> S.Atom (UInt32.to_string v))
  | C_uint64 { v; repr } -> (
      match repr with Some s -> S.Atom s | None -> S.Atom (UInt64.to_string v))
  | C_bool b -> Moon_sexp_conv.sexp_of_bool b
  | C_char c -> Basic_uchar_utils.sexp_of_uchar c
  | C_float { v; repr } -> (
      match repr with
      | Some s -> S.Atom s
      | None -> Moon_sexp_conv.sexp_of_float v)
  | C_double { v; repr } -> (
      match repr with
      | Some s -> S.Atom s
      | None -> Moon_sexp_conv.sexp_of_float v)
  | C_string s -> Moon_sexp_conv.sexp_of_string s
  | C_bytes { v; repr } -> (
      match repr with Some s -> S.Atom s | None -> S.Atom v)
  | C_bigint { v; repr } -> (
      match repr with Some s -> S.Atom s | None -> S.Atom (BigInt.to_string v))

let eval_arith (ty : Primitive.operand_type) (op : Primitive.arith_operator)
    (c1 : t) (c2 : t) =
  match (ty, c1, c2) with
  | I32, C_int { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_int i = Some (C_int { v = i; repr = None }) in
      match op with
      | Add -> make_int (Int32.add v1 v2)
      | Sub -> make_int (Int32.sub v1 v2)
      | Mul -> make_int (Int32.mul v1 v2)
      | Div -> if v2 <> 0l then make_int (Int32.div v1 v2) else None
      | Mod -> if v2 <> 0l then make_int (Int32.rem v1 v2) else None
      | Neg | Sqrt -> None)
  | I64, C_int64 { v = v1; repr = _ }, C_int64 { v = v2; repr = _ } -> (
      let make_int64 i = Some (C_int64 { v = i; repr = None }) in
      match op with
      | Add -> make_int64 (Int64.add v1 v2)
      | Sub -> make_int64 (Int64.sub v1 v2)
      | Mul -> make_int64 (Int64.mul v1 v2)
      | Div -> if v2 <> 0L then make_int64 (Int64.div v1 v2) else None
      | Mod -> if v2 <> 0L then make_int64 (Int64.rem v1 v2) else None
      | Neg | Sqrt -> None)
  | F64, C_double { v = v1; repr = _ }, C_double { v = v2; repr = _ } -> (
      let make_float i = Some (C_double { v = i; repr = None }) in
      match op with
      | Add -> make_float (v1 +. v2)
      | Sub -> make_float (v1 -. v2)
      | Mul -> make_float (v1 *. v2)
      | Div -> if v2 <> 0. then make_float (v1 /. v2) else None
      | Mod | Neg | Sqrt -> None)
  | _ -> None

let eval_comparison (ty : Primitive.operand_type) (op : Primitive.comparison)
    (c1 : t) (c2 : t) =
  let make_bool b = Some (C_bool b) [@@local] in
  match (ty, c1, c2) with
  | I32, C_int { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      match op with
      | Lt -> make_bool (Int32.compare v1 v2 < 0)
      | Gt -> make_bool (Int32.compare v1 v2 > 0)
      | Le -> make_bool (Int32.compare v1 v2 <= 0)
      | Ge -> make_bool (Int32.compare v1 v2 >= 0)
      | Eq -> make_bool (Int32.compare v1 v2 = 0)
      | Ne -> make_bool (Int32.compare v1 v2 <> 0))
  | I64, C_int64 { v = v1; repr = _ }, C_int64 { v = v2; repr = _ } -> (
      match op with
      | Lt -> make_bool (Int64.compare v1 v2 < 0)
      | Gt -> make_bool (Int64.compare v1 v2 > 0)
      | Le -> make_bool (Int64.compare v1 v2 <= 0)
      | Ge -> make_bool (Int64.compare v1 v2 >= 0)
      | Eq -> make_bool (Int64.compare v1 v2 = 0)
      | Ne -> make_bool (Int64.compare v1 v2 <> 0))
  | F64, C_double { v = v1; repr = _ }, C_double { v = v2; repr = _ } -> (
      match op with
      | Lt -> make_bool (Float.compare v1 v2 < 0)
      | Gt -> make_bool (Float.compare v1 v2 > 0)
      | Le -> make_bool (Float.compare v1 v2 <= 0)
      | Ge -> make_bool (Float.compare v1 v2 >= 0)
      | Eq -> make_bool (Float.compare v1 v2 = 0)
      | Ne -> make_bool (Float.compare v1 v2 <> 0))
  | U32, C_uint { v = v1; repr = _ }, C_uint { v = v2; repr = _ } -> (
      match op with
      | Lt -> make_bool (UInt32.compare v1 v2 < 0)
      | Gt -> make_bool (UInt32.compare v1 v2 > 0)
      | Le -> make_bool (UInt32.compare v1 v2 <= 0)
      | Ge -> make_bool (UInt32.compare v1 v2 >= 0)
      | Eq -> make_bool (UInt32.compare v1 v2 = 0)
      | Ne -> make_bool (UInt32.compare v1 v2 <> 0))
  | U64, C_uint64 { v = v1; repr = _ }, C_uint64 { v = v2; repr = _ } -> (
      match op with
      | Lt -> make_bool (UInt64.compare v1 v2 < 0)
      | Gt -> make_bool (UInt64.compare v1 v2 > 0)
      | Le -> make_bool (UInt64.compare v1 v2 <= 0)
      | Ge -> make_bool (UInt64.compare v1 v2 >= 0)
      | Eq -> make_bool (UInt64.compare v1 v2 = 0)
      | Ne -> make_bool (UInt64.compare v1 v2 <> 0))
  | _ -> None

let eval_bitwise (ty : Primitive.operand_type) (op : Primitive.bitwise_operator)
    (c1 : t) (c2 : t) =
  match (ty, c1, c2) with
  | I32, C_int { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_int i = Some (C_int { v = i; repr = None }) in
      match op with
      | Not -> make_int (Int32.lognot v1)
      | And -> make_int (Int32.logand v1 v2)
      | Or -> make_int (Int32.logor v1 v2)
      | Xor -> make_int (Int32.logxor v1 v2)
      | Shl -> make_int (Int32.shift_left v1 (Int32.to_int v2 mod 32))
      | Shr -> make_int (Int32.shift_right v1 (Int32.to_int v2 mod 32))
      | _ -> None)
  | I64, C_int64 { v = v1; repr = _ }, C_int64 { v = v2; repr = _ } -> (
      let make_int64 i = Some (C_int64 { v = i; repr = None }) in
      match op with
      | Not -> make_int64 (Int64.lognot v1)
      | And -> make_int64 (Int64.logand v1 v2)
      | Or -> make_int64 (Int64.logor v1 v2)
      | Xor -> make_int64 (Int64.logxor v1 v2)
      | _ -> None)
  | U32, C_uint { v = v1; repr = _ }, C_uint { v = v2; repr = _ } -> (
      let make_int i = Some (C_uint { v = i; repr = None }) in
      match op with
      | Not -> make_int (UInt32.lognot v1)
      | And -> make_int (UInt32.logand v1 v2)
      | Or -> make_int (UInt32.logor v1 v2)
      | Xor -> make_int (UInt32.logxor v1 v2)
      | _ -> None)
  | U64, C_uint64 { v = v1; repr = _ }, C_uint64 { v = v2; repr = _ } -> (
      let make_uint64 i = Some (C_uint64 { v = i; repr = None }) in
      match op with
      | Not -> make_uint64 (UInt64.lognot v1)
      | And -> make_uint64 (UInt64.logand v1 v2)
      | Or -> make_uint64 (UInt64.logor v1 v2)
      | Xor -> make_uint64 (UInt64.logxor v1 v2)
      | _ -> None)
  | I64, C_int64 { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_int64 i = Some (C_int64 { v = i; repr = None }) in
      match op with
      | Shl -> make_int64 (Int64.shift_left v1 (Int32.to_int v2 mod 64))
      | Shr -> make_int64 (Int64.shift_right v1 (Int32.to_int v2 mod 64))
      | _ -> None)
  | U64, C_uint64 { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_uint64 i = Some (C_uint64 { v = i; repr = None }) in
      match op with
      | Shl -> make_uint64 (UInt64.shift_left v1 (Int32.to_int v2 mod 64))
      | Shr -> make_uint64 (UInt64.shift_right v1 (Int32.to_int v2 mod 64))
      | _ -> None)
  | U32, C_uint { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_uint i = Some (C_uint { v = i; repr = None }) in
      match op with
      | Shl -> make_uint (UInt32.shift_left v1 (Int32.to_int v2 mod 32))
      | Shr -> make_uint (UInt32.shift_right v1 (Int32.to_int v2 mod 32))
      | _ -> None)
  | U32, C_int { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_uint i = Some (C_int { v = i; repr = None }) in
      match op with
      | Shr -> make_uint (Int32.shift_right_logical v1 (Int32.to_int v2 mod 32))
      | _ -> None)
  | U64, C_int64 { v = v1; repr = _ }, C_int { v = v2; repr = _ } -> (
      let make_uint64 i = Some (C_int64 { v = i; repr = None }) in
      match op with
      | Shr ->
          make_uint64 (Int64.shift_right_logical v1 (Int32.to_int v2 mod 64))
      | _ -> None)
  | _ -> None

let eval_compare c1 c2 : int option =
  match c1 with
  | C_int { v = v1; repr = _ } -> (
      match c2 with
      | C_int { v = v2; repr = _ } -> Some (Int32.compare v1 v2)
      | _ -> None)
  | C_int64 { v = v1; repr = _ } -> (
      match c2 with
      | C_int64 { v = v2; repr = _ } -> Some (Int64.compare v1 v2)
      | _ -> None)
  | C_uint { v = v1; repr = _ } -> (
      match c2 with
      | C_uint { v = v2; repr = _ } -> Some (Basic_uint32.compare v1 v2)
      | _ -> None)
  | C_uint64 { v = v1; repr = _ } -> (
      match c2 with
      | C_uint64 { v = v2; repr = _ } -> Some (Basic_uint64.compare v1 v2)
      | _ -> None)
  | C_char c1 -> (
      match c2 with C_char c2 -> Some (Uchar.compare c1 c2) | _ -> None)
  | C_float { v = v1; repr = _ } -> (
      match c2 with
      | C_float { v = v2; repr = _ } -> Some (Float.compare v1 v2)
      | _ -> None)
  | C_double { v = v1; repr = _ } -> (
      match c2 with
      | C_double { v = v2; repr = _ } -> Some (Float.compare v1 v2)
      | _ -> None)
  | C_string _ | C_bytes _ | C_bool _ | C_bigint _ -> None

let to_string (c : t) =
  match c with
  | C_bool true -> "true"
  | C_bool false -> "false"
  | C_int { repr = Some repr; v = _ }
  | C_uint { repr = Some repr; v = _ }
  | C_int64 { repr = Some repr; v = _ }
  | C_uint64 { repr = Some repr; v = _ }
  | C_float { repr = Some repr; v = _ }
  | C_double { repr = Some repr; v = _ }
  | C_bigint { repr = Some repr; v = _ }
  | C_bytes { repr = Some repr; v = _ } ->
      repr
  | C_int { v; repr = None } -> Int32.to_string v
  | C_uint { v; repr = None } -> UInt32.to_string v
  | C_int64 { v; repr = None } -> Int64.to_string v
  | C_uint64 { v; repr = None } -> UInt64.to_string v
  | C_bigint { v; repr = None } -> BigInt.to_string v
  | C_float { v; repr = None } | C_double { v; repr = None } ->
      string_of_float v
  | C_bytes { v; repr = None } ->
      let buf = Buffer.create ((String.length v * 4) + 3) in
      Buffer.add_string buf "b\"";
      for i = 0 to String.length v - 1 do
        Printf.bprintf buf "\\x%02d" (Char.code v.[i])
      done;
      Buffer.add_char buf '"';
      Buffer.contents buf
  | C_char c -> (
      let code = Uchar.to_int c in
      match Char.unsafe_chr code with
      | '\'' -> "'\\''"
      | '\\' -> "'\\\\'"
      | '\n' -> "'\\n'"
      | '\r' -> "'\\r'"
      | '\b' -> "'\\b'"
      | '\t' -> "'\\t'"
      | _ when code < 0x20 -> Printf.sprintf "'\\x%02x'" code
      | _ ->
          let buf = Buffer.create 3 in
          Buffer.add_char buf '\'';
          Buffer.add_utf_8_uchar buf c;
          Buffer.add_char buf '\'';
          Buffer.contents buf)
  | C_string s ->
      let buf = Buffer.create (String.length s + 2) in
      Buffer.add_char buf '"';
      let code_points = Utf8.from_string s in
      Vec_int.iter code_points (fun code ->
          match Char.unsafe_chr code with
          | '"' -> Buffer.add_string buf "\\\""
          | '\\' -> Buffer.add_string buf "\\\\"
          | '\n' -> Buffer.add_string buf "\\n"
          | '\r' -> Buffer.add_string buf "\\r"
          | '\b' -> Buffer.add_string buf "\\b"
          | '\t' -> Buffer.add_string buf "\\t"
          | _ when code < 0x20 -> Printf.bprintf buf "\\x%02x" code
          | _ -> Buffer.add_utf_8_uchar buf (Uchar.of_int code));
      Buffer.add_char buf '"';
      Buffer.contents buf
