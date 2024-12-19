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


module Constr_info = Basic_constr_info
module Hash_string = Basic_hash_string

type operand_type = I32 | I64 | U32 | U64 | F64 | F32 | U8
and convert_kind = Convert | Reinterpret
and arith_operator = Add | Sub | Mul | Div | Mod | Neg | Sqrt
and bitwise_operator = Not | And | Or | Xor | Shl | Shr | Ctz | Clz | Popcnt
and comparison = Lt | Le | Gt | Ge | Eq | Ne
and cast_kind = Constr_to_enum | Unfold_rec_newtype | Make_newtype

and prim =
  | Pfixedarray_length
  | Pccall of { arity : int; func_name : string }
  | Pintrinsic of Moon_intrinsic.t
  | Pgetstringitem
  | Pignore
  | Pidentity
  | Pmakebytes
  | Pbyteslength
  | Pgetbytesitem
  | Psetbytesitem
  | Pnot
  | Praise
  | Ppanic
  | Punreachable
  | Pcatch
  | Psequand
  | Psequor
  | Pstringlength
  | Pstringequal
  | Pcast of { kind : cast_kind }
  | Pconvert of { kind : convert_kind; from : operand_type; to_ : operand_type }
  | Parith of { operand_type : operand_type; operator : arith_operator }
  | Pbitwise of { operand_type : operand_type; operator : bitwise_operator }
  | Pcomparison of { operand_type : operand_type; operator : comparison }
  | Pcompare of operand_type
  | Pfixedarray_make of { kind : make_array_kind }
  | Pfixedarray_get_item of { kind : array_get_kind }
  | Pfixedarray_set_item of { set_kind : array_set_kind }
  | Penum_field of { index : int; tag : Constr_info.constr_tag }
  | Pset_enum_field of { index : int; tag : Constr_info.constr_tag }
  | Pclosure_to_extern_ref
  | Prefeq
  | Parray_make
  | Pnull
  | Pnull_string_extern
  | Pis_null
  | Pas_non_null
  | Pmake_value_or_error of { tag : Basic_constr_info.constr_tag }
  | Pprintln
  | Perror_to_string
  | Pcall_object_method of { method_index : int; method_name : string }
  | Pany_to_string

and make_array_kind = LenAndInit | EverySingleElem | Uninit
and array_get_kind = Safe | Unsafe | Rev_unsafe
and array_set_kind = Null | Default | Value | Unsafe

include struct
  let _ = fun (_ : operand_type) -> ()
  let _ = fun (_ : convert_kind) -> ()
  let _ = fun (_ : arith_operator) -> ()
  let _ = fun (_ : bitwise_operator) -> ()
  let _ = fun (_ : comparison) -> ()
  let _ = fun (_ : cast_kind) -> ()
  let _ = fun (_ : prim) -> ()
  let _ = fun (_ : make_array_kind) -> ()
  let _ = fun (_ : array_get_kind) -> ()
  let _ = fun (_ : array_set_kind) -> ()

  let rec sexp_of_operand_type =
    (function
     | I32 -> S.Atom "I32"
     | I64 -> S.Atom "I64"
     | U32 -> S.Atom "U32"
     | U64 -> S.Atom "U64"
     | F64 -> S.Atom "F64"
     | F32 -> S.Atom "F32"
     | U8 -> S.Atom "U8"
      : operand_type -> S.t)

  and sexp_of_convert_kind =
    (function
     | Convert -> S.Atom "Convert" | Reinterpret -> S.Atom "Reinterpret"
      : convert_kind -> S.t)

  and sexp_of_arith_operator =
    (function
     | Add -> S.Atom "Add"
     | Sub -> S.Atom "Sub"
     | Mul -> S.Atom "Mul"
     | Div -> S.Atom "Div"
     | Mod -> S.Atom "Mod"
     | Neg -> S.Atom "Neg"
     | Sqrt -> S.Atom "Sqrt"
      : arith_operator -> S.t)

  and sexp_of_bitwise_operator =
    (function
     | Not -> S.Atom "Not"
     | And -> S.Atom "And"
     | Or -> S.Atom "Or"
     | Xor -> S.Atom "Xor"
     | Shl -> S.Atom "Shl"
     | Shr -> S.Atom "Shr"
     | Ctz -> S.Atom "Ctz"
     | Clz -> S.Atom "Clz"
     | Popcnt -> S.Atom "Popcnt"
      : bitwise_operator -> S.t)

  and sexp_of_comparison =
    (function
     | Lt -> S.Atom "Lt"
     | Le -> S.Atom "Le"
     | Gt -> S.Atom "Gt"
     | Ge -> S.Atom "Ge"
     | Eq -> S.Atom "Eq"
     | Ne -> S.Atom "Ne"
      : comparison -> S.t)

  and sexp_of_cast_kind =
    (function
     | Constr_to_enum -> S.Atom "Constr_to_enum"
     | Unfold_rec_newtype -> S.Atom "Unfold_rec_newtype"
     | Make_newtype -> S.Atom "Make_newtype"
      : cast_kind -> S.t)

  and sexp_of_prim =
    (function
     | Pfixedarray_length -> S.Atom "Pfixedarray_length"
     | Pccall { arity = arity__002_; func_name = func_name__004_ } ->
         let bnds__001_ = ([] : _ Stdlib.List.t) in
         let bnds__001_ =
           let arg__005_ = Moon_sexp_conv.sexp_of_string func_name__004_ in
           (S.List [ S.Atom "func_name"; arg__005_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         let bnds__001_ =
           let arg__003_ = Moon_sexp_conv.sexp_of_int arity__002_ in
           (S.List [ S.Atom "arity"; arg__003_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pccall" :: bnds__001_)
     | Pintrinsic arg0__006_ ->
         let res0__007_ = Moon_intrinsic.sexp_of_t arg0__006_ in
         S.List [ S.Atom "Pintrinsic"; res0__007_ ]
     | Pgetstringitem -> S.Atom "Pgetstringitem"
     | Pignore -> S.Atom "Pignore"
     | Pidentity -> S.Atom "Pidentity"
     | Pmakebytes -> S.Atom "Pmakebytes"
     | Pbyteslength -> S.Atom "Pbyteslength"
     | Pgetbytesitem -> S.Atom "Pgetbytesitem"
     | Psetbytesitem -> S.Atom "Psetbytesitem"
     | Pnot -> S.Atom "Pnot"
     | Praise -> S.Atom "Praise"
     | Ppanic -> S.Atom "Ppanic"
     | Punreachable -> S.Atom "Punreachable"
     | Pcatch -> S.Atom "Pcatch"
     | Psequand -> S.Atom "Psequand"
     | Psequor -> S.Atom "Psequor"
     | Pstringlength -> S.Atom "Pstringlength"
     | Pstringequal -> S.Atom "Pstringequal"
     | Pcast { kind = kind__009_ } ->
         let bnds__008_ = ([] : _ Stdlib.List.t) in
         let bnds__008_ =
           let arg__010_ = sexp_of_cast_kind kind__009_ in
           (S.List [ S.Atom "kind"; arg__010_ ] :: bnds__008_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pcast" :: bnds__008_)
     | Pconvert { kind = kind__012_; from = from__014_; to_ = to___016_ } ->
         let bnds__011_ = ([] : _ Stdlib.List.t) in
         let bnds__011_ =
           let arg__017_ = sexp_of_operand_type to___016_ in
           (S.List [ S.Atom "to_"; arg__017_ ] :: bnds__011_ : _ Stdlib.List.t)
         in
         let bnds__011_ =
           let arg__015_ = sexp_of_operand_type from__014_ in
           (S.List [ S.Atom "from"; arg__015_ ] :: bnds__011_ : _ Stdlib.List.t)
         in
         let bnds__011_ =
           let arg__013_ = sexp_of_convert_kind kind__012_ in
           (S.List [ S.Atom "kind"; arg__013_ ] :: bnds__011_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pconvert" :: bnds__011_)
     | Parith { operand_type = operand_type__019_; operator = operator__021_ }
       ->
         let bnds__018_ = ([] : _ Stdlib.List.t) in
         let bnds__018_ =
           let arg__022_ = sexp_of_arith_operator operator__021_ in
           (S.List [ S.Atom "operator"; arg__022_ ] :: bnds__018_
             : _ Stdlib.List.t)
         in
         let bnds__018_ =
           let arg__020_ = sexp_of_operand_type operand_type__019_ in
           (S.List [ S.Atom "operand_type"; arg__020_ ] :: bnds__018_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Parith" :: bnds__018_)
     | Pbitwise { operand_type = operand_type__024_; operator = operator__026_ }
       ->
         let bnds__023_ = ([] : _ Stdlib.List.t) in
         let bnds__023_ =
           let arg__027_ = sexp_of_bitwise_operator operator__026_ in
           (S.List [ S.Atom "operator"; arg__027_ ] :: bnds__023_
             : _ Stdlib.List.t)
         in
         let bnds__023_ =
           let arg__025_ = sexp_of_operand_type operand_type__024_ in
           (S.List [ S.Atom "operand_type"; arg__025_ ] :: bnds__023_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pbitwise" :: bnds__023_)
     | Pcomparison
         { operand_type = operand_type__029_; operator = operator__031_ } ->
         let bnds__028_ = ([] : _ Stdlib.List.t) in
         let bnds__028_ =
           let arg__032_ = sexp_of_comparison operator__031_ in
           (S.List [ S.Atom "operator"; arg__032_ ] :: bnds__028_
             : _ Stdlib.List.t)
         in
         let bnds__028_ =
           let arg__030_ = sexp_of_operand_type operand_type__029_ in
           (S.List [ S.Atom "operand_type"; arg__030_ ] :: bnds__028_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pcomparison" :: bnds__028_)
     | Pcompare arg0__033_ ->
         let res0__034_ = sexp_of_operand_type arg0__033_ in
         S.List [ S.Atom "Pcompare"; res0__034_ ]
     | Pfixedarray_make { kind = kind__036_ } ->
         let bnds__035_ = ([] : _ Stdlib.List.t) in
         let bnds__035_ =
           let arg__037_ = sexp_of_make_array_kind kind__036_ in
           (S.List [ S.Atom "kind"; arg__037_ ] :: bnds__035_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pfixedarray_make" :: bnds__035_)
     | Pfixedarray_get_item { kind = kind__039_ } ->
         let bnds__038_ = ([] : _ Stdlib.List.t) in
         let bnds__038_ =
           let arg__040_ = sexp_of_array_get_kind kind__039_ in
           (S.List [ S.Atom "kind"; arg__040_ ] :: bnds__038_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pfixedarray_get_item" :: bnds__038_)
     | Pfixedarray_set_item { set_kind = set_kind__042_ } ->
         let bnds__041_ = ([] : _ Stdlib.List.t) in
         let bnds__041_ =
           let arg__043_ = sexp_of_array_set_kind set_kind__042_ in
           (S.List [ S.Atom "set_kind"; arg__043_ ] :: bnds__041_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pfixedarray_set_item" :: bnds__041_)
     | Penum_field { index = index__045_; tag = tag__047_ } ->
         let bnds__044_ = ([] : _ Stdlib.List.t) in
         let bnds__044_ =
           let arg__048_ = Constr_info.sexp_of_constr_tag tag__047_ in
           (S.List [ S.Atom "tag"; arg__048_ ] :: bnds__044_ : _ Stdlib.List.t)
         in
         let bnds__044_ =
           let arg__046_ = Moon_sexp_conv.sexp_of_int index__045_ in
           (S.List [ S.Atom "index"; arg__046_ ] :: bnds__044_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Penum_field" :: bnds__044_)
     | Pset_enum_field { index = index__050_; tag = tag__052_ } ->
         let bnds__049_ = ([] : _ Stdlib.List.t) in
         let bnds__049_ =
           let arg__053_ = Constr_info.sexp_of_constr_tag tag__052_ in
           (S.List [ S.Atom "tag"; arg__053_ ] :: bnds__049_ : _ Stdlib.List.t)
         in
         let bnds__049_ =
           let arg__051_ = Moon_sexp_conv.sexp_of_int index__050_ in
           (S.List [ S.Atom "index"; arg__051_ ] :: bnds__049_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pset_enum_field" :: bnds__049_)
     | Pclosure_to_extern_ref -> S.Atom "Pclosure_to_extern_ref"
     | Prefeq -> S.Atom "Prefeq"
     | Parray_make -> S.Atom "Parray_make"
     | Pnull -> S.Atom "Pnull"
     | Pnull_string_extern -> S.Atom "Pnull_string_extern"
     | Pis_null -> S.Atom "Pis_null"
     | Pas_non_null -> S.Atom "Pas_non_null"
     | Pmake_value_or_error { tag = tag__055_ } ->
         let bnds__054_ = ([] : _ Stdlib.List.t) in
         let bnds__054_ =
           let arg__056_ = Basic_constr_info.sexp_of_constr_tag tag__055_ in
           (S.List [ S.Atom "tag"; arg__056_ ] :: bnds__054_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pmake_value_or_error" :: bnds__054_)
     | Pprintln -> S.Atom "Pprintln"
     | Perror_to_string -> S.Atom "Perror_to_string"
     | Pcall_object_method
         { method_index = method_index__058_; method_name = method_name__060_ }
       ->
         let bnds__057_ = ([] : _ Stdlib.List.t) in
         let bnds__057_ =
           let arg__061_ = Moon_sexp_conv.sexp_of_string method_name__060_ in
           (S.List [ S.Atom "method_name"; arg__061_ ] :: bnds__057_
             : _ Stdlib.List.t)
         in
         let bnds__057_ =
           let arg__059_ = Moon_sexp_conv.sexp_of_int method_index__058_ in
           (S.List [ S.Atom "method_index"; arg__059_ ] :: bnds__057_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Pcall_object_method" :: bnds__057_)
     | Pany_to_string -> S.Atom "Pany_to_string"
      : prim -> S.t)

  and sexp_of_make_array_kind =
    (function
     | LenAndInit -> S.Atom "LenAndInit"
     | EverySingleElem -> S.Atom "EverySingleElem"
     | Uninit -> S.Atom "Uninit"
      : make_array_kind -> S.t)

  and sexp_of_array_get_kind =
    (function
     | Safe -> S.Atom "Safe"
     | Unsafe -> S.Atom "Unsafe"
     | Rev_unsafe -> S.Atom "Rev_unsafe"
      : array_get_kind -> S.t)

  and sexp_of_array_set_kind =
    (function
     | Null -> S.Atom "Null"
     | Default -> S.Atom "Default"
     | Value -> S.Atom "Value"
     | Unsafe -> S.Atom "Unsafe"
      : array_set_kind -> S.t)

  let _ = sexp_of_operand_type
  and _ = sexp_of_convert_kind
  and _ = sexp_of_arith_operator
  and _ = sexp_of_bitwise_operator
  and _ = sexp_of_comparison
  and _ = sexp_of_cast_kind
  and _ = sexp_of_prim
  and _ = sexp_of_make_array_kind
  and _ = sexp_of_array_get_kind
  and _ = sexp_of_array_set_kind
end

let sexp_of_prim (x : prim) : S.t =
  match x with
  | Pccall { func_name; _ } -> Atom ("@" ^ func_name)
  | Parith { operand_type; operator } ->
      (List
         (List.cons
            (Atom "Parith" : S.t)
            (List.cons
               (sexp_of_operand_type operand_type : S.t)
               ([ sexp_of_arith_operator operator ] : S.t list)))
        : S.t)
  | Pcomparison { operand_type; operator } ->
      (List
         (List.cons
            (Atom "Pcomparison" : S.t)
            (List.cons
               (sexp_of_operand_type operand_type : S.t)
               ([ sexp_of_comparison operator ] : S.t list)))
        : S.t)
  | _ -> sexp_of_prim x

let compare_int = Pcompare I32
let compare_int64 = Pcompare I64
let compare_uint = Pcompare U32
let compare_uint64 = Pcompare U64
let compare_bool = compare_int
let compare_char = compare_int
let compare_float = Pcompare F64
let compare_float32 = Pcompare F32
let equal_int = Pcomparison { operand_type = I32; operator = Eq }
let equal_int64 = Pcomparison { operand_type = I64; operator = Eq }
let equal_uint = equal_int
let equal_uint64 = equal_int64
let equal_bool = equal_int
let equal_char = equal_int
let equal_float = Pcomparison { operand_type = F32; operator = Eq }
let equal_float64 = Pcomparison { operand_type = F64; operator = Eq }
let equal_string = Pstringequal
let ge_int = Pcomparison { operand_type = I32; operator = Ge }
let le_int = Pcomparison { operand_type = I32; operator = Le }
let gt_int = Pcomparison { operand_type = I32; operator = Gt }
let lt_int = Pcomparison { operand_type = I32; operator = Lt }
let ne_int = Pcomparison { operand_type = I32; operator = Ne }
let eq_int = Pcomparison { operand_type = I32; operator = Eq }
let ge_uint = Pcomparison { operand_type = U32; operator = Ge }
let le_uint = Pcomparison { operand_type = U32; operator = Le }
let gt_uint = Pcomparison { operand_type = U32; operator = Gt }
let lt_uint = Pcomparison { operand_type = U32; operator = Lt }
let ne_uint = Pcomparison { operand_type = U32; operator = Ne }
let ge_int64 = Pcomparison { operand_type = I64; operator = Ge }
let le_int64 = Pcomparison { operand_type = I64; operator = Le }
let gt_int64 = Pcomparison { operand_type = I64; operator = Gt }
let lt_int64 = Pcomparison { operand_type = I64; operator = Lt }
let ne_int64 = Pcomparison { operand_type = I64; operator = Ne }
let eq_int64 = Pcomparison { operand_type = I64; operator = Eq }
let ge_uint64 = Pcomparison { operand_type = U64; operator = Ge }
let le_uint64 = Pcomparison { operand_type = U64; operator = Le }
let gt_uint64 = Pcomparison { operand_type = U64; operator = Gt }
let lt_uint64 = Pcomparison { operand_type = U64; operator = Lt }
let ne_uint64 = Pcomparison { operand_type = U64; operator = Ne }
let ge_float = Pcomparison { operand_type = F32; operator = Ge }
let le_float = Pcomparison { operand_type = F32; operator = Le }
let gt_float = Pcomparison { operand_type = F32; operator = Gt }
let lt_float = Pcomparison { operand_type = F32; operator = Lt }
let ne_float = Pcomparison { operand_type = F32; operator = Ne }
let ge_double = Pcomparison { operand_type = F64; operator = Ge }
let le_double = Pcomparison { operand_type = F64; operator = Le }
let gt_double = Pcomparison { operand_type = F64; operator = Gt }
let lt_double = Pcomparison { operand_type = F64; operator = Lt }
let ne_double = Pcomparison { operand_type = F64; operator = Ne }
let add_string = Pccall { func_name = "add_string"; arity = 2 }
let incref = Pccall { func_name = "incref"; arity = 1 }
let drop = Pccall { func_name = "decref"; arity = 1 }

let prim_table =
  Hash_string.of_list
    [
      ("%ignore", Pignore);
      ("%identity", Pidentity);
      ("%refeq", Prefeq);
      ("%loc_to_string", Pidentity);
      ( "%unsafe_obj_block_tag",
        Pccall { func_name = "unsafe_obj_block_tag"; arity = 1 } );
      ( "%unsafe_obj_block_length",
        Pccall { func_name = "unsafe_obj_block_length"; arity = 1 } );
      ( "%unsafe_obj_get_field",
        Pccall { func_name = "unsafe_obj_get_field"; arity = 2 } );
      ("%println", Pprintln);
      ("%abort", Praise);
      ("%panic", Ppanic);
      ("%unreachable", Punreachable);
      ("%control.catch", Pcatch);
      ("%bool_not", Pnot);
      ("%bool_eq", equal_bool);
      ("%bool_compare", compare_bool);
      ("%bool_default", Pccall { func_name = "default_bool"; arity = 0 });
      ("%loc_ghost", Pccall { arity = 0; func_name = "loc_ghost" });
      ("%loc_to_string", Pidentity);
      ("%string_length", Pstringlength);
      ("%string_get", Pgetstringitem);
      ("%string_add", add_string);
      ("%string_eq", equal_string);
      ("%string_to_string", Pidentity);
      ("%fixedarray.get", Pfixedarray_get_item { kind = Safe });
      ("%fixedarray.unsafe_get", Pfixedarray_get_item { kind = Unsafe });
      ("%fixedarray.set", Pfixedarray_set_item { set_kind = Value });
      ("%fixedarray.unsafe_set", Pfixedarray_set_item { set_kind = Unsafe });
      ("%fixedarray.set_default", Pfixedarray_set_item { set_kind = Default });
      ("%fixedarray.set_null", Pfixedarray_set_item { set_kind = Null });
      ("%fixedarray.length", Pfixedarray_length);
      ("%fixedarray.make", Pfixedarray_make { kind = LenAndInit });
      ("%fixedarray.make_uninit", Pfixedarray_make { kind = Uninit });
      ("%bytes_get", Pgetbytesitem);
      ("%bytes_set", Psetbytesitem);
      ("%bytes_make", Pmakebytes);
      ("%bytes_length", Pbyteslength);
      ("%i32_neg", Parith { operand_type = I32; operator = Neg });
      ("%i32_add", Parith { operand_type = I32; operator = Add });
      ("%i32_sub", Parith { operand_type = I32; operator = Sub });
      ("%i32_mul", Parith { operand_type = I32; operator = Mul });
      ("%i32_div", Parith { operand_type = I32; operator = Div });
      ("%i32_mod", Parith { operand_type = I32; operator = Mod });
      ("%i32_lnot", Pbitwise { operand_type = I32; operator = Not });
      ("%i32_land", Pbitwise { operand_type = I32; operator = And });
      ("%i32_lor", Pbitwise { operand_type = I32; operator = Or });
      ("%i32_lxor", Pbitwise { operand_type = I32; operator = Xor });
      ("%i32_shl", Pbitwise { operand_type = I32; operator = Shl });
      ("%i32_shr", Pbitwise { operand_type = I32; operator = Shr });
      ("%i32_ctz", Pbitwise { operand_type = I32; operator = Ctz });
      ("%i32_clz", Pbitwise { operand_type = I32; operator = Clz });
      ("%i32_popcnt", Pbitwise { operand_type = I32; operator = Popcnt });
      ("%i32_eq", Pcomparison { operand_type = I32; operator = Eq });
      ("%i32_ne", Pcomparison { operand_type = I32; operator = Ne });
      ("%i32_compare", compare_int);
      ("%i32_is_pos", Pccall { func_name = "int_is_pos"; arity = 1 });
      ("%i32_is_neg", Pccall { func_name = "int_is_neg"; arity = 1 });
      ("%i32_is_non_pos", Pccall { func_name = "int_is_non_pos"; arity = 1 });
      ("%i32_is_non_neg", Pccall { func_name = "int_is_non_neg"; arity = 1 });
      ("%i32_default", Pccall { func_name = "default_int"; arity = 0 });
      ("%u32.add", Parith { operand_type = U32; operator = Add });
      ("%u32.sub", Parith { operand_type = U32; operator = Sub });
      ("%u32.mul", Parith { operand_type = U32; operator = Mul });
      ("%u32.div", Parith { operand_type = U32; operator = Div });
      ("%u32.mod", Parith { operand_type = U32; operator = Mod });
      ("%u32.eq", Pcomparison { operand_type = U32; operator = Eq });
      ("%u32.ne", Pcomparison { operand_type = U32; operator = Ne });
      ("%u32.compare", compare_uint);
      ("%u32.bitand", Pbitwise { operand_type = U32; operator = And });
      ("%u32.bitnot", Pbitwise { operand_type = U32; operator = Not });
      ("%u32.bitor", Pbitwise { operand_type = U32; operator = Or });
      ("%u32.bitxor", Pbitwise { operand_type = U32; operator = Xor });
      ("%u32.shl", Pbitwise { operand_type = U32; operator = Shl });
      ("%u32.shr", Pbitwise { operand_type = U32; operator = Shr });
      ("%u32.clz", Pbitwise { operand_type = U32; operator = Clz });
      ("%u32.ctz", Pbitwise { operand_type = U32; operator = Ctz });
      ("%u32.popcnt", Pbitwise { operand_type = U32; operator = Popcnt });
      ("%u64.add", Parith { operand_type = U64; operator = Add });
      ("%u64.sub", Parith { operand_type = U64; operator = Sub });
      ("%u64.mul", Parith { operand_type = U64; operator = Mul });
      ("%u64.div", Parith { operand_type = U64; operator = Div });
      ("%u64.mod", Parith { operand_type = U64; operator = Mod });
      ("%u64.eq", Pcomparison { operand_type = U64; operator = Eq });
      ("%u64.ne", Pcomparison { operand_type = U64; operator = Ne });
      ("%u64.compare", compare_uint64);
      ("%u64.bitand", Pbitwise { operand_type = U64; operator = And });
      ("%u64.bitnot", Pbitwise { operand_type = U64; operator = Not });
      ("%u64.bitor", Pbitwise { operand_type = U64; operator = Or });
      ("%u64.bitxor", Pbitwise { operand_type = U64; operator = Xor });
      ("%u64.shl", Pbitwise { operand_type = U64; operator = Shl });
      ("%u64.shr", Pbitwise { operand_type = U64; operator = Shr });
      ("%u64.clz", Pbitwise { operand_type = U64; operator = Clz });
      ("%u64.ctz", Pbitwise { operand_type = U64; operator = Ctz });
      ("%u64.popcnt", Pbitwise { operand_type = U64; operator = Popcnt });
      ("%i64_neg", Parith { operand_type = I64; operator = Neg });
      ("%i64_add", Parith { operand_type = I64; operator = Add });
      ("%i64_sub", Parith { operand_type = I64; operator = Sub });
      ("%i64_mul", Parith { operand_type = I64; operator = Mul });
      ("%i64_div", Parith { operand_type = I64; operator = Div });
      ("%i64_mod", Parith { operand_type = I64; operator = Mod });
      ("%i64_lnot", Pbitwise { operand_type = I64; operator = Not });
      ("%i64_land", Pbitwise { operand_type = I64; operator = And });
      ("%i64_lor", Pbitwise { operand_type = I64; operator = Or });
      ("%i64_lxor", Pbitwise { operand_type = I64; operator = Xor });
      ("%i64_shl", Pbitwise { operand_type = I64; operator = Shl });
      ("%i64_shr", Pbitwise { operand_type = I64; operator = Shr });
      ("%i64_ctz", Pbitwise { operand_type = I64; operator = Ctz });
      ("%i64_clz", Pbitwise { operand_type = I64; operator = Clz });
      ("%i64_popcnt", Pbitwise { operand_type = I64; operator = Popcnt });
      ("%i64_eq", Pcomparison { operand_type = I64; operator = Eq });
      ("%i64_ne", Pcomparison { operand_type = I64; operator = Ne });
      ("%i64_compare", compare_int64);
      ("%i64_default", Pccall { func_name = "default_int64"; arity = 0 });
      ("%f32.neg", Parith { operand_type = F32; operator = Neg });
      ("%f32.add", Parith { operand_type = F32; operator = Add });
      ("%f32.sub", Parith { operand_type = F32; operator = Sub });
      ("%f32.mul", Parith { operand_type = F32; operator = Mul });
      ("%f32.div", Parith { operand_type = F32; operator = Div });
      ("%f32.sqrt", Parith { operand_type = F32; operator = Sqrt });
      ("%f32.eq", Pcomparison { operand_type = F32; operator = Eq });
      ("%f32.ne", Pcomparison { operand_type = F32; operator = Ne });
      ("%f32.compare", compare_float32);
      ("%i32.to_f32", Pconvert { from = I32; to_ = F32; kind = Convert });
      ("%i64.to_f32", Pconvert { from = I64; to_ = F32; kind = Convert });
      ("%f64.to_f32", Pconvert { from = F64; to_ = F32; kind = Convert });
      ("%f32.to_f64", Pconvert { from = F32; to_ = F64; kind = Convert });
      ("%byte.to_f32", Pconvert { from = U8; to_ = F32; kind = Convert });
      ("%u32.to_f32", Pconvert { from = U32; to_ = F32; kind = Convert });
      ("%u64.to_f32", Pconvert { from = U64; to_ = F32; kind = Convert });
      ("%f32.to_i32", Pconvert { from = F32; to_ = I32; kind = Convert });
      ( "%f32.to_i32_reinterpret",
        Pconvert { from = F32; to_ = I32; kind = Reinterpret } );
      ( "%i32.to_f32_reinterpret",
        Pconvert { from = I32; to_ = F32; kind = Reinterpret } );
      ("%f64_neg", Parith { operand_type = F64; operator = Neg });
      ("%f64_add", Parith { operand_type = F64; operator = Add });
      ("%f64_sub", Parith { operand_type = F64; operator = Sub });
      ("%f64_mul", Parith { operand_type = F64; operator = Mul });
      ("%f64_div", Parith { operand_type = F64; operator = Div });
      ("%f64_sqrt", Parith { operand_type = F64; operator = Sqrt });
      ("%f64_eq", Pcomparison { operand_type = F64; operator = Eq });
      ("%f64_ne", Pcomparison { operand_type = F64; operator = Ne });
      ("%f64_compare", compare_float);
      ("%f64_default", Pccall { func_name = "default_float"; arity = 0 });
      ("%char_to_int", Pidentity);
      ("%char_from_int", Pidentity);
      ("%char_eq", Pcomparison { operand_type = I32; operator = Eq });
      ("%char_compare", compare_char);
      ("%char_default", Pccall { func_name = "default_char"; arity = 0 });
      ("%i32_to_i64", Pconvert { from = I32; to_ = I64; kind = Convert });
      ("%i32_to_f64", Pconvert { from = I32; to_ = F64; kind = Convert });
      ("%i64_to_i32", Pconvert { from = I64; to_ = I32; kind = Convert });
      ("%i64_to_f64", Pconvert { from = I64; to_ = F64; kind = Convert });
      ( "%i64_to_f64_reinterpret",
        Pconvert { from = I64; to_ = F64; kind = Reinterpret } );
      ("%f64_default", Pccall { func_name = "default_float"; arity = 0 });
      ("%f64_to_i32", Pconvert { from = F64; to_ = I32; kind = Convert });
      ("%f64_to_i64", Pconvert { from = F64; to_ = I64; kind = Convert });
      ( "%f64_to_i64_reinterpret",
        Pconvert { from = F64; to_ = I64; kind = Reinterpret } );
      ("%byte_to_int", Pidentity);
      ("%byte_to_i64", Pconvert { from = U8; to_ = I64; kind = Convert });
      ("%byte_to_f64", Pconvert { from = U8; to_ = F64; kind = Convert });
      ("%i32_to_byte", Pconvert { from = I32; to_ = U8; kind = Convert });
      ("%i64_to_byte", Pconvert { from = I64; to_ = U8; kind = Convert });
      ("%f64_to_byte", Pconvert { from = F64; to_ = U8; kind = Convert });
      ( "%u32.to_i32_reinterpret",
        Pconvert { from = U32; to_ = I32; kind = Reinterpret } );
      ( "%i32.to_u32_reinterpret",
        Pconvert { from = I32; to_ = U32; kind = Reinterpret } );
      ( "%u64.to_i64_reinterpret",
        Pconvert { from = U64; to_ = I64; kind = Reinterpret } );
      ( "%i64.to_u64_reinterpret",
        Pconvert { from = I64; to_ = U64; kind = Reinterpret } );
      ("%f64.to_u32", Pconvert { from = F64; to_ = U32; kind = Convert });
      ("%u32.to_u64", Pconvert { from = U32; to_ = U64; kind = Convert });
      ("%f64.to_u64", Pconvert { from = F64; to_ = U64; kind = Convert });
      ("%u32.to_f64", Pconvert { from = U32; to_ = F64; kind = Convert });
      ("%u64.to_u32", Pconvert { from = U64; to_ = U32; kind = Convert });
      ("%u64.to_i32", Pconvert { from = U64; to_ = I32; kind = Convert });
      ("%u64.to_f64", Pconvert { from = U64; to_ = F64; kind = Convert });
      ("%error.to_string", Perror_to_string);
      ("%any.to_string", Pany_to_string);
      ("%char.to_string", Pintrinsic Char_to_string);
      ("%f64.to_string", Pintrinsic F64_to_string);
      ("%string.substring", Pintrinsic String_substring);
      ("%fixedarray.join", Pintrinsic FixedArray_join);
      ("%fixedarray.iter", Pintrinsic FixedArray_iter);
      ("%fixedarray.iteri", Pintrinsic FixedArray_iteri);
      ("%fixedarray.map", Pintrinsic FixedArray_map);
      ("%fixedarray.fold_left", Pintrinsic FixedArray_fold_left);
      ("%fixedarray.copy", Pintrinsic FixedArray_copy);
      ("%fixedarray.fill", Pintrinsic FixedArray_fill);
      ("%iter.map", Pintrinsic Iter_map);
      ("%iter.filter", Pintrinsic Iter_filter);
      ("%iter.iter", Pintrinsic Iter_iter);
      ("%iter.take", Pintrinsic Iter_take);
      ("%iter.reduce", Pintrinsic Iter_reduce);
      ("%iter.from_array", Pintrinsic Iter_from_array);
      ("%iter.flat_map", Pintrinsic Iter_flat_map);
      ("%iter.repeat", Pintrinsic Iter_repeat);
      ("%iter.concat", Pintrinsic Iter_concat);
      ("%array.length", Pintrinsic Array_length);
      ("%array.get", Pintrinsic Array_get);
      ("%array.unsafe_get", Pintrinsic Array_unsafe_get);
      ("%array.set", Pintrinsic Array_set);
      ("%array.unsafe_set", Pintrinsic Array_unsafe_set);
    ]

let find_prim (name : string) : prim option =
  Hash_string.find_opt prim_table name

let is_intrinsic (p : prim) = match p with Pintrinsic _ -> true | _ -> false

let is_pure (prim : prim) =
  match prim with
  | Pfixedarray_length | Penum_field _ | Pnot | Pstringlength | Pstringequal
  | Pignore | Pidentity | Psequor | Psequand | Pmakebytes | Pbyteslength
  | Pfixedarray_make _
  | Pfixedarray_get_item { kind = Unsafe }
  | Parray_make | Parith _ | Pbitwise _ | Pconvert _ | Pcast _ | Pcomparison _
  | Pcompare _ | Pclosure_to_extern_ref | Prefeq | Pnull | Pnull_string_extern
  | Pis_null | Pas_non_null | Pmake_value_or_error _ | Perror_to_string
  | Pany_to_string ->
      true
  | Pfixedarray_get_item _ | Pgetstringitem | Pgetbytesitem | Pccall _
  | Pintrinsic _ | Praise | Ppanic | Punreachable | Pcatch
  | Pfixedarray_set_item _ | Psetbytesitem | Pset_enum_field _ | Pprintln
  | Pcall_object_method _ ->
      false
