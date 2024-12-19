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


module Ident = Clam_ident
module Ident_set = Ident.Set
module Lst = Basic_lst
module Ltype = Ltype_gc
module Ltype_util = Ltype_gc_util
module Tid = Basic_ty_ident

module List = struct
  include List

  let cons = Wasm_util.cons_peephole

  let rec append l1 l2 =
    match l1 with [] -> l2 | hd :: tl -> cons hd (append tl l2)
end

type instr = Wasm_util.instr

let i32_to_sexp = Wasm_util.i32_to_sexp
let int32_to_sexp = Wasm_util.int32_to_sexp
let i64_to_sexp = Wasm_util.i64_to_sexp
let f32_to_sexp = Wasm_util.f32_to_sexp
let f64_to_sexp = Wasm_util.f64_to_sexp
let u32_to_sexp = Wasm_util.u32_to_sexp
let u64_to_sexp = Wasm_util.u64_to_sexp
let tid_to_string = Transl_type.tid_to_wasm_name
let tid_to_sexp = Transl_type.tid_to_sexp
let ltype_to_sexp = Transl_type.ltype_to_sexp
let id_to_string = Ident.to_wasm_name
let label_to_string = Label.to_wasm_name

let id_to_sexp (id : Ident.t) =
  match id with
  | Pident _ | Pmutable_ident _ ->
      (List
         (List.cons
            (Atom "local.get" : W.t)
            ([ Atom (id_to_string id) ] : W.t list))
        : W.t)
  | Pdot { ty; _ } ->
      if Ltype_util.is_non_nullable_ref_type ty then
        (List
           (List.cons
              (Atom "ref.as_non_null" : W.t)
              ([
                 List
                   (List.cons
                      (Atom "global.get" : W.t)
                      ([ Atom (id_to_string id) ] : W.t list));
               ]
                : W.t list))
          : W.t)
      else
        (List
           (List.cons
              (Atom "global.get" : W.t)
              ([ Atom (id_to_string id) ] : W.t list))
          : W.t)

let compile_source_type (typ : Ltype.t) =
  match typ with
  | I32_Int -> Wasm_util.compile_source_type "int"
  | I32_Char -> Wasm_util.compile_source_type "char"
  | I32_Bool -> Wasm_util.compile_source_type "bool"
  | I32_Unit -> Wasm_util.compile_source_type "unit"
  | I32_Byte -> Wasm_util.compile_source_type "byte"
  | I64 -> Wasm_util.compile_source_type "int64"
  | F64 -> Wasm_util.compile_source_type "double"
  | F32 -> Wasm_util.compile_source_type "float"
  | _ -> []

let addr_to_string = Basic_fn_address.to_wasm_name
let add_cst = Wasm_util.add_cst
let add_dummy_i32 rest = add_cst 0 rest
let int_to_string = string_of_int

type loop_info = { params : Ident.t list; break_used : bool ref }

type ctx = {
  locals : Ident_set.t ref;
  loop_info : loop_info Label.Map.t;
  join_points : Ident.t list Join.Hash.t;
}

let ( +> ) (ctx : ctx) id = ctx.locals := Ident_set.add !(ctx.locals) id

type func = W.t

let i32_zero : W.t =
  (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list)) : W.t)

let i32_minus_one : W.t =
  (List (List.cons (Atom "i32.const" : W.t) ([ Atom "-1" ] : W.t list)) : W.t)

let i64_zero : W.t =
  (List (List.cons (Atom "i64.const" : W.t) ([ Atom "0" ] : W.t list)) : W.t)

let f32_zero : W.t =
  (List (List.cons (Atom "f32.const" : W.t) ([ Atom "0" ] : W.t list)) : W.t)

let f64_zero : W.t =
  (List (List.cons (Atom "f64.const" : W.t) ([ Atom "0" ] : W.t list)) : W.t)

let zero (t : Ltype.t) : W.t =
  match t with
  | I64 -> i64_zero
  | F32 -> f32_zero
  | F64 -> f64_zero
  | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag -> i32_zero
  | I32_Option_Char -> i32_minus_one
  | Ref_extern ->
      (List (List.cons (Atom "ref.null" : W.t) ([ Atom "extern" ] : W.t list))
        : W.t)
  | Ref_string ->
      (List
         (List.cons
            (Atom "ref.null" : W.t)
            ([ Atom "$moonbit.string" ] : W.t list))
        : W.t)
  | Ref_bytes ->
      (List
         (List.cons
            (Atom "ref.null" : W.t)
            ([ Atom "$moonbit.bytes" ] : W.t list))
        : W.t)
  | Ref_lazy_init { tid } | Ref { tid } | Ref_nullable { tid } ->
      (List
         (List.cons
            (Atom "ref.null" : W.t)
            ([ Atom (tid_to_string tid) ] : W.t list))
        : W.t)
  | Ref_func ->
      (List (List.cons (Atom "ref.null" : W.t) ([ Atom "func" ] : W.t list))
        : W.t)
  | Ref_any ->
      (List (List.cons (Atom "ref.null" : W.t) ([ Atom "any" ] : W.t list))
        : W.t)

let compilePrimitive (fn : Primitive.prim) (rest : instr list) : instr list =
  match fn with
  | Pccall { func_name = "add_string"; _ } ->
      (List
         (List.cons
            (Atom "call" : W.t)
            ([ Atom "$moonbit.add_string" ] : W.t list))
        : W.t)
      :: rest
  | Pprintln ->
      (List
         (List.cons
            (Atom "call" : W.t)
            ([ Atom "$moonbit.println" ] : W.t list))
        : W.t)
      :: add_dummy_i32 rest
  | Pfixedarray_length -> (List ([ Atom "array.len" ] : W.t list) : W.t) :: rest
  | Pgetbytesitem ->
      (List
         (List.cons
            (Atom "array.get_u" : W.t)
            ([ Atom "$moonbit.bytes" ] : W.t list))
        : W.t)
      :: rest
  | Psetbytesitem ->
      (List
         (List.cons
            (Atom "array.set" : W.t)
            ([ Atom "$moonbit.bytes" ] : W.t list))
        : W.t)
      :: add_dummy_i32 rest
  | Pgetstringitem ->
      (if !Basic_config.use_js_builtin_string then
         (List
            (List.cons
               (Atom "call" : W.t)
               ([ Atom "$moonbit.js_string.charCodeAt" ] : W.t list))
           : W.t)
       else
         (List
            (List.cons
               (Atom "array.get_u" : W.t)
               ([ Atom "$moonbit.string" ] : W.t list))
           : W.t))
      :: rest
  | Pbyteslength ->
      List.cons (List ([ Atom "array.len" ] : W.t list) : W.t) (rest : W.t list)
  | Pstringlength ->
      (if !Basic_config.use_js_builtin_string then
         (List
            (List.cons
               (Atom "call" : W.t)
               ([ Atom "$moonbit.js_string.length" ] : W.t list))
           : W.t)
       else (List ([ Atom "array.len" ] : W.t list) : W.t))
      :: rest
  | Pstringequal ->
      (List
         (List.cons
            (Atom "call" : W.t)
            ([ Atom "$moonbit.string_equal" ] : W.t list))
        : W.t)
      :: rest
  | Pccall { func_name = "int_is_pos"; _ } ->
      List.cons
        (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.cons
           (List ([ Atom "i32.gt_s" ] : W.t list) : W.t)
           (rest : W.t list))
  | Pccall { func_name = "int_is_neg"; _ } ->
      List.cons
        (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.cons
           (List ([ Atom "i32.lt_s" ] : W.t list) : W.t)
           (rest : W.t list))
  | Pccall { func_name = "int_is_non_pos"; _ } ->
      List.cons
        (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.cons
           (List ([ Atom "i32.le_s" ] : W.t list) : W.t)
           (rest : W.t list))
  | Pccall { func_name = "int_is_non_neg"; _ } ->
      List.cons
        (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.cons
           (List ([ Atom "i32.ge_s" ] : W.t list) : W.t)
           (rest : W.t list))
  | Parith { operand_type; operator } ->
      Wasm_util.compile_arith operand_type operator :: rest
  | Pconvert { kind; from; to_ } -> Wasm_util.compile_convert kind from to_ rest
  | Pbitwise { operand_type; operator } ->
      Wasm_util.compile_bitwise operand_type operator rest
  | Pcomparison { operand_type; operator } ->
      Wasm_util.compile_comparison operand_type operator :: rest
  | Prefeq ->
      List.cons (List ([ Atom "ref.eq" ] : W.t list) : W.t) (rest : W.t list)
  | Pignore ->
      List.cons
        (List ([ Atom "drop" ] : W.t list) : W.t)
        (List.cons
           (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Pidentity -> rest
  | Pccall { func_name = "default_int" | "default_char" | "default_bool"; _ } ->
      i32_zero :: rest
  | Pccall { func_name = "default_int64"; _ } -> i64_zero :: rest
  | Pccall { func_name = "default_float"; _ } -> f64_zero :: rest
  | Psequand | Psequor -> assert false
  | Pnot ->
      List.cons (List ([ Atom "i32.eqz" ] : W.t list) : W.t) (rest : W.t list)
  | Praise ->
      List.cons
        (List ([ Atom "unreachable" ] : W.t list) : W.t)
        (rest : W.t list)
  | Ppanic ->
      if !Basic_config.test_mode then
        List.cons
          (List (List.cons (Atom "call" : W.t) ([ Atom "$throw" ] : W.t list))
            : W.t)
          ([ List ([ Atom "unreachable" ] : W.t list) ] : W.t list)
      else ([ List ([ Atom "unreachable" ] : W.t list) ] : W.t list)
  | Punreachable -> ([ List ([ Atom "unreachable" ] : W.t list) ] : W.t list)
  | Pnull ->
      (List (List.cons (Atom "ref.null" : W.t) ([ Atom "none" ] : W.t list))
        : W.t)
      :: rest
  | Pnull_string_extern ->
      (List (List.cons (Atom "ref.null" : W.t) ([ Atom "extern" ] : W.t list))
        : W.t)
      :: rest
  | Pas_non_null -> (List ([ Atom "ref.as_non_null" ] : W.t list) : W.t) :: rest
  | Pis_null -> (List ([ Atom "ref.is_null" ] : W.t list) : W.t) :: rest
  | Pfixedarray_make _ | Pfixedarray_get_item _ | Pfixedarray_set_item _
  | Penum_field _ | Pset_enum_field _ | Parray_make | Pintrinsic _ | Pcompare _
  | Pcatch | Pcast _ | Pclosure_to_extern_ref | Pmake_value_or_error _
  | Perror_to_string | Pany_to_string | Pcall_object_method _ | Pmakebytes
  | Pccall _ ->
      failwith
        ("Unsupported primitive ${fn |> Primitive.sexp_of_prim |> W.to_string}"
          : Stdlib.String.t)

let compileClosure ~(global_ctx : Global_ctx2.t) (it : Clam.closure) rest :
    instr list =
  match (it.captures, it.address) with
  | [], Normal fn_addr ->
      Global_ctx2.add_func_ref global_ctx fn_addr;
      let tid = it.tid in
      List.cons
        (List
           (List.cons
              (Atom "struct.new" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 ([
                    List
                      (List.cons
                         (Atom "ref.func" : W.t)
                         ([ Atom (addr_to_string fn_addr) ] : W.t list));
                  ]
                   : W.t list)))
          : W.t)
        (rest : W.t list)
  | captures, Normal address ->
      Global_ctx2.add_func_ref global_ctx address;
      let tid_capture = it.tid in
      let captures = Lst.map captures (fun c : W.t -> id_to_sexp c) in
      List.cons
        (List
           (List.cons
              (Atom "struct.new" : W.t)
              (List.cons
                 (Atom (tid_to_string tid_capture) : W.t)
                 (List.cons
                    (List
                       (List.cons
                          (Atom "ref.func" : W.t)
                          ([ Atom (addr_to_string address) ] : W.t list))
                      : W.t)
                    (captures : W.t list))))
          : W.t)
        (rest : W.t list)
  | captures, Object methods ->
      let tid = it.tid in
      let methods =
        Lst.map methods (fun method_addr ->
            Global_ctx2.add_func_ref global_ctx method_addr;
            (List
               (List.cons
                  (Atom "ref.func" : W.t)
                  ([ Atom (addr_to_string method_addr) ] : W.t list))
              : W.t))
      in
      let captures = Lst.map captures (fun c : W.t -> id_to_sexp c) in
      List.cons
        (List
           (List.cons
              (Atom "struct.new" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 (List.append (methods : W.t list) (captures : W.t list))))
          : W.t)
        (rest : W.t list)
  | _, Well_known_mut_rec -> assert false

let rec compileExpr ~(tail : bool) ~ctx ~global_ctx ~type_defs
    (body : Clam.lambda) (rest : instr list) : instr list =
  let body_loc = Clam_util.loc_of_lambda body in
  let rest = Wasm_util.prepend_end_source_pos body_loc rest in
  Wasm_util.prepend_start_source_pos body_loc
    (compileExpr0 ~tail ~ctx ~global_ctx ~type_defs body rest)

and compileExpr0 ~(tail : bool) ~ctx ~global_ctx ~type_defs (body : Clam.lambda)
    (rest : instr list) : instr list =
  let got body rest =
    compileExpr ~ctx ~global_ctx ~tail ~type_defs body rest
      [@@inline]
  in
  let gon body rest =
    compileExpr ~ctx ~global_ctx ~tail:false ~type_defs body rest
      [@@inline]
  in
  let gos_non_tail args rest = List.fold_right gon args rest [@@inline] in
  let new_ident ~ty name =
    let ptr = Ident.fresh name ~ty in
    ctx +> ptr;
    ptr
      [@@inline]
  in
  let new_label name = Label.fresh name [@@inline] in
  match body with
  | Levent { expr; _ } -> got expr rest
  | Lconst constant -> (
      match constant with
      | C_int i -> add_cst (Int32.to_int i.v) rest
      | C_int64 i ->
          let iv = i.v in
          (i64_to_sexp iv : W.t) :: rest
      | C_uint { v; _ } -> (u32_to_sexp v : W.t) :: rest
      | C_uint64 { v; _ } -> (u64_to_sexp v : W.t) :: rest
      | C_bool false -> add_cst 0 rest
      | C_bool true -> add_cst 1 rest
      | C_char c ->
          let c = Uchar.to_int c in
          (i32_to_sexp c : W.t) :: rest
      | C_string s -> Global_ctx2.compile_string_literal ~global_ctx s :: rest
      | C_bytes { v; _ } ->
          Global_ctx2.compile_bytes_literal ~global_ctx v :: rest
      | C_float { v; _ } -> (f32_to_sexp v : W.t) :: rest
      | C_double f ->
          let fv = f.v in
          (f64_to_sexp fv : W.t) :: rest
      | C_bigint _ -> assert false)
  | Lvar { var } -> List.cons (id_to_sexp var : W.t) (rest : W.t list)
  | Llet { name = Pdot _ as name; e; body } ->
      let rest = got body rest in
      gon e
        (List.cons
           (List
              (List.cons
                 (Atom "global.set" : W.t)
                 ([ Atom (id_to_string name) ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Llet { name = (Pident _ | Pmutable_ident _) as name; e; body } ->
      ctx +> name;
      let rest = got body rest in
      gon e
        (List.cons
           (List
              (List.cons
                 (Atom "local.set" : W.t)
                 ([ Atom (id_to_string name) ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lassign { var; e } ->
      gon e
        (List.cons
           (List
              (List.cons
                 (Atom "local.set" : W.t)
                 ([ Atom (id_to_string var) ] : W.t list))
             : W.t)
           (List.cons
              (List
                 (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
                : W.t)
              (rest : W.t list)))
  | Lsequence { expr1; expr2 } ->
      let rest = got expr2 rest in
      gon expr1
        (List.cons (List ([ Atom "drop" ] : W.t list) : W.t) (rest : W.t list))
  | Lif { pred; ifso; ifnot; type_ } -> (
      match (type_, tail) with
      | I32_Unit, false ->
          let ifso =
            gon ifso ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
          in
          let ifnot =
            gon ifnot ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
          in
          let branches =
            if ifnot = [] then
              List.cons
                (List
                   (List.cons
                      (Atom "if" : W.t)
                      ([
                         List (List.cons (Atom "then" : W.t) (ifso : W.t list));
                       ]
                        : W.t list))
                  : W.t)
                (List.cons
                   (List
                      (List.cons
                         (Atom "i32.const" : W.t)
                         ([ Atom "0" ] : W.t list))
                     : W.t)
                   (rest : W.t list))
            else
              List.cons
                (List
                   (List.cons
                      (Atom "if" : W.t)
                      (List.cons
                         (List (List.cons (Atom "then" : W.t) (ifso : W.t list))
                           : W.t)
                         ([
                            List
                              (List.cons (Atom "else" : W.t) (ifnot : W.t list));
                          ]
                           : W.t list)))
                  : W.t)
                (List.cons
                   (List
                      (List.cons
                         (Atom "i32.const" : W.t)
                         ([ Atom "0" ] : W.t list))
                     : W.t)
                   (rest : W.t list))
          in
          gon pred branches
      | _ ->
          let ifso = got ifso [] in
          let ifnot = got ifnot [] in
          gon pred
            (List.cons
               (List
                  (List.cons
                     (Atom "if" : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "result" : W.t)
                              ([ ltype_to_sexp type_ ] : W.t list))
                          : W.t)
                        (List.cons
                           (List
                              (List.cons (Atom "then" : W.t) (ifso : W.t list))
                             : W.t)
                           ([
                              List
                                (List.cons
                                   (Atom "else" : W.t)
                                   (ifnot : W.t list));
                            ]
                             : W.t list))))
                 : W.t)
               (rest : W.t list)))
  | Lclosure closure -> compileClosure ~global_ctx closure rest
  | Lget_field { kind = Enum _; obj; index; tid } ->
      let index = index + 1 in
      gon obj
        (List.cons
           (List
              (List.cons
                 (Atom "struct.get" : W.t)
                 (List.cons
                    (Atom (tid_to_string tid) : W.t)
                    ([ Atom (int_to_string index) ] : W.t list)))
             : W.t)
           (rest : W.t list))
  | Lget_field { kind = Tuple | Struct; obj; index; tid } ->
      gon obj
        (List.cons
           (List
              (List.cons
                 (Atom "struct.get" : W.t)
                 (List.cons
                    (Atom (tid_to_string tid) : W.t)
                    ([ Atom (int_to_string index) ] : W.t list)))
             : W.t)
           (rest : W.t list))
  | Lclosure_field { obj; index; tid } ->
      let index =
        match[@warning "-fragile-match"] Tid.Hash.find_exn type_defs tid with
        | Ltype.Ref_closure { fn_sig_tid; captures = _ } -> (
            match Tid.Hash.find_exn type_defs fn_sig_tid with
            | Ref_closure_abstract _ -> index + 1
            | Ref_object { methods } -> index + List.length methods
            | Ref_struct _ | Ref_late_init_struct _ | Ref_constructor _
            | Ref_array _ | Ref_closure _ ->
                assert false)
        | _ -> assert false
      in
      gon obj
        (List.cons
           (List
              (List.cons
                 (Atom "struct.get" : W.t)
                 (List.cons
                    (Atom (tid_to_string tid) : W.t)
                    ([ Atom (int_to_string index) ] : W.t list)))
             : W.t)
           (rest : W.t list))
  | Lset_field { kind; obj; field; index; tid } ->
      let index = match kind with Enum _ -> index + 1 | _ -> index in
      gon obj
        (gon field
           (List.cons
              (List
                 (List.cons
                    (Atom "struct.set" : W.t)
                    (List.cons
                       (Atom (tid_to_string tid) : W.t)
                       ([ Atom (int_to_string index) ] : W.t list)))
                : W.t)
              (List.cons
                 (List
                    (List.cons
                       (Atom "i32.const" : W.t)
                       ([ Atom "0" ] : W.t list))
                   : W.t)
                 (rest : W.t list))))
  | Lapply { fn = Dynamic fn; args; prim = _ } -> (
      let tid =
        match Ident.get_type fn with
        | Ref_lazy_init { tid } -> tid
        | Ltype.Ref { tid } -> tid
        | _ -> assert false
      in
      let tid_code_ptr = Tid.code_pointer_of_closure tid in
      let args = gos_non_tail (Clam.Lvar { var = fn } :: args) [] in
      match fn with
      | Pident _ | Pdot _ ->
          List.append
            (args : W.t list)
            (List.cons
               (List
                  (List.cons
                     (Atom "struct.get" : W.t)
                     (List.cons
                        (Atom (tid_to_string tid) : W.t)
                        (List.cons
                           (Atom "0" : W.t)
                           ([ id_to_sexp fn ] : W.t list))))
                 : W.t)
               (List.cons
                  (List
                     (List.cons
                        (Atom "call_ref" : W.t)
                        ([ Atom (tid_to_string tid_code_ptr) ] : W.t list))
                    : W.t)
                  (rest : W.t list)))
      | Pmutable_ident _ ->
          let fn_tmp =
            new_ident ~ty:(Ref_lazy_init { tid = tid_code_ptr }) "*fn_tmp"
          in
          List.cons
            (List
               (List.cons
                  (Atom "struct.get" : W.t)
                  (List.cons
                     (Atom (tid_to_string tid) : W.t)
                     (List.cons (Atom "0" : W.t) ([ id_to_sexp fn ] : W.t list))))
              : W.t)
            (List.cons
               (List
                  (List.cons
                     (Atom "local.set" : W.t)
                     ([ Atom (id_to_string fn_tmp) ] : W.t list))
                 : W.t)
               (List.append
                  (args : W.t list)
                  (List.cons
                     (List
                        (List.cons
                           (Atom "local.get" : W.t)
                           ([ Atom (id_to_string fn_tmp) ] : W.t list))
                       : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "call_ref" : W.t)
                              ([ Atom (tid_to_string tid_code_ptr) ] : W.t list))
                          : W.t)
                        (rest : W.t list))))))
  | Lapply { fn = Object { obj; method_index; method_ty = _ }; args; prim = _ }
    ->
      let tid =
        match Ident.get_type obj with
        | Ref_lazy_init { tid } -> tid
        | Ref { tid } -> tid
        | _ -> assert false
      in
      let tid_code_ptr = Tid.method_of_object tid method_index in
      gos_non_tail
        (Clam.Lvar { var = obj } :: args)
        (List.cons
           (List
              (List.cons
                 (Atom "struct.get" : W.t)
                 (List.cons
                    (Atom (tid_to_string tid) : W.t)
                    (List.cons
                       (Atom (int_to_string method_index) : W.t)
                       ([ id_to_sexp obj ] : W.t list))))
             : W.t)
           (List.cons
              (List
                 (List.cons
                    (Atom "call_ref" : W.t)
                    ([ Atom (tid_to_string tid_code_ptr) ] : W.t list))
                : W.t)
              (rest : W.t list)))
  | Lapply { fn = _; args; prim = Some (FixedArray_copy { src_tid; dst_tid }) }
    ->
      let rest = add_dummy_i32 rest in
      gos_non_tail args
        (List.cons
           (List
              (List.cons
                 (Atom "array.copy" : W.t)
                 (List.cons
                    (Atom (tid_to_string dst_tid) : W.t)
                    ([ Atom (tid_to_string src_tid) ] : W.t list)))
             : W.t)
           (rest : W.t list))
  | Lapply { fn = _; args; prim = Some (FixedArray_fill { tid = arr_tid }) } ->
      let rest = add_dummy_i32 rest in
      gos_non_tail args
        (List.cons
           (List
              (List.cons
                 (Atom "array.fill" : W.t)
                 ([ Atom (tid_to_string arr_tid) ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lapply { fn = StaticFn addr; args; prim = _ } ->
      gos_non_tail args
        (List.cons
           (List
              (List.cons
                 (Atom "call" : W.t)
                 ([ Atom (addr_to_string addr) ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lstub_call { fn = func_stubs; params_ty; return_ty; args } ->
      let name =
        match func_stubs with
        | Import { module_name; func_name } ->
            Global_ctx2.add_import global_ctx
              { module_name; func_name; params_ty; return_ty }
        | Internal { func_name } -> (func_name : Stdlib.String.t)
        | Inline_code_sexp { language; func_body } ->
            if language <> "wasm" then
              failwith
                ("extern \"" ^ language
                 ^ "\" is not supported in wasm-gc backend"
                  : Stdlib.String.t);
            Global_ctx2.add_inline global_ctx func_body
        | Inline_code_text _ -> assert false
      in
      let rest =
        match return_ty with None -> add_dummy_i32 rest | _ -> rest
      in
      gos_non_tail args
        (List.cons
           (List (List.cons (Atom "call" : W.t) ([ Atom name ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lprim { fn = Parith { operand_type = I32; operator = Neg }; args = a :: [] }
    ->
      let a = gon a [] in
      List.cons
        (List (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.append
           (a : W.t list)
           (List.cons
              (List ([ Atom "i32.sub" ] : W.t list) : W.t)
              (rest : W.t list)))
  | Lprim { fn = Parith { operand_type = I64; operator = Neg }; args = a :: [] }
    ->
      let a = gon a [] in
      List.cons
        (List (List.cons (Atom "i64.const" : W.t) ([ Atom "0" ] : W.t list))
          : W.t)
        (List.append
           (a : W.t list)
           (List.cons
              (List ([ Atom "i64.sub" ] : W.t list) : W.t)
              (rest : W.t list)))
  | Lprim { fn = Psequand; args } -> (
      match[@warning "-fragile-match"] args with
      | [ a; b ] ->
          let a = gon a [] in
          let b = got b [] in
          List.cons
            (List
               (List.cons
                  (Atom "block" : W.t)
                  (List.cons
                     (List
                        (List.cons
                           (Atom "result" : W.t)
                           ([ Atom "i32" ] : W.t list))
                       : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "i32.const" : W.t)
                              ([ Atom "0" ] : W.t list))
                          : W.t)
                        (List.append
                           (a : W.t list)
                           (List.cons
                              (List ([ Atom "i32.eqz" ] : W.t list) : W.t)
                              (List.cons
                                 (List
                                    (List.cons
                                       (Atom "br_if" : W.t)
                                       ([ Atom "0" ] : W.t list))
                                   : W.t)
                                 (List.cons
                                    (List ([ Atom "drop" ] : W.t list) : W.t)
                                    (b : W.t list))))))))
              : W.t)
            (rest : W.t list)
      | _ -> assert false)
  | Lprim { fn = Psequor; args } -> (
      match[@warning "-fragile-match"] args with
      | [ a; b ] ->
          let a = gon a [] in
          let b = got b [] in
          List.cons
            (List
               (List.cons
                  (Atom "block" : W.t)
                  (List.cons
                     (List
                        (List.cons
                           (Atom "result" : W.t)
                           ([ Atom "i32" ] : W.t list))
                       : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "i32.const" : W.t)
                              ([ Atom "1" ] : W.t list))
                          : W.t)
                        (List.append
                           (a : W.t list)
                           (List.cons
                              (List
                                 (List.cons
                                    (Atom "br_if" : W.t)
                                    ([ Atom "0" ] : W.t list))
                                : W.t)
                              (List.cons
                                 (List ([ Atom "drop" ] : W.t list) : W.t)
                                 (b : W.t list)))))))
              : W.t)
            (rest : W.t list)
      | _ -> assert false)
  | Lprim { fn = Pcompare operand_type; args = [ a1; a2 ] } ->
      let compile_arg (arg : Clam.lambda) k =
        match arg with
        | Lvar _ | Lconst _ -> (
            match[@warning "-fragile-match"] gon arg [] with
            | v :: [] -> k v
            | _ -> assert false)
        | _ ->
            let ty : Ltype.t =
              match operand_type with
              | I32 | U32 -> I32_Int
              | I64 | U64 -> I64
              | F32 -> F32
              | F64 -> F64
              | U8 -> assert false
            in
            let tmp = Ident.fresh ~ty "tmp" in
            ctx +> tmp;
            gon arg
              ((List
                  (List.cons
                     (Atom "local.set" : W.t)
                     ([ Atom (id_to_string tmp) ] : W.t list))
                 : W.t)
              :: k (id_to_sexp tmp))
      in
      compile_arg a1 (fun v1 ->
          compile_arg a2 (fun v2 ->
              Wasm_util.compile_compare operand_type v1 v2 :: rest))
  | Lmake_array { kind = LenAndInit; tid; elems } -> (
      match[@warning "-fragile-match"] elems with
      | [ len; init ] ->
          let init = gon init [] in
          let len = gon len [] in
          List.append
            (init : W.t list)
            (List.append
               (len : W.t list)
               (List.cons
                  (List
                     (List.cons
                        (Atom "array.new" : W.t)
                        ([ Atom (tid_to_string tid) ] : W.t list))
                    : W.t)
                  (rest : W.t list)))
      | _ -> assert false)
  | Lmake_array { kind = Uninit; tid; elems } -> (
      match[@warning "-fragile-match"] elems with
      | len :: [] -> (
          let len = gon len [] in
          match Global_ctx2.find_default_elem global_ctx type_defs tid with
          | Some id ->
              List.cons
                (List
                   (List.cons
                      (Atom "global.get" : W.t)
                      ([ Atom id ] : W.t list))
                  : W.t)
                (List.append
                   (len : W.t list)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "array.new" : W.t)
                            ([ Atom (tid_to_string tid) ] : W.t list))
                        : W.t)
                      (rest : W.t list)))
          | None ->
              List.append
                (len : W.t list)
                (List.cons
                   (List
                      (List.cons
                         (Atom "array.new_default" : W.t)
                         ([ Atom (tid_to_string tid) ] : W.t list))
                     : W.t)
                   (rest : W.t list)))
      | _ -> assert false)
  | Larray_get_item { arr; index; tid; kind; need_non_null_cast } -> (
      let has_default_elem tid =
        match Global_ctx2.find_default_elem global_ctx type_defs tid with
        | Some _ -> true
        | None -> false
      in
      let is_non_nullable_elem tid =
        match Tid.Hash.find_opt type_defs tid with
        | Some (Ref_array { elem = Ref_nullable _ }) -> false
        | _ -> true
      in
      let rest =
        if
          need_non_null_cast
          && (not (has_default_elem tid))
          && is_non_nullable_elem tid
        then
          List.cons
            (List
               (List.cons
                  (Atom "array.get" : W.t)
                  ([ Atom (tid_to_string tid) ] : W.t list))
              : W.t)
            (List.cons
               (List ([ Atom "ref.as_non_null" ] : W.t list) : W.t)
               (rest : W.t list))
        else
          List.cons
            (List
               (List.cons
                  (Atom "array.get" : W.t)
                  ([ Atom (tid_to_string tid) ] : W.t list))
              : W.t)
            (rest : W.t list)
      in
      match kind with
      | Safe | Unsafe -> gon arr (gon index rest)
      | Rev_unsafe ->
          let arr = gon arr [] in
          let rev_index = gon index [] in
          List.append
            (arr : W.t list)
            (List.cons
               (List
                  (List.cons
                     (Atom "i32.sub" : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "i32.sub" : W.t)
                              (List.cons
                                 (List
                                    (List.cons
                                       (Atom "array.len" : W.t)
                                       (arr : W.t list))
                                   : W.t)
                                 (rev_index : W.t list)))
                          : W.t)
                        ([
                           List
                             (List.cons
                                (Atom "i32.const" : W.t)
                                ([ Atom "1" ] : W.t list));
                         ]
                          : W.t list)))
                 : W.t)
               (rest : W.t list)))
  | Larray_set_item { tid; kind = Null; arr; index; item = _ } -> (
      let elem_type = Ltype_util.get_arr_elem tid type_defs in
      match elem_type with
      | I32_Unit | I32_Int | I32_Char | I32_Bool | I32_Byte | I32_Tag
      | I32_Option_Char | I64 | F32 | F64 ->
          add_dummy_i32 rest
      | Ref _ | Ref_string | Ref_bytes | Ref_func | Ref_any | Ref_nullable _
      | Ref_lazy_init _ | Ref_extern ->
          let arr = gon arr [] in
          let index = gon index [] in
          let zero =
            match Global_ctx2.find_default_elem global_ctx type_defs tid with
            | Some id ->
                (List
                   (List.cons
                      (Atom "global.get" : W.t)
                      ([ Atom id ] : W.t list))
                  : W.t)
            | None -> zero elem_type
          in
          let rest = add_dummy_i32 rest in
          List.append
            (arr : W.t list)
            (List.append
               (index : W.t list)
               (List.cons
                  (zero : W.t)
                  (List.cons
                     (List
                        (List.cons
                           (Atom "array.set" : W.t)
                           ([ Atom (tid_to_string tid) ] : W.t list))
                       : W.t)
                     (rest : W.t list)))))
  | Larray_set_item { tid; kind = Value | Unsafe; arr; index; item } ->
      let item = Option.get item in
      gos_non_tail [ arr; index; item ]
        (List.cons
           (List
              (List.cons
                 (Atom "array.set" : W.t)
                 ([ Atom (tid_to_string tid) ] : W.t list))
             : W.t)
           (List.cons
              (List
                 (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
                : W.t)
              (rest : W.t list)))
  | Larray_set_item { tid; kind = Default; arr; index; item = _ } ->
      let default =
        match Global_ctx2.find_default_elem global_ctx type_defs tid with
        | Some id ->
            (List (List.cons (Atom "global.get" : W.t) ([ Atom id ] : W.t list))
              : W.t)
        | None ->
            let elem_type = Ltype_util.get_arr_elem tid type_defs in
            let zero = zero elem_type in
            (zero : W.t)
      in
      gos_non_tail [ arr; index ]
        (default
        :: (List
              (List.cons
                 (Atom "array.set" : W.t)
                 ([ Atom (tid_to_string tid) ] : W.t list))
             : W.t)
        :: add_dummy_i32 rest)
  | Lallocate { kind = Enum { tag }; fields = []; tid = _ } ->
      let tag = tag.index in
      Global_ctx2.compile_constant_constr ~global_ctx ~tag :: rest
  | Lallocate { kind; fields; tid } ->
      let fields = gos_non_tail fields [] in
      let fields =
        match kind with
        | Enum { tag = { index; _ } } ->
            List.cons (i32_to_sexp index : W.t) (fields : W.t list)
        | _ -> fields
      in
      List.cons
        (List
           (List.cons
              (Atom "struct.new" : W.t)
              (List.cons (Atom (tid_to_string tid) : W.t) (fields : W.t list)))
          : W.t)
        (rest : W.t list)
  | Lmake_array { kind = EverySingleElem; tid; elems } ->
      let exception Not_constant in
      let n = List.length elems in
      let normal () =
        if n > 100 then
          let init = gon (List.hd elems) [] in
          let arr_id = new_ident ~ty:(Ref { tid }) "arr" in
          let rest =
            (List
               (List.cons
                  (Atom "local.get" : W.t)
                  ([ Atom (id_to_string arr_id) ] : W.t list))
              : W.t)
            :: rest
          in
          let set_elem =
            Lst.flat_mapi_append (List.tl elems) ~init:rest ~f:(fun i a ->
                let arg = gon a [] in
                let i = i + 1 in
                List.cons
                  (List
                     (List.cons
                        (Atom "local.get" : W.t)
                        ([ Atom (id_to_string arr_id) ] : W.t list))
                    : W.t)
                  (List.cons
                     (i32_to_sexp i : W.t)
                     (List.append
                        (arg : W.t list)
                        ([
                           List
                             (List.cons
                                (Atom "array.set" : W.t)
                                ([ Atom (tid_to_string tid) ] : W.t list));
                         ]
                          : W.t list))))
          in
          List.append
            (init : W.t list)
            (List.cons
               (i32_to_sexp n : W.t)
               (List.cons
                  (List
                     (List.cons
                        (Atom "array.new" : W.t)
                        ([ Atom (tid_to_string tid) ] : W.t list))
                    : W.t)
                  (List.cons
                     (List
                        (List.cons
                           (Atom "local.set" : W.t)
                           ([ Atom (id_to_string arr_id) ] : W.t list))
                       : W.t)
                     (set_elem : W.t list))))
        else
          gos_non_tail elems
            (List.cons
               (List
                  (List.cons
                     (Atom "array.new_fixed" : W.t)
                     (List.cons
                        (Atom (tid_to_string tid) : W.t)
                        ([ Atom (int_to_string n) ] : W.t list)))
                 : W.t)
               (rest : W.t list))
          [@@local]
      in
      if n >= 5 then
        let t = Ltype_util.get_arr_elem tid type_defs in
        match t with
        | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Tag -> (
            match
              Lst.map elems (fun x ->
                  match Clam_util.no_located x with
                  | Lconst (C_int { v; repr = _ }) -> v
                  | _ -> raise_notrace Not_constant)
            with
            | xs ->
                Global_ctx2.compile_int32_array_literal ~global_ctx xs tid
                :: rest
            | exception Not_constant -> normal ())
        | I64 -> (
            match
              Lst.map elems (fun x ->
                  match Clam_util.no_located x with
                  | Lconst (C_int64 { v; repr = _ }) -> v
                  | _ -> raise_notrace Not_constant)
            with
            | xs ->
                Global_ctx2.compile_int64_array_literal ~global_ctx xs tid
                :: rest
            | exception Not_constant -> normal ())
        | I32_Byte -> (
            let buf = Buffer.create 16 in
            match
              Lst.map elems (fun x ->
                  match Clam_util.no_located x with
                  | Lconst (C_int { v; repr = _ }) ->
                      Buffer.add_int8 buf (Int32.to_int v)
                  | _ -> raise_notrace Not_constant)
            with
            | _ ->
                Global_ctx2.compile_bytes_literal ~global_ctx
                  (Buffer.contents buf)
                :: rest
            | exception Not_constant -> normal ())
        | _ -> normal ()
      else normal ()
  | Lprim { fn = Pclosure_to_extern_ref; args = Lvar { var; _ } :: [] } ->
      let closure_ffi_name =
        Global_ctx2.add_import global_ctx Global_ctx2.Import.make_closure
      in
      let tid =
        match Ident.get_type var with
        | Ref_lazy_init { tid } | Ltype.Ref { tid } -> tid
        | _ -> assert false
      in
      let tid_base = tid in
      List.cons
        (List
           (List.cons
              (Atom "struct.get" : W.t)
              (List.cons
                 (Atom (tid_to_string tid_base) : W.t)
                 (List.cons (Atom "0" : W.t) ([ id_to_sexp var ] : W.t list))))
          : W.t)
        (List.cons
           (id_to_sexp var : W.t)
           (List.cons
              (List
                 (List.cons
                    (Atom "call" : W.t)
                    ([ Atom closure_ffi_name ] : W.t list))
                : W.t)
              (rest : W.t list)))
  | Lcast { expr; target_type = Ref_any } -> got expr rest
  | Lcast { expr; target_type } ->
      gon expr
        (List.cons
           (List
              (List.cons
                 (Atom "ref.cast" : W.t)
                 ([ ltype_to_sexp target_type ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lcatch { body; on_exception; type_ } ->
      assert !Basic_config.test_mode;
      let body = gon body [] in
      let on_exception = gon on_exception [] in
      let label_exit = new_label "exit" in
      let label_catch = new_label "catch" in
      let tag = Ltype.tag_name in
      List.cons
        (List
           (List.cons
              (Atom "block" : W.t)
              (List.cons
                 (Atom (label_to_string label_exit) : W.t)
                 (List.cons
                    (List
                       (List.cons
                          (Atom "result" : W.t)
                          ([ ltype_to_sexp type_ ] : W.t list))
                      : W.t)
                    (List.cons
                       (List
                          (List.cons
                             (Atom "block" : W.t)
                             (List.cons
                                (Atom (label_to_string label_catch) : W.t)
                                ([
                                   List
                                     (List.cons
                                        (Atom "try_table" : W.t)
                                        (List.cons
                                           (List
                                              (List.cons
                                                 (Atom "catch" : W.t)
                                                 (List.cons
                                                    (Atom tag : W.t)
                                                    ([
                                                       Atom
                                                         (label_to_string
                                                            label_catch);
                                                     ]
                                                      : W.t list)))
                                             : W.t)
                                           (List.append
                                              (body : W.t list)
                                              ([
                                                 List
                                                   (List.cons
                                                      (Atom "br" : W.t)
                                                      ([
                                                         Atom
                                                           (label_to_string
                                                              label_exit);
                                                       ]
                                                        : W.t list));
                                               ]
                                                : W.t list))));
                                 ]
                                  : W.t list)))
                         : W.t)
                       (on_exception : W.t list)))))
          : W.t)
        (rest : W.t list)
  | Lprim { fn = Pmakebytes; args } -> (
      match[@warning "-fragile-match"] args with
      | [ size; v ] ->
          let size = gon size [] in
          let v = gon v [] in
          List.append
            (v : W.t list)
            (List.append
               (size : W.t list)
               (List.cons
                  (List
                     (List.cons
                        (Atom "array.new" : W.t)
                        ([ Atom "$moonbit.bytes" ] : W.t list))
                    : W.t)
                  (rest : W.t list)))
      | _ -> assert false)
  | Lprim { fn; args } -> gos_non_tail args (compilePrimitive fn rest)
  | Lswitch { obj; cases; default; type_ } ->
      let push_drop = match type_ with I32_Unit -> not tail | _ -> false in
      let compile_branch branch =
        if push_drop then
          gon branch ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
        else got branch []
          [@@inline]
      in
      let tag = new_ident ~ty:I32_Tag "tag" in
      let obj_tag = (id_to_sexp tag : W.t) in
      let default = compile_branch default in
      let cases =
        Lst.map cases (fun (tag, action) -> (tag.index, compile_branch action))
      in
      let switches =
        Wasm_util.compile_int_switch
          ~result_ty:
            (if push_drop then []
             else
               ([
                  List
                    (List.cons
                       (Atom "result" : W.t)
                       ([ ltype_to_sexp type_ ] : W.t list));
                ]
                 : W.t list))
          ~obj:obj_tag ~cases ~default
          (if push_drop then add_cst 0 rest else rest)
      in
      let tid_enum = Ltype.tid_enum in
      List.cons
        (id_to_sexp obj : W.t)
        (List.cons
           (List
              (List.cons
                 (Atom "struct.get" : W.t)
                 (List.cons
                    (Atom (tid_to_string tid_enum) : W.t)
                    ([ Atom "0" ] : W.t list)))
             : W.t)
           (List.cons
              (List
                 (List.cons
                    (Atom "local.set" : W.t)
                    ([ Atom (id_to_string tag) ] : W.t list))
                : W.t)
              (switches : W.t list)))
  | Lswitchint { obj; cases; default; type_ } ->
      let push_drop = match type_ with I32_Unit -> not tail | _ -> false in
      let compile_branch branch =
        if push_drop then
          gon branch ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
        else got branch []
          [@@inline]
      in
      Wasm_util.compile_int_switch
        ~result_ty:
          (if push_drop then []
           else
             ([
                List
                  (List.cons
                     (Atom "result" : W.t)
                     ([ ltype_to_sexp type_ ] : W.t list));
              ]
               : W.t list))
        ~obj:(id_to_sexp obj : W.t)
        ~cases:(List.map (fun (c, action) -> (c, compile_branch action)) cases)
        ~default:(compile_branch default)
        (if push_drop then add_cst 0 rest else rest)
  | Lswitchstring { obj; cases; default; type_ } ->
      let push_drop = match type_ with I32_Unit -> not tail | _ -> false in
      let result_ty =
        if push_drop then []
        else
          ([
             List
               (List.cons
                  (Atom "result" : W.t)
                  ([ ltype_to_sexp type_ ] : W.t list));
           ]
            : W.t list)
      in
      let compile_branch branch =
        if push_drop then
          gon branch ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
        else got branch []
          [@@inline]
      in
      let tag = new_ident ~ty:Ref_string "tag" in
      let obj_tag = (id_to_sexp tag : W.t) in
      let switches =
        List.fold_right
          (fun ((tag : string), act) acc ->
            let act = compile_branch act in
            let tagExpr = Global_ctx2.compile_string_literal ~global_ctx tag in
            List.cons
              (obj_tag : W.t)
              (List.cons
                 (tagExpr : W.t)
                 (List.cons
                    (List
                       (List.cons
                          (Atom "call" : W.t)
                          ([ Atom "$moonbit.string_equal" ] : W.t list))
                      : W.t)
                    ([
                       List
                         (List.cons
                            (Atom "if" : W.t)
                            (List.append
                               (result_ty : W.t list)
                               (List.cons
                                  (List
                                     (List.cons
                                        (Atom "then" : W.t)
                                        (act : W.t list))
                                    : W.t)
                                  ([
                                     List
                                       (List.cons
                                          (Atom "else" : W.t)
                                          (acc : W.t list));
                                   ]
                                    : W.t list))));
                     ]
                      : W.t list))))
          cases (compile_branch default)
      in
      let rest = if push_drop then add_cst 0 rest else rest in
      gon obj
        (List.cons
           (List
              (List.cons
                 (Atom "local.set" : W.t)
                 ([ Atom (id_to_string tag) ] : W.t list))
             : W.t)
           (List.append (switches : W.t list) (rest : W.t list)))
  | Lletrec { names = name :: []; fns = closure :: []; body } ->
      ctx +> name;
      let rest = got body rest in
      compileClosure ~global_ctx closure
        (List.cons
           (List
              (List.cons
                 (Atom "local.set" : W.t)
                 ([ Atom (id_to_string name) ] : W.t list))
             : W.t)
           (rest : W.t list))
  | Lletrec { names; fns; body } ->
      Lst.iter names (fun name -> ctx +> name);
      let names_fns = List.combine names fns in
      let rest = got body rest in
      let alloc_closure =
        Lst.concat_map names_fns (fun (name, { address; _ }) ->
            let tid_capture =
              match address with
              | Normal address -> Tid.capture_of_function address
              | Well_known_mut_rec -> (
                  match Ident.get_type name with
                  | Ref { tid } -> tid
                  | _ -> assert false)
              | Object _ -> assert false
            in
            List.cons
              (List
                 (List.cons
                    (Atom "struct.new_default" : W.t)
                    ([ Atom (tid_to_string tid_capture) ] : W.t list))
                : W.t)
              ([
                 List
                   (List.cons
                      (Atom "local.set" : W.t)
                      ([ Atom (id_to_string name) ] : W.t list));
               ]
                : W.t list))
      in
      let write_closure =
        Lst.concat_map names_fns (fun (name, { captures; address; tid }) ->
            let tid_capture = tid in
            match address with
            | Normal address ->
                Global_ctx2.add_func_ref global_ctx address;
                let closure =
                  (List
                     (List.cons
                        (Atom "ref.cast" : W.t)
                        (List.cons
                           (tid_to_sexp tid_capture : W.t)
                           ([ id_to_sexp name ] : W.t list)))
                    : W.t)
                in
                (List
                   (List.cons
                      (Atom "struct.set" : W.t)
                      (List.cons
                         (Atom (tid_to_string tid_capture) : W.t)
                         (List.cons
                            (Atom "0" : W.t)
                            (List.cons
                               (closure : W.t)
                               ([
                                  List
                                    (List.cons
                                       (Atom "ref.func" : W.t)
                                       ([ Atom (addr_to_string address) ]
                                         : W.t list));
                                ]
                                 : W.t list)))))
                  : W.t)
                :: Lst.mapi captures (fun i c ->
                       let i = i + 1 in
                       (List
                          (List.cons
                             (Atom "struct.set" : W.t)
                             (List.cons
                                (Atom (tid_to_string tid_capture) : W.t)
                                (List.cons
                                   (Atom (int_to_string i) : W.t)
                                   (List.cons
                                      (closure : W.t)
                                      ([ id_to_sexp c ] : W.t list)))))
                         : W.t))
            | Well_known_mut_rec ->
                Lst.mapi captures (fun i c ->
                    let closure =
                      (List
                         (List.cons
                            (Atom "ref.cast" : W.t)
                            (List.cons
                               (tid_to_sexp tid_capture : W.t)
                               ([ id_to_sexp name ] : W.t list)))
                        : W.t)
                    in
                    (List
                       (List.cons
                          (Atom "struct.set" : W.t)
                          (List.cons
                             (Atom (tid_to_string tid_capture) : W.t)
                             (List.cons
                                (Atom (int_to_string i) : W.t)
                                (List.cons
                                   (closure : W.t)
                                   ([ id_to_sexp c ] : W.t list)))))
                      : W.t))
            | Object _ -> assert false)
      in
      List.append
        (alloc_closure : W.t list)
        (List.append (write_closure : W.t list) (rest : W.t list))
  | Ljoinlet { name; e; body; params; kind = _; type_ } ->
      Lst.iter params (fun param -> ctx +> param);
      let result_param_types =
        Lst.map params (fun id ->
            let t = Transl_type.ltype_to_sexp (Ident.get_type id) in
            (List (List.cons (Atom "result" : W.t) ([ t ] : W.t list)) : W.t))
      in
      let result_types =
        Lst.map type_ (fun t : W.t ->
            List
              (List.cons (Atom "result" : W.t) ([ ltype_to_sexp t ] : W.t list)))
      in
      let label = Join.to_wasm_label name in
      Join.Hash.add ctx.join_points name params;
      let def e =
        Lst.fold_left params e (fun acc param_name ->
            List.cons
              (List
                 (List.cons
                    (Atom "local.set" : W.t)
                    ([ Atom (id_to_string param_name) ] : W.t list))
                : W.t)
              (acc : W.t list))
      in
      if tail then
        let body =
          got body ([ List ([ Atom "return" ] : W.t list) ] : W.t list)
        in
        let e = def (got e rest) in
        List.cons
          (List
             (List.cons
                (Atom "block" : W.t)
                (List.cons
                   (Atom label : W.t)
                   (List.append
                      (result_param_types : W.t list)
                      (body : W.t list))))
            : W.t)
          (e : W.t list)
      else
        let body =
          gon body
            ([ List (List.cons (Atom "br" : W.t) ([ Atom "1" ] : W.t list)) ]
              : W.t list)
        in
        let e = def (gon e []) in
        List.cons
          (List
             (List.cons
                (Atom "block" : W.t)
                (List.append
                   (result_types : W.t list)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "block" : W.t)
                            (List.cons
                               (Atom label : W.t)
                               (List.append
                                  (result_param_types : W.t list)
                                  (body : W.t list))))
                        : W.t)
                      (e : W.t list))))
            : W.t)
          (rest : W.t list)
  | Ljoinapply { name; args } ->
      let label = Join.to_wasm_label name in
      let jump =
        ([ List (List.cons (Atom "br" : W.t) ([ Atom label ] : W.t list)) ]
          : W.t list)
      in
      gos_non_tail args jump
  | Lbreak { arg; label = loop_label } -> (
      let label = Label.to_wasm_label_break loop_label in
      let loop_info = Label.Map.find_exn ctx.loop_info loop_label in
      loop_info.break_used := true;
      match arg with
      | None ->
          ([ List (List.cons (Atom "br" : W.t) ([ Atom label ] : W.t list)) ]
            : W.t list)
      | Some arg ->
          gon arg
            ([ List (List.cons (Atom "br" : W.t) ([ Atom label ] : W.t list)) ]
              : W.t list))
  | Lcontinue { args; label = loop_label } -> (
      let label = Label.to_wasm_label_loop loop_label in
      match Label.Map.find_exn ctx.loop_info loop_label with
      | { params; _ } ->
          if !Basic_config.use_block_params then
            gos_non_tail args
              ([
                 List (List.cons (Atom "br" : W.t) ([ Atom label ] : W.t list));
               ]
                : W.t list)
          else
            let set_params =
              Lst.fold_left params [] (fun acc param ->
                  List.cons
                    (List
                       (List.cons
                          (Atom "local.set" : W.t)
                          ([ Atom (id_to_string param) ] : W.t list))
                      : W.t)
                    (acc : W.t list))
            in
            gos_non_tail args
              (List.append
                 (set_params : W.t list)
                 ([
                    List
                      (List.cons (Atom "br" : W.t) ([ Atom label ] : W.t list));
                  ]
                   : W.t list)))
  | Lreturn e ->
      if tail then got e rest
      else gon e ([ List ([ Atom "return" ] : W.t list) ] : W.t list)
  | Lloop { params; body; args; label = loop_label; type_ } ->
      Lst.iter params (fun param -> ctx +> param);
      let break_label = Label.to_wasm_label_break loop_label in
      let label = Label.to_wasm_label_loop loop_label in
      let args = gos_non_tail args [] in
      let no_value = type_ = I32_Unit in
      let break_used = ref false in
      let new_loop_info =
        Label.Map.add ctx.loop_info loop_label { params; break_used }
      in
      let new_ctx = { ctx with loop_info = new_loop_info } in
      let body =
        compileExpr ~global_ctx ~ctx:new_ctx body [] ~tail:false ~type_defs
      in
      let body =
        if no_value then
          List.append
            (body : W.t list)
            ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
        else body
      in
      let result =
        if no_value then []
        else
          ([
             List
               (List.cons
                  (Atom "result" : W.t)
                  ([ ltype_to_sexp type_ ] : W.t list));
           ]
            : W.t list)
      in
      let set_params =
        Lst.fold_left params [] (fun acc param ->
            List.cons
              (List
                 (List.cons
                    (Atom "local.set" : W.t)
                    ([ Atom (id_to_string param) ] : W.t list))
                : W.t)
              (acc : W.t list))
      in
      let loop =
        if !Basic_config.use_block_params then
          let param_types =
            Lst.map params (fun param ->
                let t = Transl_type.ltype_to_sexp (Ident.get_type param) in
                (List (List.cons (Atom "param" : W.t) ([ t ] : W.t list)) : W.t))
          in
          List.append
            (args : W.t list)
            ([
               List
                 (List.cons
                    (Atom "loop" : W.t)
                    (List.cons
                       (Atom label : W.t)
                       (List.append
                          (param_types : W.t list)
                          (List.append
                             (result : W.t list)
                             (List.append
                                (set_params : W.t list)
                                (body : W.t list))))));
             ]
              : W.t list)
        else
          List.append
            (args : W.t list)
            (List.append
               (set_params : W.t list)
               ([
                  List
                    (List.cons
                       (Atom "loop" : W.t)
                       (List.cons
                          (Atom label : W.t)
                          (List.append (result : W.t list) (body : W.t list))));
                ]
                 : W.t list))
      in
      let loop =
        if !break_used then
          ([
             List
               (List.cons
                  (Atom "block" : W.t)
                  (List.cons
                     (Atom break_label : W.t)
                     (List.append (result : W.t list) (loop : W.t list))));
           ]
            : W.t list)
        else loop
      in
      if no_value then
        List.append
          (loop : W.t list)
          (List.cons
             (List
                (List.cons (Atom "i32.const" : W.t) ([ Atom "0" ] : W.t list))
               : W.t)
             (rest : W.t list))
      else List.append (loop : W.t list) (rest : W.t list)

let compileFunc ~(global_ctx : Global_ctx2.t) (top : Clam.top_func_item)
    ~type_defs : func list =
  let addr = top.binder in
  let ({ body = fn_body; params = fn_params; return_type_; _ } : Clam.fn) =
    top.fn
  in
  let ctx =
    {
      locals = ref Ident_set.empty;
      loop_info = Label.Map.empty;
      join_points = Join.Hash.create 17;
    }
  in
  let body = compileExpr ~ctx ~global_ctx fn_body [] ~tail:true ~type_defs in
  let locals =
    Ident_set.map_to_list !(ctx.locals) (fun local ->
        let ty = Ident.get_type local in
        let source_name =
          Wasm_util.compile_source_name (Ident.source_name local)
        in
        let source_type = compile_source_type ty in
        (List
           (List.cons
              (Atom "local" : W.t)
              (List.cons
                 (Atom (id_to_string local) : W.t)
                 (List.append
                    (source_name : W.t list)
                    (List.append
                       (source_type : W.t list)
                       ([ ltype_to_sexp ty ] : W.t list)))))
          : W.t))
  in
  let params =
    Lst.map fn_params (fun x ->
        let ty = Ident.get_type x in
        let source_name = Wasm_util.compile_source_name (Ident.source_name x) in
        let source_type = compile_source_type ty in
        (List
           (List.cons
              (Atom "param" : W.t)
              (List.cons
                 (Atom (id_to_string x) : W.t)
                 (List.append
                    (source_name : W.t list)
                    (List.append
                       (source_type : W.t list)
                       ([ ltype_to_sexp ty ] : W.t list)))))
          : W.t))
  in
  let result =
    Lst.map return_type_ (fun x : W.t ->
        List (List.cons (Atom "result" : W.t) ([ ltype_to_sexp x ] : W.t list)))
  in
  let export =
    match top.fn_kind_ with
    | Top_pub s ->
        let s = Basic_strutil.esc_quote s in
        ([ List (List.cons (Atom "export" : W.t) ([ Atom s ] : W.t list)) ]
          : W.t list)
    | Top_private -> []
  in
  let source_name = Wasm_util.compile_fn_source_name top.binder in
  let prologue_end = Wasm_util.compile_prologue_end () in
  let type_ =
    match top.tid with
    | None -> []
    | Some tid ->
        ([
           List
             (List.cons
                (Atom "type" : W.t)
                ([ Atom (tid_to_string tid) ] : W.t list));
         ]
          : W.t list)
  in
  match return_type_ with
  | I32_Unit :: [] when export <> [] ->
      let name_wrapper =
        Basic_fn_address.fresh (Basic_fn_address.to_string addr ^ "_wrapper")
      in
      let args =
        Lst.mapi params (fun i _ ->
            let i = Int.to_string i in
            (List (List.cons (Atom "local.get" : W.t) ([ Atom i ] : W.t list))
              : W.t))
      in
      List.cons
        (List
           (List.cons
              (Atom "func" : W.t)
              (List.cons
                 (Atom (addr_to_string name_wrapper) : W.t)
                 (List.append
                    (export : W.t list)
                    (List.append
                       (params : W.t list)
                       (List.cons
                          (List
                             (List.cons
                                (Atom "call" : W.t)
                                (List.cons
                                   (Atom (addr_to_string addr) : W.t)
                                   (args : W.t list)))
                            : W.t)
                          ([ List ([ Atom "drop" ] : W.t list) ] : W.t list))))))
          : W.t)
        ([
           List
             (List.cons
                (Atom "func" : W.t)
                (List.cons
                   (Atom (addr_to_string addr) : W.t)
                   (List.append
                      (source_name : W.t list)
                      (List.append
                         (type_ : W.t list)
                         (List.append
                            (params : W.t list)
                            (List.append
                               (result : W.t list)
                               (List.append
                                  (locals : W.t list)
                                  (List.append
                                     (prologue_end : W.t list)
                                     (body : W.t list)))))))));
         ]
          : W.t list)
  | _ ->
      ([
         List
           (List.cons
              (Atom "func" : W.t)
              (List.cons
                 (Atom (addr_to_string addr) : W.t)
                 (List.append
                    (source_name : W.t list)
                    (List.append
                       (export : W.t list)
                       (List.append
                          (type_ : W.t list)
                          (List.append
                             (params : W.t list)
                             (List.append
                                (result : W.t list)
                                (List.append
                                   (locals : W.t list)
                                   (List.append
                                      (prologue_end : W.t list)
                                      (body : W.t list))))))))));
       ]
        : W.t list)

let compile (prog : Clam.prog) : W.t list =
  let global_ctx = Global_ctx2.create () in
  let fns : W.t list =
    Lst.concat_map prog.fns (fun top ->
        compileFunc ~global_ctx top ~type_defs:prog.type_defs)
  in
  let compile_expr addr body =
    let ctx =
      {
        locals = ref Ident_set.empty;
        loop_info = Label.Map.empty;
        join_points = Join.Hash.create 17;
      }
    in
    let body =
      compileExpr ~ctx ~global_ctx body
        ([ List ([ Atom "drop" ] : W.t list) ] : W.t list)
        ~tail:false ~type_defs:prog.type_defs
    in
    let locals =
      Ident_set.map_to_list !(ctx.locals) (fun local ->
          let ty = Ident.get_type local in
          let source_name =
            Wasm_util.compile_source_name (Ident.source_name local)
          in
          let source_type = compile_source_type ty in
          (List
             (List.cons
                (Atom "local" : W.t)
                (List.cons
                   (Atom (id_to_string local) : W.t)
                   (List.append
                      (source_name : W.t list)
                      (List.append
                         (source_type : W.t list)
                         ([ ltype_to_sexp ty ] : W.t list)))))
            : W.t))
    in
    let prologue_end = Wasm_util.compile_prologue_end () in
    (List
       (List.cons
          (Atom "func" : W.t)
          (List.cons
             (Atom (addr_to_string addr) : W.t)
             (List.append
                (locals : W.t list)
                (List.append (prologue_end : W.t list) (body : W.t list)))))
      : W.t)
  in
  let main =
    match prog.main with
    | Some main_code ->
        let main = Basic_fn_address.main () in
        let main_code = compile_expr main main_code in
        List.cons
          (main_code : W.t)
          ([
             List
               (List.cons
                  (Atom "export" : W.t)
                  (List.cons
                     (Atom "\"_start\"" : W.t)
                     ([
                        List
                          (List.cons
                             (Atom "func" : W.t)
                             ([ Atom (addr_to_string main) ] : W.t list));
                      ]
                       : W.t list)));
           ]
            : W.t list)
    | None -> []
  in
  let init = Basic_fn_address.init () in
  let init_code = compile_expr init prog.init in
  let types = Transl_type.compile_group_type_defs prog.type_defs in
  let data_section = Global_ctx2.compile_to_data global_ctx in
  let table = Global_ctx2.compile_func_ref_declare global_ctx in
  let custom_imports = Global_ctx2.compile_imports global_ctx in
  let inline_wasm = Global_ctx2.compile_inlines global_ctx in
  let runtime =
    match
      if !Basic_config.use_js_builtin_string then
        Runtime_gc_js_string_api.runtime_gc_sexp
      else Runtime_gc.runtime_gc_sexp
    with
    | (List (Atom "module" :: body) : W.t) -> body
    | _ -> assert false
  in
  let globals =
    Lst.map prog.globals (fun (x, initial) ->
        match initial with
        | Some (C_bool false) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons (Atom "i32" : W.t) ([ i32_zero ] : W.t list))))
              : W.t)
        | Some (C_bool true) ->
            let i = 1 in
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i32" : W.t)
                        ([ i32_to_sexp i ] : W.t list))))
              : W.t)
        | Some (C_char c) ->
            let i = Uchar.to_int c in
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i32" : W.t)
                        ([ i32_to_sexp i ] : W.t list))))
              : W.t)
        | Some (C_int { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i32" : W.t)
                        ([ int32_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_int64 { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i64" : W.t)
                        ([ i64_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_uint { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i32" : W.t)
                        ([ u32_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_uint64 { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "i64" : W.t)
                        ([ u64_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_double { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "f64" : W.t)
                        ([ f64_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_float { v; _ }) ->
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (Atom "f32" : W.t)
                        ([ f32_to_sexp v ] : W.t list))))
              : W.t)
        | Some (C_string s) ->
            let v = Global_ctx2.compile_string_literal ~global_ctx s in
            (List
               (List.cons
                  (Atom "global" : W.t)
                  (List.cons
                     (Atom (id_to_string x) : W.t)
                     (List.cons
                        (List
                           (List.cons
                              (Atom "ref" : W.t)
                              ([ Atom "extern" ] : W.t list))
                          : W.t)
                        ([ v ] : W.t list))))
              : W.t)
        | Some (C_bytes _) | Some (C_bigint _) -> assert false
        | None -> (
            match Ident.get_type x with
            | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
            | I32_Option_Char ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "i32" ] : W.t list))
                              : W.t)
                            ([ i32_zero ] : W.t list))))
                  : W.t)
            | I64 ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "i64" ] : W.t list))
                              : W.t)
                            ([ i64_zero ] : W.t list))))
                  : W.t)
            | F32 ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "f32" ] : W.t list))
                              : W.t)
                            ([ f32_zero ] : W.t list))))
                  : W.t)
            | F64 ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "f64" ] : W.t list))
                              : W.t)
                            ([ f64_zero ] : W.t list))))
                  : W.t)
            | Ref_lazy_init { tid } | Ref { tid } | Ref_nullable { tid } ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([
                                     List
                                       (List.cons
                                          (Atom "ref" : W.t)
                                          (List.cons
                                             (Atom "null" : W.t)
                                             ([ Atom (tid_to_string tid) ]
                                               : W.t list)));
                                   ]
                                    : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom (tid_to_string tid) ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)
            | Ref_extern ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "externref" ] : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom "extern" ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)
            | Ref_string ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([
                                     List
                                       (List.cons
                                          (Atom "ref" : W.t)
                                          (List.cons
                                             (Atom "null" : W.t)
                                             ([ Atom "$moonbit.string" ]
                                               : W.t list)));
                                   ]
                                    : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom "$moonbit.string" ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)
            | Ref_bytes ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([
                                     List
                                       (List.cons
                                          (Atom "ref" : W.t)
                                          (List.cons
                                             (Atom "null" : W.t)
                                             ([ Atom "$moonbit.bytes" ]
                                               : W.t list)));
                                   ]
                                    : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom "$moonbit.bytes" ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)
            | Ref_func ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "funcref" ] : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom "func" ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)
            | Ref_any ->
                (List
                   (List.cons
                      (Atom "global" : W.t)
                      (List.cons
                         (Atom (id_to_string x) : W.t)
                         (List.cons
                            (List
                               (List.cons
                                  (Atom "mut" : W.t)
                                  ([ Atom "anyref" ] : W.t list))
                              : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "ref.null" : W.t)
                                    ([ Atom "any" ] : W.t list));
                             ]
                              : W.t list))))
                  : W.t)))
  in
  let global_ctx_section = Global_ctx2.compile_to_globals global_ctx in
  let tags =
    if !Basic_config.test_mode then
      let tag = Ltype.tag_name in
      List.cons
        (List
           (List.cons
              (Atom "import" : W.t)
              (List.cons
                 (Atom "\"exception\"" : W.t)
                 (List.cons
                    (Atom "\"tag\"" : W.t)
                    ([
                       List
                         (List.cons
                            (Atom "tag" : W.t)
                            ([ Atom tag ] : W.t list));
                     ]
                      : W.t list))))
          : W.t)
        ([
           List
             (List.cons
                (Atom "import" : W.t)
                (List.cons
                   (Atom "\"exception\"" : W.t)
                   (List.cons
                      (Atom "\"throw\"" : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "func" : W.t)
                              ([ Atom "$throw" ] : W.t list));
                       ]
                        : W.t list))));
         ]
          : W.t list)
    else []
  in
  let mem =
    let export =
      match !Basic_config.export_memory_name with
      | Some mem_name ->
          let s = Basic_strutil.esc_quote mem_name in
          ([ List (List.cons (Atom "export" : W.t) ([ Atom s ] : W.t list)) ]
            : W.t list)
      | None -> []
    in
    let import =
      match
        (!Basic_config.import_memory_module, !Basic_config.import_memory_name)
      with
      | Some module_name, Some mem_name ->
          let s1 = Basic_strutil.esc_quote module_name in
          let s2 = Basic_strutil.esc_quote mem_name in
          ([
             List
               (List.cons
                  (Atom "import" : W.t)
                  (List.cons (Atom s1 : W.t) ([ Atom s2 ] : W.t list)));
           ]
            : W.t list)
      | _ -> []
    in
    ([
       List
         (List.cons
            (Atom "memory" : W.t)
            (List.cons
               (Atom "$moonbit.memory" : W.t)
               (List.append
                  (export : W.t list)
                  (List.append (import : W.t list) ([ Atom "1" ] : W.t list)))));
     ]
      : W.t list)
  in
  List.append
    (data_section : W.t list)
    (List.append
       (tags : W.t list)
       (List.append
          (custom_imports : W.t list)
          (List.append
             (mem : W.t list)
             (List.append
                (runtime : W.t list)
                (List.append
                   (types : W.t list)
                   (List.append
                      (table : W.t list)
                      (List.append
                         (globals : W.t list)
                         (List.append
                            (global_ctx_section : W.t list)
                            (List.append
                               (inline_wasm : W.t list)
                               (List.append
                                  (fns : W.t list)
                                  (List.cons
                                     (List
                                        (List.cons
                                           (Atom "start" : W.t)
                                           ([ Atom (addr_to_string init) ]
                                             : W.t list))
                                       : W.t)
                                     (List.cons
                                        (init_code : W.t)
                                        (main : W.t list)))))))))))))
