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


module Ty_ident = Basic_ty_ident
module Hashf = Basic_hashf

type t =
  | I32_Int
  | U32
  | I32_Char
  | I32_Bool
  | I32_Unit
  | I32_Byte
  | I32_Tag
  | I32_Option_Char
  | I64
  | U64
  | F32
  | F64
  | Ref of { tid : Ty_ident.t }
  | Ref_lazy_init of { tid : Ty_ident.t }
  | Ref_nullable of { tid : Ty_ident.t }
  | Ref_extern
  | Ref_string
  | Ref_bytes
  | Ref_func
  | Ref_any

include struct
  let _ = fun (_ : t) -> ()

  let sexp_of_t =
    (function
     | I32_Int -> S.Atom "I32_Int"
     | U32 -> S.Atom "U32"
     | I32_Char -> S.Atom "I32_Char"
     | I32_Bool -> S.Atom "I32_Bool"
     | I32_Unit -> S.Atom "I32_Unit"
     | I32_Byte -> S.Atom "I32_Byte"
     | I32_Tag -> S.Atom "I32_Tag"
     | I32_Option_Char -> S.Atom "I32_Option_Char"
     | I64 -> S.Atom "I64"
     | U64 -> S.Atom "U64"
     | F32 -> S.Atom "F32"
     | F64 -> S.Atom "F64"
     | Ref { tid = tid__002_ } ->
         let bnds__001_ = ([] : _ Stdlib.List.t) in
         let bnds__001_ =
           let arg__003_ = Ty_ident.sexp_of_t tid__002_ in
           (S.List [ S.Atom "tid"; arg__003_ ] :: bnds__001_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref" :: bnds__001_)
     | Ref_lazy_init { tid = tid__005_ } ->
         let bnds__004_ = ([] : _ Stdlib.List.t) in
         let bnds__004_ =
           let arg__006_ = Ty_ident.sexp_of_t tid__005_ in
           (S.List [ S.Atom "tid"; arg__006_ ] :: bnds__004_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_lazy_init" :: bnds__004_)
     | Ref_nullable { tid = tid__008_ } ->
         let bnds__007_ = ([] : _ Stdlib.List.t) in
         let bnds__007_ =
           let arg__009_ = Ty_ident.sexp_of_t tid__008_ in
           (S.List [ S.Atom "tid"; arg__009_ ] :: bnds__007_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_nullable" :: bnds__007_)
     | Ref_extern -> S.Atom "Ref_extern"
     | Ref_string -> S.Atom "Ref_string"
     | Ref_bytes -> S.Atom "Ref_bytes"
     | Ref_func -> S.Atom "Ref_func"
     | Ref_any -> S.Atom "Ref_any"
      : t -> S.t)

  let _ = sexp_of_t

  let equal =
    (fun a__010_ b__011_ ->
       if Stdlib.( == ) a__010_ b__011_ then true
       else
         match (a__010_, b__011_) with
         | I32_Int, I32_Int -> true
         | I32_Int, _ -> false
         | _, I32_Int -> false
         | U32, U32 -> true
         | U32, _ -> false
         | _, U32 -> false
         | I32_Char, I32_Char -> true
         | I32_Char, _ -> false
         | _, I32_Char -> false
         | I32_Bool, I32_Bool -> true
         | I32_Bool, _ -> false
         | _, I32_Bool -> false
         | I32_Unit, I32_Unit -> true
         | I32_Unit, _ -> false
         | _, I32_Unit -> false
         | I32_Byte, I32_Byte -> true
         | I32_Byte, _ -> false
         | _, I32_Byte -> false
         | I32_Tag, I32_Tag -> true
         | I32_Tag, _ -> false
         | _, I32_Tag -> false
         | I32_Option_Char, I32_Option_Char -> true
         | I32_Option_Char, _ -> false
         | _, I32_Option_Char -> false
         | I64, I64 -> true
         | I64, _ -> false
         | _, I64 -> false
         | U64, U64 -> true
         | U64, _ -> false
         | _, U64 -> false
         | F32, F32 -> true
         | F32, _ -> false
         | _, F32 -> false
         | F64, F64 -> true
         | F64, _ -> false
         | _, F64 -> false
         | Ref _a__012_, Ref _b__013_ ->
             Ty_ident.equal _a__012_.tid _b__013_.tid
         | Ref _, _ -> false
         | _, Ref _ -> false
         | Ref_lazy_init _a__014_, Ref_lazy_init _b__015_ ->
             Ty_ident.equal _a__014_.tid _b__015_.tid
         | Ref_lazy_init _, _ -> false
         | _, Ref_lazy_init _ -> false
         | Ref_nullable _a__016_, Ref_nullable _b__017_ ->
             Ty_ident.equal _a__016_.tid _b__017_.tid
         | Ref_nullable _, _ -> false
         | _, Ref_nullable _ -> false
         | Ref_extern, Ref_extern -> true
         | Ref_extern, _ -> false
         | _, Ref_extern -> false
         | Ref_string, Ref_string -> true
         | Ref_string, _ -> false
         | _, Ref_string -> false
         | Ref_bytes, Ref_bytes -> true
         | Ref_bytes, _ -> false
         | _, Ref_bytes -> false
         | Ref_func, Ref_func -> true
         | Ref_func, _ -> false
         | _, Ref_func -> false
         | Ref_any, Ref_any -> true
      : t -> t -> bool)

  let _ = equal

  let (hash_fold_t : Ppx_base.state -> t -> Ppx_base.state) =
    (fun hsv arg ->
       match arg with
       | I32_Int -> Ppx_base.hash_fold_int hsv 0
       | U32 -> Ppx_base.hash_fold_int hsv 1
       | I32_Char -> Ppx_base.hash_fold_int hsv 2
       | I32_Bool -> Ppx_base.hash_fold_int hsv 3
       | I32_Unit -> Ppx_base.hash_fold_int hsv 4
       | I32_Byte -> Ppx_base.hash_fold_int hsv 5
       | I32_Tag -> Ppx_base.hash_fold_int hsv 6
       | I32_Option_Char -> Ppx_base.hash_fold_int hsv 7
       | I64 -> Ppx_base.hash_fold_int hsv 8
       | U64 -> Ppx_base.hash_fold_int hsv 9
       | F32 -> Ppx_base.hash_fold_int hsv 10
       | F64 -> Ppx_base.hash_fold_int hsv 11
       | Ref _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 12 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_lazy_init _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 13 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_nullable _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 14 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_extern -> Ppx_base.hash_fold_int hsv 15
       | Ref_string -> Ppx_base.hash_fold_int hsv 16
       | Ref_bytes -> Ppx_base.hash_fold_int hsv 17
       | Ref_func -> Ppx_base.hash_fold_int hsv 18
       | Ref_any -> Ppx_base.hash_fold_int hsv 19
      : Ppx_base.state -> t -> Ppx_base.state)

  let _ = hash_fold_t

  let (hash : t -> Ppx_base.hash_value) =
    let func arg =
      Ppx_base.get_hash_value
        (let hsv = Ppx_base.create () in
         hash_fold_t hsv arg)
    in
    fun x -> func x

  let _ = hash
end

type return_type = Ret_single of t | Ret_error of { ok_ty : t; err_ty : t }

include struct
  let _ = fun (_ : return_type) -> ()

  let sexp_of_return_type =
    (function
     | Ret_single arg0__018_ ->
         let res0__019_ = sexp_of_t arg0__018_ in
         S.List [ S.Atom "Ret_single"; res0__019_ ]
     | Ret_error { ok_ty = ok_ty__021_; err_ty = err_ty__023_ } ->
         let bnds__020_ = ([] : _ Stdlib.List.t) in
         let bnds__020_ =
           let arg__024_ = sexp_of_t err_ty__023_ in
           (S.List [ S.Atom "err_ty"; arg__024_ ] :: bnds__020_
             : _ Stdlib.List.t)
         in
         let bnds__020_ =
           let arg__022_ = sexp_of_t ok_ty__021_ in
           (S.List [ S.Atom "ok_ty"; arg__022_ ] :: bnds__020_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ret_error" :: bnds__020_)
      : return_type -> S.t)

  let _ = sexp_of_return_type

  let equal_return_type =
    (fun a__025_ b__026_ ->
       if Stdlib.( == ) a__025_ b__026_ then true
       else
         match (a__025_, b__026_) with
         | Ret_single _a__027_, Ret_single _b__028_ -> equal _a__027_ _b__028_
         | Ret_single _, _ -> false
         | _, Ret_single _ -> false
         | Ret_error _a__029_, Ret_error _b__030_ ->
             Stdlib.( && )
               (equal _a__029_.ok_ty _b__030_.ok_ty)
               (equal _a__029_.err_ty _b__030_.err_ty)
      : return_type -> return_type -> bool)

  let _ = equal_return_type

  let (hash_fold_return_type : Ppx_base.state -> return_type -> Ppx_base.state)
      =
    (fun hsv arg ->
       match arg with
       | Ret_single _a0 ->
           let hsv = Ppx_base.hash_fold_int hsv 0 in
           let hsv = hsv in
           hash_fold_t hsv _a0
       | Ret_error _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 1 in
           let hsv =
             let hsv = hsv in
             hash_fold_t hsv _ir.ok_ty
           in
           hash_fold_t hsv _ir.err_ty
      : Ppx_base.state -> return_type -> Ppx_base.state)

  let _ = hash_fold_return_type

  let (hash_return_type : return_type -> Ppx_base.hash_value) =
    let func arg =
      Ppx_base.get_hash_value
        (let hsv = Ppx_base.create () in
         hash_fold_return_type hsv arg)
    in
    fun x -> func x

  let _ = hash_return_type
end

type fn_sig = { params : t list; ret : return_type }

include struct
  let _ = fun (_ : fn_sig) -> ()

  let sexp_of_fn_sig =
    (fun { params = params__032_; ret = ret__034_ } ->
       let bnds__031_ = ([] : _ Stdlib.List.t) in
       let bnds__031_ =
         let arg__035_ = sexp_of_return_type ret__034_ in
         (S.List [ S.Atom "ret"; arg__035_ ] :: bnds__031_ : _ Stdlib.List.t)
       in
       let bnds__031_ =
         let arg__033_ = Moon_sexp_conv.sexp_of_list sexp_of_t params__032_ in
         (S.List [ S.Atom "params"; arg__033_ ] :: bnds__031_ : _ Stdlib.List.t)
       in
       S.List bnds__031_
      : fn_sig -> S.t)

  let _ = sexp_of_fn_sig

  let equal_fn_sig =
    (fun a__036_ b__037_ ->
       if Stdlib.( == ) a__036_ b__037_ then true
       else
         Stdlib.( && )
           (Ppx_base.equal_list
              (fun a__038_ b__039_ -> equal a__038_ b__039_)
              a__036_.params b__037_.params)
           (equal_return_type a__036_.ret b__037_.ret)
      : fn_sig -> fn_sig -> bool)

  let _ = equal_fn_sig

  let (hash_fold_fn_sig : Ppx_base.state -> fn_sig -> Ppx_base.state) =
   fun hsv arg ->
    let hsv =
      let hsv = hsv in
      Ppx_base.hash_fold_list hash_fold_t hsv arg.params
    in
    hash_fold_return_type hsv arg.ret

  let _ = hash_fold_fn_sig

  let (hash_fn_sig : fn_sig -> Ppx_base.hash_value) =
    let func arg =
      Ppx_base.get_hash_value
        (let hsv = Ppx_base.create () in
         hash_fold_fn_sig hsv arg)
    in
    fun x -> func x

  let _ = hash_fn_sig
end

let sexp_of_fn_sig (t : fn_sig) =
  match t with
  | { params; ret = Ret_single ret } ->
      let params = Basic_lst.map params sexp_of_t in
      let ret = sexp_of_t ret in
      (List
         (List.cons
            (List
               (List.cons
                  (Atom "params" : S.t)
                  ([ List (params : S.t list) ] : S.t list))
              : S.t)
            ([ List (List.cons (Atom "ret" : S.t) ([ ret ] : S.t list)) ]
              : S.t list))
        : S.t)
  | _ -> sexp_of_fn_sig t

type def =
  | Ref_array of { elem : t }
  | Ref_struct of { fields : (t * bool) list }
  | Ref_late_init_struct of { fields : t list }
  | Ref_constructor of { args : (t * bool) list [@list] }
  | Ref_closure_abstract of { fn_sig : fn_sig }
  | Ref_object of { methods : fn_sig list }
  | Ref_closure of { fn_sig_tid : Ty_ident.t; captures : t list }

include struct
  let _ = fun (_ : def) -> ()

  let sexp_of_def =
    (function
     | Ref_array { elem = elem__041_ } ->
         let bnds__040_ = ([] : _ Stdlib.List.t) in
         let bnds__040_ =
           let arg__042_ = sexp_of_t elem__041_ in
           (S.List [ S.Atom "elem"; arg__042_ ] :: bnds__040_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_array" :: bnds__040_)
     | Ref_struct { fields = fields__044_ } ->
         let bnds__043_ = ([] : _ Stdlib.List.t) in
         let bnds__043_ =
           let arg__045_ =
             Moon_sexp_conv.sexp_of_list
               (fun (arg0__046_, arg1__047_) ->
                 let res0__048_ = sexp_of_t arg0__046_
                 and res1__049_ = Moon_sexp_conv.sexp_of_bool arg1__047_ in
                 S.List [ res0__048_; res1__049_ ])
               fields__044_
           in
           (S.List [ S.Atom "fields"; arg__045_ ] :: bnds__043_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_struct" :: bnds__043_)
     | Ref_late_init_struct { fields = fields__051_ } ->
         let bnds__050_ = ([] : _ Stdlib.List.t) in
         let bnds__050_ =
           let arg__052_ = Moon_sexp_conv.sexp_of_list sexp_of_t fields__051_ in
           (S.List [ S.Atom "fields"; arg__052_ ] :: bnds__050_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_late_init_struct" :: bnds__050_)
     | Ref_constructor { args = args__055_ } ->
         let bnds__053_ = ([] : _ Stdlib.List.t) in
         let bnds__053_ =
           if match args__055_ with [] -> true | _ -> false then bnds__053_
           else
             let arg__061_ =
               (Moon_sexp_conv.sexp_of_list (fun (arg0__056_, arg1__057_) ->
                    let res0__058_ = sexp_of_t arg0__056_
                    and res1__059_ = Moon_sexp_conv.sexp_of_bool arg1__057_ in
                    S.List [ res0__058_; res1__059_ ]))
                 args__055_
             in
             let bnd__060_ = S.List [ S.Atom "args"; arg__061_ ] in
             (bnd__060_ :: bnds__053_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_constructor" :: bnds__053_)
     | Ref_closure_abstract { fn_sig = fn_sig__063_ } ->
         let bnds__062_ = ([] : _ Stdlib.List.t) in
         let bnds__062_ =
           let arg__064_ = sexp_of_fn_sig fn_sig__063_ in
           (S.List [ S.Atom "fn_sig"; arg__064_ ] :: bnds__062_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_closure_abstract" :: bnds__062_)
     | Ref_object { methods = methods__066_ } ->
         let bnds__065_ = ([] : _ Stdlib.List.t) in
         let bnds__065_ =
           let arg__067_ =
             Moon_sexp_conv.sexp_of_list sexp_of_fn_sig methods__066_
           in
           (S.List [ S.Atom "methods"; arg__067_ ] :: bnds__065_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_object" :: bnds__065_)
     | Ref_closure { fn_sig_tid = fn_sig_tid__069_; captures = captures__071_ }
       ->
         let bnds__068_ = ([] : _ Stdlib.List.t) in
         let bnds__068_ =
           let arg__072_ =
             Moon_sexp_conv.sexp_of_list sexp_of_t captures__071_
           in
           (S.List [ S.Atom "captures"; arg__072_ ] :: bnds__068_
             : _ Stdlib.List.t)
         in
         let bnds__068_ =
           let arg__070_ = Ty_ident.sexp_of_t fn_sig_tid__069_ in
           (S.List [ S.Atom "fn_sig_tid"; arg__070_ ] :: bnds__068_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_closure" :: bnds__068_)
      : def -> S.t)

  let _ = sexp_of_def
end

let tid_string = Ty_ident.of_string "moonbit.string"
let tid_bytes = Ty_ident.of_string "moonbit.bytes"
let tid_enum = Ty_ident.of_string "moonbit.enum"
let tid_array_i32 = Ty_ident.of_string "moonbit.array_i32"
let tid_array_i64 = Ty_ident.of_string "moonbit.array_i64"
let tid_array_f64 = Ty_ident.of_string "moonbit.array_f64"
let tag_name = "$moonbit.tag"
let ref_enum = Ref { tid = tid_enum }
let ref_array_i32 = Ref { tid = tid_array_i32 }
let ref_array_i64 = Ref { tid = tid_array_i64 }
let ref_array_f64 = Ref { tid = tid_array_f64 }
let def_array_i32 = Ref_array { elem = I32_Int }
let def_array_i64 = Ref_array { elem = I64 }
let def_array_f64 = Ref_array { elem = F64 }

let predefs =
  [
    (tid_array_i32, def_array_i32);
    (tid_array_i64, def_array_i64);
    (tid_array_f64, def_array_f64);
  ]

module FnSigHash = Hashf.Make (struct
  type t = fn_sig

  include struct
    let _ = fun (_ : t) -> ()
    let sexp_of_t = (sexp_of_fn_sig : t -> S.t)
    let _ = sexp_of_t
    let equal = (equal_fn_sig : t -> t -> bool)
    let _ = equal

    let (hash_fold_t : Ppx_base.state -> t -> Ppx_base.state) = hash_fold_fn_sig

    and (hash : t -> Ppx_base.hash_value) =
      let func = hash_fn_sig in
      fun x -> func x

    let _ = hash_fold_t
    and _ = hash
  end
end)

module Hash = Hashf.Make (struct
  type nonrec t = t

  let sexp_of_t = sexp_of_t
  let equal = equal
  let hash = hash
end)

type type_defs = def Ty_ident.Hash.t

include struct
  let _ = fun (_ : type_defs) -> ()

  let sexp_of_type_defs =
    (fun x__075_ -> Ty_ident.Hash.sexp_of_t sexp_of_def x__075_
      : type_defs -> S.t)

  let _ = sexp_of_type_defs
end

type type_defs_with_context = {
  defs : type_defs;
  fn_sig_tbl : Ty_ident.t FnSigHash.t;
}

include struct
  let _ = fun (_ : type_defs_with_context) -> ()

  let sexp_of_type_defs_with_context =
    (fun { defs = defs__077_; fn_sig_tbl = fn_sig_tbl__079_ } ->
       let bnds__076_ = ([] : _ Stdlib.List.t) in
       let bnds__076_ =
         let arg__080_ =
           FnSigHash.sexp_of_t Ty_ident.sexp_of_t fn_sig_tbl__079_
         in
         (S.List [ S.Atom "fn_sig_tbl"; arg__080_ ] :: bnds__076_
           : _ Stdlib.List.t)
       in
       let bnds__076_ =
         let arg__078_ = sexp_of_type_defs defs__077_ in
         (S.List [ S.Atom "defs"; arg__078_ ] :: bnds__076_ : _ Stdlib.List.t)
       in
       S.List bnds__076_
      : type_defs_with_context -> S.t)

  let _ = sexp_of_type_defs_with_context
end
