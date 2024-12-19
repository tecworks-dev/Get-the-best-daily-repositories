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
  | I32_Char
  | I32_Bool
  | I32_Unit
  | I32_Byte
  | I32_Tag
  | I32_Option_Char
  | I64
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
     | I32_Char -> S.Atom "I32_Char"
     | I32_Bool -> S.Atom "I32_Bool"
     | I32_Unit -> S.Atom "I32_Unit"
     | I32_Byte -> S.Atom "I32_Byte"
     | I32_Tag -> S.Atom "I32_Tag"
     | I32_Option_Char -> S.Atom "I32_Option_Char"
     | I64 -> S.Atom "I64"
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
       | I32_Char -> Ppx_base.hash_fold_int hsv 1
       | I32_Bool -> Ppx_base.hash_fold_int hsv 2
       | I32_Unit -> Ppx_base.hash_fold_int hsv 3
       | I32_Byte -> Ppx_base.hash_fold_int hsv 4
       | I32_Tag -> Ppx_base.hash_fold_int hsv 5
       | I32_Option_Char -> Ppx_base.hash_fold_int hsv 6
       | I64 -> Ppx_base.hash_fold_int hsv 7
       | F32 -> Ppx_base.hash_fold_int hsv 8
       | F64 -> Ppx_base.hash_fold_int hsv 9
       | Ref _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 10 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_lazy_init _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 11 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_nullable _ir ->
           let hsv = Ppx_base.hash_fold_int hsv 12 in
           let hsv = hsv in
           Ty_ident.hash_fold_t hsv _ir.tid
       | Ref_extern -> Ppx_base.hash_fold_int hsv 13
       | Ref_string -> Ppx_base.hash_fold_int hsv 14
       | Ref_bytes -> Ppx_base.hash_fold_int hsv 15
       | Ref_func -> Ppx_base.hash_fold_int hsv 16
       | Ref_any -> Ppx_base.hash_fold_int hsv 17
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

type fn_sig = { params : t list; ret : t list }

include struct
  let _ = fun (_ : fn_sig) -> ()

  let sexp_of_fn_sig =
    (fun { params = params__019_; ret = ret__021_ } ->
       let bnds__018_ = ([] : _ Stdlib.List.t) in
       let bnds__018_ =
         let arg__022_ = Moon_sexp_conv.sexp_of_list sexp_of_t ret__021_ in
         (S.List [ S.Atom "ret"; arg__022_ ] :: bnds__018_ : _ Stdlib.List.t)
       in
       let bnds__018_ =
         let arg__020_ = Moon_sexp_conv.sexp_of_list sexp_of_t params__019_ in
         (S.List [ S.Atom "params"; arg__020_ ] :: bnds__018_ : _ Stdlib.List.t)
       in
       S.List bnds__018_
      : fn_sig -> S.t)

  let _ = sexp_of_fn_sig

  let equal_fn_sig =
    (fun a__023_ b__024_ ->
       if Stdlib.( == ) a__023_ b__024_ then true
       else
         Stdlib.( && )
           (Ppx_base.equal_list
              (fun a__025_ b__026_ -> equal a__025_ b__026_)
              a__023_.params b__024_.params)
           (Ppx_base.equal_list
              (fun a__027_ b__028_ -> equal a__027_ b__028_)
              a__023_.ret b__024_.ret)
      : fn_sig -> fn_sig -> bool)

  let _ = equal_fn_sig

  let (hash_fold_fn_sig : Ppx_base.state -> fn_sig -> Ppx_base.state) =
   fun hsv arg ->
    let hsv =
      let hsv = hsv in
      Ppx_base.hash_fold_list hash_fold_t hsv arg.params
    in
    Ppx_base.hash_fold_list hash_fold_t hsv arg.ret

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
  | { params; ret = ret :: [] } ->
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
     | Ref_array { elem = elem__030_ } ->
         let bnds__029_ = ([] : _ Stdlib.List.t) in
         let bnds__029_ =
           let arg__031_ = sexp_of_t elem__030_ in
           (S.List [ S.Atom "elem"; arg__031_ ] :: bnds__029_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_array" :: bnds__029_)
     | Ref_struct { fields = fields__033_ } ->
         let bnds__032_ = ([] : _ Stdlib.List.t) in
         let bnds__032_ =
           let arg__034_ =
             Moon_sexp_conv.sexp_of_list
               (fun (arg0__035_, arg1__036_) ->
                 let res0__037_ = sexp_of_t arg0__035_
                 and res1__038_ = Moon_sexp_conv.sexp_of_bool arg1__036_ in
                 S.List [ res0__037_; res1__038_ ])
               fields__033_
           in
           (S.List [ S.Atom "fields"; arg__034_ ] :: bnds__032_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_struct" :: bnds__032_)
     | Ref_late_init_struct { fields = fields__040_ } ->
         let bnds__039_ = ([] : _ Stdlib.List.t) in
         let bnds__039_ =
           let arg__041_ = Moon_sexp_conv.sexp_of_list sexp_of_t fields__040_ in
           (S.List [ S.Atom "fields"; arg__041_ ] :: bnds__039_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_late_init_struct" :: bnds__039_)
     | Ref_constructor { args = args__044_ } ->
         let bnds__042_ = ([] : _ Stdlib.List.t) in
         let bnds__042_ =
           if match args__044_ with [] -> true | _ -> false then bnds__042_
           else
             let arg__050_ =
               (Moon_sexp_conv.sexp_of_list (fun (arg0__045_, arg1__046_) ->
                    let res0__047_ = sexp_of_t arg0__045_
                    and res1__048_ = Moon_sexp_conv.sexp_of_bool arg1__046_ in
                    S.List [ res0__047_; res1__048_ ]))
                 args__044_
             in
             let bnd__049_ = S.List [ S.Atom "args"; arg__050_ ] in
             (bnd__049_ :: bnds__042_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_constructor" :: bnds__042_)
     | Ref_closure_abstract { fn_sig = fn_sig__052_ } ->
         let bnds__051_ = ([] : _ Stdlib.List.t) in
         let bnds__051_ =
           let arg__053_ = sexp_of_fn_sig fn_sig__052_ in
           (S.List [ S.Atom "fn_sig"; arg__053_ ] :: bnds__051_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_closure_abstract" :: bnds__051_)
     | Ref_object { methods = methods__055_ } ->
         let bnds__054_ = ([] : _ Stdlib.List.t) in
         let bnds__054_ =
           let arg__056_ =
             Moon_sexp_conv.sexp_of_list sexp_of_fn_sig methods__055_
           in
           (S.List [ S.Atom "methods"; arg__056_ ] :: bnds__054_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_object" :: bnds__054_)
     | Ref_closure { fn_sig_tid = fn_sig_tid__058_; captures = captures__060_ }
       ->
         let bnds__057_ = ([] : _ Stdlib.List.t) in
         let bnds__057_ =
           let arg__061_ =
             Moon_sexp_conv.sexp_of_list sexp_of_t captures__060_
           in
           (S.List [ S.Atom "captures"; arg__061_ ] :: bnds__057_
             : _ Stdlib.List.t)
         in
         let bnds__057_ =
           let arg__059_ = Ty_ident.sexp_of_t fn_sig_tid__058_ in
           (S.List [ S.Atom "fn_sig_tid"; arg__059_ ] :: bnds__057_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ref_closure" :: bnds__057_)
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
    (fun x__064_ -> Ty_ident.Hash.sexp_of_t sexp_of_def x__064_
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
    (fun { defs = defs__066_; fn_sig_tbl = fn_sig_tbl__068_ } ->
       let bnds__065_ = ([] : _ Stdlib.List.t) in
       let bnds__065_ =
         let arg__069_ =
           FnSigHash.sexp_of_t Ty_ident.sexp_of_t fn_sig_tbl__068_
         in
         (S.List [ S.Atom "fn_sig_tbl"; arg__069_ ] :: bnds__065_
           : _ Stdlib.List.t)
       in
       let bnds__065_ =
         let arg__067_ = sexp_of_type_defs defs__066_ in
         (S.List [ S.Atom "defs"; arg__067_ ] :: bnds__065_ : _ Stdlib.List.t)
       in
       S.List bnds__065_
      : type_defs_with_context -> S.t)

  let _ = sexp_of_type_defs_with_context
end
