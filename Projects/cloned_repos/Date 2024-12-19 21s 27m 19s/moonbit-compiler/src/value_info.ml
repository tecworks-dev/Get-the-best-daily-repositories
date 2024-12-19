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


module Ident = Basic_ident

type value_kind = Normal | Prim of Primitive.prim | Const of Constant.t

include struct
  let _ = fun (_ : value_kind) -> ()

  let sexp_of_value_kind =
    (function
     | Normal -> S.Atom "Normal"
     | Prim arg0__001_ ->
         let res0__002_ = Primitive.sexp_of_prim arg0__001_ in
         S.List [ S.Atom "Prim"; res0__002_ ]
     | Const arg0__003_ ->
         let res0__004_ = Constant.sexp_of_t arg0__003_ in
         S.List [ S.Atom "Const"; res0__004_ ]
      : value_kind -> S.t)

  let _ = sexp_of_value_kind
end

type direct_use_loc =
  | Not_direct_use
  | Explicit_import of Loc.t
  | Implicit_import_all of Loc.t

include struct
  let _ = fun (_ : direct_use_loc) -> ()

  let sexp_of_direct_use_loc =
    (function
     | Not_direct_use -> S.Atom "Not_direct_use"
     | Explicit_import arg0__005_ ->
         let res0__006_ = Loc.sexp_of_t arg0__005_ in
         S.List [ S.Atom "Explicit_import"; res0__006_ ]
     | Implicit_import_all arg0__007_ ->
         let res0__008_ = Loc.sexp_of_t arg0__007_ in
         S.List [ S.Atom "Implicit_import_all"; res0__008_ ]
      : direct_use_loc -> S.t)

  let _ = sexp_of_direct_use_loc
end

let is_no_direct_use = function Not_direct_use -> true | _ -> false

type toplevel = {
  id : Basic_qual_ident.t;
  typ : Stype.t;
  pub : bool;
  kind : value_kind;
      [@sexp_drop_if function Normal -> true | Prim _ | Const _ -> false]
  loc_ : Loc.t;
  doc_ : Docstring.t; [@sexp_drop_if Docstring.is_empty]
  ty_params_ : Tvar_env.t; [@sexp_drop_if Tvar_env.is_empty]
  arity_ : Fn_arity.t option;
      [@sexp_drop_if
        function Some arity -> Fn_arity.is_simple arity | None -> true]
  param_names_ : string list;
  direct_use_loc_ : direct_use_loc; [@sexp_drop_if is_no_direct_use]
}

include struct
  let _ = fun (_ : toplevel) -> ()

  let sexp_of_toplevel =
    (let (drop_if__017_ : value_kind -> Stdlib.Bool.t) = function
       | Normal -> true
       | Prim _ | Const _ -> false
     and (drop_if__024_ : Docstring.t -> Stdlib.Bool.t) = Docstring.is_empty
     and (drop_if__029_ : Tvar_env.t -> Stdlib.Bool.t) = Tvar_env.is_empty
     and (drop_if__034_ : Fn_arity.t option -> Stdlib.Bool.t) = function
       | Some arity -> Fn_arity.is_simple arity
       | None -> true
     and (drop_if__041_ : direct_use_loc -> Stdlib.Bool.t) = is_no_direct_use in
     fun {
           id = id__010_;
           typ = typ__012_;
           pub = pub__014_;
           kind = kind__018_;
           loc_ = loc___021_;
           doc_ = doc___025_;
           ty_params_ = ty_params___030_;
           arity_ = arity___035_;
           param_names_ = param_names___038_;
           direct_use_loc_ = direct_use_loc___042_;
         } ->
       let bnds__009_ = ([] : _ Stdlib.List.t) in
       let bnds__009_ =
         if drop_if__041_ direct_use_loc___042_ then bnds__009_
         else
           let arg__044_ = sexp_of_direct_use_loc direct_use_loc___042_ in
           let bnd__043_ = S.List [ S.Atom "direct_use_loc_"; arg__044_ ] in
           (bnd__043_ :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         let arg__039_ =
           Moon_sexp_conv.sexp_of_list Moon_sexp_conv.sexp_of_string
             param_names___038_
         in
         (S.List [ S.Atom "param_names_"; arg__039_ ] :: bnds__009_
           : _ Stdlib.List.t)
       in
       let bnds__009_ =
         if drop_if__034_ arity___035_ then bnds__009_
         else
           let arg__037_ =
             (Moon_sexp_conv.sexp_of_option Fn_arity.sexp_of_t) arity___035_
           in
           let bnd__036_ = S.List [ S.Atom "arity_"; arg__037_ ] in
           (bnd__036_ :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         if drop_if__029_ ty_params___030_ then bnds__009_
         else
           let arg__032_ = Tvar_env.sexp_of_t ty_params___030_ in
           let bnd__031_ = S.List [ S.Atom "ty_params_"; arg__032_ ] in
           (bnd__031_ :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         if drop_if__024_ doc___025_ then bnds__009_
         else
           let arg__027_ = Docstring.sexp_of_t doc___025_ in
           let bnd__026_ = S.List [ S.Atom "doc_"; arg__027_ ] in
           (bnd__026_ :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         let arg__022_ = Loc.sexp_of_t loc___021_ in
         (S.List [ S.Atom "loc_"; arg__022_ ] :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         if drop_if__017_ kind__018_ then bnds__009_
         else
           let arg__020_ = sexp_of_value_kind kind__018_ in
           let bnd__019_ = S.List [ S.Atom "kind"; arg__020_ ] in
           (bnd__019_ :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         let arg__015_ = Moon_sexp_conv.sexp_of_bool pub__014_ in
         (S.List [ S.Atom "pub"; arg__015_ ] :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         let arg__013_ = Stype.sexp_of_t typ__012_ in
         (S.List [ S.Atom "typ"; arg__013_ ] :: bnds__009_ : _ Stdlib.List.t)
       in
       let bnds__009_ =
         let arg__011_ = Basic_qual_ident.sexp_of_t id__010_ in
         (S.List [ S.Atom "id"; arg__011_ ] :: bnds__009_ : _ Stdlib.List.t)
       in
       S.List bnds__009_
      : toplevel -> S.t)

  let _ = sexp_of_toplevel
end

type t =
  | Local_imm of { id : Ident.t; typ : Stype.t; loc_ : Rloc.t }
  | Local_mut of { id : Ident.t; typ : Stype.t; loc_ : Rloc.t }
  | Toplevel_value of toplevel

include struct
  let _ = fun (_ : t) -> ()

  let sexp_of_t =
    (function
     | Local_imm { id = id__046_; typ = typ__048_; loc_ = loc___050_ } ->
         let bnds__045_ = ([] : _ Stdlib.List.t) in
         let bnds__045_ =
           let arg__051_ = Rloc.sexp_of_t loc___050_ in
           (S.List [ S.Atom "loc_"; arg__051_ ] :: bnds__045_ : _ Stdlib.List.t)
         in
         let bnds__045_ =
           let arg__049_ = Stype.sexp_of_t typ__048_ in
           (S.List [ S.Atom "typ"; arg__049_ ] :: bnds__045_ : _ Stdlib.List.t)
         in
         let bnds__045_ =
           let arg__047_ = Ident.sexp_of_t id__046_ in
           (S.List [ S.Atom "id"; arg__047_ ] :: bnds__045_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Local_imm" :: bnds__045_)
     | Local_mut { id = id__053_; typ = typ__055_; loc_ = loc___057_ } ->
         let bnds__052_ = ([] : _ Stdlib.List.t) in
         let bnds__052_ =
           let arg__058_ = Rloc.sexp_of_t loc___057_ in
           (S.List [ S.Atom "loc_"; arg__058_ ] :: bnds__052_ : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__056_ = Stype.sexp_of_t typ__055_ in
           (S.List [ S.Atom "typ"; arg__056_ ] :: bnds__052_ : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__054_ = Ident.sexp_of_t id__053_ in
           (S.List [ S.Atom "id"; arg__054_ ] :: bnds__052_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "Local_mut" :: bnds__052_)
     | Toplevel_value arg0__059_ ->
         let res0__060_ = sexp_of_toplevel arg0__059_ in
         S.List [ S.Atom "Toplevel_value"; res0__060_ ]
      : t -> S.t)

  let _ = sexp_of_t
end
