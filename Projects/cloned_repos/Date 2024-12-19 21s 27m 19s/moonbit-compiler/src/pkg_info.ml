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


type mi_view = {
  values : (string * Value_info.toplevel) array;
  external_types : (string * Typedecl_info.t) array;
  external_traits : (string * Trait_decl.t) array;
  external_constrs : Typedecl_info.constructor list Basic_hash_string.t;
  external_type_alias : Typedecl_info.alias Basic_hash_string.t;
  method_env : Method_env.t;
  ext_method_env : Ext_method_env.t;
  trait_impls : Trait_impl.t;
  name : string;
}

include struct
  let _ = fun (_ : mi_view) -> ()

  let sexp_of_mi_view =
    (fun {
           values = values__002_;
           external_types = external_types__008_;
           external_traits = external_traits__014_;
           external_constrs = external_constrs__020_;
           external_type_alias = external_type_alias__022_;
           method_env = method_env__024_;
           ext_method_env = ext_method_env__026_;
           trait_impls = trait_impls__028_;
           name = name__030_;
         } ->
       let bnds__001_ = ([] : _ Stdlib.List.t) in
       let bnds__001_ =
         let arg__031_ = Moon_sexp_conv.sexp_of_string name__030_ in
         (S.List [ S.Atom "name"; arg__031_ ] :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__029_ = Trait_impl.sexp_of_t trait_impls__028_ in
         (S.List [ S.Atom "trait_impls"; arg__029_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__027_ = Ext_method_env.sexp_of_t ext_method_env__026_ in
         (S.List [ S.Atom "ext_method_env"; arg__027_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__025_ = Method_env.sexp_of_t method_env__024_ in
         (S.List [ S.Atom "method_env"; arg__025_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__023_ =
           Basic_hash_string.sexp_of_t Typedecl_info.sexp_of_alias
             external_type_alias__022_
         in
         (S.List [ S.Atom "external_type_alias"; arg__023_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__021_ =
           Basic_hash_string.sexp_of_t
             (Moon_sexp_conv.sexp_of_list Typedecl_info.sexp_of_constructor)
             external_constrs__020_
         in
         (S.List [ S.Atom "external_constrs"; arg__021_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__015_ =
           Moon_sexp_conv.sexp_of_array
             (fun (arg0__016_, arg1__017_) ->
               let res0__018_ = Moon_sexp_conv.sexp_of_string arg0__016_
               and res1__019_ = Trait_decl.sexp_of_t arg1__017_ in
               S.List [ res0__018_; res1__019_ ])
             external_traits__014_
         in
         (S.List [ S.Atom "external_traits"; arg__015_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__009_ =
           Moon_sexp_conv.sexp_of_array
             (fun (arg0__010_, arg1__011_) ->
               let res0__012_ = Moon_sexp_conv.sexp_of_string arg0__010_
               and res1__013_ = Typedecl_info.sexp_of_t arg1__011_ in
               S.List [ res0__012_; res1__013_ ])
             external_types__008_
         in
         (S.List [ S.Atom "external_types"; arg__009_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__003_ =
           Moon_sexp_conv.sexp_of_array
             (fun (arg0__004_, arg1__005_) ->
               let res0__006_ = Moon_sexp_conv.sexp_of_string arg0__004_
               and res1__007_ = Value_info.sexp_of_toplevel arg1__005_ in
               S.List [ res0__006_; res1__007_ ])
             values__002_
         in
         (S.List [ S.Atom "values"; arg__003_ ] :: bnds__001_ : _ Stdlib.List.t)
       in
       S.List bnds__001_
      : mi_view -> S.t)

  let _ = sexp_of_mi_view
end
