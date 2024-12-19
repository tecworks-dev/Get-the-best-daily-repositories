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


module Path_def = struct
  type t = access list

  and access =
    | Field of int
    | Constr_field of { tag_index : int; arg_index : int }
    | Last_field of int
    | Field_slice
    | Casted_constr of { tag_index : int }
    | Map_elem of { key : Constant.t }
    | Error_constr_field of {
        tag : Basic_constr_info.constr_tag;
        arg_index : int;
      }

  include struct
    let _ = fun (_ : t) -> ()
    let _ = fun (_ : access) -> ()

    let rec equal =
      (fun a__001_ b__002_ ->
         Ppx_base.equal_list
           (fun a__003_ b__004_ -> equal_access a__003_ b__004_)
           a__001_ b__002_
        : t -> t -> bool)

    and equal_access =
      (fun a__005_ b__006_ ->
         if Stdlib.( == ) a__005_ b__006_ then true
         else
           match (a__005_, b__006_) with
           | Field _a__007_, Field _b__008_ ->
               Stdlib.( = ) (_a__007_ : int) _b__008_
           | Field _, _ -> false
           | _, Field _ -> false
           | Constr_field _a__009_, Constr_field _b__010_ ->
               Stdlib.( && )
                 (Stdlib.( = ) (_a__009_.tag_index : int) _b__010_.tag_index)
                 (Stdlib.( = ) (_a__009_.arg_index : int) _b__010_.arg_index)
           | Constr_field _, _ -> false
           | _, Constr_field _ -> false
           | Last_field _a__011_, Last_field _b__012_ ->
               Stdlib.( = ) (_a__011_ : int) _b__012_
           | Last_field _, _ -> false
           | _, Last_field _ -> false
           | Field_slice, Field_slice -> true
           | Field_slice, _ -> false
           | _, Field_slice -> false
           | Casted_constr _a__013_, Casted_constr _b__014_ ->
               Stdlib.( = ) (_a__013_.tag_index : int) _b__014_.tag_index
           | Casted_constr _, _ -> false
           | _, Casted_constr _ -> false
           | Map_elem _a__015_, Map_elem _b__016_ ->
               Constant.equal _a__015_.key _b__016_.key
           | Map_elem _, _ -> false
           | _, Map_elem _ -> false
           | Error_constr_field _a__017_, Error_constr_field _b__018_ ->
               Stdlib.( && )
                 (Basic_constr_info.equal_constr_tag _a__017_.tag _b__018_.tag)
                 (Stdlib.( = ) (_a__017_.arg_index : int) _b__018_.arg_index)
        : access -> access -> bool)

    let _ = equal
    and _ = equal_access

    let rec compare =
      (fun a__019_ b__020_ ->
         Ppx_base.compare_list
           (fun a__021_ b__022_ -> compare_access a__021_ b__022_)
           a__019_ b__020_
        : t -> t -> int)

    and compare_access =
      (fun a__023_ b__024_ ->
         if Stdlib.( == ) a__023_ b__024_ then 0
         else
           match (a__023_, b__024_) with
           | Field _a__025_, Field _b__026_ ->
               Stdlib.compare (_a__025_ : int) _b__026_
           | Field _, _ -> -1
           | _, Field _ -> 1
           | Constr_field _a__027_, Constr_field _b__028_ -> (
               match
                 Stdlib.compare (_a__027_.tag_index : int) _b__028_.tag_index
               with
               | 0 ->
                   Stdlib.compare (_a__027_.arg_index : int) _b__028_.arg_index
               | n -> n)
           | Constr_field _, _ -> -1
           | _, Constr_field _ -> 1
           | Last_field _a__029_, Last_field _b__030_ ->
               Stdlib.compare (_a__029_ : int) _b__030_
           | Last_field _, _ -> -1
           | _, Last_field _ -> 1
           | Field_slice, Field_slice -> 0
           | Field_slice, _ -> -1
           | _, Field_slice -> 1
           | Casted_constr _a__031_, Casted_constr _b__032_ ->
               Stdlib.compare (_a__031_.tag_index : int) _b__032_.tag_index
           | Casted_constr _, _ -> -1
           | _, Casted_constr _ -> 1
           | Map_elem _a__033_, Map_elem _b__034_ ->
               Constant.compare _a__033_.key _b__034_.key
           | Map_elem _, _ -> -1
           | _, Map_elem _ -> 1
           | Error_constr_field _a__035_, Error_constr_field _b__036_ -> (
               match
                 Basic_constr_info.compare_constr_tag _a__035_.tag _b__036_.tag
               with
               | 0 ->
                   Stdlib.compare (_a__035_.arg_index : int) _b__036_.arg_index
               | n -> n)
        : access -> access -> int)

    let _ = compare
    and _ = compare_access

    let rec sexp_of_t =
      (fun x__037_ -> Moon_sexp_conv.sexp_of_list sexp_of_access x__037_
        : t -> S.t)

    and sexp_of_access =
      (function
       | Field arg0__038_ ->
           let res0__039_ = Moon_sexp_conv.sexp_of_int arg0__038_ in
           S.List [ S.Atom "Field"; res0__039_ ]
       | Constr_field
           { tag_index = tag_index__041_; arg_index = arg_index__043_ } ->
           let bnds__040_ = ([] : _ Stdlib.List.t) in
           let bnds__040_ =
             let arg__044_ = Moon_sexp_conv.sexp_of_int arg_index__043_ in
             (S.List [ S.Atom "arg_index"; arg__044_ ] :: bnds__040_
               : _ Stdlib.List.t)
           in
           let bnds__040_ =
             let arg__042_ = Moon_sexp_conv.sexp_of_int tag_index__041_ in
             (S.List [ S.Atom "tag_index"; arg__042_ ] :: bnds__040_
               : _ Stdlib.List.t)
           in
           S.List (S.Atom "Constr_field" :: bnds__040_)
       | Last_field arg0__045_ ->
           let res0__046_ = Moon_sexp_conv.sexp_of_int arg0__045_ in
           S.List [ S.Atom "Last_field"; res0__046_ ]
       | Field_slice -> S.Atom "Field_slice"
       | Casted_constr { tag_index = tag_index__048_ } ->
           let bnds__047_ = ([] : _ Stdlib.List.t) in
           let bnds__047_ =
             let arg__049_ = Moon_sexp_conv.sexp_of_int tag_index__048_ in
             (S.List [ S.Atom "tag_index"; arg__049_ ] :: bnds__047_
               : _ Stdlib.List.t)
           in
           S.List (S.Atom "Casted_constr" :: bnds__047_)
       | Map_elem { key = key__051_ } ->
           let bnds__050_ = ([] : _ Stdlib.List.t) in
           let bnds__050_ =
             let arg__052_ = Constant.sexp_of_t key__051_ in
             (S.List [ S.Atom "key"; arg__052_ ] :: bnds__050_
               : _ Stdlib.List.t)
           in
           S.List (S.Atom "Map_elem" :: bnds__050_)
       | Error_constr_field { tag = tag__054_; arg_index = arg_index__056_ } ->
           let bnds__053_ = ([] : _ Stdlib.List.t) in
           let bnds__053_ =
             let arg__057_ = Moon_sexp_conv.sexp_of_int arg_index__056_ in
             (S.List [ S.Atom "arg_index"; arg__057_ ] :: bnds__053_
               : _ Stdlib.List.t)
           in
           let bnds__053_ =
             let arg__055_ = Basic_constr_info.sexp_of_constr_tag tag__054_ in
             (S.List [ S.Atom "tag"; arg__055_ ] :: bnds__053_
               : _ Stdlib.List.t)
           in
           S.List (S.Atom "Error_constr_field" :: bnds__053_)
        : access -> S.t)

    let _ = sexp_of_t
    and _ = sexp_of_access
  end
end

include Path_def
module Map = Basic_mapf.Make (Path_def)
