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


module type S = sig
  type key
  type +'a t

  val empty : 'a t
  val is_empty : 'a t -> bool
  val mem : 'a t -> key -> bool
  val to_sorted_array : 'a t -> (key * 'a) array
  val to_sorted_array_with_f : 'a t -> (key -> 'a -> 'b) -> 'b array
  val add : 'a t -> key -> 'a -> 'a t
  val adjust : 'a t -> key -> ('a option -> 'a) -> 'a t
  val singleton : key -> 'a -> 'a t
  val remove : 'a t -> key -> 'a t
  val disjoint_merge_exn : 'a t -> 'a t -> (key -> 'a -> 'a -> exn) -> 'a t
  val iter : 'a t -> (key -> 'a -> unit) -> unit
  val fold : 'a t -> 'b -> (key -> 'a -> 'b -> 'b) -> 'b
  val for_all : 'a t -> (key -> 'a -> bool) -> bool
  val exists : 'a t -> (key -> 'a -> bool) -> bool
  val cardinal : 'a t -> int
  val bindings : 'a t -> (key * 'a) list
  val keys : 'a t -> key list
  val find_exn : 'a t -> key -> 'a
  val find_opt : 'a t -> key -> 'a option
  val find_default : 'a t -> key -> 'a -> 'a
  val map : 'a t -> ('a -> 'b) -> 'b t
  val mapi : 'a t -> (key -> 'a -> 'b) -> 'b t
  val of_list : (key * 'a) list -> 'a t
  val of_array : (key * 'a) array -> 'a t
  val add_list : (key * 'b) list -> 'b t -> 'b t
  val sexp_of_t : ('a -> S.t) -> 'a t -> S.t
end

module type OrderedType = sig
  type t

  include sig
    [@@@ocaml.warning "-32"]

    val sexp_of_t : t -> S.t
    val compare : t -> t -> int
    val equal : t -> t -> bool
  end
end
