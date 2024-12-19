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
  type t

  val create : int -> t
  val clear : t -> unit
  val reset : t -> unit
  val remove : t -> key -> unit
  val add : t -> key -> unit
  val of_array : key array -> t
  val check_add : t -> key -> bool
  val find_or_add : t -> key -> key
  val mem : t -> key -> bool
  val iter : t -> (key -> unit) -> unit
  val to_iter : t -> key Basic_iter.t
  val fold : t -> 'b -> (key -> 'b -> 'b) -> 'b
  val length : t -> int
  val to_list : t -> key list
end
