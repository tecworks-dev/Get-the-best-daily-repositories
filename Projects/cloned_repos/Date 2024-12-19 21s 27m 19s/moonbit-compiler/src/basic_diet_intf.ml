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


module type ELT = sig
  type t

  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pred : t -> t
  val succ : t -> t
  val sub : t -> t -> t
  val add : t -> t -> t
end

module type INTERVAL_SET = sig
  type elt
  type t

  val empty : t
  val is_empty : t -> bool
  val has_single_element : t -> bool
  val singleton : elt -> elt -> t
  val mem : elt -> t -> bool
  val diff : t -> t -> t
  val iter_range : t -> (elt * elt -> unit) -> unit
  val iter : t -> (elt -> unit) -> unit
end
