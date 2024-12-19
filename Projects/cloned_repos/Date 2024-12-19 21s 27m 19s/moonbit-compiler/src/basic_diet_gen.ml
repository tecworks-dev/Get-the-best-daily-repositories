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


type 'a t = Empty | Node of { x : 'a; y : 'a; l : 'a t; r : 'a t; h : int }

let height = function Empty -> 0 | Node n -> n.h [@@inline]

let create x y l r =
  let h = Int.max (height l) (height r) + 1 in
  Node { x; y; l; r; h }
[@@inline]

let rec bal x y l r =
  let hl = height l in
  let hr = height r in
  if hl > hr + 2 then
    match[@warning "-fragile-match"] l with
    | Node { x = lx; y = ly; l = ll; r = lr; _ } -> (
        if height ll >= height lr then bal lx ly ll (bal x y lr r)
        else
          match[@warning "-fragile-match"] lr with
          | Node { x = lrx; y = lry; l = lrl; r = lrr; _ } ->
              bal lrx lry (bal lx ly ll lrl) (bal x y lrr r)
          | _ -> assert false)
    | _ -> assert false
  else if hr > hl + 2 then
    match[@warning "-fragile-match"] r with
    | Node { x = rx; y = ry; l = rl; r = rr; _ } -> (
        if height rr >= height rl then bal rx ry (bal x y l rl) rr
        else
          match[@warning "-fragile-match"] rl with
          | Node { x = rlx; y = rly; l = rll; r = rlr; _ } ->
              bal rlx rly (bal x y l rll) (bal rx ry rlr rr)
          | _ -> assert false)
    | _ -> assert false
  else create x y l r

let empty = Empty
let is_empty = function Empty -> true | _ -> false

let rec fold f t acc =
  match t with
  | Empty -> acc
  | Node n ->
      let acc = fold f n.l acc in
      let acc = f acc ~lo:n.x ~hi:n.y in
      fold f n.r acc

let rec splitMax = function
  | Node { x; y; l; r = Empty; _ } -> (x, y, l)
  | Node ({ r; _ } as n) ->
      let u, v, r' = splitMax r in
      (u, v, bal n.x n.y n.l r')
  | Empty -> assert false

let rec iter_range t f =
  match t with
  | Empty -> ()
  | Node { x; y; l; r; _ } ->
      iter_range l f;
      f (x, y);
      iter_range r f
