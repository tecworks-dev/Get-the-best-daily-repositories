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


type ('a, 'b) t = Nil | Cons of 'a * 'b * ('a, 'b) t

let nil = Nil
let cons a b xs = Cons (a, b, xs)

let rec iter xs f =
  match xs with
  | Nil -> ()
  | Cons (a0, b0, Nil) -> f a0 b0
  | Cons (a0, b0, Cons (a1, b1, Nil)) ->
      f a0 b0;
      f a1 b1
  | Cons (a0, b0, Cons (a1, b1, Cons (a2, b2, Nil))) ->
      f a0 b0;
      f a1 b1;
      f a2 b2
  | Cons (a0, b0, Cons (a1, b1, Cons (a2, b2, Cons (a3, b3, xs)))) ->
      f a0 b0;
      f a1 b1;
      f a2 b2;
      f a3 b3;
      iter xs f
