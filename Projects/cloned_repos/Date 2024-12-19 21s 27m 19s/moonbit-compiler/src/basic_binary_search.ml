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


open Basic_unsafe_external

let search_range (x : int) (a : int array) : int =
  let rec go lower_bound upper_bound =
    if lower_bound > upper_bound then -1
    else
      let m = lower_bound + ((upper_bound - lower_bound) / 2) in
      let x0 = a.!(2 * m) in
      let x1 = a.!((2 * m) + 1) in
      if x < x0 then go lower_bound (m - 1)
      else if x > x1 then go (m + 1) upper_bound
      else m
  in
  go 0 ((Array.length a / 2) - 1)
