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


let encode s =
  String.init
    (String.length s * 2)
    (fun index ->
      let chr = Char.code s.[index / 2] in
      let value = if index mod 2 = 0 then chr / 16 else chr mod 16 in
      if value >= 10 then Char.chr (value - 10 + Char.code 'a')
      else Char.chr (value + Char.code '0'))
