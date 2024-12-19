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


type t =
  | Int
  | Uint
  | Char
  | Bool
  | Unit
  | Byte
  | Int64
  | UInt64
  | Float
  | Double

include struct
  let _ = fun (_ : t) -> ()
  let equal = (Stdlib.( = ) : t -> t -> bool)
  let _ = equal

  let (hash_fold_t : Ppx_base.state -> t -> Ppx_base.state) =
    (fun hsv arg ->
       Ppx_base.hash_fold_int hsv
         (match arg with
         | Int -> 0
         | Uint -> 1
         | Char -> 2
         | Bool -> 3
         | Unit -> 4
         | Byte -> 5
         | Int64 -> 6
         | UInt64 -> 7
         | Float -> 8
         | Double -> 9)
      : Ppx_base.state -> t -> Ppx_base.state)

  let _ = hash_fold_t

  let (hash : t -> Ppx_base.hash_value) =
    let func arg =
      Ppx_base.get_hash_value
        (let hsv = Ppx_base.create () in
         hash_fold_t hsv arg)
    in
    fun x -> func x

  let _ = hash

  let sexp_of_t =
    (function
     | Int -> S.Atom "Int"
     | Uint -> S.Atom "Uint"
     | Char -> S.Atom "Char"
     | Bool -> S.Atom "Bool"
     | Unit -> S.Atom "Unit"
     | Byte -> S.Atom "Byte"
     | Int64 -> S.Atom "Int64"
     | UInt64 -> S.Atom "UInt64"
     | Float -> S.Atom "Float"
     | Double -> S.Atom "Double"
      : t -> S.t)

  let _ = sexp_of_t
end

let name = function
  | Int -> "Int"
  | Uint -> "Uint"
  | Char -> "Char"
  | Bool -> "Bool"
  | Unit -> "Unit"
  | Byte -> "Byte"
  | Int64 -> "Int64"
  | UInt64 -> "UInt64"
  | Float -> "Float"
  | Double -> "Double"

let byte_size = function
  | Int | Uint | Char | Bool | Unit | Byte | Float -> 4
  | Int64 | UInt64 | Double -> 8
