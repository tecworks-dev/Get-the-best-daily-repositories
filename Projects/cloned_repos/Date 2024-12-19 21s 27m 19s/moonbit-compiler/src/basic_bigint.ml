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


type sign = Neg | Pos
type t = { sign : sign; digits : int array }

let of_string str =
  let radix = 1000_000_000L in
  if String.length str = 0 then raise (Invalid_argument "empty string");
  let sign, left =
    match str.[0] with '-' -> (Neg, 1) | '+' -> (Pos, 1) | _ -> (Pos, 0)
  in
  let str_len = Int.to_float (String.length str) in
  let base, dec_len, left =
    match str.[left] with
    | '0' when String.length str - left > 2 -> (
        match String.sub str left 2 with
        | "0x" | "0X" ->
            ( 16L,
              (str_len -. (Int.to_float left +. 2.0) +. 1.0) /. 0.830,
              left + 2 )
        | "0o" | "0O" ->
            ( 8L,
              (str_len -. (Int.to_float left +. 2.0) +. 1.0) /. 1.107,
              left + 2 )
        | "0b" | "0B" ->
            ( 2L,
              (str_len -. (Int.to_float left +. 2.0) +. 1.0) /. 3.321,
              left + 2 )
        | _ -> (10L, str_len, left))
    | _ -> (10L, str_len, left)
  in
  let digits_len = Float.to_int (Float.ceil (dec_len /. 9.0)) + 1 in
  let digits = Array.make digits_len 0 in
  for i = left to String.length str - 1 do
    if str.[i] <> '_' then
      let n =
        let c = str.[i] in
        match c with
        | ('0' | '1') when base = 2L -> Char.code c - Char.code '0'
        | '0' .. '7' when base = 8L -> Char.code c - Char.code '0'
        | '0' .. '9' when base = 10L || base = 16L ->
            Char.code c - Char.code '0'
        | 'a' .. 'f' when base = 16L -> Char.code c - Char.code 'a' + 10
        | 'A' .. 'F' when base = 16L -> Char.code c - Char.code 'A' + 10
        | _ -> raise (Invalid_argument "invalid character")
      in
      let carry = ref (Int64.of_int n) in
      for j = digits_len - 1 downto 0 do
        carry := Int64.add !carry (Int64.mul base (Int64.of_int digits.(j)));
        digits.(j) <- Int64.to_int (Int64.rem !carry radix);
        carry := Int64.div !carry radix
      done
  done;
  { sign; digits }

let of_string_opt str =
  try Some (of_string str) with Invalid_argument _ -> None

let to_string { sign; digits } =
  let left = ref 0 in
  while !left < Array.length digits && digits.(!left) = 0 do
    incr left
  done;
  let str = ref "" in
  for i = !left to Array.length digits - 1 do
    if !str = "" then str := Printf.sprintf "%d" digits.(i)
    else str := !str ^ Printf.sprintf "%09d" digits.(i)
  done;
  if !str = "" then "0" else match sign with Neg -> "-" ^ !str | Pos -> !str

let compare x y =
  let compare_digits digits1 digits2 =
    match Int.compare (Array.length digits1) (Array.length digits2) with
    | 0 ->
        let rec aux i =
          if i < Array.length digits1 then
            match Int.compare digits1.(i) digits2.(i) with
            | 0 -> aux (i + 1)
            | n -> n
          else 0
        in
        aux 0
    | r -> r
  in
  match (x.sign, y.sign) with
  | Neg, Pos -> -1
  | Neg, Neg -> -compare_digits x.digits y.digits
  | Pos, Pos -> compare_digits x.digits y.digits
  | Pos, Neg -> 1

let equal x y =
  if x.sign <> y.sign then false
  else
    let digits1 = x.digits in
    let digits2 = y.digits in
    match Int.equal (Array.length digits1) (Array.length digits2) with
    | true ->
        let rec aux i =
          if i < Array.length digits1 then
            match Int.equal digits1.(i) digits2.(i) with
            | true -> aux (i + 1)
            | false -> false
          else true
        in
        aux 0
    | false -> false

let zero = of_string "0"
