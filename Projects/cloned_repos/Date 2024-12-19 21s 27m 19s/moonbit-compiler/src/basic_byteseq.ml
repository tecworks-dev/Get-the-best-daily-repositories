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


module Vec = Basic_vec

let max_bytes_length = 128

type marker = { mutable transitive_offset : int }

type t =
  | Bytes of string
  | Concat of int * t * t
  | Marker of marker
  | Deferred of int * (int -> string)

let length = function
  | Bytes x -> String.length x
  | Concat (len, _, _) -> len
  | Marker _ -> 0
  | Deferred (len, _) -> len

let of_string s = Bytes s
let empty = Bytes ""
let is_empty t = length t = 0

let concat a b =
  match (a, b) with
  | Bytes a, Bytes b when String.length a + String.length b <= max_bytes_length
    ->
      Bytes (a ^ b)
  | _ -> Concat (length a + length b, a, b)

let create_marker () = { transitive_offset = -1 }
let get_transitive_offset marker = marker.transitive_offset
let of_marker marker = Marker marker
let deferred len f = Deferred (len, f)

let to_string t =
  let len = length t in
  let bytes = Bytes.make len '\000' in
  let defer = Vec.empty () in
  let rec fill ~base t =
    match t with
    | Bytes s -> Bytes.blit_string s 0 bytes base (String.length s)
    | Concat (_, t1, t2) ->
        fill ~base:(base + length t1) t2;
        fill ~base t1
    | Marker marker -> marker.transitive_offset <- base
    | Deferred (len, f) -> Vec.push defer (base, len, f)
  in
  fill ~base:0 t;
  Vec.iter defer (fun (base, len, f) ->
      let encoded = f len in
      assert (String.length encoded <= len);
      Bytes.blit_string encoded 0 bytes base (String.length encoded));
  Bytes.unsafe_to_string bytes

module O = struct
  let ( ^^ ) = concat
  let ( ^^= ) t x = t := !t ^^ x
end
