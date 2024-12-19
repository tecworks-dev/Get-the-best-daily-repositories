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

let alphabet =
  Bytes.unsafe_of_string
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

let inv c =
  if 'A' <= c && c <= 'Z' then Char.code c - Char.code 'A'
  else if 'a' <= c && c <= 'z' then Char.code c - Char.code 'a' + 26
  else if '0' <= c && c <= '9' then Char.code c - Char.code '0' + 52
  else if c = '-' then 62
  else if c = '_' then 63
  else raise (Invalid_argument (Printf.sprintf "Base64.inv: `%c`" c))

let pad = '='

let encode s =
  let m1 = 63 lsl 18 in
  let m2 = 63 lsl 12 in
  let m3 = 63 lsl 6 in
  let len = String.length s in
  let n = (len / 3 * 4) + if len mod 3 <> 0 then 4 else 0 in
  let arr = Bytes.unsafe_of_string s in
  let ret = Bytes.create n in
  let a = ref 0 in
  let pos = ref 0 in
  let l = ref len in
  let get = Bytes.get_uint8 arr in
  let add c =
    Bytes.unsafe_blit alphabet c ret !pos 1;
    (incr [@inlined always]) pos
      [@@inline always]
  in
  let padding () =
    Bytes.unsafe_fill ret !pos 1 pad;
    (incr [@inlined always]) pos
      [@@inline always]
  in
  while !l > 2 do
    let d = (get !a lsl 16) lor (get (!a + 1) lsl 8) lor get (!a + 2) in
    add ((d land m1) lsr 18);
    add ((d land m2) lsr 12);
    add ((d land m3) lsr 6);
    add (d land 63);
    a := !a + 3;
    l := !l - 3
  done;
  (match !l with
  | 2 ->
      let d = (get !a lsl 16) lor (get (!a + 1) lsl 8) in
      add ((d land m1) lsr 18);
      add ((d land m2) lsr 12);
      add ((d land m3) lsr 6);
      padding ()
  | 1 ->
      let d = get !a lsl 16 in
      add ((d land m1) lsr 18);
      add ((d land m2) lsr 12);
      padding ();
      padding ()
  | 0 -> ()
  | _ -> assert false);
  Bytes.unsafe_to_string ret

let decode s =
  let len = String.length s in
  if len mod 4 <> 0 then
    raise
      (Invalid_argument
         (Printf.sprintf "Base64.decode: length `%d` not multiple of 4" len))
  else
    let two_bits = 3 in
    let four_bits = 15 in
    let pad_len =
      if s.![len - 2] = pad then 2 else if s.![len - 1] = pad then 1 else 0
    in
    let n = (len / 4 * 3) - pad_len in
    let ret = Bytes.create n in
    let pos = ref 0 in
    let iter = ref 0 in
    let get i = inv s.![i] [@@inline always] in
    let add c =
      Bytes.unsafe_set ret !iter (Char.unsafe_chr c);
      (incr [@inlined always]) iter
        [@@inline always]
    in
    while !pos < len - 4 do
      let c1 = get !pos in
      let c2 = get (!pos + 1) in
      let c3 = get (!pos + 2) in
      let c4 = get (!pos + 3) in
      add ((c1 lsl 2) lor (c2 lsr 4));
      add (((c2 land four_bits) lsl 4) lor (c3 lsr 2));
      add (((c3 land two_bits) lsl 6) lor c4);
      pos := !pos + 4
    done;
    (match pad_len with
    | 0 ->
        let c1 = get !pos in
        let c2 = get (!pos + 1) in
        let c3 = get (!pos + 2) in
        let c4 = get (!pos + 3) in
        add ((c1 lsl 2) lor (c2 lsr 4));
        add (((c2 land four_bits) lsl 4) lor (c3 lsr 2));
        add (((c3 land two_bits) lsl 6) lor c4)
    | 1 ->
        let c1 = get !pos in
        let c2 = get (!pos + 1) in
        let c3 = get (!pos + 2) in
        add ((c1 lsl 2) lor (c2 lsr 4));
        add (((c2 land four_bits) lsl 4) lor (c3 lsr 2))
    | 2 ->
        let c1 = get !pos in
        let c2 = get (!pos + 1) in
        add ((c1 lsl 2) lor (c2 lsr 4))
    | _ -> assert false);
    Bytes.unsafe_to_string ret
