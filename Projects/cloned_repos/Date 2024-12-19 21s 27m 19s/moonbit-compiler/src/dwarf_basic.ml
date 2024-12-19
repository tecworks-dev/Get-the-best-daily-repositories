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


module Byteseq = Basic_byteseq

let ( ^^ ) = Byteseq.O.( ^^ )

let int_uleb128 = Basic_encoders.int_uleb128
and byte = Basic_encoders.byte
and int_32_le = Basic_encoders.int_32_le
and with_length_preceded = Basic_encoders.with_length_preceded

let dw_form_string = Dwarf_consts.dw_form_string
and dw_form_udata = Dwarf_consts.dw_form_udata
and dw_form_sec_offset = Dwarf_consts.dw_form_sec_offset
and dw_form_exprloc = Dwarf_consts.dw_form_exprloc
and dw_form_ref4 = Dwarf_consts.dw_form_ref4

type relocatable_pc = int ref * int
type relative_offset = Byteseq.marker * Byteseq.marker

type attr_value =
  | String of string
  | Unsigned of int
  | SecOffset of int
  | Exprloc of Byteseq.t
  | Reference of relative_offset

let attr_form_of_value value =
  match value with
  | String _ -> dw_form_string
  | Unsigned _ -> dw_form_udata
  | SecOffset _ -> dw_form_sec_offset
  | Exprloc _ -> dw_form_exprloc
  | Reference _ -> dw_form_ref4

let attr_value = function
  | Unsigned x -> int_uleb128 x
  | SecOffset x -> int_32_le x
  | String x -> Byteseq.of_string x ^^ byte 0x00
  | Exprloc e -> with_length_preceded ~f:int_uleb128 e
  | Reference (base, target) ->
      Byteseq.deferred 4 (fun _ ->
          int_32_le
            (Byteseq.get_transitive_offset target
            - Byteseq.get_transitive_offset base)
          |> Byteseq.to_string)

let absolute_pc_of ((base, offset) : relocatable_pc) = base.contents + offset
let list l f = Basic_lst.fold_left l Byteseq.empty (fun t x -> t ^^ f x)
let count_list l f = int_uleb128 (List.length l) ^^ list l f
