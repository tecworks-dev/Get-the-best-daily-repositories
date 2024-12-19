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


let runtime_gc_sexp =
  (List
     (List.cons
        (Atom "module" : W.t)
        (List.cons
           (List
              (List.cons
                 (Atom "func" : W.t)
                 (List.cons
                    (Atom "$printc" : W.t)
                    (List.cons
                       (List
                          (List.cons
                             (Atom "import" : W.t)
                             (List.cons
                                (Atom "\"spectest\"" : W.t)
                                ([ Atom "\"print_char\"" ] : W.t list)))
                         : W.t)
                       ([
                          List
                            (List.cons
                               (Atom "param" : W.t)
                               (List.cons
                                  (Atom "$i" : W.t)
                                  ([ Atom "i32" ] : W.t list)));
                        ]
                         : W.t list))))
             : W.t)
           (List.cons
              (List
                 (List.cons
                    (Atom "func" : W.t)
                    (List.cons
                       (Atom "$moonbit.js_string.cast" : W.t)
                       (List.cons
                          (List
                             (List.cons
                                (Atom "import" : W.t)
                                (List.cons
                                   (Atom "\"wasm:js-string\"" : W.t)
                                   ([ Atom "\"test\"" ] : W.t list)))
                            : W.t)
                          (List.cons
                             (List
                                (List.cons
                                   (Atom "param" : W.t)
                                   ([ Atom "externref" ] : W.t list))
                               : W.t)
                             ([
                                List
                                  (List.cons
                                     (Atom "result" : W.t)
                                     ([ Atom "i32" ] : W.t list));
                              ]
                               : W.t list)))))
                : W.t)
              (List.cons
                 (List
                    (List.cons
                       (Atom "func" : W.t)
                       (List.cons
                          (Atom "$moonbit.js_string.length" : W.t)
                          (List.cons
                             (List
                                (List.cons
                                   (Atom "import" : W.t)
                                   (List.cons
                                      (Atom "\"wasm:js-string\"" : W.t)
                                      ([ Atom "\"length\"" ] : W.t list)))
                               : W.t)
                             (List.cons
                                (List
                                   (List.cons
                                      (Atom "param" : W.t)
                                      ([ Atom "externref" ] : W.t list))
                                  : W.t)
                                ([
                                   List
                                     (List.cons
                                        (Atom "result" : W.t)
                                        ([ Atom "i32" ] : W.t list));
                                 ]
                                  : W.t list)))))
                   : W.t)
                 (List.cons
                    (List
                       (List.cons
                          (Atom "func" : W.t)
                          (List.cons
                             (Atom "$moonbit.js_string.charCodeAt" : W.t)
                             (List.cons
                                (List
                                   (List.cons
                                      (Atom "import" : W.t)
                                      (List.cons
                                         (Atom "\"wasm:js-string\"" : W.t)
                                         ([ Atom "\"charCodeAt\"" ] : W.t list)))
                                  : W.t)
                                (List.cons
                                   (List
                                      (List.cons
                                         (Atom "param" : W.t)
                                         ([ Atom "externref" ] : W.t list))
                                     : W.t)
                                   (List.cons
                                      (List
                                         (List.cons
                                            (Atom "param" : W.t)
                                            ([ Atom "i32" ] : W.t list))
                                        : W.t)
                                      ([
                                         List
                                           (List.cons
                                              (Atom "result" : W.t)
                                              ([ Atom "i32" ] : W.t list));
                                       ]
                                        : W.t list))))))
                      : W.t)
                    (List.cons
                       (List
                          (List.cons
                             (Atom "func" : W.t)
                             (List.cons
                                (Atom "$moonbit.string_equal" : W.t)
                                (List.cons
                                   (List
                                      (List.cons
                                         (Atom "import" : W.t)
                                         (List.cons
                                            (Atom "\"wasm:js-string\"" : W.t)
                                            ([ Atom "\"equals\"" ] : W.t list)))
                                     : W.t)
                                   (List.cons
                                      (List
                                         (List.cons
                                            (Atom "param" : W.t)
                                            ([ Atom "externref" ] : W.t list))
                                        : W.t)
                                      (List.cons
                                         (List
                                            (List.cons
                                               (Atom "param" : W.t)
                                               ([ Atom "externref" ] : W.t list))
                                           : W.t)
                                         ([
                                            List
                                              (List.cons
                                                 (Atom "result" : W.t)
                                                 ([ Atom "i32" ] : W.t list));
                                          ]
                                           : W.t list))))))
                         : W.t)
                       (List.cons
                          (List
                             (List.cons
                                (Atom "func" : W.t)
                                (List.cons
                                   (Atom "$moonbit.add_string" : W.t)
                                   (List.cons
                                      (List
                                         (List.cons
                                            (Atom "import" : W.t)
                                            (List.cons
                                               (Atom "\"wasm:js-string\"" : W.t)
                                               ([ Atom "\"concat\"" ]
                                                 : W.t list)))
                                        : W.t)
                                      (List.cons
                                         (List
                                            (List.cons
                                               (Atom "param" : W.t)
                                               ([ Atom "externref" ] : W.t list))
                                           : W.t)
                                         (List.cons
                                            (List
                                               (List.cons
                                                  (Atom "param" : W.t)
                                                  ([ Atom "externref" ]
                                                    : W.t list))
                                              : W.t)
                                            ([
                                               List
                                                 (List.cons
                                                    (Atom "result" : W.t)
                                                    ([
                                                       List
                                                         (List.cons
                                                            (Atom "ref" : W.t)
                                                            ([ Atom "extern" ]
                                                              : W.t list));
                                                     ]
                                                      : W.t list));
                                             ]
                                              : W.t list))))))
                            : W.t)
                          (List.cons
                             (List
                                (List.cons
                                   (Atom "func" : W.t)
                                   (List.cons
                                      (Atom "$moonbit.js_string.substring"
                                        : W.t)
                                      (List.cons
                                         (List
                                            (List.cons
                                               (Atom "import" : W.t)
                                               (List.cons
                                                  (Atom "\"wasm:js-string\""
                                                    : W.t)
                                                  ([ Atom "\"substring\"" ]
                                                    : W.t list)))
                                           : W.t)
                                         (List.cons
                                            (List
                                               (List.cons
                                                  (Atom "param" : W.t)
                                                  ([ Atom "externref" ]
                                                    : W.t list))
                                              : W.t)
                                            (List.cons
                                               (List
                                                  (List.cons
                                                     (Atom "param" : W.t)
                                                     ([ Atom "i32" ] : W.t list))
                                                 : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "param" : W.t)
                                                        ([ Atom "i32" ]
                                                          : W.t list))
                                                    : W.t)
                                                  ([
                                                     List
                                                       (List.cons
                                                          (Atom "result" : W.t)
                                                          ([
                                                             List
                                                               (List.cons
                                                                  (Atom "ref"
                                                                    : W.t)
                                                                  ([
                                                                     Atom
                                                                       "extern";
                                                                   ]
                                                                    : W.t list));
                                                           ]
                                                            : W.t list));
                                                   ]
                                                    : W.t list)))))))
                               : W.t)
                             (List.cons
                                (List
                                   (List.cons
                                      (Atom "func" : W.t)
                                      (List.cons
                                         (Atom "$moonbit.js_string.fromCharCode"
                                           : W.t)
                                         (List.cons
                                            (List
                                               (List.cons
                                                  (Atom "import" : W.t)
                                                  (List.cons
                                                     (Atom "\"wasm:js-string\""
                                                       : W.t)
                                                     ([
                                                        Atom "\"fromCharCode\"";
                                                      ]
                                                       : W.t list)))
                                              : W.t)
                                            (List.cons
                                               (List
                                                  (List.cons
                                                     (Atom "param" : W.t)
                                                     ([ Atom "i32" ] : W.t list))
                                                 : W.t)
                                               ([
                                                  List
                                                    (List.cons
                                                       (Atom "result" : W.t)
                                                       ([
                                                          List
                                                            (List.cons
                                                               (Atom "ref"
                                                                 : W.t)
                                                               ([
                                                                  Atom "extern";
                                                                ]
                                                                 : W.t list));
                                                        ]
                                                         : W.t list));
                                                ]
                                                 : W.t list)))))
                                  : W.t)
                                (List.cons
                                   (List
                                      (List.cons
                                         (Atom "func" : W.t)
                                         (List.cons
                                            (Atom
                                               "$moonbit.js_string.fromCharCodeArray"
                                              : W.t)
                                            (List.cons
                                               (List
                                                  (List.cons
                                                     (Atom "import" : W.t)
                                                     (List.cons
                                                        (Atom
                                                           "\"wasm:js-string\""
                                                          : W.t)
                                                        ([
                                                           Atom
                                                             "\"fromCharCodeArray\"";
                                                         ]
                                                          : W.t list)))
                                                 : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "param" : W.t)
                                                        ([
                                                           List
                                                             (List.cons
                                                                (Atom "ref"
                                                                  : W.t)
                                                                (List.cons
                                                                   (Atom "null"
                                                                     : W.t)
                                                                   ([
                                                                      Atom
                                                                        "$moonbit.string";
                                                                    ]
                                                                     : W.t list)));
                                                         ]
                                                          : W.t list))
                                                    : W.t)
                                                  (List.cons
                                                     (List
                                                        (List.cons
                                                           (Atom "param" : W.t)
                                                           ([ Atom "i32" ]
                                                             : W.t list))
                                                       : W.t)
                                                     (List.cons
                                                        (List
                                                           (List.cons
                                                              (Atom "param"
                                                                : W.t)
                                                              ([ Atom "i32" ]
                                                                : W.t list))
                                                          : W.t)
                                                        ([
                                                           List
                                                             (List.cons
                                                                (Atom "result"
                                                                  : W.t)
                                                                ([
                                                                   List
                                                                     (List.cons
                                                                        (Atom
                                                                           "ref"
                                                                          : W.t)
                                                                        ([
                                                                           Atom
                                                                             "extern";
                                                                         ]
                                                                          : W.t
                                                                            list));
                                                                 ]
                                                                  : W.t list));
                                                         ]
                                                          : W.t list)))))))
                                     : W.t)
                                   (List.cons
                                      (List
                                         (List.cons
                                            (Atom "func" : W.t)
                                            (List.cons
                                               (Atom "$moonbit.println" : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "import" : W.t)
                                                        (List.cons
                                                           (Atom "\"console\""
                                                             : W.t)
                                                           ([ Atom "\"log\"" ]
                                                             : W.t list)))
                                                    : W.t)
                                                  ([
                                                     List
                                                       (List.cons
                                                          (Atom "param" : W.t)
                                                          (List.cons
                                                             (Atom "$str" : W.t)
                                                             ([
                                                                List
                                                                  (List.cons
                                                                     (Atom "ref"
                                                                       : W.t)
                                                                     ([
                                                                        Atom
                                                                          "extern";
                                                                      ]
                                                                       : W.t
                                                                         list));
                                                              ]
                                                               : W.t list)));
                                                   ]
                                                    : W.t list))))
                                        : W.t)
                                      (List.cons
                                         (List
                                            (List.cons
                                               (Atom "type" : W.t)
                                               (List.cons
                                                  (Atom "$moonbit.enum" : W.t)
                                                  ([
                                                     List
                                                       (List.cons
                                                          (Atom "sub" : W.t)
                                                          ([
                                                             List
                                                               (List.cons
                                                                  (Atom "struct"
                                                                    : W.t)
                                                                  ([
                                                                     List
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "field"
                                                                            : W
                                                                              .t)
                                                                          ([
                                                                             Atom
                                                                               "i32";
                                                                           ]
                                                                            : W
                                                                              .t
                                                                              list));
                                                                   ]
                                                                    : W.t list));
                                                           ]
                                                            : W.t list));
                                                   ]
                                                    : W.t list)))
                                           : W.t)
                                         (List.cons
                                            (List
                                               (List.cons
                                                  (Atom "type" : W.t)
                                                  (List.cons
                                                     (Atom "$moonbit.string"
                                                       : W.t)
                                                     ([
                                                        List
                                                          (List.cons
                                                             (Atom "array"
                                                               : W.t)
                                                             ([
                                                                List
                                                                  (List.cons
                                                                     (Atom "mut"
                                                                       : W.t)
                                                                     ([
                                                                        Atom
                                                                          "i16";
                                                                      ]
                                                                       : W.t
                                                                         list));
                                                              ]
                                                               : W.t list));
                                                      ]
                                                       : W.t list)))
                                              : W.t)
                                            (List.cons
                                               (List
                                                  (List.cons
                                                     (Atom "type" : W.t)
                                                     (List.cons
                                                        (Atom "$moonbit.bytes"
                                                          : W.t)
                                                        ([
                                                           List
                                                             (List.cons
                                                                (Atom "array"
                                                                  : W.t)
                                                                ([
                                                                   List
                                                                     (List.cons
                                                                        (Atom
                                                                           "mut"
                                                                          : W.t)
                                                                        ([
                                                                           Atom
                                                                             "i8";
                                                                         ]
                                                                          : W.t
                                                                            list));
                                                                 ]
                                                                  : W.t list));
                                                         ]
                                                          : W.t list)))
                                                 : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "type" : W.t)
                                                        (List.cons
                                                           (Atom
                                                              "$moonbit.open_empty_struct"
                                                             : W.t)
                                                           ([
                                                              List
                                                                (List.cons
                                                                   (Atom "sub"
                                                                     : W.t)
                                                                   ([
                                                                      List
                                                                        ([
                                                                           Atom
                                                                             "struct";
                                                                         ]
                                                                          : W.t
                                                                            list);
                                                                    ]
                                                                     : W.t list));
                                                            ]
                                                             : W.t list)))
                                                    : W.t)
                                                  (List.cons
                                                     (List
                                                        (List.cons
                                                           (Atom "func" : W.t)
                                                           (List.cons
                                                              (Atom
                                                                 "$moonbit.unsafe_make_string"
                                                                : W.t)
                                                              (List.cons
                                                                 (List
                                                                    (List.cons
                                                                       (Atom
                                                                          "param"
                                                                         : W.t)
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "$len"
                                                                            : W
                                                                              .t)
                                                                          ([
                                                                             Atom
                                                                               "i32";
                                                                           ]
                                                                            : W
                                                                              .t
                                                                              list)))
                                                                   : W.t)
                                                                 (List.cons
                                                                    (List
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "param"
                                                                            : W
                                                                              .t)
                                                                          (List
                                                                           .cons
                                                                             (Atom
                                                                                "$val"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             ([
                                                                                Atom
                                                                                "i32";
                                                                              ]
                                                                               :
                                                                               W
                                                                               .t
                                                                               list)))
                                                                      : W.t)
                                                                    (List.cons
                                                                       (List
                                                                          (List
                                                                           .cons
                                                                             (Atom
                                                                                "result"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "extern";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                              ]
                                                                               :
                                                                               W
                                                                               .t
                                                                               list))
                                                                         : W.t)
                                                                       (List
                                                                        .cons
                                                                          (List
                                                                             (List
                                                                              .cons
                                                                                (Atom
                                                                                "array.new"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.string"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$val";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$len";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))
                                                                            : W
                                                                              .t)
                                                                          (List
                                                                           .cons
                                                                             (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "0";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                               :
                                                                               W
                                                                               .t)
                                                                             (List
                                                                              .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$len";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "call"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.js_string.fromCharCodeArray";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))))))))
                                                       : W.t)
                                                     (List.cons
                                                        (List
                                                           (List.cons
                                                              (Atom "func"
                                                                : W.t)
                                                              (List.cons
                                                                 (Atom
                                                                    "$moonbit.unsafe_bytes_blit"
                                                                   : W.t)
                                                                 (List.cons
                                                                    (List
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "param"
                                                                            : W
                                                                              .t)
                                                                          (List
                                                                           .cons
                                                                             (Atom
                                                                                "$dst"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.bytes";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                              ]
                                                                               :
                                                                               W
                                                                               .t
                                                                               list)))
                                                                      : W.t)
                                                                    (List.cons
                                                                       (List
                                                                          (List
                                                                           .cons
                                                                             (Atom
                                                                                "param"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             (List
                                                                              .cons
                                                                                (Atom
                                                                                "$dst_offset"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                         : W.t)
                                                                       (List
                                                                        .cons
                                                                          (List
                                                                             (List
                                                                              .cons
                                                                                (Atom
                                                                                "param"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$src"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.bytes";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                            : W
                                                                              .t)
                                                                          (List
                                                                           .cons
                                                                             (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "param"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$src_offset"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                               :
                                                                               W
                                                                               .t)
                                                                             (List
                                                                              .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "param"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$length"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.copy"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.bytes"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.bytes"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$dst";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$dst_offset";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$src";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$src_offset";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$length";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))
                                                          : W.t)
                                                        ([
                                                           List
                                                             (List.cons
                                                                (Atom "func"
                                                                  : W.t)
                                                                (List.cons
                                                                   (Atom
                                                                      "$moonbit.unsafe_bytes_sub_string"
                                                                     : W.t)
                                                                   (List.cons
                                                                      (List
                                                                         (List
                                                                          .cons
                                                                            (Atom
                                                                               "param"
                                                                              : W
                                                                                .t)
                                                                            (List
                                                                             .cons
                                                                               (Atom
                                                                                "$src"
                                                                                :
                                                                                W
                                                                                .t)
                                                                               ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.bytes";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                :
                                                                                W
                                                                                .t
                                                                                list)))
                                                                        : W.t)
                                                                      (List.cons
                                                                         (List
                                                                            (List
                                                                             .cons
                                                                               (Atom
                                                                                "param"
                                                                                :
                                                                                W
                                                                                .t)
                                                                               (List
                                                                                .cons
                                                                                (Atom
                                                                                "$offset"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                           : W.t)
                                                                         (List
                                                                          .cons
                                                                            (List
                                                                               (List
                                                                                .cons
                                                                                (Atom
                                                                                "param"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$length"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                              : W
                                                                                .t)
                                                                            (List
                                                                             .cons
                                                                               (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "result"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "extern";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                :
                                                                                W
                                                                                .t)
                                                                               (List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$dst"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.string";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$strlen"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$ch"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$i"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$j"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$strlen"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.shr_s"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$length";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "1";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$dst"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.new"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.string"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "0";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$strlen";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "loop"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$loop"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "if"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.lt_s"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$i";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$strlen";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "then"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$j"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.add"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$offset";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.shl"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$i";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "1";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$ch"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.or"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.get_u"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.bytes"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$src";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$j";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.shl"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.get_u"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.bytes"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$src";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.add"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$j";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "1";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "8";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.string"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$dst";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$i";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$ch";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$i"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.add"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$i";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "1";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "br"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$loop";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                ([
                                                                                Atom
                                                                                "else";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list);
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$dst";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.const"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "0";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$strlen";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "call"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.js_string.fromCharCodeArray";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))))))))))));
                                                         ]
                                                          : W.t list))))))))))))))))))
    : W.t)
