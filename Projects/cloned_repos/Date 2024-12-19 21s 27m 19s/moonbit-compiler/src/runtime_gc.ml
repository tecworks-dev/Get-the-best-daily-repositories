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
                    (Atom "$print_i32" : W.t)
                    (List.cons
                       (List
                          (List.cons
                             (Atom "import" : W.t)
                             (List.cons
                                (Atom "\"spectest\"" : W.t)
                                ([ Atom "\"print_i32\"" ] : W.t list)))
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
                       (Atom "$print_i64" : W.t)
                       (List.cons
                          (List
                             (List.cons
                                (Atom "import" : W.t)
                                (List.cons
                                   (Atom "\"spectest\"" : W.t)
                                   ([ Atom "\"print_i64\"" ] : W.t list)))
                            : W.t)
                          ([
                             List
                               (List.cons
                                  (Atom "param" : W.t)
                                  (List.cons
                                     (Atom "$i" : W.t)
                                     ([ Atom "i64" ] : W.t list)));
                           ]
                            : W.t list))))
                : W.t)
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
                             (Atom "$print_f64" : W.t)
                             (List.cons
                                (List
                                   (List.cons
                                      (Atom "import" : W.t)
                                      (List.cons
                                         (Atom "\"spectest\"" : W.t)
                                         ([ Atom "\"print_f64\"" ] : W.t list)))
                                  : W.t)
                                ([
                                   List
                                     (List.cons
                                        (Atom "param" : W.t)
                                        (List.cons
                                           (Atom "$i" : W.t)
                                           ([ Atom "f64" ] : W.t list)));
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
                                                (Atom "struct" : W.t)
                                                ([
                                                   List
                                                     (List.cons
                                                        (Atom "field" : W.t)
                                                        ([ Atom "i32" ]
                                                          : W.t list));
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
                                   (Atom "$moonbit.string" : W.t)
                                   ([
                                      List
                                        (List.cons
                                           (Atom "array" : W.t)
                                           ([
                                              List
                                                (List.cons
                                                   (Atom "mut" : W.t)
                                                   ([ Atom "i16" ] : W.t list));
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
                                      (Atom "$moonbit.bytes" : W.t)
                                      ([
                                         List
                                           (List.cons
                                              (Atom "array" : W.t)
                                              ([
                                                 List
                                                   (List.cons
                                                      (Atom "mut" : W.t)
                                                      ([ Atom "i8" ] : W.t list));
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
                                         (Atom "$moonbit.open_empty_struct"
                                           : W.t)
                                         ([
                                            List
                                              (List.cons
                                                 (Atom "sub" : W.t)
                                                 ([
                                                    List
                                                      ([ Atom "struct" ]
                                                        : W.t list);
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
                                            (Atom "$moonbit.string_pool_type"
                                              : W.t)
                                            ([
                                               List
                                                 (List.cons
                                                    (Atom "array" : W.t)
                                                    ([
                                                       List
                                                         (List.cons
                                                            (Atom "mut" : W.t)
                                                            ([
                                                               List
                                                                 (List.cons
                                                                    (Atom "ref"
                                                                      : W.t)
                                                                    (List.cons
                                                                       (Atom
                                                                          "null"
                                                                         : W.t)
                                                                       ([
                                                                          Atom
                                                                            "$moonbit.string";
                                                                        ]
                                                                         : W.t
                                                                           list)));
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
                                            (Atom "func" : W.t)
                                            (List.cons
                                               (Atom "$moonbit.string_equal"
                                                 : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "param" : W.t)
                                                        (List.cons
                                                           (Atom "$stra" : W.t)
                                                           ([
                                                              List
                                                                (List.cons
                                                                   (Atom "ref"
                                                                     : W.t)
                                                                   ([
                                                                      Atom
                                                                        "$moonbit.string";
                                                                    ]
                                                                     : W.t list));
                                                            ]
                                                             : W.t list)))
                                                    : W.t)
                                                  (List.cons
                                                     (List
                                                        (List.cons
                                                           (Atom "param" : W.t)
                                                           (List.cons
                                                              (Atom "$strb"
                                                                : W.t)
                                                              ([
                                                                 List
                                                                   (List.cons
                                                                      (Atom
                                                                         "ref"
                                                                        : W.t)
                                                                      ([
                                                                         Atom
                                                                           "$moonbit.string";
                                                                       ]
                                                                        : W.t
                                                                          list));
                                                               ]
                                                                : W.t list)))
                                                       : W.t)
                                                     (List.cons
                                                        (List
                                                           (List.cons
                                                              (Atom "result"
                                                                : W.t)
                                                              ([ Atom "i32" ]
                                                                : W.t list))
                                                          : W.t)
                                                        (List.cons
                                                           (List
                                                              (List.cons
                                                                 (Atom "local"
                                                                   : W.t)
                                                                 (List.cons
                                                                    (Atom
                                                                       "$counter"
                                                                      : W.t)
                                                                    ([
                                                                       Atom
                                                                         "i32";
                                                                     ]
                                                                      : W.t list)))
                                                             : W.t)
                                                           (List.cons
                                                              (List
                                                                 (List.cons
                                                                    (Atom
                                                                       "local"
                                                                      : W.t)
                                                                    (List.cons
                                                                       (Atom
                                                                          "$stra_len"
                                                                         : W.t)
                                                                       ([
                                                                          Atom
                                                                            "i32";
                                                                        ]
                                                                         : W.t
                                                                           list)))
                                                                : W.t)
                                                              (List.cons
                                                                 (List
                                                                    (List.cons
                                                                       (Atom
                                                                          "local.set"
                                                                         : W.t)
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "$stra_len"
                                                                            : W
                                                                              .t)
                                                                          ([
                                                                             List
                                                                               (List
                                                                                .cons
                                                                                (Atom
                                                                                "array.len"
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
                                                                                "$stra";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                           ]
                                                                            : W
                                                                              .t
                                                                              list)))
                                                                   : W.t)
                                                                 ([
                                                                    List
                                                                      (List.cons
                                                                         (Atom
                                                                            "if"
                                                                           : W.t)
                                                                         (List
                                                                          .cons
                                                                            (List
                                                                               (List
                                                                                .cons
                                                                                (Atom
                                                                                "result"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                              : W
                                                                                .t)
                                                                            (List
                                                                             .cons
                                                                               (List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "i32.eq"
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
                                                                                "$stra_len";
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
                                                                                "array.len"
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
                                                                                "$strb";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "then"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
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
                                                                                (
                                                                                List
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
                                                                                Atom
                                                                                "i32";
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
                                                                                "result"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
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
                                                                                "$counter";
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
                                                                                "$stra_len";
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
                                                                                "result"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "i32";
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
                                                                                "i32.eq"
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
                                                                                "$stra";
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
                                                                                "$counter";
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
                                                                                "array.get_u"
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
                                                                                "$strb";
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
                                                                                "$counter";
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
                                                                                "$counter"
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
                                                                                "$counter";
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
                                                                                list)))
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "else"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "return"
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
                                                                                "0";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))));
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
                                                                                "else"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "return"
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
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))));
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
                                                                                "else"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "return"
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
                                                                                "0";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))));
                                                                  ]
                                                                   : W.t list)))))))))
                                        : W.t)
                                      (List.cons
                                         (List
                                            (List.cons
                                               (Atom "func" : W.t)
                                               (List.cons
                                                  (Atom "$moonbit.println"
                                                    : W.t)
                                                  (List.cons
                                                     (List
                                                        (List.cons
                                                           (Atom "param" : W.t)
                                                           (List.cons
                                                              (Atom "$str"
                                                                : W.t)
                                                              ([
                                                                 List
                                                                   (List.cons
                                                                      (Atom
                                                                         "ref"
                                                                        : W.t)
                                                                      ([
                                                                         Atom
                                                                           "$moonbit.string";
                                                                       ]
                                                                        : W.t
                                                                          list));
                                                               ]
                                                                : W.t list)))
                                                       : W.t)
                                                     (List.cons
                                                        (List
                                                           (List.cons
                                                              (Atom "local"
                                                                : W.t)
                                                              (List.cons
                                                                 (Atom
                                                                    "$counter"
                                                                   : W.t)
                                                                 ([ Atom "i32" ]
                                                                   : W.t list)))
                                                          : W.t)
                                                        (List.cons
                                                           (List
                                                              (List.cons
                                                                 (Atom "loop"
                                                                   : W.t)
                                                                 (List.cons
                                                                    (Atom
                                                                       "$loop"
                                                                      : W.t)
                                                                    ([
                                                                       List
                                                                         (List
                                                                          .cons
                                                                            (Atom
                                                                               "if"
                                                                              : W
                                                                                .t)
                                                                            (List
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
                                                                                "$counter";
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
                                                                                "array.len"
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
                                                                                "$str";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "call"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$printc"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
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
                                                                                "$str";
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
                                                                                "$counter";
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
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$counter"
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
                                                                                "$counter";
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
                                                                                list))))
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
                                                                      : W.t list)))
                                                             : W.t)
                                                           ([
                                                              List
                                                                (List.cons
                                                                   (Atom "call"
                                                                     : W.t)
                                                                   (List.cons
                                                                      (Atom
                                                                         "$printc"
                                                                        : W.t)
                                                                      ([
                                                                         List
                                                                           (List
                                                                            .cons
                                                                              (Atom
                                                                                "i32.const"
                                                                                :
                                                                                W
                                                                                .t)
                                                                              ([
                                                                                Atom
                                                                                "10";
                                                                               ]
                                                                                :
                                                                                W
                                                                                .t
                                                                                list));
                                                                       ]
                                                                        : W.t
                                                                          list)));
                                                            ]
                                                             : W.t list))))))
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
                                                              (Atom "param"
                                                                : W.t)
                                                              (List.cons
                                                                 (Atom "$len"
                                                                   : W.t)
                                                                 ([ Atom "i32" ]
                                                                   : W.t list)))
                                                          : W.t)
                                                        (List.cons
                                                           (List
                                                              (List.cons
                                                                 (Atom "param"
                                                                   : W.t)
                                                                 (List.cons
                                                                    (Atom "$val"
                                                                      : W.t)
                                                                    ([
                                                                       Atom
                                                                         "i32";
                                                                     ]
                                                                      : W.t list)))
                                                             : W.t)
                                                           (List.cons
                                                              (List
                                                                 (List.cons
                                                                    (Atom
                                                                       "result"
                                                                      : W.t)
                                                                    ([
                                                                       List
                                                                         (List
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
                                                                      : W.t list))
                                                                : W.t)
                                                              ([
                                                                 List
                                                                   (List.cons
                                                                      (Atom
                                                                         "array.new"
                                                                        : W.t)
                                                                      (List.cons
                                                                         (Atom
                                                                            "$moonbit.string"
                                                                           : W.t)
                                                                         (List
                                                                          .cons
                                                                            (List
                                                                               (List
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
                                                                                list))));
                                                               ]
                                                                : W.t list))))))
                                              : W.t)
                                            (List.cons
                                               (List
                                                  (List.cons
                                                     (Atom "func" : W.t)
                                                     (List.cons
                                                        (Atom
                                                           "$moonbit.add_string"
                                                          : W.t)
                                                        (List.cons
                                                           (List
                                                              (List.cons
                                                                 (Atom "param"
                                                                   : W.t)
                                                                 (List.cons
                                                                    (Atom "$x"
                                                                      : W.t)
                                                                    ([
                                                                       List
                                                                         (List
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
                                                                      : W.t list)))
                                                             : W.t)
                                                           (List.cons
                                                              (List
                                                                 (List.cons
                                                                    (Atom
                                                                       "param"
                                                                      : W.t)
                                                                    (List.cons
                                                                       (Atom
                                                                          "$y"
                                                                         : W.t)
                                                                       ([
                                                                          List
                                                                            (List
                                                                             .cons
                                                                               (Atom
                                                                                "ref"
                                                                                :
                                                                                W
                                                                                .t)
                                                                               ([
                                                                                Atom
                                                                                "$moonbit.string";
                                                                                ]
                                                                                :
                                                                                W
                                                                                .t
                                                                                list));
                                                                        ]
                                                                         : W.t
                                                                           list)))
                                                                : W.t)
                                                              (List.cons
                                                                 (List
                                                                    (List.cons
                                                                       (Atom
                                                                          "result"
                                                                         : W.t)
                                                                       ([
                                                                          List
                                                                            (List
                                                                             .cons
                                                                               (Atom
                                                                                "ref"
                                                                                :
                                                                                W
                                                                                .t)
                                                                               ([
                                                                                Atom
                                                                                "$moonbit.string";
                                                                                ]
                                                                                :
                                                                                W
                                                                                .t
                                                                                list));
                                                                        ]
                                                                         : W.t
                                                                           list))
                                                                   : W.t)
                                                                 (List.cons
                                                                    (List
                                                                       (List
                                                                        .cons
                                                                          (Atom
                                                                             "local"
                                                                            : W
                                                                              .t)
                                                                          (List
                                                                           .cons
                                                                             (Atom
                                                                                "$lenx"
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
                                                                                "local"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             (List
                                                                              .cons
                                                                                (Atom
                                                                                "$leny"
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
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
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
                                                                            : W
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
                                                                                "$ptr"
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
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$lenx"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.len"
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
                                                                                "$x";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$leny"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.len"
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
                                                                                "$y";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$len"
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
                                                                                "$lenx";
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
                                                                                "$leny";
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
                                                                                "$ptr"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.new_default"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.string"
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
                                                                                "array.copy"
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
                                                                                "$ptr";
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
                                                                                "$x";
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
                                                                                "$lenx";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))
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
                                                                                "array.copy"
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
                                                                                "$ptr";
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
                                                                                "$lenx";
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
                                                                                "$y";
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
                                                                                "$leny";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))
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
                                                                                "$ptr";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))))))))))
                                                 : W.t)
                                               (List.cons
                                                  (List
                                                     (List.cons
                                                        (Atom "func" : W.t)
                                                        (List.cons
                                                           (Atom
                                                              "$moonbit.string_literal"
                                                             : W.t)
                                                           (List.cons
                                                              (List
                                                                 (List.cons
                                                                    (Atom
                                                                       "param"
                                                                      : W.t)
                                                                    (List.cons
                                                                       (Atom
                                                                          "$index"
                                                                         : W.t)
                                                                       ([
                                                                          Atom
                                                                            "i32";
                                                                        ]
                                                                         : W.t
                                                                           list)))
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
                                                                                "$length"
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
                                                                                "$moonbit.string";
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
                                                                                "local"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$cached"
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
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "null"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.string";
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
                                                                                "$new_string"
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
                                                                                "i32.eqz"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref.is_null"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "local.tee"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$cached"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.get"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$moonbit.string_pool_type"
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
                                                                                "global.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.string_pool";
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
                                                                                "$index";
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
                                                                                list)));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "then"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "return"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "ref.as_non_null"
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
                                                                                "$cached";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
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
                                                                                "local.set"
                                                                                : W
                                                                                .t)
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "$new_string"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                List
                                                                                (
                                                                                List
                                                                                .cons
                                                                                (Atom
                                                                                "array.new_data"
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
                                                                                (Atom
                                                                                "$moonbit.const_data"
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
                                                                                list)))));
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
                                                                                "$moonbit.string_pool_type"
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
                                                                                "global.get"
                                                                                : W
                                                                                .t)
                                                                                ([
                                                                                Atom
                                                                                "$moonbit.string_pool";
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
                                                                                "$index";
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
                                                                                "$new_string";
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
                                                                                "$new_string";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))))))))))))
                                                    : W.t)
                                                  (List.cons
                                                     (List
                                                        (List.cons
                                                           (Atom "func" : W.t)
                                                           (List.cons
                                                              (Atom
                                                                 "$moonbit.unsafe_bytes_blit"
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
                                                                             "$dst"
                                                                            : W
                                                                              .t)
                                                                          ([
                                                                             List
                                                                               (List
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
                                                                                "$dst_offset"
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
                                                                                "param"
                                                                               :
                                                                               W
                                                                               .t)
                                                                             (List
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
                                                                               :
                                                                               W
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
                                                                               :
                                                                               W
                                                                               .t
                                                                               list))))))))
                                                       : W.t)
                                                     ([
                                                        List
                                                          (List.cons
                                                             (Atom "func" : W.t)
                                                             (List.cons
                                                                (Atom
                                                                   "$moonbit.unsafe_bytes_sub_string"
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
                                                                                "$offset"
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
                                                                                "param"
                                                                                :
                                                                                W
                                                                                .t)
                                                                               (List
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
                                                                           : W.t)
                                                                         (List
                                                                          .cons
                                                                            (List
                                                                               (List
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
                                                                                "$moonbit.string";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list))
                                                                              : W
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
                                                                                "$dst";
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list));
                                                                                ]
                                                                                : W
                                                                                .t
                                                                                list)))))))))))))));
                                                      ]
                                                       : W.t list)))))))))))))))))
    : W.t)
